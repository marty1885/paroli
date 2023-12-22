#include <array>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <espeak-ng/speak_lib.h>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "piper.hpp"
#include "utf8.h"
#include "wavfile.hpp"

#include "rknn-inferer.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

namespace piper {

#ifdef _PIPER_VERSION
// https://stackoverflow.com/questions/47346133/how-to-use-a-define-inside-a-format-string
#define _STR(x) #x
#define STR(x) _STR(x)
const std::string VERSION = STR(_PIPER_VERSION);
#else
const std::string VERSION = "";
#endif

// Maximum value for 16-bit signed WAV sample
const float MAX_WAV_VALUE = 32767.0f;

const std::string instanceName{"piper"};

std::string getVersion() { return VERSION; }

// True if the string is a single UTF-8 codepoint
bool isSingleCodepoint(std::string s) {
  return utf8::distance(s.begin(), s.end()) == 1;
}

// Get the first UTF-8 codepoint of a string
Phoneme getCodepoint(std::string s) {
  utf8::iterator character_iter(s.begin(), s.begin(), s.end());
  return *character_iter;
}

// Load JSON config information for phonemization
void parsePhonemizeConfig(json &configRoot, PhonemizeConfig &phonemizeConfig) {
  // {
  //     "espeak": {
  //         "voice": "<language code>"
  //     },
  //     "phoneme_type": "<espeak or text>",
  //     "phoneme_map": {
  //         "<from phoneme>": ["<to phoneme 1>", "<to phoneme 2>", ...]
  //     },
  //     "phoneme_id_map": {
  //         "<phoneme>": [<id1>, <id2>, ...]
  //     }
  // }

  if (configRoot.contains("espeak")) {
    auto espeakValue = configRoot["espeak"];
    if (espeakValue.contains("voice")) {
      phonemizeConfig.eSpeak.voice = espeakValue["voice"].get<std::string>();
    }
  }

  if (configRoot.contains("phoneme_type")) {
    auto phonemeTypeStr = configRoot["phoneme_type"].get<std::string>();
    if (phonemeTypeStr == "text") {
      phonemizeConfig.phonemeType = TextPhonemes;
    }
  }

  // phoneme to [id] map
  // Maps phonemes to one or more phoneme ids (required).
  if (configRoot.contains("phoneme_id_map")) {
    auto phonemeIdMapValue = configRoot["phoneme_id_map"];
    for (auto &fromPhonemeItem : phonemeIdMapValue.items()) {
      std::string fromPhoneme = fromPhonemeItem.key();
      if (!isSingleCodepoint(fromPhoneme)) {
        std::stringstream idsStr;
        for (auto &toIdValue : fromPhonemeItem.value()) {
          PhonemeId toId = toIdValue.get<PhonemeId>();
          idsStr << toId << ",";
        }

        spdlog::error("\"{}\" is not a single codepoint (ids={})", fromPhoneme,
                      idsStr.str());
        throw std::runtime_error(
            "Phonemes must be one codepoint (phoneme id map)");
      }

      auto fromCodepoint = getCodepoint(fromPhoneme);
      for (auto &toIdValue : fromPhonemeItem.value()) {
        PhonemeId toId = toIdValue.get<PhonemeId>();
        phonemizeConfig.phonemeIdMap[fromCodepoint].push_back(toId);
      }
    }
  }

  // phoneme to [phoneme] map
  // Maps phonemes to one or more other phonemes (not normally used).
  if (configRoot.contains("phoneme_map")) {
    if (!phonemizeConfig.phonemeMap) {
      phonemizeConfig.phonemeMap.emplace();
    }

    auto phonemeMapValue = configRoot["phoneme_map"];
    for (auto &fromPhonemeItem : phonemeMapValue.items()) {
      std::string fromPhoneme = fromPhonemeItem.key();
      if (!isSingleCodepoint(fromPhoneme)) {
        spdlog::error("\"{}\" is not a single codepoint", fromPhoneme);
        throw std::runtime_error(
            "Phonemes must be one codepoint (phoneme map)");
      }

      auto fromCodepoint = getCodepoint(fromPhoneme);
      for (auto &toPhonemeValue : fromPhonemeItem.value()) {
        std::string toPhoneme = toPhonemeValue.get<std::string>();
        if (!isSingleCodepoint(toPhoneme)) {
          throw std::runtime_error(
              "Phonemes must be one codepoint (phoneme map)");
        }

        auto toCodepoint = getCodepoint(toPhoneme);
        (*phonemizeConfig.phonemeMap)[fromCodepoint].push_back(toCodepoint);
      }
    }
  }

} /* parsePhonemizeConfig */

// Load JSON config for audio synthesis
void parseSynthesisConfig(json &configRoot, SynthesisConfig &synthesisConfig) {
  // {
  //     "audio": {
  //         "sample_rate": 22050
  //     },
  //     "inference": {
  //         "noise_scale": 0.667,
  //         "length_scale": 1,
  //         "noise_w": 0.8,
  //         "phoneme_silence": {
  //           "<phoneme>": <seconds of silence>,
  //           ...
  //         }
  //     }
  // }

  if (configRoot.contains("audio")) {
    auto audioValue = configRoot["audio"];
    if (audioValue.contains("sample_rate")) {
      // Default sample rate is 22050 Hz
      synthesisConfig.sampleRate = audioValue.value("sample_rate", 22050);
    }
  }

  if (configRoot.contains("inference")) {
    // Overrides default inference settings
    auto inferenceValue = configRoot["inference"];
    if (inferenceValue.contains("noise_scale")) {
      synthesisConfig.noiseScale = inferenceValue.value("noise_scale", 0.667f);
    }

    if (inferenceValue.contains("length_scale")) {
      synthesisConfig.lengthScale = inferenceValue.value("length_scale", 1.0f);
    }

    if (inferenceValue.contains("noise_w")) {
      synthesisConfig.noiseW = inferenceValue.value("noise_w", 0.8f);
    }

    if (inferenceValue.contains("phoneme_silence")) {
      // phoneme -> seconds of silence to add after
      synthesisConfig.phonemeSilenceSeconds.emplace();
      auto phonemeSilenceValue = inferenceValue["phoneme_silence"];
      for (auto &phonemeItem : phonemeSilenceValue.items()) {
        std::string phonemeStr = phonemeItem.key();
        if (!isSingleCodepoint(phonemeStr)) {
          spdlog::error("\"{}\" is not a single codepoint", phonemeStr);
          throw std::runtime_error(
              "Phonemes must be one codepoint (phoneme silence)");
        }

        auto phoneme = getCodepoint(phonemeStr);
        (*synthesisConfig.phonemeSilenceSeconds)[phoneme] =
            phonemeItem.value().get<float>();
      }

    } // if phoneme_silence

  } // if inference

} /* parseSynthesisConfig */

void parseModelConfig(json &configRoot, ModelConfig &modelConfig) {

  modelConfig.numSpeakers = configRoot["num_speakers"].get<SpeakerId>();

  if (configRoot.contains("speaker_id_map")) {
    if (!modelConfig.speakerIdMap) {
      modelConfig.speakerIdMap.emplace();
    }

    auto speakerIdMapValue = configRoot["speaker_id_map"];
    for (auto &speakerItem : speakerIdMapValue.items()) {
      std::string speakerName = speakerItem.key();
      (*modelConfig.speakerIdMap)[speakerName] =
          speakerItem.value().get<SpeakerId>();
    }
  }

} /* parseModelConfig */

void initialize(PiperConfig &config) {
  if (config.useESpeak) {
    // Set up espeak-ng for calling espeak_TextToPhonemesWithTerminator
    // See: https://github.com/rhasspy/espeak-ng
    spdlog::debug("Initializing eSpeak");
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,
                                   /*buflength*/ 0,
                                   /*path*/ config.eSpeakDataPath.c_str(),
                                   /*options*/ 0);
    if (result < 0) {
      throw std::runtime_error("Failed to initialize eSpeak-ng");
    }

    spdlog::debug("Initialized eSpeak");
  }

  // Load onnx model for libtashkeel
  // https://github.com/mush42/libtashkeel/
  if (config.useTashkeel) {
    spdlog::debug("Using libtashkeel for diacritization");
    if (!config.tashkeelModelPath) {
      throw std::runtime_error("No path to libtashkeel model");
    }

    spdlog::debug("Loading libtashkeel model from {}",
                  config.tashkeelModelPath.value());
    config.tashkeelState = std::make_unique<tashkeel::State>();
    tashkeel::tashkeel_load(config.tashkeelModelPath.value(),
                            *config.tashkeelState);
    spdlog::debug("Initialized libtashkeel");
  }

  spdlog::info("Initialized piper");
}

void terminate(PiperConfig &config) {
  if (config.useESpeak) {
    // Clean up espeak-ng
    spdlog::debug("Terminating eSpeak");
    espeak_Terminate();
    spdlog::debug("Terminated eSpeak");
  }

  spdlog::info("Terminated piper");
}

// Load Onnx model and JSON config file
void loadVoice(PiperConfig &config, std::string modelPath,
               std::string encoderPath, std::string decoderPath,
               std::string modelConfigPath, Voice &voice,
               std::optional<SpeakerId> &speakerId, std::string accelerator) {
  spdlog::debug("Parsing voice config at {}", modelConfigPath);
  std::ifstream modelConfigFile(modelConfigPath);
  voice.configRoot = json::parse(modelConfigFile);

  parsePhonemizeConfig(voice.configRoot, voice.phonemizeConfig);
  parseSynthesisConfig(voice.configRoot, voice.synthesisConfig);
  parseModelConfig(voice.configRoot, voice.modelConfig);

  if (voice.modelConfig.numSpeakers > 1) {
    // Multi-speaker model
    if (speakerId) {
      voice.synthesisConfig.speakerId = speakerId;
    } else {
      // Default speaker
      voice.synthesisConfig.speakerId = 0;
    }
  }

  spdlog::debug("Voice contains {} speaker(s)", voice.modelConfig.numSpeakers);

  voice.encoder.load(encoderPath, accelerator);

  // TODO: Parse wtih std::filesystem
  auto extension = std::filesystem::path(decoderPath).extension();
  if(extension == ".rknn")
      voice.decoder = std::make_unique<RknnDecoderInferer>();
  else
      voice.decoder = std::make_unique<OnnxDecoderInferer>();
  voice.decoder->load(decoderPath, accelerator);

} /* loadVoice */

void OnnxDecoderInferer::load(std::string path, std::string accelerator)
{
    spdlog::debug("Loading decoder onnx model from {}", path);
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                             instanceName.c_str());
    env.DisableTelemetryEvents();
    
    if (accelerator == "cuda") {
      // Use CUDA provider
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      options.AppendExecutionProvider_CUDA(cuda_options);
    }
    
    //options.DisableCpuMemArena();
    //options.DisableMemPattern();
    onnx = Ort::Session(env, path.c_str(), options);
}

std::vector<int16_t> OnnxDecoderInferer::infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const xt::xarray<float>& g)
{
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::vector<Ort::Value> inputTensors;
  // HACK: Fix this later
  const std::array<std::string, 3> paramNames = {"z", "y_mask", "g"};
  for(auto& name : paramNames) {
    const xt::xarray<float>* ptr = nullptr;
    if(name == "z")
      ptr = &z;
    else if(name == "y_mask")
      ptr = &y_mask;
    else if(name == "g")
      ptr = &g;
    else
      throw std::runtime_error("Invalid parameter name");
    auto& arr = *ptr;
    std::vector<int64_t> shape(arr.shape().begin(), arr.shape().end());
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)arr.data(), arr.size(), shape.data(),
        shape.size()));
  }

  std::array<const char *, 3> inputNames = {"z", "y_mask", "g"};
  std::array<const char *, 1> outputNames = {"output"};

  auto startTime = std::chrono::steady_clock::now();
  auto outputTensors = onnx.Run(
      Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
      inputTensors.size(), outputNames.data(), outputNames.size());
  auto endTime = std::chrono::steady_clock::now();

  if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor())) {
    throw std::runtime_error("Invalid output tensors");
  }
  std::vector<int16_t> output;
  output.resize(outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount());
  auto ortOutPtr = outputTensors.front().GetTensorData<float>();
  for(size_t i = 0; i < output.size(); i++) {
      float val = std::min(std::max(ortOutPtr[i], -1.0f), 1.0f);
      output[i] = val * MAX_WAV_VALUE;
  }
  spdlog::debug("Decoder inference took {} seconds", std::chrono::duration<double>(endTime - startTime).count());
  return output;
}


void EncoderInferer::load(std::string path, std::string accelerator)
{
    spdlog::debug("Loading encoder onnx model from {}", path);
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                             instanceName.c_str());
    env.DisableTelemetryEvents();
    
    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.DisableProfiling();
    if (accelerator == "cuda") {
      // Use CUDA provider
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      options.AppendExecutionProvider_CUDA(cuda_options);
    }
    
    // Makes encoder slower
    //options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    onnx = Ort::Session(env, path.c_str(), options);
}

std::map<std::string, xt::xarray<float>> EncoderInferer::infer(const std::vector<int64_t> &phonemeIds,
             int64_t inputLength,
             std::optional<int64_t> sid,
             float noiseScale,
             float lengthScale,
             float noiseW)
{
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // Allocate
  std::vector<int64_t> phonemeIdLengths{(int64_t)phonemeIds.size()};
  std::vector<float> scales{noiseScale,
                            lengthScale,
                            noiseW};

  std::vector<Ort::Value> inputTensors;
  std::vector<int64_t> phonemeIdsShape{1, (int64_t)phonemeIds.size()};
  inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo, (int64_t*)phonemeIds.data(), phonemeIds.size(), phonemeIdsShape.data(),
      phonemeIdsShape.size()));

  std::vector<int64_t> phomemeIdLengthsShape{(int64_t)phonemeIdLengths.size()};
  inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo, phonemeIdLengths.data(), phonemeIdLengths.size(),
      phomemeIdLengthsShape.data(), phomemeIdLengthsShape.size()));

  std::vector<int64_t> scalesShape{(int64_t)scales.size()};
  inputTensors.push_back(
      Ort::Value::CreateTensor<float>(memoryInfo, scales.data(), scales.size(),
                                      scalesShape.data(), scalesShape.size()));

  // Add speaker id.
  // NOTE: These must be kept outside the "if" below to avoid being deallocated.
  std::vector<int64_t> speakerId{sid.value_or(0)};
  std::vector<int64_t> speakerIdShape{(int64_t)speakerId.size()};

  if (sid.has_value()) {
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, speakerId.data(), speakerId.size(), speakerIdShape.data(),
        speakerIdShape.size()));
  }

  // From export_onnx.py
  std::array<const char *, 4> inputNames = {"input", "input_lengths", "scales",
                                            "sid"};
  // TODO: Just use all outputs
  std::array<const char *, 3> outputNames = {"z", "y_mask", "g"};

  // Infer
  auto startTime = std::chrono::steady_clock::now();
  auto outputTensors = onnx.Run(
      Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
      inputTensors.size(), outputNames.data(), outputNames.size());
  auto endTime = std::chrono::steady_clock::now();

  if(outputTensors.size() != outputNames.size())
    throw std::runtime_error("Number of output tensors does not match number of output names");

  std::map<std::string, xt::xarray<float>> output;
  for(int i = 0; i < outputTensors.size(); i++) {
      if(!outputTensors[i].IsTensor())
        throw std::runtime_error("Output tensor is not a tensor");

      xt::xarray<float> arr = xt::adapt(outputTensors[i].GetTensorMutableData<float>(),
                                        outputTensors[i].GetTensorTypeAndShapeInfo().GetShape());
      output[outputNames[i]] = std::move(arr);
  }

  auto inferDuration = std::chrono::duration<double>(endTime - startTime);
  auto inferSeconds = inferDuration.count();
  // clean up
  for (std::size_t i = 0; i < outputTensors.size(); i++) {
    Ort::detail::OrtRelease(outputTensors[i].release());
  }
  for (std::size_t i = 0; i < inputTensors.size(); i++) {
    Ort::detail::OrtRelease(inputTensors[i].release());
  }

  spdlog::debug("Encoder inference took {} seconds", inferSeconds);
  return output;
}

// ----------------------------------------------------------------------------

// Phonemize text and synthesize audio
void textToAudio(PiperConfig &config, Voice &voice, std::string text,
                 std::vector<int16_t> &audioBuffer, SynthesisResult &result,
                 const std::function<void()> &audioCallback,
                 std::optional<size_t> speakerId,
                 std::optional<float> noiseScale,
                 std::optional<float> lengthScale,
                 std::optional<float> noiseW) {

  std::cout << (bool)audioCallback << std::endl;
  std::size_t sentenceSilenceSamples = 0;
  if (voice.synthesisConfig.sentenceSilenceSeconds > 0) {
    sentenceSilenceSamples = (std::size_t)(
        voice.synthesisConfig.sentenceSilenceSeconds *
        voice.synthesisConfig.sampleRate * voice.synthesisConfig.channels);
  }

  if (config.useTashkeel) {
    if (!config.tashkeelState) {
      throw std::runtime_error("Tashkeel model is not loaded");
    }

    spdlog::debug("Diacritizing text with libtashkeel: {}", text);
    text = tashkeel::tashkeel_run(text, *config.tashkeelState);
  }

  // Phonemes for each sentence
  spdlog::debug("Phonemizing text: {}", text);
  std::vector<std::vector<Phoneme>> phonemes;

  if (voice.phonemizeConfig.phonemeType == eSpeakPhonemes) {
    // Use espeak-ng for phonemization
    static std::mutex espeakMutex; // espak-ng is not thread-safe
    std::lock_guard<std::mutex> lock(espeakMutex);
    eSpeakPhonemeConfig eSpeakConfig;
    eSpeakConfig.voice = voice.phonemizeConfig.eSpeak.voice;
    phonemize_eSpeak(text, eSpeakConfig, phonemes);
  } else {
    // Use UTF-8 codepoints as "phonemes"
    CodepointsPhonemeConfig codepointsConfig;
    phonemize_codepoints(text, codepointsConfig, phonemes);
  }

  // Synthesize each sentence independently.
  std::vector<PhonemeId> phonemeIds;
  std::map<Phoneme, std::size_t> missingPhonemes;
  for (auto phonemesIter = phonemes.begin(); phonemesIter != phonemes.end();
       ++phonemesIter) {
    std::vector<Phoneme> &sentencePhonemes = *phonemesIter;

    if (spdlog::should_log(spdlog::level::debug)) {
      // DEBUG log for phonemes
      std::string phonemesStr;
      for (auto phoneme : sentencePhonemes) {
        utf8::append(phoneme, std::back_inserter(phonemesStr));
      }

      spdlog::debug("Converting {} phoneme(s) to ids: {}",
                    sentencePhonemes.size(), phonemesStr);
    }

    std::vector<std::shared_ptr<std::vector<Phoneme>>> phrasePhonemes;
    std::vector<SynthesisResult> phraseResults;
    std::vector<size_t> phraseSilenceSamples;

    // Use phoneme/id map from config
    PhonemeIdConfig idConfig;
    idConfig.phonemeIdMap =
        std::make_shared<PhonemeIdMap>(voice.phonemizeConfig.phonemeIdMap);

    if (voice.synthesisConfig.phonemeSilenceSeconds) {
      // Split into phrases
      std::map<Phoneme, float> &phonemeSilenceSeconds =
          *voice.synthesisConfig.phonemeSilenceSeconds;

      auto currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
      phrasePhonemes.push_back(currentPhrasePhonemes);

      for (auto sentencePhonemesIter = sentencePhonemes.begin();
           sentencePhonemesIter != sentencePhonemes.end();
           sentencePhonemesIter++) {
        Phoneme &currentPhoneme = *sentencePhonemesIter;
        currentPhrasePhonemes->push_back(currentPhoneme);

        if (phonemeSilenceSeconds.count(currentPhoneme) > 0) {
          // Split at phrase boundary
          phraseSilenceSamples.push_back(
              (std::size_t)(phonemeSilenceSeconds[currentPhoneme] *
                            voice.synthesisConfig.sampleRate *
                            voice.synthesisConfig.channels));

          currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
          phrasePhonemes.push_back(currentPhrasePhonemes);
        }
      }
    } else {
      // Use all phonemes
      phrasePhonemes.push_back(
          std::make_shared<std::vector<Phoneme>>(sentencePhonemes));
    }

    // Ensure results/samples are the same size
    while (phraseResults.size() < phrasePhonemes.size()) {
      phraseResults.emplace_back();
    }

    while (phraseSilenceSamples.size() < phrasePhonemes.size()) {
      phraseSilenceSamples.push_back(0);
    }

    // phonemes -> ids -> audio
    for (size_t phraseIdx = 0; phraseIdx < phrasePhonemes.size(); phraseIdx++) {
      if (phrasePhonemes[phraseIdx]->size() <= 0) {
        continue;
      }

      // phonemes -> ids
      phonemes_to_ids(*(phrasePhonemes[phraseIdx]), idConfig, phonemeIds,
                      missingPhonemes);
      if (spdlog::should_log(spdlog::level::debug)) {
        // DEBUG log for phoneme ids
        std::stringstream phonemeIdsStr;
        for (auto phonemeId : phonemeIds) {
          phonemeIdsStr << phonemeId << ", ";
        }

        spdlog::debug("Converted {} phoneme(s) to {} phoneme id(s): {}",
                      phrasePhonemes[phraseIdx]->size(), phonemeIds.size(),
                      phonemeIdsStr.str());
      }

      // ids -> audio
      auto encode_start = std::chrono::steady_clock::now();
      std::optional<size_t> sid = speakerId;
      if(!sid && voice.synthesisConfig.speakerId)
        sid = voice.synthesisConfig.speakerId;
      auto params = voice.encoder.infer(phonemeIds, phrasePhonemes[phraseIdx]->size(),
                          sid,
                          noiseScale.value_or(voice.synthesisConfig.noiseScale),
                          lengthScale.value_or(voice.synthesisConfig.lengthScale),
                          noiseW.value_or(voice.synthesisConfig.noiseW));
      auto encode_end = std::chrono::steady_clock::now();
      float encode_seconds = std::chrono::duration<double>(encode_end - encode_start).count();
      auto& g = params["g"]; // g could be missing. TODO: Fix this
      auto& y_mask = params["y_mask"];
      auto& z = params["z"];

      size_t nslices = z.shape()[2];
      if(nslices != y_mask.shape()[2])
        throw std::runtime_error("z and y_mask must have the same number of slices");

      const size_t chunkSize = 45;
      const size_t padding = 5;

      float audioSeconds = 0;
      float inferSeconds = encode_seconds;

      // Too small to chunk, just pass it through
      if(nslices < chunkSize + padding * 2) {
          auto t0 = std::chrono::steady_clock::now();
          audioBuffer = voice.decoder->infer(z, y_mask, g);
          auto t1 = std::chrono::steady_clock::now();
          inferSeconds += std::chrono::duration<double>(t1 - t0).count();
          audioSeconds = (double)audioBuffer.size() / (double)voice.synthesisConfig.sampleRate;
      }
      else {
        for(size_t i=0,idx=0;i<nslices;i+=chunkSize,idx++) {
          size_t start = i > padding ? i - padding : 0;
          size_t end = std::min(nslices, i + chunkSize + padding);
          auto z_chunk = xt::view(z, xt::all(), xt::all(), xt::range(start, end));
          auto y_mask_chunk = xt::view(y_mask, xt::all(), xt::all(), xt::range(start, end));

          auto t0 = std::chrono::steady_clock::now();
          auto chunk_audio = voice.decoder->infer(z_chunk, y_mask_chunk, g);
          auto t1 = std::chrono::steady_clock::now();

          auto real_start = chunk_audio.begin() + (i - start) * 256;
          auto end_pad = padding;
          if(i+chunkSize >= nslices)
            end_pad = 0;
          else if(i+chunkSize+padding >= nslices)
            end_pad = nslices - (i+chunkSize);

          // HACK: compare the end of the previous chunk and the start of the next chunk to determine the best
          // place to stitch them together
          // This is 99% good. Still get pops rarely.
          constexpr size_t compare_window = 24;
          constexpr size_t search_window = 44;
          static_assert(compare_window < search_window, "compare_window must be less than search_window");
          const bool do_depop = audioBuffer.size() > compare_window && chunk_audio.size() > search_window * 2;
          if(do_depop) {
            auto prev_chunk_end = audioBuffer.end() - compare_window;
            auto next_chunk_start = real_start;
            next_chunk_start -= std::min(std::distance(chunk_audio.begin(), next_chunk_start), (ptrdiff_t)compare_window);
            size_t min_diff = std::numeric_limits<size_t>::max();
            // increment by 4 to speed up the search
            for(size_t j=0;j<search_window*2;j+=4) {
              size_t diff = 0;
              for(size_t k=0;k<compare_window;k++) {
                diff += std::abs(prev_chunk_end[k] - next_chunk_start[j+k]);
              }
              if(diff < min_diff) {
                min_diff = diff;
                real_start = next_chunk_start + j + compare_window;
              }
            }
            // average the samples in the compare window to smooth out the transition even more
            auto prev_base_ptr = audioBuffer.end() - compare_window;
            auto next_base_ptr = real_start - compare_window;
            for(size_t j=0;j<compare_window;j++) {
                prev_base_ptr[j] = (prev_base_ptr[j] + next_base_ptr[j]) / 2;
            }
          }

          auto real_end = chunk_audio.end() - end_pad * 256;
          audioBuffer.insert(audioBuffer.end(), real_start, real_end);
          float chunk_audio_seconds = (double)chunk_audio.size() / (double)voice.synthesisConfig.sampleRate;
          float chunk_infer_seconds = std::chrono::duration<double>(t1 - t0).count();

          if(audioCallback && audioBuffer.size() > compare_window) {
            std::vector<int16_t> tmp;
            tmp.insert(tmp.end(), audioBuffer.begin(), audioBuffer.end() - compare_window);
            audioBuffer.resize(compare_window);
            audioCallback();
            audioBuffer.resize(tmp.size());
            memcpy(audioBuffer.data(), tmp.data(), tmp.size() * sizeof(int16_t));
          }

          audioSeconds += chunk_audio_seconds;
          inferSeconds += chunk_infer_seconds;
          auto rtf = chunk_infer_seconds / chunk_audio_seconds;
          spdlog::debug("Chunk {} took {} seconds, RTF: {}", idx, std::chrono::duration<double>(t1 - t0).count(), rtf);

          if(i == 0 && phraseIdx == 0) {
            auto t = std::chrono::steady_clock::now();
            auto first_chunk_duration = std::chrono::duration<double>(t - encode_start).count();
            spdlog::debug("First chunk latency: {} seconds", first_chunk_duration);
          }
        }
        phraseResults[phraseIdx].audioSeconds = audioSeconds;
        phraseResults[phraseIdx].inferSeconds = inferSeconds;
      }

      // Add end of phrase silence
      for (std::size_t i = 0; i < phraseSilenceSamples[phraseIdx]; i++) {
        audioBuffer.push_back(0);
      }

      result.audioSeconds += phraseResults[phraseIdx].audioSeconds;
      result.inferSeconds += phraseResults[phraseIdx].inferSeconds;

      phonemeIds.clear();
    }

    // Add end of sentence silence
    if (sentenceSilenceSamples > 0) {
      for (std::size_t i = 0; i < sentenceSilenceSamples; i++) {
        audioBuffer.push_back(0);
      }
    }

    if (audioCallback) {
      // Call back must copy audio since it is cleared afterwards.
      audioCallback();
      audioBuffer.clear();
    }

    phonemeIds.clear();
  }

  if (missingPhonemes.size() > 0) {
    spdlog::warn("Missing {} phoneme(s) from phoneme/id map!",
                 missingPhonemes.size());

    for (auto phonemeCount : missingPhonemes) {
      std::string phonemeStr;
      utf8::append(phonemeCount.first, std::back_inserter(phonemeStr));
      spdlog::warn("Missing \"{}\" (\\u{:04X}): {} time(s)", phonemeStr,
                   (uint32_t)phonemeCount.first, phonemeCount.second);
    }
  }

  if (result.audioSeconds > 0) {
    result.realTimeFactor = result.inferSeconds / result.audioSeconds;
  }

} /* textToAudio */

// Phonemize text and synthesize audio to WAV file
void textToWavFile(PiperConfig &config, Voice &voice, std::string text,
                   std::ostream &audioFile, SynthesisResult &result,
                   std::optional<size_t> speakerId,
                   std::optional<float> noiseScale,
                   std::optional<float> lengthScale,
                   std::optional<float> noiseW) {

  std::vector<int16_t> audioBuffer;
  textToAudio(config, voice, text, audioBuffer, result, NULL, noiseScale,
              lengthScale, noiseW);

  // Write WAV
  auto synthesisConfig = voice.synthesisConfig;
  writeWavHeader(synthesisConfig.sampleRate, synthesisConfig.sampleWidth,
                 synthesisConfig.channels, (int32_t)audioBuffer.size(),
                 audioFile);

  audioFile.write((const char *)audioBuffer.data(),
                  sizeof(int16_t) * audioBuffer.size());

} /* textToWavFile */

} // namespace piper

