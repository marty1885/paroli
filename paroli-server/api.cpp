#include <drogon/drogon.h>
#include <drogon/HttpController.h>
#include <drogon/WebSocketController.h>
#include <trantor/net/EventLoopThreadPool.h>

#include <span>
#include <bit>
#include <atomic>
#include <coroutine>
#include <cmath>

#include "piper.hpp"
#include "OggOpusEncoder.hpp"
#include <nlohmann/json.hpp>
#include <soxr.h>

using namespace drogon;
extern piper::PiperConfig piperConfig;
extern piper::Voice voice;
extern std::string authToken;

static constexpr size_t MAX_TEXT_LENGTH = 64 * 1024; // 64 KiB
trantor::EventLoopThreadPool synthesizerThreadPool(3, "synehesizer thread pool");

template<typename Func>
requires std::is_invocable_v<Func, const std::span<const short>>
[[nodiscard]]
auto speak(const std::string& text, std::optional<size_t> speaker_id, Func cb, std::optional<float> length_scale
        , std::optional<float> noise_scale, std::optional<float> noise_w) -> bool
{
    std::vector<short> audioBuffer;
    piper::SynthesisResult result;
    auto callback = [&audioBuffer, cb=std::move(cb)]() {
        auto view = std::span(audioBuffer);
        cb(view);
    };

    try {
        piper::textToAudio(piperConfig, voice, text, audioBuffer, result, callback, speaker_id,
            noise_scale, length_scale, noise_w);
    }
    catch(const std::exception& e) {
        LOG_ERROR << "Exception thrown while generating speach: " << e.what();
        return false;
    }
    return true;
}

struct SynthesisApiParams
{
    std::string text;
    std::optional<int64_t> speaker_id;
    std::optional<float> length_scale;
    std::optional<float> noise_scale;
    std::optional<float> noise_w;
    std::optional<std::string> audio_format;
};

static std::string replaceAll(std::string_view str, std::string_view from, std::string_view to)
{
    std::string result;
    result.reserve(str.size());
    size_t last = 0;
    while(true) {
        auto next = str.find(from, last);
        if(next == std::string_view::npos) {
            result += str.substr(last);
            break;
        }
        result += str.substr(last, next - last);
        result += to;
        last = next + from.size();
    }
    return result;
}

static std::string piperTextPreprocess(std::string text)
{
    // trim leading and tailing spaces
    auto first = text.find_first_not_of(" \n\r\t");
    if(first != std::string::npos)
        text = text.substr(first);
    auto last = text.find_last_not_of(" \n\r\t");
    if(last != std::string::npos)
        text = text.substr(0, last + 1);

    if(text.empty()) {
        text = ",";
        return text;
    }

    // append a comma if the text does not end with a punctuation
    const char* punctuation = ".,!?;:";
    bool has_punctuation = (strchr(punctuation, text.back()) != nullptr);
    if(!has_punctuation)
        text += ",";

    // Piper have no idea how to process ... and .. so we convert them to ,
    std::string result;
    size_t i = 0;
    while(i < text.size()) {
        auto ch = text[i];
        if(ch == '.') {
            size_t count = 1;
            while(i + count < text.size() && text[i + count] == '.')
                count++;
            if(count == 1)
                result += '.';
            else
                result += ',';
            i += count - 1;
        } else {
            result += ch;
        }
        i++;
    }

    // Handle stupid unicode characters that piper can't handle
    result = replaceAll(result, "…", ",");
    result = replaceAll(result, "“", "\"");
    result = replaceAll(result, "”", "\"");
    result = replaceAll(result, "‘", "'");
    result = replaceAll(result, "’", "'");
    result = replaceAll(result, "—", ", ");
    result = replaceAll(result, " - ", ", ");
    return result;
}

SynthesisApiParams parseSynthesisApiParams(const std::string_view json_txt)
{
    auto res = SynthesisApiParams{};
    auto json = nlohmann::json::parse(json_txt);
    if(!json.contains("text"))
        throw std::runtime_error("Missing 'text' field");
    res.text = json["text"].get<std::string>();
    if(res.text.size() > MAX_TEXT_LENGTH)
        throw std::runtime_error("Text too long");
    if(json.contains("speaker_id") && json["speaker_id"].is_null() == false)
        res.speaker_id = json["speaker_id"].get<int64_t>();
    if(json.contains("speaker")) {
        const auto& speaker_id_map = voice.modelConfig.speakerIdMap;
        auto speaker = json["speaker"].get<std::string>();
        if(speaker_id_map.has_value() == false)
            throw std::runtime_error("Speaker ID map is not available");

        if(!speaker_id_map->contains(speaker))
            throw std::runtime_error("Unknown speaker name " + speaker);
        res.speaker_id = speaker_id_map->at(speaker);
    }
    if(json.contains("length_scale")) {
        if(json["length_scale"].is_number() == false)
            throw std::runtime_error("length_scale must be a number");
        res.length_scale = json["length_scale"].get<float>();
        if(!std::isfinite(*res.length_scale) || *res.length_scale <= 0.0f || *res.length_scale > 100.0f)
            throw std::runtime_error("length_scale out of range");
    }
    if(json.contains("noise_scale")) {
        if(json["noise_scale"].is_number() == false)
            throw std::runtime_error("noise_scale must be a number");
        res.noise_scale = json["noise_scale"].get<float>();
        if(!std::isfinite(*res.noise_scale) || *res.noise_scale < 0.0f || *res.noise_scale > 100.0f)
            throw std::runtime_error("noise_scale out of range");
    }
    if(json.contains("noise_w")) {
        if(json["noise_w"].is_number() == false)
            throw std::runtime_error("noise_w must be a number");
        res.noise_w = json["noise_w"].get<float>();
        if(!std::isfinite(*res.noise_w) || *res.noise_w < 0.0f || *res.noise_w > 100.0f)
            throw std::runtime_error("noise_w out of range");
    }
    if(json.contains("audio_format")) {
        if(json["audio_format"].is_string() == false)
            throw std::runtime_error("audio_format must be a string");
        res.audio_format = json["audio_format"].get<std::string>();
    }

    if(res.speaker_id.has_value() && (*res.speaker_id < 0 || *res.speaker_id >= voice.modelConfig.numSpeakers))
        throw std::runtime_error("Speaker ID is out of range");

    res.text = piperTextPreprocess(res.text);
    return res;
}

static std::vector<short> resample(std::span<const short> input, size_t orig_sr, size_t out_sr, int channels)
{
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_INT16_I, SOXR_INT16_I);
    soxr_quality_spec_t q_spec = soxr_quality_spec(SOXR_MQ, 0);
    soxr_error_t error;
    soxr_t soxr = soxr_create(orig_sr, out_sr, channels, &error, &io_spec, &q_spec, NULL);
    if(error != NULL) {
        throw std::runtime_error("soxr_create failed");
    }
    std::vector<short> output(input.size() * out_sr / orig_sr);
    size_t idone, odone;
    error = soxr_process(soxr, input.data(), input.size(), &idone, output.data(), output.size(), &odone);
    if(error != NULL) {
        soxr_delete(soxr);
        throw std::runtime_error("soxr_process failed");
    }
    output.resize(odone);
    
    soxr_delete(soxr);
    return output;
}

HttpResponsePtr makeBadRequestResponse(const std::string &msg)
{
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k400BadRequest);
    resp->setContentTypeCode(CT_TEXT_PLAIN);
    resp->setBody(msg);
    return resp;
}

// Awaiter that dispatches per-sentence synthesize() calls across the thread pool.
// The last task to complete (atomic counter == N) resumes the suspended coroutine.
struct ParallelSynthAwaiter : drogon::CallbackAwaiter<void>
{
    ParallelSynthAwaiter(
        trantor::EventLoopThreadPool& pool,
        piper::Voice& voice,
        const piper::PhonemeData& phonemeData,
        std::vector<std::vector<int16_t>>& sentenceAudio,
        std::vector<piper::SynthesisResult>& sentenceResults,
        std::optional<size_t> speakerId,
        std::optional<float> noiseScale,
        std::optional<float> lengthScale,
        std::optional<float> noiseW)
        : pool_(pool), voice_(voice), phonemeData_(phonemeData),
          sentenceAudio_(sentenceAudio), sentenceResults_(sentenceResults),
          speakerId_(speakerId), noiseScale_(noiseScale),
          lengthScale_(lengthScale), noiseW_(noiseW) {}

    void await_suspend(std::coroutine_handle<> handle)
    {
        const size_t n = phonemeData_.sentences.size();
        for (size_t i = 0; i < n; i++) {
            pool_.getNextLoop()->queueInLoop([this, i, n, handle]() {
                try {
                    piper::PhonemeData single;
                    single.sentences.push_back(phonemeData_.sentences[i]);
                    piper::synthesize(voice_, single, sentenceAudio_[i],
                                     sentenceResults_[i], nullptr,
                                     speakerId_, noiseScale_, lengthScale_, noiseW_);
                } catch (...) {
                    if (!exceptionCaptured_.test_and_set(std::memory_order_acq_rel))
                        setException(std::current_exception());
                }
                if (completed_.fetch_add(1, std::memory_order_acq_rel) + 1 == n)
                    handle.resume();
            });
        }
    }

private:
    trantor::EventLoopThreadPool& pool_;
    piper::Voice& voice_;
    const piper::PhonemeData& phonemeData_;
    std::vector<std::vector<int16_t>>& sentenceAudio_;
    std::vector<piper::SynthesisResult>& sentenceResults_;
    std::optional<size_t> speakerId_;
    std::optional<float> noiseScale_;
    std::optional<float> lengthScale_;
    std::optional<float> noiseW_;
    std::atomic<size_t> completed_{0};
    std::atomic_flag exceptionCaptured_{};
};

// Phonemize sequentially, then synthesize sentences in parallel across the pool.
// Single-sentence texts skip the awaiter and synthesize inline.
static Task<std::vector<int16_t>> doSynthesis(
    const std::string& text,
    std::optional<size_t> speakerId,
    std::optional<float> noiseScale,
    std::optional<float> lengthScale,
    std::optional<float> noiseW)
{
    auto phonemeData = piper::phonemize(piperConfig, voice, text);
    const size_t sentenceCount = phonemeData.sentences.size();
    if (sentenceCount == 0)
        co_return {};

    // Fast path: single sentence, synthesize inline without queueInLoop overhead
    if (sentenceCount == 1) {
        std::vector<int16_t> audioBuffer;
        piper::SynthesisResult result{};
        audioBuffer.reserve(voice.synthesisConfig.sampleRate);
        piper::synthesize(voice, phonemeData, audioBuffer, result, nullptr,
                         speakerId, noiseScale, lengthScale, noiseW);
        co_return audioBuffer;
    }

    // Multi-sentence: dispatch each sentence to the pool in parallel
    std::vector<std::vector<int16_t>> sentenceAudio(sentenceCount);
    std::vector<piper::SynthesisResult> sentenceResults(sentenceCount);

    co_await ParallelSynthAwaiter(
        synthesizerThreadPool, voice, phonemeData,
        sentenceAudio, sentenceResults,
        speakerId, noiseScale, lengthScale, noiseW);

    // Stitch audio in sentence order (sentence silence already embedded by synthesize)
    size_t totalSamples = 0;
    for (const auto& audio : sentenceAudio)
        totalSamples += audio.size();

    std::vector<int16_t> stitched;
    stitched.reserve(totalSamples);
    for (auto& audio : sentenceAudio)
        stitched.insert(stitched.end(), audio.begin(), audio.end());

    // Accumulate stats
    piper::SynthesisResult totalResult{};
    for (const auto& r : sentenceResults) {
        totalResult.inferSeconds += r.inferSeconds;
        totalResult.audioSeconds += r.audioSeconds;
    }
    if (totalResult.audioSeconds > 0)
        totalResult.realTimeFactor = totalResult.inferSeconds / totalResult.audioSeconds;

    co_return stitched;
}

namespace api
{
struct v1 : public HttpController<v1>
{
    v1()
    {
        synthesizerThreadPool.start();
    }
    METHOD_LIST_BEGIN
    METHOD_ADD(v1::synthesise, "/synthesise", {Post, Options});
    METHOD_ADD(v1::speakers, "/speakers", Get);
    METHOD_LIST_END

    Task<HttpResponsePtr> synthesise(const HttpRequestPtr req);
    Task<HttpResponsePtr> speakers(const HttpRequestPtr req);
};

struct v1ws : public WebSocketController<v1ws>
{
   void handleNewConnection(const HttpRequestPtr& req, const WebSocketConnectionPtr& wsConnPtr) override;
   void handleNewMessage(const WebSocketConnectionPtr& wsConnPtr, std::string&& message, const WebSocketMessageType& type) override;
   void handleConnectionClosed(const WebSocketConnectionPtr& wsConnPtr) override {}


   Task<> handleNewMessageAsync(WebSocketConnectionPtr wsConnPtr, std::string message, WebSocketMessageType type);
   WS_PATH_LIST_BEGIN
   WS_PATH_ADD("/api/v1/stream", Get);
   WS_PATH_LIST_END
};

void v1ws::handleNewConnection(const HttpRequestPtr& req, const WebSocketConnectionPtr& wsConnPtr)
{
    if(authToken.empty())
        return;
    auto auth = req->getHeader("Authorization");
    if(auth.empty() || auth != "Bearer " + authToken) {
        wsConnPtr->forceClose();
        return;
    }
}

void v1ws::handleNewMessage(const WebSocketConnectionPtr& wsConnPtr, std::string&& message, const WebSocketMessageType& type)
{
    synthesizerThreadPool.getNextLoop()->queueInLoop(async_func([=, this]() mutable -> Task<> {
        co_await handleNewMessageAsync(wsConnPtr, std::move(message), type);
    }));
}
Task<> v1ws::handleNewMessageAsync(WebSocketConnectionPtr wsConnPtr, std::string message, WebSocketMessageType type)
{
    if(type != WebSocketMessageType::Text)
        co_return;

    SynthesisApiParams params;
    try {
        params = parseSynthesisApiParams(message);
    }
    catch (const std::exception& e) {
        nlohmann::json resp;
        resp["status"] = "failed";
        resp["message"] = std::string(e.what());
        wsConnPtr->send(resp.dump());
        co_return;
    }
    bool send_opus = params.audio_format.value_or("opus") == "opus";

    StreamingOggOpusEncoder encoder(24000, 1);
    bool ok = speak(params.text, params.speaker_id, [&](const std::span<const short> view) {
        if(view.empty())
            return;
        if(send_opus) {
            auto pcm = resample(view, voice.synthesisConfig.sampleRate, 24000, 1);
            auto opus = encoder.encode(pcm);
            if(!opus.empty())
                wsConnPtr->send((char*)opus.data(), opus.size(), WebSocketMessageType::Binary);
            return;
        }

        if constexpr (std::endian::native == std::endian::big) {
            // Convert to little endian
            std::vector<int16_t> pcm(view.begin(), view.end());
            for(int16_t& sample : pcm)
                sample = (sample >> 8) | (sample << 8);
            wsConnPtr->send((char*)pcm.data(), pcm.size() * sizeof(int16_t), WebSocketMessageType::Binary);
            return;
        }
        wsConnPtr->send((char*)view.data(), view.size() * sizeof(int16_t), WebSocketMessageType::Binary);
    }, params.length_scale, params.noise_scale, params.noise_w);

    if(!ok) {
        wsConnPtr->send(R"({"status":"failed", "message":"failed to synthesis"})");
        co_return;

    }
    if(send_opus) {
        auto opus = encoder.finish();
        if(opus.empty() == false)
            wsConnPtr->send((char*)opus.data(), opus.size(), WebSocketMessageType::Binary);
    }
    wsConnPtr->send(R"({"status":"ok", "message":"finished"})");
}

Task<HttpResponsePtr> v1::synthesise(const HttpRequestPtr req)
{
    if(req->method() == Options) {
        auto resp = HttpResponse::newHttpResponse();
        resp->addHeader("Access-Control-Allow-Origin", "*");
        resp->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
        co_return resp;
    }

    if (req->getContentType() != CT_APPLICATION_JSON)
        co_return makeBadRequestResponse("Content-Type must be application/json");

    if(!authToken.empty()) {
        auto auth = req->getHeader("Authorization");
        if(auth.empty() || auth != "Bearer " + authToken)
            co_return makeBadRequestResponse("Invalid Authorization");
    }

    auto loop = synthesizerThreadPool.getNextLoop();
    co_await switchThreadCoro(loop);


    SynthesisApiParams params;
    try {
        params = parseSynthesisApiParams(req->getBody());
    }
    catch (const std::exception& e) {
        co_return makeBadRequestResponse(e.what());
    }

    std::vector<int16_t> audio;
    try {
        audio = co_await doSynthesis(params.text, params.speaker_id,
                                     params.noise_scale, params.length_scale, params.noise_w);
    }
    catch (const std::exception& e) {
        co_return makeBadRequestResponse(e.what());
    }

    auto resp = HttpResponse::newHttpResponse();
    if(params.audio_format.value_or("opus") == "opus") {
        auto pcm = resample(audio, voice.synthesisConfig.sampleRate, 24000, 1);
        auto opus = encodeOgg(pcm, 24000, 1);
        resp->setContentTypeString("audio/ogg; codecs=opus");
        resp->setBody(std::string(reinterpret_cast<const char*>(opus.data()), opus.size()));
        co_return resp;
    }

    resp->setStatusCode(k200OK);
    resp->setContentTypeString("audio/raw");
    resp->setBody(std::string(reinterpret_cast<const char*>(audio.data()), audio.size() * sizeof(int16_t)));
    co_return resp;
}

Task<HttpResponsePtr> v1::speakers(const HttpRequestPtr req)
{
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k200OK);
    resp->setContentTypeCode(CT_APPLICATION_JSON);


    const auto& speakerIdMap = voice.modelConfig.speakerIdMap;
    if(speakerIdMap.has_value() == false) {
        resp->setBody("{}");
    }
    else {
        resp->setBody(nlohmann::json(speakerIdMap.value()).dump());
    }
    co_return resp;
}

} // namespace api

namespace v1
{
struct audio : public HttpController<audio>
{
    METHOD_LIST_BEGIN
    METHOD_ADD(audio::speech, "/speech", {Post, Options});
    METHOD_LIST_END

    Task<HttpResponsePtr> speech(const HttpRequestPtr req);
};

Task<HttpResponsePtr> audio::speech(const HttpRequestPtr req)
{
    if(req->method() == Options) {
        auto resp = HttpResponse::newHttpResponse();
        resp->addHeader("Access-Control-Allow-Origin", "*");
        resp->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
        co_return resp;
    }

    if (req->getContentType() != CT_APPLICATION_JSON)
        co_return makeBadRequestResponse("Content-Type must be application/json");

    if(!authToken.empty()) {
        auto auth = req->getHeader("Authorization");
        if(auth.empty() || auth != "Bearer " + authToken)
            co_return makeBadRequestResponse("Invalid Authorization");
    }

    auto loop = synthesizerThreadPool.getNextLoop();
    co_await switchThreadCoro(loop);

    // Translate OpenAI request format into SynthesisApiParams
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(req->getBody());
    }
    catch (const std::exception&) {
        co_return makeBadRequestResponse("Invalid JSON");
    }

    if(!json.contains("input") || !json["input"].is_string())
        co_return makeBadRequestResponse("Missing or invalid 'input' field");

    SynthesisApiParams params;
    auto inputText = json["input"].get<std::string>();
    if(inputText.size() > MAX_TEXT_LENGTH)
        co_return makeBadRequestResponse("Text too long");
    params.text = piperTextPreprocess(std::move(inputText));

    // Resolve "voice" to speaker_id (numeric string or case-insensitive name)
    if(json.contains("voice") && json["voice"].is_string()) {
        auto v = json["voice"].get<std::string>();
        char* end = nullptr;
        long id = std::strtol(v.c_str(), &end, 10);
        if(end != v.c_str() && *end == '\0' && id >= 0) {
            params.speaker_id = static_cast<int64_t>(id);
        } else {
            const auto& map = voice.modelConfig.speakerIdMap;
            if(map.has_value()) {
                for(const auto& [name, sid] : *map) {
                    if(name.size() == v.size() && std::equal(name.begin(), name.end(), v.begin(),
                        [](char a, char b){ return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); })) {
                        params.speaker_id = static_cast<int64_t>(sid);
                        break;
                    }
                }
            }
        }
    }

    std::vector<int16_t> audioBuffer;
    try {
        audioBuffer = co_await doSynthesis(params.text, params.speaker_id,
                                           params.noise_scale, params.length_scale, params.noise_w);
    }
    catch (const std::exception& e) {
        co_return makeBadRequestResponse(std::string("Synthesis failed: ") + e.what());
    }

    auto pcm = resample(audioBuffer, voice.synthesisConfig.sampleRate, 24000, 1);
    auto opus = encodeOgg(pcm, 24000, 1);

    auto resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("audio/ogg; codecs=opus");
    resp->setBody(std::string(reinterpret_cast<const char*>(opus.data()), opus.size()));
    co_return resp;
}
} // namespace v1

