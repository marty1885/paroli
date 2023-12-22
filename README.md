# piper-streamer

Streaming mode implementation of the Piper TTS system in C++ with (optional) RK3588 NPU acceleration support

## How to use

Before building, you will need to fulfill the following dependencies

* xtensor
* spdlog
* libfmt
* piper-phoenomize
* onnxruntime == 1.14

(RKNN support)
* rknnrt >= 1.6.0

In which `piper-phoenomize` and `onnxruntime` binary (not the source! Unless you want to build yourselves!) likely needs to be downloaded and decompressed manually. Afterwards run CMake and point to the folders you recompressed them.

```bash
mkdir build
cd build
cmake .. -DORT_ROOT=/path/to/your/onnxruntime-linux-aarch64-1.14.1 -DPIPER_PHONEMIZE_ROOT=/path/to/your/piper-phonemize-2023-11-14 -DCMAKE_BUILD_TYPE=Release
make -j
```

**TODO:** Explain how to get the models
Afterwards run `piper-cli` and type into the console to synthesize speech.

```plaintext
./piper-cli --encoder /path/to/your/encoder.onnx --decoder /path/to/your/decoder.rknn -c /path/to/your/model.json
```
