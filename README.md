# Paroli on Orange Pi

Streaming mode implementation of the Piper TTS system in C++ with RK3588/3566 NPU acceleration support. 

## How to use

Assuming you are running Ubuntu/Debian clean on your Orange Pi RK3588/3566 ([Orange Pi 5 series](https://orangepi.vn/tu-khoa-san-pham/opi5series)/ [Orange Pi 3B](https://orangepi.net/product-tag/orange-pi-3b))

1. You first need to instal rknpu lib, the fastest way is using Petrolus ezrknn installer

```bash
https://github.com/Pelochus/ezrknn-toolkit2
cd ezrknn-toolkit2
sudo bash install.sh
```

2. then you need to install git-lfs to clone models (which is large size)
```bash
(. /etc/lsb-release && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo env os=ubuntu dist="${DISTRIB_CODENAME}" bash)
sudo apt-get install git-lfs
```

3. Install some dependencies
```bash
sudo apt install -y cmake build-essential
sudo apt install -y libxtensor-dev nlohmann-json3-dev libspdlog-dev libopus-dev libfmt-dev libjsoncpp-dev
sudo apt install -y espeak-ng libespeak-ng-dev libogg-dev libsoxr-dev
```

4. Install drogon
```bash
cd ~
sudo apt install libssl-dev pkg-config libbotan-2-dev libc-ares-dev uuid-dev doxygen
git clone https://github.com/drogonframework/drogon
cd drogon
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

5. Install libopusenc
```bash
cd ~
wget https://archive.mozilla.org/pub/opus/libopusenc-0.2.1.tar.gz
tar -xvzf libopusenc-0.2.1.tar.gz
cd libopusenc-0.2.1
./configure
make -j4
sudo make install
```
this part is optional, some time the ldconfig not update your /usr/local/lib, then you need to do it manually
`sudo nano /etc/ld.so.conf.d/local.conf`
add this line `/usr/local/lib` then save and exit

7. Cloning models
```bash
cd ~
https://huggingface.co/thanhtantran/piper-paroli-rknn-model
```

8. Clone onnxruntime
```bash
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-aarch64-1.21.0.tgz
tar -xvf onnxruntime-linux-aarch64-1.21.0.tgz
```

10. Clone piper-phoemize
```bash
cd ~
wget https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_linux_aarch64.tar.gz
tar -xvf piper-phonemize_linux_aarch64.tar.gz
```

12. Now build the app
```bash
git clone https://github.com/thanhtantran/paroli
cd paroli
mkdir build && cd build
cmake .. -DUSE_RKNN=ON -DORT_ROOT=/home/orangepi/onnxruntime-linux-aarch64-1.21.0 -DPIPER_PHONEMIZE_ROOT=/home/orangepi/piper_phonemize -DCMAKE_BUILD_TYPE=Release
make -j4
```
After cmake run, you will see wherether the lib `librknnrt.so` is loaded or not, normally it should be in `/usr/lib/`. If you see not loaded, that mean the lib still not installed. And you can not run with RKNPU accelerator.

After all, the program is compiled, you can use it inside the build folder. But still need one more step, this is important. Copy the `espeak-ng-data` from `piper_phonemize` into the build folder:
```bash
cp -r ~/piper_phonemize/share/espeak-ng-data/ .
```

### The Command to transform text to wav

```plaintext
./paroli-cli --encoder /home/orangepi/piper-paroli-rknn-model/encoder.onnx --decoder /home/orangepi/piper-paroli-rknn-model/decoder.rknn -c ~/piper-paroli-rknn-model/config.json
```
After piper loaded, paste the text into the shell, then it will transform to wav.

Change the `decoder.rknn` to your Orange Pi model, for example with Orange Pi 5 series, you will use `decoder-3588.rknn`; if you are using Orange Pi 3B, use `decoder-3566.rknn`

### The API server

An web API server is also provided so other applications can easily perform text to speech. For details, please refer to the [web API document](paroli-server/docs/web_api.md) for details. By default, a demo UI can be accessed at the root of the URL. The API server supports both responding with compressed audio to reduce bandwidth requirement and streaming audio via WebSocket. 

To run it:

```bash
./paroli-server --encoder /home/orangepi/piper-paroli-rknn-model/encoder.onnx --decoder /home/orangepi/piper-paroli-rknn-model/decoder.rknn -c ~/piper-paroli-rknn-model/config.json --ip 0.0.0.0 --port 8848
```
Same as the CLI, change the `decoder.rknn` to your Orange Pi model, for example with Orange Pi 5 series, you will use `decoder-3588.rknn`; if you are using Orange Pi 3B, use `decoder-3566.rknn`

And to invoke TSS

```bash
curl http://your.server.address:8848/api/v1/synthesise -X POST -H 'Content-Type: application/json' -d '{"text": "To be or not to be, that is the question"}' > test.opus
```

### Demo

Video wait here ...

#### Authentication

To enable use cases where the service is exposed for whatever reason. The API server supports a basic authentication scheme. The `--auth` flag will generate a bearer token that is different every time and both websocket and HTTP synthesis API will only work if enabled. `--auth [YOUR_TOKEN]` will set the token to YOUR_TOKEN. Furthermore setting the `PAROLI_TOKEN` environment variable will set the bearer token to whatever the environment variable is set to.

```plaintext
Authentication: Bearer <insert the token>
```

**The Web UI will not work when authentication is enabled**

## Training models

To obtain the encoder and decoder models, you'll either need to download them or creating one from checkpoints. Checkpoints are the trained raw model piper generates. Please refer to [piper's TRAINING.md](https://github.com/rhasspy/piper/blob/master/TRAINING.md) for details. To convert checkpoints into ONNX file pairs, you'll need [mush42's piper fork and the streaming branch](https://github.com/mush42/piper/tree/streaming). Run

```bash
python3 -m piper_train.export_onnx_streaming /path/to/your/traning/lighting_logs/version_0/checkpoints/blablablas.ckpt /path/to/output/directory
```

### Downloading models

Some 100% legal models are provided on [HuggingFace](https://huggingface.co/thanhtantran/piper-paroli-rknn-model)

### Converting model to Rockchip NPU 

Also, converting ONNX to RKNN has to be done on an x64 computer. As of writing this document, you likely want to install the version for Python 3.10 as this is the same version that works with upstream piper. rknn-toolkit2 version 1.6.0 is required.

```bash
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
sudo cp aarch64/librknnrt.so /usr/lib/
sudo cp include/* /usr/include/
```

Then convert your model

```bash
# Install rknn-toolkit2
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/tree/master/rknn-toolkit2/packages
pip install rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl

# Run the conversion script
python tools/decoder2rknn.py /path/to/model/decoder.onnx /path/to/model/decoder.rknn
```

## Credit

- [Piper](https://github.com/rhasspy/piper)
- [Paroli original](https://github.com/marty1885/paroli)
- [Petrolus ezrknn toolkit](https://github.com/Pelochus/ezrknn-toolkit2) 


