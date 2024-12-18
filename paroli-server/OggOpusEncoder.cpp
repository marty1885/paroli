#include <ogg/ogg.h>
#include <opus/opus.h>
#include <opus/opusenc.h>

#include "OggOpusEncoder.hpp"

#include <vector>
#include <cstdint>
#include <unistd.h>
#include <span>
#include <stdexcept>
#include <memory>
#include <cstring>

std::vector<uint8_t> encodeOgg(const std::vector<short>& data, size_t sr, size_t nchannels, size_t bitrate)
{
    StreamingOggOpusEncoder encoder(sr, nchannels, bitrate);
    std::vector<uint8_t> oggBuffer = encoder.encode(data);
    auto extra = encoder.finish();
    oggBuffer.insert(oggBuffer.end(), extra.begin(), extra.end());
    return oggBuffer;
}

StreamingOggOpusEncoder::StreamingOggOpusEncoder(size_t sr, size_t nchannels, size_t bitrate)
    :sr(sr), nchannels(nchannels)
{
    auto write_func = [](void* user_data, const unsigned char* ptr, opus_int32 size) -> int {
        StreamingOggOpusEncoder* self = (StreamingOggOpusEncoder*)user_data;
        auto buffer = &self->oggBuffer;
        auto old_size = buffer->size();
        buffer->resize(old_size + size);
        memcpy(buffer->data() + old_size, ptr, size);
        return 0;
    };
    auto close_func = [](void* user_data) -> int {
        return 0;
    };

    OpusEncCallbacks callbacks {
        .write = write_func,
        .close = close_func,
    };

    comments = std::shared_ptr<OggOpusComments>(ope_comments_create(), &ope_comments_destroy);
    ope_comments_add(comments.get(), "ENCODER", "libopusenc");

    int err = 0;
    encoder = std::shared_ptr<OggOpusEnc>(
            ope_encoder_create_callbacks(&callbacks, this, comments.get(), sr, nchannels, 0, &err),
            &ope_encoder_destroy);
    if(err != OPE_OK)
        throw std::runtime_error("Failed to create encoder");
    if(ope_encoder_ctl(encoder.get(), OPUS_SET_BITRATE(bitrate)) != OPE_OK)
        std::cerr << "Failed to set bitrate to " << bitrate << std::endl;
}

std::vector<uint8_t> StreamingOggOpusEncoder::encode(const std::vector<short>& data)
{
    oggBuffer.clear();
    audioBuffer.insert(audioBuffer.end(), data.begin(), data.end());
    const size_t frame_size = 960;
    if(audioBuffer.size() < frame_size * nchannels)
        return {};
    size_t i = 0;
    size_t end = (audioBuffer.size() / frame_size) * frame_size;
    for(i = 0; i < end; i += frame_size) {
        int err = ope_encoder_write(encoder.get(), audioBuffer.data() + i, frame_size);
        if(err != 0)
            throw std::runtime_error("opusenc failed to encode");
    }
    audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + end);

    return oggBuffer;
}

std::vector<uint8_t> StreamingOggOpusEncoder::finish()
{
    oggBuffer.clear();
    ope_encoder_drain(encoder.get());
    return oggBuffer;
}

