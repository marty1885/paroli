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
    const size_t estimated_size = ((double)data.size() / sr) * bitrate * 1.03;
    std::vector<uint8_t> oggBuffer;
    oggBuffer.reserve(estimated_size);
    auto write_func = [](void* user_data, const unsigned char* ptr, opus_int32 size) -> int {
        auto buffer = (std::vector<uint8_t>*)user_data;
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

    auto comments = std::shared_ptr<OggOpusComments>(ope_comments_create(), &ope_comments_destroy);
    ope_comments_add(comments.get(), "ENCODER", "libopusenc");

    int err = 0;
    auto enc = std::shared_ptr<OggOpusEnc>(
            ope_encoder_create_callbacks(&callbacks, &oggBuffer, comments.get(), sr, nchannels, 0, &err),
            &ope_encoder_destroy);
    if(err != OPE_OK)
        throw std::runtime_error("Failed to create encoder");
    if(ope_encoder_ctl(enc.get(), OPUS_SET_BITRATE(bitrate)) != OPE_OK)
        std::cerr << "Failed to set bitrate to " << bitrate << std::endl;

    const size_t frame_size = sr * 0.02 * nchannels;
    for(size_t i = 0; i < data.size(); i += frame_size) {
        size_t idx_begin = i;
        size_t idx_end = std::min(i + frame_size, data.size());
        size_t samples_per_channel = (idx_end - idx_begin) / nchannels;
        // should not happen. but just in case to avoid OOB access
        if(samples_per_channel == 0)
            break;
        ope_encoder_write(enc.get(), data.data() + idx_begin, samples_per_channel);
    }

    ope_encoder_drain(enc.get());
    
    return oggBuffer;
}

StreamingOggOpusEncoder::StreamingOggOpusEncoder(size_t sr, size_t nchannels, size_t bitrate)
    :sr(sr), nchannels(nchannels)
{
    auto write_func = [](void* user_data, const unsigned char* ptr, opus_int32 size) -> int {
        StreamingOggOpusEncoder* self = (StreamingOggOpusEncoder*)user_data;
        auto buffer = (std::vector<uint8_t>*)&self->oggBuffer;
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
    audioBuffer.insert(audioBuffer.end(), data.begin(), data.end());
    const size_t frame_size = sr * 0.02 * nchannels;
    size_t i = 0;
    for(i = 0; i < audioBuffer.size(); i += frame_size) {
        size_t idx_begin = i;
        size_t idx_end = std::min(i + frame_size, audioBuffer.size());
        size_t samples_per_channel = (idx_end - idx_begin) / nchannels;
        // should not happen. but just in case to avoid OOB access
        if(samples_per_channel == 0)
            break;
        ope_encoder_write(encoder.get(), audioBuffer.data() + idx_begin, samples_per_channel);
    }
    audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + i);

    return oggBuffer;
}

std::vector<uint8_t> StreamingOggOpusEncoder::finish()
{
    ope_encoder_drain(encoder.get());
    return oggBuffer;
}

