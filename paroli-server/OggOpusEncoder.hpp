#pragma once

#include <vector>
#include <cstdint>
#include <unistd.h>
#include <opus/opusenc.h>

std::vector<uint8_t> encodeOgg(const std::vector<short>& data, size_t sr, size_t nchannels, size_t bitrate = 96000);

struct StreamingOggOpusEncoder {
    StreamingOggOpusEncoder(size_t sr, size_t nchannels, size_t bitrate = 96000);

    std::vector<uint8_t> encode(const std::vector<short>& data);
    std::vector<uint8_t> finish();

    std::vector<short> audioBuffer;
    std::shared_ptr<OggOpusEnc> encoder;
    OpusEncCallbacks callbacks;
    size_t sr;
    size_t nchannels;
    std::vector<uint8_t> oggBuffer;
    std::shared_ptr<OggOpusComments> comments;
};

