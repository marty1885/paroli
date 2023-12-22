#pragma once

#include <vector>
#include <cstdint>
#include <unistd.h>
#include <opus/opusenc.h>

std::vector<uint8_t> encodeOgg(const std::vector<short>& data, size_t sr, size_t nchannels, size_t bitrate = 96000);

