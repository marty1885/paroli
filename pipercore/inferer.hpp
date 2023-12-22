#pragma once

#include <string>
#include <vector>

#include <xtensor/xarray.hpp>

struct DecoderInferer {
  virtual ~DecoderInferer() = default;
  virtual std::vector<int16_t> infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const xt::xarray<float>& g) = 0;
  virtual void load(std::string modelPath, std::string accelerator) = 0;
};
