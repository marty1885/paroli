#pragma once

#include <rknn_api.h>

#include "inferer.hpp"

struct RknnDecoderInferer : public DecoderInferer {
  rknn_context ctx = 0;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  std::vector<rknn_input> inputs;
  std::vector<rknn_output> outputs;

  std::vector<int16_t> infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const xt::xarray<float>& g) override;
  void load(std::string modelPath) override;
};
