#pragma once

#include <rknn_api.h>

#include "inferer.hpp"

#include <memory>
#include <mutex>

struct RknnDecoderInfererImpl {
  RknnDecoderInfererImpl() = delete;
  RknnDecoderInfererImpl(const RknnDecoderInfererImpl&) = delete;
  RknnDecoderInfererImpl& operator=(const RknnDecoderInfererImpl&) = delete;
  RknnDecoderInfererImpl(RknnDecoderInfererImpl&&);

  RknnDecoderInfererImpl(rknn_context ctx);
  ~RknnDecoderInfererImpl();
  rknn_context ctx = 0;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  std::vector<rknn_input> inputs;
  std::vector<rknn_output> outputs;

  std::vector<int16_t> infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const xt::xarray<float>& g);
};

struct RknnDecoderInferer : public DecoderInferer {
  std::vector<RknnDecoderInfererImpl> impls;
  std::mutex mtx;
  std::vector<int> implTracker;
  std::condition_variable cv;
  bool flag = false;

  std::vector<int16_t> infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const xt::xarray<float>& g) override;
  void load(std::string modelPath, std::string accelerator) override;
};
