#include "rknn-inferer.hpp"

#include <spdlog/spdlog.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

static std::vector<uint8_t> loadModel(std::string modelPath)
{
    std::ifstream ifs(modelPath, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cout << "open model file failed." << std::endl;
        return std::vector<uint8_t>();
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    ifs.close();

    return buf;
}

void RknnDecoderInferer::load(std::string modelPath)
{
    auto model = loadModel(modelPath);
    if (model.empty())
        throw std::runtime_error("load model failed.");

    // enabling sram seems to help with reducing the variance of inference time. no hard evidence though.
    auto ret = rknn_init(&ctx, model.data(), model.size(), RKNN_FLAG_ENABLE_SRAM, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_init failed. Error code: " + std::to_string(ret));

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_query failed. Error code: " + std::to_string(ret));

    input_attrs.resize(io_num.n_input);
    output_attrs.resize(io_num.n_output);
    inputs.resize(io_num.n_input);
    outputs.resize(io_num.n_output);
    // for good measure
    memset(input_attrs.data(), 0, sizeof(rknn_tensor_attr) * io_num.n_input);
    memset(output_attrs.data(), 0, sizeof(rknn_tensor_attr) * io_num.n_output);
    memset(inputs.data(), 0, sizeof(rknn_input) * io_num.n_input);
    memset(outputs.data(), 0, sizeof(rknn_output) * io_num.n_output);

    spdlog::info("input num: {}, output num: {}", io_num.n_input, io_num.n_output);
    for(int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
            throw std::runtime_error("rknn_query failed. Error code: " + std::to_string(ret));
    }

    for(int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
            throw std::runtime_error("rknn_query failed. Error code: " + std::to_string(ret));
    }
}

std::vector<int16_t> RknnDecoderInferer::infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const xt::xarray<float>& g)
{
    std::vector<const xt::xarray<float>*> sources = {&z, &y_mask, &g};
    std::vector<xt::xarray<float>> reshaped_sources;
    const auto sz = z.shape()[2];
    if(sz > 55)
        throw std::runtime_error("z shape[2] > 55");
    if(sz != 55) {
        reshaped_sources.resize(2);
        xt::xarray<float> buf = xt::zeros<float>({z.shape()[0], z.shape()[1], size_t(55)});
        xt::view(buf, xt::all(), xt::all(), xt::range(0, sz)) = z;
        reshaped_sources[0] = std::move(buf);
        sources[0] = &reshaped_sources[0];

        buf = xt::zeros<float>({y_mask.shape()[0], y_mask.shape()[1], size_t(55)});
        xt::view(buf, xt::all(), xt::all(), xt::range(0, sz)) = y_mask;
        reshaped_sources[1] = std::move(buf);
        sources[1] = &reshaped_sources[1];
    }
    std::vector<xt::xarray<__fp16>> buffers;
    buffers.reserve(sources.size());
    for (const auto& source : sources)
        buffers.emplace_back(xt::cast<__fp16>(*source));
    // release the potentialls used sources to reduce memory usage
    reshaped_sources.clear();

    for(int i = 0; i < input_attrs.size(); i++) {
        auto& attr = input_attrs[i];
        auto& buffer = buffers[i];

        inputs[i].index = attr.index;
        inputs[i].size = buffer.size() * sizeof(__fp16);
        inputs[i].type = RKNN_TENSOR_FLOAT16;
        inputs[i].fmt = RKNN_TENSOR_UNDEFINED;
        inputs[i].buf = buffer.data();
    }

    auto ret = rknn_inputs_set(ctx, input_attrs.size(), inputs.data());
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_inputs_set failed. Error code: " + std::to_string(ret));

    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_run failed. Error code: " + std::to_string(ret));

    ret = rknn_outputs_get(ctx, output_attrs.size(), outputs.data(), nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_outputs_get failed. Error code: " + std::to_string(ret));

    // output sanity check
    auto& outattr = output_attrs[outputs[0].index];
    if (outattr.type != RKNN_TENSOR_FLOAT16)
        throw std::runtime_error("output type mismatch. Expected RKNN_TENSOR_FLOAT16");

    size_t outsize = outattr.n_elems * (float)sz/55;

    __fp16* outbuf = reinterpret_cast<__fp16*>(outputs[0].buf);
    std::vector<int16_t> out(outsize);
    for (size_t i = 0; i < outsize; i++) {
        float val = static_cast<float>(outbuf[i]);
        val = std::min(std::max(val, -1.0f), 1.0f) * SHRT_MAX;
        out[i] = static_cast<int16_t>(val);
    }
    return out;
}
