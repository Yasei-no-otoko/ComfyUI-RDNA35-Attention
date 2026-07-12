#include <torch/extension.h>

#include <hip/hip_version.h>
#include <pybind11/pybind11.h>
#include <torch/version.h>

#include <cstdint>
#include <tuple>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pisa_block_stats(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pisa_pack_spatial_qkv(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);
torch::Tensor pisa_unpack_spatial_output(torch::Tensor output, int64_t batch, int64_t heads);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "block_stats",
        &pisa_block_stats,
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"));
    module.def(
        "pack_spatial_qkv",
        &pisa_pack_spatial_qkv,
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"));
    module.def(
        "unpack_spatial_output",
        &pisa_unpack_spatial_output,
        pybind11::arg("output"),
        pybind11::arg("batch"),
        pybind11::arg("heads"));
    module.def("build_info", []() {
        pybind11::dict info;
        info["api"] = RDNA35_PISA_CK_API;
        info["architecture"] = "gfx1151";
        info["ck_commit"] = "4975bd0c8e17a54bdc27c746527a385e7383bb07";
        info["implementation"] = "ck_stats_flexattention_wmma";
        info["torch_build_version"] = TORCH_VERSION;
        info["hip_build_version"] = pybind11::make_tuple(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);
        info["block_size"] = 64;
        info["head_dim"] = 128;
        info["max_blocks"] = 144;
        return info;
    });
}
