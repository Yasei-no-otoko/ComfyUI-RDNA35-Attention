#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>

#include "ck_tile/core.hpp"


namespace {

constexpr int kBlockSize = 64;
constexpr int kHeadDim = 128;
constexpr int kMaxBlocks = 144;
constexpr int kMaxTokens = kBlockSize * kMaxBlocks;
constexpr int kSpatialSide = 96;
constexpr int kSpatialTile = 8;
constexpr int kCopyBytes = 16;
constexpr int kCopyElements = kCopyBytes / sizeof(ck_tile::bf16_t);
constexpr int kVectorsPerHead = kHeadDim / kCopyElements;
constexpr int kCopyThreads = 256;
constexpr int kTokensPerCopyBlock = kCopyThreads / kVectorsPerHead;
constexpr int kTokenCopyBlocks = kMaxTokens / kTokensPerCopyBlock;
constexpr int kPackTokensPerTile = 8;
constexpr int kPackHeadsPerTile = 8;
constexpr int kPackTokenTiles = kMaxTokens / kPackTokensPerTile;
constexpr int kPackTileVectors = kPackTokensPerTile * kPackHeadsPerTile * kVectorsPerHead;
constexpr int kPackIterations = kPackTileVectors / kCopyThreads;

static_assert(kMaxTokens == kSpatialSide * kSpatialSide);
static_assert(kHeadDim % kCopyElements == 0);
static_assert(kCopyThreads % kVectorsPerHead == 0);
static_assert(kMaxTokens % kTokensPerCopyBlock == 0);
static_assert(kPackTokensPerTile == kSpatialTile);
static_assert(kMaxTokens % kPackTokensPerTile == 0);
static_assert(kPackTileVectors % kCopyThreads == 0);
static_assert(kPackIterations == 4);
static_assert(kPackTileVectors * sizeof(uint4) == 16 * 1024);
static_assert(sizeof(uint4) == kCopyBytes);
static_assert(alignof(uint4) == kCopyBytes);


template <typename T>
CK_TILE_DEVICE float load_float(const T* pointer, int64_t index)
{
    return ck_tile::type_convert<float>(pointer[index]);
}


CK_TILE_DEVICE int ordered_to_raster(int ordered)
{
    const int block = ordered / (kSpatialTile * kSpatialTile);
    const int intra = ordered % (kSpatialTile * kSpatialTile);
    const int tiles_per_row = kSpatialSide / kSpatialTile;
    const int tile_row = block / tiles_per_row;
    const int tile_col = block % tiles_per_row;
    return (tile_row * kSpatialTile + intra / kSpatialTile) * kSpatialSide
        + tile_col * kSpatialTile + intra % kSpatialTile;
}


CK_TILE_DEVICE uint4 load_copy_vector(const ck_tile::bf16_t* pointer, int64_t element_offset)
{
    return *reinterpret_cast<const uint4*>(pointer + element_offset);
}


CK_TILE_DEVICE void store_copy_vector(ck_tile::bf16_t* pointer, int64_t element_offset, uint4 value)
{
    *reinterpret_cast<uint4*>(pointer + element_offset) = value;
}


template <bool SynchronizeAfterStore>
CK_TILE_DEVICE void transpose_pack_tile(
    const ck_tile::bf16_t* source,
    ck_tile::bf16_t* destination,
    uint4* tile,
    int64_t source_base,
    int64_t destination_base,
    int64_t heads,
    int valid_heads)
{
#pragma unroll
    for(int iteration = 0; iteration < kPackIterations; ++iteration)
    {
        const int linear = threadIdx.x + iteration * kCopyThreads;
        const int vector = linear % kVectorsPerHead;
        const int head = (linear / kVectorsPerHead) % kPackHeadsPerTile;
        const int token = linear / (kVectorsPerHead * kPackHeadsPerTile);
        const int64_t source_offset = source_base
            + (static_cast<int64_t>(token) * heads + head) * kHeadDim
            + static_cast<int64_t>(vector) * kCopyElements;
        if(head < valid_heads)
        {
            tile[linear] = load_copy_vector(source, source_offset);
        }
    }
    __syncthreads();

#pragma unroll
    for(int iteration = 0; iteration < kPackIterations; ++iteration)
    {
        const int linear = threadIdx.x + iteration * kCopyThreads;
        const int vector = linear % kVectorsPerHead;
        const int token = (linear / kVectorsPerHead) % kPackTokensPerTile;
        const int head = linear / (kVectorsPerHead * kPackTokensPerTile);
        const int tile_offset = (token * kPackHeadsPerTile + head) * kVectorsPerHead + vector;
        const int64_t destination_offset = destination_base
            + (static_cast<int64_t>(head) * kMaxTokens + token) * kHeadDim
            + static_cast<int64_t>(vector) * kCopyElements;
        if(head < valid_heads)
        {
            store_copy_vector(destination, destination_offset, tile[tile_offset]);
        }
    }

    if constexpr(SynchronizeAfterStore)
    {
        __syncthreads();
    }
}


template <typename T>
__global__ void block_stats_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ q_centroids,
    float* __restrict__ k_centroids,
    T* __restrict__ v_means,
    int tokens,
    int blocks)
{
    const int group = blockIdx.x;
    const int bh = group / blocks;
    const int sequence_block = group - bh * blocks;
    const int dim = threadIdx.x;
    const int start = sequence_block * kBlockSize;
    const int length = min(kBlockSize, tokens - start);

    const float inverse_length = 1.0f / static_cast<float>(length);
    float q_mean = 0.0f;
    float k_mean = 0.0f;
    float v_mean = 0.0f;
    for(int row = 0; row < length; ++row)
    {
        const int64_t offset = (static_cast<int64_t>(bh) * tokens + start + row) * kHeadDim + dim;
        q_mean += load_float(q, offset) * inverse_length;
        k_mean += load_float(k, offset) * inverse_length;
        v_mean += load_float(v, offset) * inverse_length;
    }

    const int64_t output = (static_cast<int64_t>(bh) * blocks + sequence_block) * kHeadDim + dim;
    q_centroids[output] = ck_tile::type_convert<T>(q_mean);
    k_centroids[output] = k_mean;
    v_means[output] = ck_tile::type_convert<T>(v_mean);
}


__global__ void pack_spatial_qkv_kernel(
    const ck_tile::bf16_t* __restrict__ q,
    const ck_tile::bf16_t* __restrict__ k,
    const ck_tile::bf16_t* __restrict__ v,
    ck_tile::bf16_t* __restrict__ packed_q,
    ck_tile::bf16_t* __restrict__ packed_k,
    ck_tile::bf16_t* __restrict__ packed_v,
    int64_t heads)
{
    __shared__ uint4 tile[kPackTileVectors];

    const int64_t batch = static_cast<int64_t>(blockIdx.z);
    const int ordered_tile = static_cast<int>(blockIdx.y);
    const int64_t head_tile = static_cast<int64_t>(blockIdx.x);
    const int ordered_base = ordered_tile * kPackTokensPerTile;
    const int raster_base = ordered_to_raster(ordered_base);
    const int64_t head_base = head_tile * kPackHeadsPerTile;
    const int64_t remaining_heads = heads - head_base;
    const int valid_heads = remaining_heads < kPackHeadsPerTile ? static_cast<int>(remaining_heads) : kPackHeadsPerTile;
    const int64_t source_base = ((batch * kMaxTokens + raster_base) * heads + head_base) * kHeadDim;
    const int64_t destination_base = ((batch * heads + head_base) * kMaxTokens + ordered_base) * kHeadDim;

    transpose_pack_tile<true>(q, packed_q, tile, source_base, destination_base, heads, valid_heads);
    transpose_pack_tile<true>(k, packed_k, tile, source_base, destination_base, heads, valid_heads);
    transpose_pack_tile<false>(v, packed_v, tile, source_base, destination_base, heads, valid_heads);
}


__global__ void unpack_spatial_output_kernel(
    const ck_tile::bf16_t* __restrict__ output,
    ck_tile::bf16_t* __restrict__ unpacked,
    int64_t heads)
{
    const int64_t batch_head = static_cast<int64_t>(blockIdx.x) / kTokenCopyBlocks;
    const int token_block = static_cast<int>(blockIdx.x % kTokenCopyBlocks);
    const int token_in_block = threadIdx.x / kVectorsPerHead;
    const int vector = threadIdx.x % kVectorsPerHead;
    const int ordered = token_block * kTokensPerCopyBlock + token_in_block;
    const int raster = ordered_to_raster(ordered);
    const int64_t batch = batch_head / heads;
    const int64_t head = batch_head % heads;
    const int64_t vector_offset = static_cast<int64_t>(vector) * kCopyElements;
    const int64_t source_offset = (batch_head * kMaxTokens + ordered) * kHeadDim + vector_offset;
    const int64_t destination_offset = ((batch * kMaxTokens + raster) * heads + head) * kHeadDim + vector_offset;

    store_copy_vector(unpacked, destination_offset, load_copy_vector(output, source_offset));
}


__global__ void fuse_spatial_epilogue_kernel(
    const ck_tile::bf16_t* __restrict__ exact_output,
    const float* __restrict__ exact_lse,
    const ck_tile::bf16_t* __restrict__ approximate_output,
    const float* __restrict__ approximate_lse,
    const float* __restrict__ correction,
    ck_tile::bf16_t* __restrict__ output,
    int64_t heads)
{
    __shared__ float exact_weights[kTokensPerCopyBlock];
    __shared__ float approximate_weights[kTokensPerCopyBlock];
    const int64_t batch_head = static_cast<int64_t>(blockIdx.x) / kTokenCopyBlocks;
    const int token_block = static_cast<int>(blockIdx.x % kTokenCopyBlocks);
    const int token_in_block = threadIdx.x / kVectorsPerHead;
    const int vector = threadIdx.x % kVectorsPerHead;
    const int ordered = token_block * kTokensPerCopyBlock + token_in_block;
    const int raster = ordered_to_raster(ordered);
    const int64_t batch = batch_head / heads;
    const int64_t head = batch_head % heads;
    const int64_t source_offset = (batch_head * kMaxTokens + ordered) * kHeadDim;
    const int64_t destination_offset = ((batch * kMaxTokens + raster) * heads + head) * kHeadDim;
    if(vector == 0)
    {
        const float exact_logsumexp = exact_lse[batch_head * kMaxTokens + ordered];
        const float approximate_logsumexp = approximate_lse[batch_head * kMaxTokens + ordered];
        const float maximum = fmaxf(exact_logsumexp, approximate_logsumexp);
        const float total_logsumexp = maximum + logf(expf(exact_logsumexp - maximum) + expf(approximate_logsumexp - maximum));
        exact_weights[token_in_block] = expf(exact_logsumexp - total_logsumexp);
        approximate_weights[token_in_block] = expf(approximate_logsumexp - total_logsumexp);
    }
    __syncthreads();
    const float exact_weight = exact_weights[token_in_block];
    const float approximate_weight = approximate_weights[token_in_block];
    const int element_base = vector * kCopyElements;

#pragma unroll
    for(int element = 0; element < kCopyElements; ++element)
    {
        const int64_t offset = source_offset + element_base + element;
        const float merged = load_float(exact_output, offset) * exact_weight
            + load_float(approximate_output, offset) * approximate_weight
            + correction[offset] * approximate_weight;
        output[destination_offset + element_base + element] = ck_tile::type_convert<ck_tile::bf16_t>(merged);
    }
}


void check_launch()
{
    const hipError_t error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "PISA block statistics failed: ", hipGetErrorString(error));
}


void check_copy_launch(const char* operation)
{
    const hipError_t error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, operation, " failed: ", hipGetErrorString(error));
}


void check_gfx1151(const torch::Tensor& tensor, const char* operation, hipDeviceProp_t& properties)
{
    const hipError_t error = hipGetDeviceProperties(&properties, tensor.get_device());
    TORCH_CHECK(error == hipSuccess, operation, " could not query HIP device properties: ", hipGetErrorString(error));
    const bool is_gfx1151 = std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0
        && (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':');
    TORCH_CHECK(is_gfx1151, operation, " only supports gfx1151.");
}


void check_aligned(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % kCopyBytes == 0,
        name,
        " must be 16-byte aligned.");
}


void check_spatial_bhtd_stride(const torch::Tensor& tensor, const char* name, int64_t heads)
{
    TORCH_CHECK(heads <= std::numeric_limits<int64_t>::max() / kHeadDim, name, " stride size overflow.");
    const int64_t token_stride = heads * kHeadDim;
    TORCH_CHECK(token_stride <= std::numeric_limits<int64_t>::max() / kMaxTokens, name, " stride size overflow.");
    const int64_t batch_stride = kMaxTokens * token_stride;
    TORCH_CHECK(
        tensor.stride(0) == batch_stride
            && tensor.stride(1) == kHeadDim
            && tensor.stride(2) == token_stride
            && tensor.stride(3) == 1,
        name,
        " must be a [B,H,9216,128] view over contiguous [B,9216,H,128] storage.");
}


int64_t copy_grid_blocks(int64_t batch_heads, const hipDeviceProp_t& properties, const char* operation)
{
    TORCH_CHECK(
        batch_heads <= std::numeric_limits<int64_t>::max() / kTokenCopyBlocks,
        operation,
        " grid size overflow.");
    const int64_t blocks = batch_heads * kTokenCopyBlocks;
    TORCH_CHECK(blocks <= properties.maxGridSize[0], operation, " input is too large for the gfx1151 launch grid.");
    return blocks;
}


dim3 pack_grid(int64_t batch, int64_t heads, const hipDeviceProp_t& properties, const char* operation)
{
    const int64_t head_tiles = (heads + kPackHeadsPerTile - 1) / kPackHeadsPerTile;
    TORCH_CHECK(head_tiles <= properties.maxGridSize[0], operation, " head count is too large for the gfx1151 launch grid.");
    TORCH_CHECK(kPackTokenTiles <= properties.maxGridSize[1], operation, " token count is too large for the gfx1151 launch grid.");
    TORCH_CHECK(batch <= properties.maxGridSize[2], operation, " batch size is too large for the gfx1151 launch grid.");
    return dim3(
        static_cast<std::uint32_t>(head_tiles),
        static_cast<std::uint32_t>(kPackTokenTiles),
        static_cast<std::uint32_t>(batch));
}


template <typename T>
void launch_block_stats(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    torch::Tensor& q_centroids,
    torch::Tensor& k_centroids,
    torch::Tensor& v_means,
    int tokens,
    int blocks,
    hipStream_t stream)
{
    const int batch_heads = static_cast<int>(q.size(0));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(block_stats_kernel<T>),
        dim3(batch_heads * blocks),
        dim3(kHeadDim),
        0,
        stream,
        reinterpret_cast<const T*>(q.data_ptr()),
        reinterpret_cast<const T*>(k.data_ptr()),
        reinterpret_cast<const T*>(v.data_ptr()),
        reinterpret_cast<T*>(q_centroids.data_ptr()),
        k_centroids.data_ptr<float>(),
        reinterpret_cast<T*>(v_means.data_ptr()),
        tokens,
        blocks);
    check_launch();
}


void launch_pack_spatial_qkv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    torch::Tensor& packed_q,
    torch::Tensor& packed_k,
    torch::Tensor& packed_v,
    int64_t heads,
    dim3 grid,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        pack_spatial_qkv_kernel,
        grid,
        dim3(kCopyThreads),
        0,
        stream,
        reinterpret_cast<const ck_tile::bf16_t*>(q.data_ptr()),
        reinterpret_cast<const ck_tile::bf16_t*>(k.data_ptr()),
        reinterpret_cast<const ck_tile::bf16_t*>(v.data_ptr()),
        reinterpret_cast<ck_tile::bf16_t*>(packed_q.data_ptr()),
        reinterpret_cast<ck_tile::bf16_t*>(packed_k.data_ptr()),
        reinterpret_cast<ck_tile::bf16_t*>(packed_v.data_ptr()),
        heads);
    check_copy_launch("PISA spatial QKV pack");
}


void launch_unpack_spatial_output(
    const torch::Tensor& output,
    torch::Tensor& unpacked,
    int64_t heads,
    int64_t grid_blocks,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        unpack_spatial_output_kernel,
        dim3(static_cast<std::uint32_t>(grid_blocks)),
        dim3(kCopyThreads),
        0,
        stream,
        reinterpret_cast<const ck_tile::bf16_t*>(output.data_ptr()),
        reinterpret_cast<ck_tile::bf16_t*>(unpacked.data_ptr()),
        heads);
    check_copy_launch("PISA spatial output unpack");
}


void launch_fuse_spatial_epilogue(
    const torch::Tensor& exact_output,
    const torch::Tensor& exact_lse,
    const torch::Tensor& approximate_output,
    const torch::Tensor& approximate_lse,
    const torch::Tensor& correction,
    torch::Tensor& output,
    int64_t heads,
    int64_t grid_blocks,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        fuse_spatial_epilogue_kernel,
        dim3(static_cast<std::uint32_t>(grid_blocks)),
        dim3(kCopyThreads),
        0,
        stream,
        reinterpret_cast<const ck_tile::bf16_t*>(exact_output.data_ptr()),
        exact_lse.data_ptr<float>(),
        reinterpret_cast<const ck_tile::bf16_t*>(approximate_output.data_ptr()),
        approximate_lse.data_ptr<float>(),
        correction.data_ptr<float>(),
        reinterpret_cast<ck_tile::bf16_t*>(output.data_ptr()),
        heads);
    check_copy_launch("PISA spatial fused epilogue");
}

} // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pisa_block_stats(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "PISA CK requires HIP device tensors.");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, and v must be on the same device.");
    TORCH_CHECK(q.dim() == 3 && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "PISA CK expects matching [BH,T,D] q/k/v.");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) == kHeadDim, "PISA CK requires BH>0, T>0, and D=128.");
    TORCH_CHECK(q.size(1) <= kMaxTokens, "PISA CK supports T<=9216.");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "PISA CK requires contiguous q/k/v.");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q, k, and v must have the same dtype.");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "PISA CK supports bfloat16 only.");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(), "PISA CK is forward-only.");

    const int tokens = static_cast<int>(q.size(1));
    const int blocks = (tokens + kBlockSize - 1) / kBlockSize;
    hipDeviceProp_t properties{};
    check_gfx1151(q, "PISA block statistics", properties);
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() / blocks, "PISA block statistics grid size overflow.");
    const int64_t grid_blocks = q.size(0) * blocks;
    TORCH_CHECK(grid_blocks <= properties.maxGridSize[0], "PISA block statistics input is too large for the gfx1151 launch grid.");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    auto q_centroids = torch::empty({q.size(0), blocks, kHeadDim}, q.options());
    auto k_centroids = torch::empty({q.size(0), blocks, kHeadDim}, q.options().dtype(torch::kFloat32));
    auto v_means = torch::empty_like(q_centroids);

    launch_block_stats<ck_tile::bf16_t>(q, k, v, q_centroids, k_centroids, v_means, tokens, blocks, stream);

    return {q_centroids, k_centroids, v_means};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pisa_pack_spatial_qkv(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "PISA spatial QKV pack requires HIP device tensors.");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, and v must be on the same device.");
    TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "PISA spatial QKV pack expects matching [B,H,T,D] q/k/v.");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) == kMaxTokens && q.size(3) == kHeadDim, "PISA spatial QKV pack requires B>0, H>0, T=9216, and D=128.");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q, k, and v must have the same dtype.");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "PISA spatial QKV pack supports bfloat16 only.");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(), "PISA spatial QKV pack is forward-only.");

    const int64_t batch = q.size(0);
    const int64_t heads = q.size(1);
    TORCH_CHECK(batch <= std::numeric_limits<int64_t>::max() / heads, "PISA spatial QKV pack batch-head size overflow.");
    const int64_t batch_heads = batch * heads;
    check_spatial_bhtd_stride(q, "q", heads);
    check_spatial_bhtd_stride(k, "k", heads);
    check_spatial_bhtd_stride(v, "v", heads);
    check_aligned(q, "q");
    check_aligned(k, "k");
    check_aligned(v, "v");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    hipDeviceProp_t properties{};
    check_gfx1151(q, "PISA spatial QKV pack", properties);
    const dim3 grid = pack_grid(batch, heads, properties, "PISA spatial QKV pack");
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    auto packed_q = torch::empty({batch_heads, kMaxTokens, kHeadDim}, q.options());
    auto packed_k = torch::empty_like(packed_q);
    auto packed_v = torch::empty_like(packed_q);
    check_aligned(packed_q, "packed q");
    check_aligned(packed_k, "packed k");
    check_aligned(packed_v, "packed v");

    launch_pack_spatial_qkv(q, k, v, packed_q, packed_k, packed_v, heads, grid, stream);
    return {packed_q, packed_k, packed_v};
}


torch::Tensor pisa_unpack_spatial_output(torch::Tensor output, int64_t batch, int64_t heads)
{
    TORCH_CHECK(output.is_cuda(), "PISA spatial output unpack requires a HIP device tensor.");
    TORCH_CHECK(batch > 0 && heads > 0, "PISA spatial output unpack requires positive batch and heads.");
    TORCH_CHECK(batch <= std::numeric_limits<int64_t>::max() / heads, "PISA spatial output unpack batch-head size overflow.");
    const int64_t batch_heads = batch * heads;
    TORCH_CHECK(
        output.dim() == 3
            && output.size(0) == batch_heads
            && output.size(1) == kMaxTokens
            && output.size(2) == kHeadDim,
        "PISA spatial output unpack expects contiguous [B*H,9216,128] input.");
    TORCH_CHECK(output.is_contiguous(), "PISA spatial output unpack requires contiguous input.");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "PISA spatial output unpack supports bfloat16 only.");
    TORCH_CHECK(!output.requires_grad(), "PISA spatial output unpack is forward-only.");
    TORCH_CHECK(heads <= std::numeric_limits<int64_t>::max() / kHeadDim, "PISA spatial output unpack head size overflow.");
    check_aligned(output, "output");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{output.device()};
    hipDeviceProp_t properties{};
    check_gfx1151(output, "PISA spatial output unpack", properties);
    const int64_t grid_blocks = copy_grid_blocks(batch_heads, properties, "PISA spatial output unpack");
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    auto unpacked = torch::empty({batch, kMaxTokens, heads * kHeadDim}, output.options());
    check_aligned(unpacked, "unpacked output");

    launch_unpack_spatial_output(output, unpacked, heads, grid_blocks, stream);
    return unpacked;
}


torch::Tensor pisa_fuse_spatial_epilogue(
    torch::Tensor exact_output,
    torch::Tensor exact_lse,
    torch::Tensor approximate_output,
    torch::Tensor approximate_lse,
    torch::Tensor correction,
    int64_t batch,
    int64_t heads)
{
    TORCH_CHECK(batch > 0 && heads > 0, "PISA spatial fused epilogue requires positive batch and heads.");
    TORCH_CHECK(batch <= std::numeric_limits<int64_t>::max() / heads, "PISA spatial fused epilogue batch-head size overflow.");
    const int64_t batch_heads = batch * heads;
    const auto check_output = [batch_heads](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP device tensor.");
        TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == batch_heads && tensor.size(1) == kMaxTokens && tensor.size(2) == kHeadDim, name, " must have shape [B*H,9216,128].");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
        TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16.");
        TORCH_CHECK(!tensor.requires_grad(), name, " is forward-only.");
    };
    const auto check_lse = [batch_heads](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP device tensor.");
        TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == batch_heads && tensor.size(1) == kMaxTokens, name, " must have shape [B*H,9216].");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
        TORCH_CHECK(!tensor.requires_grad(), name, " is forward-only.");
    };
    check_output(exact_output, "exact output");
    check_output(approximate_output, "approximate output");
    TORCH_CHECK(correction.is_cuda(), "correction must be a HIP device tensor.");
    TORCH_CHECK(correction.dim() == 3 && correction.size(0) == batch_heads && correction.size(1) == kMaxTokens && correction.size(2) == kHeadDim, "correction must have shape [B*H,9216,128].");
    TORCH_CHECK(correction.is_contiguous(), "correction must be contiguous.");
    TORCH_CHECK(correction.scalar_type() == at::kFloat, "correction must be float32.");
    TORCH_CHECK(!correction.requires_grad(), "correction is forward-only.");
    check_lse(exact_lse, "exact LSE");
    check_lse(approximate_lse, "approximate LSE");
    TORCH_CHECK(
        exact_output.device() == approximate_output.device()
            && exact_output.device() == correction.device()
            && exact_output.device() == exact_lse.device()
            && exact_output.device() == approximate_lse.device(),
        "PISA spatial fused epilogue tensors must share one device.");
    check_aligned(exact_output, "exact output");
    check_aligned(approximate_output, "approximate output");
    check_aligned(correction, "correction");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{exact_output.device()};
    hipDeviceProp_t properties{};
    check_gfx1151(exact_output, "PISA spatial fused epilogue", properties);
    const int64_t grid_blocks = copy_grid_blocks(batch_heads, properties, "PISA spatial fused epilogue");
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    auto output = torch::empty({batch, kMaxTokens, heads * kHeadDim}, exact_output.options());
    check_aligned(output, "fused output");
    launch_fuse_spatial_epilogue(
        exact_output,
        exact_lse,
        approximate_output,
        approximate_lse,
        correction,
        output,
        heads,
        grid_blocks,
        stream);
    return output;
}
