#include <cuda_fp16.h>
#include <stddef.h>

extern "C" __global__ void tei_geglu_f16(
    const __half* input, __half* output, size_t count, size_t width) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    size_t row = i / width;
    size_t col = i % width;
    float x = __half2float(input[row * (2 * width) + col]);
    __half gate = input[row * (2 * width) + width + col];
    float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
        (x + 0.044715f * x * x * x)));
    // Retain the FP16 activation boundary before the FP16 product.
    output[i] = __hmul(__float2half_rn(gelu), gate);
}

extern "C" __global__ void tei_geglu_2048_half2(
    const __half2* input, __half2* output, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    size_t row = i >> 10;
    size_t col = i & 1023;
    float2 x = __half22float2(input[row * 2048 + col]);
    __half2 gate = input[row * 2048 + 1024 + col];
    float a = 0.5f * x.x * (1.0f + tanhf(0.7978845608028654f *
        (x.x + 0.044715f * x.x * x.x * x.x)));
    float b = 0.5f * x.y * (1.0f + tanhf(0.7978845608028654f *
        (x.y + 0.044715f * x.y * x.y * x.y)));
    output[i] = __hmul2(__floats2half2_rn(a, b), gate);
}
