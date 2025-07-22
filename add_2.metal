#include <metal_stdlib>
using namespace metal;

// Kernel execution time: 13.384 ms

kernel void add_vectors(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]
) {

    const uint BLOCKSIZE = 512;

    const uint x = threadgroup_position_in_grid.x * BLOCKSIZE + (thread_position_in_threadgroup.x / BLOCKSIZE);
    const uint y = threadgroup_position_in_grid.y * BLOCKSIZE + (thread_position_in_threadgroup.y % BLOCKSIZE);

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (uint i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
