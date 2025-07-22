#include <metal_stdlib>
using namespace metal;

// Kernel execution time: 10.379 ms

kernel void add_vectors(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    threadgroup float* As [[threadgroup(0)]],
    threadgroup float* Bs [[threadgroup(1)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]
) {

    const uint BLOCKSIZE = 32;

    const int threadRow = thread_position_in_threadgroup.y;
    const int threadCol = thread_position_in_threadgroup.x;

    const uint cRow = threadgroup_position_in_grid.y;
    const uint cCol = threadgroup_position_in_grid.x;

    device const float* A_tile = A + cRow * BLOCKSIZE * K;           // row = cRow, col = 0
    device const float* B_tile = B + cCol * BLOCKSIZE;               // row = 0, col = cCol
    device float* C_tile = C + cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;

    // Loop over all tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Load tile from A into shared memory
        As[threadRow * BLOCKSIZE + threadCol] = 
            A_tile[threadRow * K + threadCol];

        // Load tile from B into shared memory
        Bs[threadRow * BLOCKSIZE + threadCol] = 
            B_tile[threadRow * N + threadCol];

        // Synchronize to make sure the tile is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply the two tiles together
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
        tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Move to next tile in A and B
        A_tile += BLOCKSIZE;
        B_tile += BLOCKSIZE * N;
    }

    int row = cRow * BLOCKSIZE + threadRow;
    int col = cCol * BLOCKSIZE + threadCol;

    // Write result to global memory if within bounds
    if (row < M && col < N) {
        C_tile[threadRow * N + threadCol] =
            alpha * tmp + beta * C_tile[threadRow * N + threadCol];
    }
}

