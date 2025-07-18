#include <mach/mach_time.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

int main() {
    // Array size
    const int M = 4096;
    const int N = 4096;
    const int bufferSize = M * N * sizeof(float);

    // 1. Create Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("Metal not supported\n");
        return 1;
    }

    // 2. Create command queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        printf("Failed to create command queue\n");
        return 1;
    }

    // 3. Load Metal shader
    NSError *error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
        printf("Failed to create library\n");
        return 1;
    }
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"add_vectors"];
    if (!kernelFunction) {
        printf("Failed to find kernel function\n");
        return 1;
    }

    // 4. Create compute pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!pipeline) {
        printf("Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
        return 1;
    }

    // 5. Create buffers and initialize data
    id<MTLBuffer> bufferA = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    if (!bufferA || !bufferB || !bufferC) {
        printf("Failed to create buffers\n");
        return 1;
    }

    float *a = (float *)[bufferA contents];
    float *b = (float *)[bufferB contents];
    for (int i = 0; i < M * N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i*2);
    }
    id<MTLBuffer> bufferM = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferN = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferK = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferAlpha = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferBeta = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    *((int *)[bufferM contents]) = M;
    *((int *)[bufferN contents]) = N;
    *((int *)[bufferK contents]) = M;
    *((float *)[bufferAlpha contents]) = 1.0;
    *((float *)[bufferBeta contents]) = 0.0;

    // 6. Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !computeEncoder) {
        printf("Failed to create command buffer or encoder\n");
        return 1;
    }

    // 7. Set pipeline and buffers
    [computeEncoder setComputePipelineState:pipeline];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferM offset:0 atIndex:3];
    [computeEncoder setBuffer:bufferN offset:0 atIndex:4];
    [computeEncoder setBuffer:bufferK offset:0 atIndex:5];
    [computeEncoder setBuffer:bufferAlpha offset:0 atIndex:6];
    [computeEncoder setBuffer:bufferBeta offset:0 atIndex:7];

    uint BLOCKSIZE = 32;
    NSUInteger sharedMemSize = BLOCKSIZE * BLOCKSIZE * sizeof(float);
    [computeEncoder setThreadgroupMemoryLength:sharedMemSize atIndex:0]; // As
    [computeEncoder setThreadgroupMemoryLength:sharedMemSize atIndex:1]; // Bs


    // 8. Configure and dispatch threads
    MTLSize gridSize = MTLSizeMake(M, N, 1);
    MTLSize threadgroup = MTLSizeMake(32, 32, 1);
    uint64_t start = mach_absolute_time();
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroup];

    // 9. End encoding and commit
    [computeEncoder endEncoding];
    [commandBuffer commit];

    // 10. Wait and read results
    [commandBuffer waitUntilCompleted];
    uint64_t end = mach_absolute_time();

    // Convert to nanoseconds
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t elapsed_ns = (end - start) * timebase.numer / timebase.denom;
    double elapsed_ms = elapsed_ns / 1000000.0;

    printf("Kernel execution time: %.3f ms\n", elapsed_ms);


    float *c = (float *)[bufferC contents];
    for (int i = 0; i < 10; i++) { // Print first 10 results
        printf("C[%d] = %.1f\n", i, c[i]);
    }

    // 11. Cleanup
    [commandQueue release];
    [library release];
    [kernelFunction release];
    [pipeline release];
    [bufferA release];
    [bufferB release];
    [bufferC release];
    [device release];

    return 0;
}