#include <mach/mach_time.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

int main() {
    // Array size
    const int M = 32;
    const int N = 32;
    const int P = 32;
    const int bufferSize = M * N * sizeof(float);

    // 1. Create Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("Metal not supported\n");
        return 1;
    }

       // ========== ADD DEVICE PROPERTIES HERE ==========
   printf("=== Metal Device Properties ===\n");
   printf("Device name: %s\n", [[device name] UTF8String]);
   printf("Max threadgroup memory: %lu bytes\n", (unsigned long)[device maxThreadgroupMemoryLength]);
   printf("Max threads per threadgroup: %lu\n", (unsigned long)[device maxThreadsPerThreadgroup].width);
   printf("Max buffer length: %llu bytes (%.2f GB)\n", 
          (unsigned long long)[device maxBufferLength], 
          [device maxBufferLength] / (1024.0 * 1024.0 * 1024.0));
   printf("Unified memory: %s\n", [device hasUnifiedMemory] ? "Yes" : "No");
   printf("================================\n\n");

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
    id<MTLBuffer> bufferA = [device newBufferWithLength:(M * P * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:(P * N * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:(M * N * sizeof(float)) options:MTLResourceStorageModeShared];
    if (!bufferA || !bufferB || !bufferC) {
        printf("Failed to create buffers\n");
        return 1;
    }

    // Fix data initialization
    float *a = (float *)[bufferA contents];
    float *b = (float *)[bufferB contents];
    float *c_init = (float *)[bufferC contents];

    // Initialize A[M×P] properly
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < P; k++) {
            a[i * P + k] = (float)((i*P+k) % 5);  // contiguous layout
        }
    }

    // Initialize B[P×N] properly  
    for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
            b[k * N + j] = (float)((k*N+j) % 5);  // contiguous layout
        }
    }

    // Initialize C to zero
    for (int i = 0; i < M * N; i++) {
        c_init[i] = 0.0f;
    }

    id<MTLBuffer> bufferM = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferN = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferK = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferAlpha = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferBeta = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    *((int *)[bufferM contents]) = M;
    *((int *)[bufferN contents]) = N;
    *((int *)[bufferK contents]) = P;
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

    // 8. grid & threadblock settings
    MTLSize blockSize = MTLSizeMake(32, 32, 1); // maxes out at 1024
    MTLSize gridSize = MTLSizeMake(M/32, N/32, 1);

    uint64_t start = mach_absolute_time();
    [computeEncoder dispatchThreadgroups:gridSize
                threadsPerThreadgroup:blockSize];

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
    printf("<<<<");
    printf("Kernel execution time: %.3f ms\n", elapsed_ms);
    printf("CHANGE\n");

    float *c = (float *)[bufferC contents];
    bool correct = true;
    printf("Beginning correct checking loop\n");
    // Change this verification code:
    for (int i = 0; i < M; i++) {  // Changed from N to M
        printf("%d/%d\n", i, M);
        for (int j = 0; j < N; j++) {  // Changed from P to N
            double sum = 0.0;
            for (int k = 0; k < P; k++) {  // This stays P (which equals K)
                sum += a[i * P + k] * b[k * N + j];  // Changed indexing
            }
            if (sum != c[i * N + j]) {  // Changed from P to N
                if (c[i * N + j] != 0) {
                    printf("index: %d MISMATCH actual %.3f | expected %.3f\n", 
                    (i * N + j), c[i * N + j], sum);
                }
                // printf("index: %d MISMATCH actual %.3f | expected %.3f\n", 
                    // (i * N + j), c[i * N + j], sum);
                correct = false;
            }
        }
    }
    printf("Out of correct checking loop\n\n");
    if (!correct) {
        printf("WRONG");
    } else {
        printf("CORRECT");
    }


    // 11. Cleanup
    [commandQueue release];
    [library release];
    [kernelFunction release];
    [pipeline release];
    [bufferA release];
    [bufferB release];
    [bufferC release];
    [bufferM release];
    [bufferN release];
    [bufferK release];
    [device release];

    return 0;
}