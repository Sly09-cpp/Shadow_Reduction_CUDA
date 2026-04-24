#include "ShadowReduction.cuh"

/**
* Two-level parallel reduction on the GPU.  Works for any array size.
* First pass: reduce each block to a partial sum.
* Second pass: reduce partial sums to a scalar (on host if <= MAX_BLOCKS).
**/
float GpuSum(const float* d_arr, float* d_tmp, int total)
{
    int n_blocks = min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
    ReduceSum<<<n_blocks, BLOCK_SIZE>>>(d_arr, d_tmp, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Finish on CPU for the small partial-sum array
    float* h_partial = (float*)malloc(n_blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_tmp, n_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (int i = 0; i < n_blocks; ++i)
    {
        sum += h_partial[i];
    } 

    free(h_partial);
    return sum;
}

/**
* Compact all Otsu's method and thresholding procedures into on helper function for readibility
**/
int RunOtsu( const float* d_img, float* d_mask, unsigned int* d_hist, float* d_omega, float* d_mu, float* d_mu_total, float* d_sigma_b, int* d_threshold, int total)
{
    cudaEvent_t t0, t1;

    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    
    // Set all values in histogram to 0
    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(unsigned int)));
    
    int n_blocks = min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
    
    // Generate histograms
    CUDA_CHECK(cudaEventRecord(t0));
    GenerateHistogram<<<n_blocks, BLOCK_SIZE>>>(d_img, d_hist, total);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
    printf("\tGenerateHistogram compute time: %.6f ms\n", elapsed);
    
    // Omega(k) and mu(k) prefix scans
    CUDA_CHECK(cudaEventRecord(t0));
    ObtainOmegaAndMu<<<1, HIST_BINS>>>(d_hist, d_omega, d_mu, d_mu_total, total);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
    printf("\tObtainOmegaAndMu compute time: %.6f ms\n", elapsed);

    // Between-class variance 
    CUDA_CHECK(cudaEventRecord(t0));
    BetweenClassVariance<<<1, HIST_BINS>>>(d_omega, d_mu, d_mu_total, d_sigma_b, HIST_BINS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
    printf("\tBetweenClassVariance compute time: %.6f ms\n", elapsed);

    // Argmax
    CUDA_CHECK(cudaEventRecord(t0));
    Argmax<<<1, HIST_BINS>>>(d_sigma_b, d_threshold, HIST_BINS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
    printf("\tArgmax compute time: %.6f ms\n", elapsed);

    // Generate mask
    CUDA_CHECK(cudaEventRecord(t0));
    ThresholdMask<<<n_blocks, BLOCK_SIZE>>>(d_img, d_mask, d_threshold, total);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
    printf("\tThresholdMask compute time: %.6f ms\n", elapsed);

    // Retrieve threshold for diagnostics
    int h_threshold = 0;
    CUDA_CHECK(cudaMemcpy(&h_threshold, d_threshold, sizeof(int), cudaMemcpyDeviceToHost));
    return h_threshold;
}


uint8_t* ReadPPM(const char* path, int* w, int* h)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    { 
        perror("fopen");
        return NULL; 
    }

    char magic[3]; 
    int maxval;
    if (fscanf(f, "%2s\n%d %d\n%d\n", magic, w, h, &maxval) != 4 || magic[1] != '6') 
    {
        fprintf(stderr, "Not a P6 PPM.\n"); fclose(f); 
        return NULL;
    }

    size_t size = (size_t)(*w) * (*h) * 3;
    uint8_t* buf = (uint8_t*)malloc(size);
    fread(buf, 1, size, f);
    fclose(f);
    return buf;
}

void WritePPM(const char* path, const uint8_t* data, int w, int h)
{
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)w * h * 3, f);
    fclose(f);
}