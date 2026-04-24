// Local includes
#include "ShadowReduction.cuh"

/** 
*   Program arguments:
*   1. Input PPM file to be operated on
*   2. Name of the result that will be exported
**/
int main(int argc, char** argv)
{
    if (argc != 3 ) 
    {
        fprintf(stderr, "Invalid arguments! \nUsage: %s <input.ppm> <output.ppm>\n", argv[0]);
        return 1;
    }

    // Read image and obtain its properties
    int img_width = 0, img_height = 0;
    uint8_t* h_rgb = ReadPPM(argv[1], &img_width, &img_height);
    if (!h_rgb) return 1;
    printf("Image: %d x %d (%d MP)\n", img_width, img_height, (img_width * img_height) / 1000);

    int total_img_size = img_width * img_height;

    // Allocate GPU resources: 

    // Given that each pixel equals one byte, to load the image I will use uint8_t's...
    uint8_t *d_rgb_in, *d_rgb_out;
    
    // However, all the operations will be performed with floating points which will then be converted back to one byte data entries
    float *d_r, *d_g, *d_b;
    float *d_yuv_u, *d_gray_ci;
    float *d_yuv_mask, *d_gray_mask;
    float *d_conv_tmp, *d_smooth_mask;
    float *d_eroded_light, *d_eroded_shadow;
    float *d_light_r, *d_light_g, *d_light_b;
    float *d_shadow_r, *d_shadow_g, *d_shadow_b;
    float *d_out_r, *d_out_g, *d_out_b;
    float *d_reduce_tmp;

    // Otsu working buffers
    unsigned int *d_hist;
    float *d_omega, *d_mu, *d_mu_total_img_size, *d_sigma_b;
    int   *d_threshold;

    float float_size = total_img_size * sizeof(float);
    int byte_size = total_img_size * 3 * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&d_rgb_in,       byte_size));
    CUDA_CHECK(cudaMalloc(&d_rgb_out,      byte_size));
    CUDA_CHECK(cudaMalloc(&d_r,            float_size));
    CUDA_CHECK(cudaMalloc(&d_g,            float_size));
    CUDA_CHECK(cudaMalloc(&d_b,            float_size));
    CUDA_CHECK(cudaMalloc(&d_yuv_u,        float_size));
    CUDA_CHECK(cudaMalloc(&d_gray_ci,      float_size));
    CUDA_CHECK(cudaMalloc(&d_yuv_mask,     float_size));
    CUDA_CHECK(cudaMalloc(&d_gray_mask,    float_size));
    CUDA_CHECK(cudaMalloc(&d_conv_tmp,     float_size));
    CUDA_CHECK(cudaMalloc(&d_smooth_mask,  float_size));
    CUDA_CHECK(cudaMalloc(&d_eroded_light, float_size));
    CUDA_CHECK(cudaMalloc(&d_eroded_shadow,float_size));
    CUDA_CHECK(cudaMalloc(&d_light_r,      float_size));
    CUDA_CHECK(cudaMalloc(&d_light_g,      float_size));
    CUDA_CHECK(cudaMalloc(&d_light_b,      float_size));
    CUDA_CHECK(cudaMalloc(&d_shadow_r,     float_size));
    CUDA_CHECK(cudaMalloc(&d_shadow_g,     float_size));
    CUDA_CHECK(cudaMalloc(&d_shadow_b,     float_size));
    CUDA_CHECK(cudaMalloc(&d_out_r,        float_size));
    CUDA_CHECK(cudaMalloc(&d_out_g,        float_size));
    CUDA_CHECK(cudaMalloc(&d_out_b,        float_size));

    int max_blocks = min((total_img_size + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
    CUDA_CHECK(cudaMalloc(&d_reduce_tmp, max_blocks * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_hist,      HIST_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_omega,     HIST_BINS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu,        HIST_BINS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu_total_img_size,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma_b,   HIST_BINS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_threshold, sizeof(int)));

    // Copy image to device resource buffer
    CUDA_CHECK(cudaMemcpy(d_rgb_in, h_rgb, byte_size, cudaMemcpyHostToDevice));

    // Create events to test compute timing performance
    cudaEvent_t total_t0, total_t1;
    
    CUDA_CHECK(cudaEventCreate(&total_t0));
    CUDA_CHECK(cudaEventCreate(&total_t1));
    
    // Create events to test performance between kernels (Stage 2 will be handled inside helper function)
    cudaEvent_t stage1_t0, stage1_t1;
    cudaEvent_t stage3_t0, stage3_t1;
    cudaEvent_t stage4_t0, stage4_t1;
    cudaEvent_t stage5_t0, stage5_t1;
    
    CUDA_CHECK(cudaEventCreate(&stage1_t0)); 
    CUDA_CHECK(cudaEventCreate(&stage1_t1));
    CUDA_CHECK(cudaEventCreate(&stage3_t0)); 
    CUDA_CHECK(cudaEventCreate(&stage3_t1));
    CUDA_CHECK(cudaEventCreate(&stage4_t0)); 
    CUDA_CHECK(cudaEventCreate(&stage4_t1));
    CUDA_CHECK(cudaEventCreate(&stage5_t0)); 
    CUDA_CHECK(cudaEventCreate(&stage5_t1));
    
    // Record start timestamp
    CUDA_CHECK(cudaEventRecord(total_t0));

    // Stage 1 - Colorspace Transformation
    {
        int n_blk = min((total_img_size + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        
        // Record stage 1 start timer
        CUDA_CHECK(cudaEventRecord(stage1_t0));
        
        ColorSpaceTransformation<<<n_blk, BLOCK_SIZE>>>(d_rgb_in, d_r, d_g, d_b, d_yuv_u, d_gray_ci, img_width, img_height);
        
        // Record stage 1 end timer
        float elapsed = 0;
        
        CUDA_CHECK(cudaEventRecord(stage1_t1));
        CUDA_CHECK(cudaEventSynchronize(stage1_t1));

        CUDA_CHECK(cudaEventElapsedTime(&elapsed, stage1_t0, stage1_t1));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        printf("\n[Stage 1] - Colorspace transform complete. Compute time: %.6f ms\n", elapsed);
    }

    // Stage 2 - Thresholding & Otsu's Method
    {
        printf("\n[Stage 2] - Otsu's method and Thresholding\n");
        
        int t_yuv = RunOtsu(d_yuv_u, d_yuv_mask, d_hist, d_omega, d_mu, d_mu_total_img_size, d_sigma_b, d_threshold, total_img_size);
        CUDA_CHECK(cudaGetLastError());
        
        int t_gray = RunOtsu(d_gray_ci, d_gray_mask, d_hist, d_omega, d_mu, d_mu_total_img_size, d_sigma_b, d_threshold, total_img_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 3 - 1D Separable Convolution
    {
        // Shared memory size = tile + 2 * radius
        int row_mem_size = (BLOCK_SIZE + 2 * CONV_RADIUS) * sizeof(float);
        int col_mem_size = (BLOCK_SIZE + 2 * CONV_RADIUS) * sizeof(float);

        // Calculate dimensions for both kernels
        dim3 row_bDim(BLOCK_SIZE, 1);
        dim3 row_gDim(min((img_width + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS), min(img_height, MAX_BLOCKS));
        dim3 col_bDim(1, BLOCK_SIZE);
        dim3 col_gDim(min(img_width, MAX_BLOCKS), min((img_height + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS));
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((img_width  + BLOCK_SIZE - 1) / BLOCK_SIZE, (img_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

        cudaStream_t s1, s2;

        cudaStreamCreate(&s1);
        cudaStreamCreate(&s2);

        // Record event start
        CUDA_CHECK(cudaEventRecord(stage3_t0));

        ConvRow<<<row_gDim, row_bDim, row_mem_size, s1>>>(d_yuv_mask, d_conv_tmp, img_width, img_height);
        ConvCol<<<col_gDim, col_bDim, col_mem_size, s2>>>(d_conv_tmp, d_smooth_mask, img_width, img_height);

        // Record stop event
        CUDA_CHECK(cudaEventRecord(stage3_t1));
        CUDA_CHECK(cudaEventSynchronize(stage3_t1));

        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);

        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, stage3_t0, stage3_t1));

        printf("\n[Stage 3] - Convolution complete. Compute time: %.6f ms\n", elapsed);
    }

    // Stage 4 - Erosion 
    {
        dim3 bDim(GRID_DIM, GRID_DIM);
        dim3 gDim(min((img_width + GRID_DIM - 1) / GRID_DIM, MAX_BLOCKS), min((img_height + GRID_DIM - 1) / GRID_DIM, MAX_BLOCKS));

        cudaStream_t s1, s2;

        cudaStreamCreate(&s1);
        cudaStreamCreate(&s2);

        // Record start event 
        CUDA_CHECK(cudaEventRecord(stage4_t0));
        
        ErosionSharedMem<<<gDim, bDim, 0, s1>>>(d_gray_mask,  d_eroded_light,  img_width, img_height);
        ErosionSharedMem<<<gDim, bDim, 0, s2>>>(d_yuv_mask,   d_eroded_shadow, img_width, img_height);
        
        // Record stop event
        CUDA_CHECK(cudaEventRecord(stage4_t1));
        CUDA_CHECK(cudaEventSynchronize(stage4_t1));
        
        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
        
        CUDA_CHECK(cudaGetLastError());
        
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, stage4_t0, stage4_t1));

        printf("[Stage 4] - Erosion complete. Compute time: %.6f ms\n", elapsed);
    }

    // Stage 5 - Result Integration
    {
        int n_blk = min((total_img_size + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);

        // Map pixels by mask
        MapMaskedPixels<<<n_blk, BLOCK_SIZE>>>(d_r, d_g, d_b,
            d_eroded_light, d_eroded_shadow,
            d_light_r,  d_light_g,  d_light_b,
            d_shadow_r, d_shadow_g, d_shadow_b,
            total_img_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        /// Compute averages via parallel reduction with streaming
        float sum_light_r  = GpuSum(d_light_r,  d_reduce_tmp, total_img_size);
        float sum_light_g  = GpuSum(d_light_g,  d_reduce_tmp, total_img_size);
        float sum_light_b  = GpuSum(d_light_b,  d_reduce_tmp, total_img_size);
        float sum_shadow_r = GpuSum(d_shadow_r, d_reduce_tmp, total_img_size);
        float sum_shadow_g = GpuSum(d_shadow_g, d_reduce_tmp, total_img_size);
        float sum_shadow_b = GpuSum(d_shadow_b, d_reduce_tmp, total_img_size);
        float sum_light_mask  = GpuSum(d_eroded_light,  d_reduce_tmp, total_img_size);
        float sum_shadow_mask = GpuSum(d_eroded_shadow, d_reduce_tmp, total_img_size);

        float avg_light_r  = (sum_light_mask  > 0) ? sum_light_r  / sum_light_mask  : 1.0f;
        float avg_light_g  = (sum_light_mask  > 0) ? sum_light_g  / sum_light_mask  : 1.0f;
        float avg_light_b  = (sum_light_mask  > 0) ? sum_light_b  / sum_light_mask  : 1.0f;
        float avg_shadow_r = (sum_shadow_mask > 0) ? sum_shadow_r / sum_shadow_mask : 1.0f;
        float avg_shadow_g = (sum_shadow_mask > 0) ? sum_shadow_g / sum_shadow_mask : 1.0f;
        float avg_shadow_b = (sum_shadow_mask > 0) ? sum_shadow_b / sum_shadow_mask : 1.0f;

        // Compute ratios
        float ratio_r = (avg_shadow_r > 1.0f) ? avg_light_r / avg_shadow_r : 1.0f;
        float ratio_g = (avg_shadow_g > 1.0f) ? avg_light_g / avg_shadow_g : 1.0f;
        float ratio_b = (avg_shadow_b > 1.0f) ? avg_light_b / avg_shadow_b : 1.0f;
        printf("[Stage 5] - Ratios  R=%.4f  G=%.4f  B=%.4f\n", ratio_r, ratio_g, ratio_b);

        // Apply ratios to remove shadow
        ApplyRatio<<<n_blk, BLOCK_SIZE>>>(
            d_r, d_g, d_b,
            d_eroded_light, d_eroded_shadow,
            ratio_r, ratio_g, ratio_b,
            d_out_r, d_out_g, d_out_b,
            total_img_size);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        printf("[Stage 5] - Result integration complete.\n");
    }

    // Export result image
    {
        int n_blk = min((total_img_size + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        PackOutput<<<n_blk, BLOCK_SIZE>>>(d_out_r, d_out_g, d_out_b, d_rgb_out, total_img_size);
    }

    CUDA_CHECK(cudaEventRecord(total_t1));
    CUDA_CHECK(cudaEventSynchronize(total_t1));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, total_t0, total_t1));
    printf("Total Compute time: %.6f ms\n", ms);

    uint8_t* h_out = (uint8_t*)malloc(total_img_size * 3);
    CUDA_CHECK(cudaMemcpy(h_out, d_rgb_out, total_img_size * 3, cudaMemcpyDeviceToHost));
    WritePPM(argv[2], h_out, img_width, img_height);
    printf("Output written to %s\n", argv[2]);

    // Release host resources:
    free(h_rgb);
    free(h_out);

    // Release GPU resources:
    cudaFree(d_rgb_in);
    cudaFree(d_rgb_out);

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);

    cudaFree(d_yuv_u);
    cudaFree(d_yuv_mask);
    
    cudaFree(d_gray_ci);
    cudaFree(d_gray_mask);

    cudaFree(d_conv_tmp);
    cudaFree(d_smooth_mask);
    
    cudaFree(d_eroded_light);
    cudaFree(d_eroded_shadow);

    cudaFree(d_light_r);
    cudaFree(d_light_g);
    cudaFree(d_light_b);

    cudaFree(d_shadow_r);
    cudaFree(d_shadow_g);
    cudaFree(d_shadow_b);

    cudaFree(d_out_r);
    cudaFree(d_out_g);
    cudaFree(d_out_b);

    cudaFree(d_reduce_tmp);

    cudaFree(d_hist);
    cudaFree(d_omega);
    cudaFree(d_mu);

    cudaFree(d_mu_total_img_size);
    cudaFree(d_sigma_b);
    cudaFree(d_threshold);

    cudaEventDestroy(total_t0);
    cudaEventDestroy(total_t1);

    return 0;
}