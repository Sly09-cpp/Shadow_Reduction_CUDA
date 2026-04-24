#ifndef SHADOWREDUCTION_CUH
#define SHADOWREDUCTION_CUH

// Global/Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Defines
#define HIST_BINS          1024                                       // Total of number of bins 
#define GRID_DIM           32                                         // Tile side length for 2D kernels
#define BLOCK_SIZE         128                                        // Threads per block for 1D kernels
#define CONV_MASK_SIZE     5
#define CONV_RADIUS        1                                          // Half-width of 5x5 convolution kernel
#define CONV_SIZE          3
#define CONV_TILE          (GRID_DIM + 2 * CONV_RADIUS)
#define EROSION_RADIUS     1
#define EROSION_SIZE       3
#define EROSION_TILE_W     (GRID_DIM + 2 * EROSION_RADIUS)
#define EROSION_TILE_H     (GRID_DIM + 2 * EROSION_RADIUS)
#define EROSION_TILE_SIZE  (EROSION_TILE_W * EROSION_TILE_H)
#define MAX_BLOCKS         65535                                      // Max grid dimension

// Check for errors
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t  e = (call);                                              \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(e));               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

typedef struct 
{
    int    width, height;
    float *r, *g, *b;
} Image;

// Used to deetermne how much each neighboring pixel contributes to the output
__constant__ float ConvMask[CONV_MASK_SIZE * CONV_MASK_SIZE] = 
{
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
};

// Helper functions
float GpuSum(const float* d_arr, float* d_tmp, int total);
int RunOtsu( const float* d_img, float* d_mask, unsigned int* d_hist, float* d_omega, float* d_mu, float* d_mu_total, float* d_sigma_b, int* d_threshold, int total);
uint8_t* ReadPPM(const char* path, int* w, int* h);
void WritePPM(const char* path, const uint8_t* data, int w, int h);

// Stage 1 - Color Space Transformation
__global__ void ColorSpaceTransformation(const uint8_t* rgb_in, float* R_out, float* G_out, float* B_out, float* yuv_u_out, float* gray_ci_out, int width, int height);

// Stage 2 - Thresholding and Otsu's Method
__global__ void GenerateHistogram(const float* data, unsigned int* hist, int total);
__global__ void ObtainOmegaAndMu(const unsigned int* hist, float* omega, float* mu, float* mu_total, int total_pixels);
__global__ void BetweenClassVariance(const float* omega, const float* mu, const float* mu_total, float* sigma_b, int n_bins);
__global__ void Argmax(const float* sigma_b, int* threshold_out, int n_bins);
__global__ void ThresholdMask(const float* data, float* mask, const int* threshold_dev, int total);

// Stage 3 - Convolution
__global__ void ConvNaive(const float* in, float* out, int width, int height); // Naive 
__global__ void ConvRow(const float* in, float* out, int width, int height);
__global__ void ConvCol(const float* in,float* out, int width, int height);

// Stage 4 - Erosion
__global__ void ErosionSharedMem(const float* in, float* out, int width, int height);
__global__ void ErosionOptimized(const float* in, float* out, int width, int height); 

// Stage 5 - Result Integration
__global__ void MapMaskedPixels(const float* R,const float* G,const float* B, const float* light_mask, const float* shadow_mask, float* light_R, float* light_G, float* light_B, float* shadow_R,float* shadow_G,float* shadow_B, int total);
__global__ void ReduceSum(const float* in, float* partial_sums, int total);
__global__ void ApplyRatio(const float* in_R, const float* in_G, const float* in_B, const float* light_mask, const float* shadow_mask, float ratio_R, float ratio_G, float ratio_B, float* out_R, float* out_G, float* out_B, int total);
__global__ void PackOutput(const float* R, const float* G, const float* B, uint8_t* rgb_out, int total);

#endif // SHADOWREDUCTION_CUH