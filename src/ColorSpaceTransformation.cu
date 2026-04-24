#include "ShadowReduction.cuh"

/**
*  Reads the packed RGB byte image once; in a single pass produces:
*    - Planar float R, G, B arrays (coalesced, for later reuse)
*    - U channel of YUV (chromaticity component)
*    - Grayscale from the color-invariant (CI) image
*
*  CI image: ci = atan(R / max(G, B))
*  YUV U:    U  = ((-38 * (r) - 74 * (g) + 112 * (b) + 128) >> 8) + 128
*  Grayscale: avg of the three CI channels
**/
__global__ void ColorSpaceTransformation(const uint8_t* rgb_in, float* R_out, float* G_out, float* B_out, float* yuv_u_out, float* gray_ci_out, int width, int height)
{
    // Get global x index
    int global_x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop - handles images of any size with any grid config 
    int total = width * height;
    for (int i = global_x_idx; i < total; i += blockDim.x * gridDim.x)
    {
        int r = rgb_in[3 * i + 0];
        int g = rgb_in[3 * i + 1];
        int b = rgb_in[3 * i + 2];

        // Store planar (coalesced write layout)
        R_out[i] = r;
        G_out[i] = g;
        B_out[i] = b;

        // YUV U component
        yuv_u_out[i] = ((-38 * (r) - 74 * (g) + 112 * (b) + 128) >> 8) + 128;

        // Color-invariant image (one channel per original channel)
        float ci_r = atan2f(r, fmaxf(g, b));
        float ci_g = atan2f(g, fmaxf(r, b));
        float ci_b = atan2f(b, fmaxf(r, g));

        gray_ci_out[i] = (ci_r + ci_g + ci_b);
    }
}