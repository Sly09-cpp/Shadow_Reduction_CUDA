#include "ShadowReduction.cuh"

/**
* Produces 6 arrays:  light_{R,G,B}  and  shadow_{R,G,B}
* Grid-stride loop for arbitrary size.
**/
__global__ void MapMaskedPixels(const float* R,const float* G,const float* B,
                                         const float* light_mask, 
                                         const float* shadow_mask,
                                         float* light_R, float* light_G, float* light_B,
                                         float* shadow_R,float* shadow_G,float* shadow_B,
                                         int total)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
    {
        float lm = light_mask[idx];
        float sm = shadow_mask[idx];
        light_R[idx]  = R[idx] * lm;
        light_G[idx]  = G[idx] * lm;
        light_B[idx]  = B[idx] * lm;
        shadow_R[idx] = R[idx] * sm;
        shadow_G[idx] = G[idx] * sm;
        shadow_B[idx] = B[idx] * sm;
    }
}

/**
* Handles arrays larger than a single block via grid-stride pre-accumulation
* followed by a shared-memory reduction.
*
* Call with ceil(total/BLOCK_SIZE) blocks and repeat if partial sums remain.
* For very large images (>4K) a "two-pass" reduction is used by the host.
**/
__global__ void ReduceSum(const float* in, float* partial_sums, int total)
{
    // Initialize shared buffer
    __shared__ float sh_data[BLOCK_SIZE];
    
    // Get IDs
    int global_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = global_x_idx; i < total; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }
    sh_data[tid] = sum;
    
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1) 
    {
        if (tid < stride) sh_data[tid] += sh_data[tid + stride];
        
        __syncthreads();
    }

    // Handle last warp 
    if (tid < 32) 
    {
        float* vs = sh_data;
        
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }

    if (tid == 0) partial_sums[blockIdx.x] = sh_data[0];
}

/**
* Per-pixel shadow removal.
* out_channel[p] = in_channel[p] * (shadow_mask[p]*ratio + light_mask[p])
*
* Clamps output to [0, 255].
**/
__global__ void ApplyRatio(
    const float* in_R, const float* in_G, const float* in_B, const float* light_mask, const float* shadow_mask,
    float ratio_R, float ratio_G, float ratio_B,
    float* out_R, float* out_G, float* out_B, int total)
{

    // Get global x index ID for thread
    int global_x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = global_x_idx; idx < total; idx += blockDim.x * gridDim.x)
    {
        float lm = light_mask[idx];
        float sm = shadow_mask[idx];
        
        float scale_R = sm * ratio_R + lm;
        float scale_G = sm * ratio_G + lm;
        float scale_B = sm * ratio_B + lm;
        
        out_R[idx] = fminf(255.0f, fmaxf(0.0f, in_R[idx] * scale_R));
        out_G[idx] = fminf(255.0f, fmaxf(0.0f, in_G[idx] * scale_G));
        out_B[idx] = fminf(255.0f, fmaxf(0.0f, in_B[idx] * scale_B));
    }
}

/**
* Convert back to byte RGB for exporting.
**/
__global__ void PackOutput(const float* R, const float* G, const float* B, uint8_t* rgb_out, int total)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
    {
        rgb_out[3 * idx + 0] = R[idx];
        rgb_out[3 * idx + 1] = G[idx];
        rgb_out[3 * idx + 2] = B[idx];
    }
}