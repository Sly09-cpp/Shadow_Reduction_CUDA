#include "ShadowReduction.cuh"

/**
* Each thread-block accumulates a private histogram in shared memory
* (privatisation avoids most global atomicAdd collisions).  Final merge to
* global histogram uses one atomicAdd per bin per block.
*
* Grid-stride loop handles images larger than the grid.
**/
__global__ void GenerateHistogram(const float* data, unsigned int* hist, int total)
{
    __shared__ unsigned int sh_histo[HIST_BINS];

    int global_x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise shared histogram
    for (int i = threadIdx.x; i < HIST_BINS; i += blockDim.x) 
    {
        sh_histo[i] = 0;
    }
    
    __syncthreads();

    // Accumulate into histogram
    for (int i = global_x_idx; i < total; i += blockDim.x * gridDim.x)
    {
        int bin = data[i];
        atomicAdd(&sh_histo[bin], 1);
    }

    __syncthreads();

    // Transfer from shared buffer back to global
    for (int i = threadIdx.x; i < HIST_BINS; i += blockDim.x)
    {
        if (sh_histo[i] > 0) atomicAdd(&hist[i], sh_histo[i]);
    }
}

/**
* Single-block kernel computes prefix sums for omega(k) and mu(k)
* over the histogram on GPU, avoiding a CPU round-trip.
*
* omega(k) = sum(i=1 to k) p_i         (cumulative probability up to k)
* mu(k) = sigma(i=1 to k) i * p_i      (cumulative first-order moment up to k)
**/
__global__ void ObtainOmegaAndMu(const unsigned int* hist, float* omega, float* mu, float* mu_total, int total_pixels)
{
    __shared__ float sh_omega[HIST_BINS];
    __shared__ float sh_mu[HIST_BINS];

    int tid = threadIdx.x;

    float p = (float)hist[tid] / (float)total_pixels;
    sh_omega[tid] = p;
    sh_mu[tid] = (float)tid * p;
    
    __syncthreads();

    // Inclusive prefix sum
    for (int stride = 1; stride < HIST_BINS; stride = stride * 2) 
    {
        float v_omega = 0.0f, v_mu = 0.0f;
        if (tid >= stride) 
        {
            v_omega = sh_omega[tid - stride];
            v_mu = sh_mu[tid - stride];
        }
        __syncthreads();

        sh_omega[tid] += v_omega;
        sh_mu[tid] += v_mu;
        
        __syncthreads();
    }

    omega[tid] = sh_omega[tid];
    mu[tid]    = sh_mu[tid];

    if (tid == HIST_BINS - 1) *mu_total = sh_mu[HIST_BINS - 1];
}

/**
* Pick the most appropriate threshold based on the foreground and background variance
**/
__global__ void BetweenClassVariance(const float* omega, const float* mu, const float* mu_total, float* sigma_b, int n_bins)
{
    int tid = threadIdx.x;
    
    if (tid >= n_bins) return;

    float w = omega[tid];
    float m = mu[tid];
    float mT = *mu_total;

    float denom = w * (1.0f - w);
    if (denom < 1e-10f) 
    {
        sigma_b[tid] = 0.0f;
    } 
    else 
    {
        float num = (mT * w - m);
        sigma_b[tid] = (num * num) / denom;
    }
}

/**
* Find the unique global maximum using shared memory via tree reduction.
**/
__global__ void Argmax(const float* sigma_b, int* threshold_out, int n_bins)
{
    __shared__ float sh_val[HIST_BINS];
    __shared__ int   sh_idx[HIST_BINS];

    int tid = threadIdx.x;

    // Store into shared memory
    sh_val[tid] = sigma_b[tid];
    sh_idx[tid] = tid;
   
    __syncthreads();

    // Parallel reduction max
    for (int stride = HIST_BINS >> 1; stride > 0; stride >>= 1) 
    {
        if (tid < stride && sh_val[tid + stride] > sh_val[tid]) 
        {
            sh_val[tid] = sh_val[tid + stride];
            sh_idx[tid] = sh_idx[tid + stride];
        }
       
        __syncthreads();
    }

    // Put result back into global memory
    if (tid == 0) *threshold_out = sh_idx[0];
}

/**
* "Binarises" a single-channel float image using a scalar threshold.
* Pixels > threshold == foreground, else it's the background.
*
* Grid-stride loop for arbitrary image size.
**/
__global__ void ThresholdMask(const float* data, float* mask, const int* threshold_dev, int total)
{
    int global_x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float threshold = float(*threshold_dev);
    for (int i = global_x_idx; i < total; i += blockDim.x * gridDim.x)
    {
        mask[i] = (data[i] > threshold) ? 1.0f : 0.0f;
    }
}