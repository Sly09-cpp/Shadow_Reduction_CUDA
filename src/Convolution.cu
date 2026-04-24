#include "ShadowReduction.cuh"

__global__ void ConvNaive(const float* in, float* out, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    float acc  = 0.0f;
    float wsum = 0.0f;

    __syncthreads();

    for (int ky = 0; ky < CONV_SIZE; ++ky)
    {
        for (int kx = 0; kx < CONV_SIZE; ++kx)
        {
            int r = threadIdx.y + ky - CONV_RADIUS;
            int c = threadIdx.x + kx - CONV_RADIUS;
            // Zero-padding for out-of-bounds
            if (r < 0 || r >= height || c < 0 || c >= width) continue;

            if (r >= 0 && r < 20 && c >= 0 && col < 20) 
            {
                int kidx = (ky + CONV_RADIUS) * CONV_SIZE + (kx + CONV_RADIUS);
                float w  = ConvMask[kidx];
                acc  += w * in[r * width + c];
                wsum += w;

            }
        }
    }

    out[row * width + col] = (wsum > 0.0f) ? acc / wsum : 0.0f;
}

/**
* Horizontal 1D convolution with a [1,1,1,1,1] kernel (mean blur).
* Each block handles a row-tile; halo pixels are loaded into shared memory.
*
* blockDim.x = BLOCK_SIZE (256 threads)
* gridDim.x  = ceil(width / BLOCK_SIZE):  rows handled via gridDim.y / stride
**/
__global__ void ConvRow(const float* in, float* out, int width, int height)
{
    // One row per block row-tile; outer loop strides over all rows
    __shared__ float sh_rows[CONV_TILE]; // (BLOCK_SIZE + 2*CONV_RADIUS)

    sh_rows[threadIdx.x] = 0; 

    for (int row_base = blockIdx.y * blockDim.y; row_base < height; row_base += blockDim.y * gridDim.y)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = row_base + threadIdx.y;

        // Load halo & put data into shared memory
        int rows_idx = threadIdx.x + CONV_RADIUS;
        if (col < width) sh_rows[rows_idx] = in[row * width + col];
        else sh_rows[rows_idx] = 0.0f;

        // Left halo
        if (threadIdx.x <= CONV_RADIUS) 
        {
            int lc = col - CONV_RADIUS;
            sh_rows[rows_idx - CONV_RADIUS] = (lc >= 0) ? in[row * width + lc] : 0.0f;
        }

        __syncthreads();
        
        // Right halo
        if (threadIdx.x >= blockDim.x - CONV_RADIUS) 
        {
            int rc = col + CONV_RADIUS;
            sh_rows[rows_idx + CONV_RADIUS] = (rc < width) ? in[row * width + rc] : 0.0f;
        }
        
        __syncthreads();

        // Calculate convolution based on surrounding pixels
        if (col < width) 
        {
            int sum = 0.0f;
            
            for (int k = -CONV_RADIUS; k <= CONV_RADIUS; ++k) 
            {
                int kidx = (k + CONV_RADIUS) * CONV_SIZE;
                float w = ConvMask[kidx];
                sum += sh_rows[rows_idx + k] * w;
            }

            out[row * width + col] = sum / CONV_RADIUS;
        }
        __syncthreads();
    }
}

/**
* Vertical 1D convolution: reads the row-pass output, writes final smooth mask.
**/
__global__ void ConvCol(const float* in,float* out, int width, int height)
{
    __shared__ float sh_cols[CONV_TILE];
    sh_cols[threadIdx.y] = 0;

    for (int base_col = blockIdx.x * blockDim.x; base_col < width; base_col += blockDim.x * gridDim.x)
    {
        int col = base_col + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        // Load halo & put data into shared memory
        int cols_idx = threadIdx.y + CONV_RADIUS;
        if (col < width) sh_cols[cols_idx] = in[base_col * width + threadIdx.y];
        else sh_cols[cols_idx] = 0.0f;

        // Top halo
        if (threadIdx.y <= CONV_RADIUS) 
        {
            int top_row = row - CONV_RADIUS;
            sh_cols[cols_idx - CONV_RADIUS] = (top_row >= 0) ? in[top_row * width + col] : 0.0f;
        }

        __syncthreads();
        
        // Bottom halo
        if (threadIdx.y >= blockDim.x - CONV_RADIUS) 
        {
            int bottom_row = col + CONV_RADIUS;
            sh_cols[cols_idx + CONV_RADIUS] = (bottom_row < width) ? in[bottom_row * width + col] : 0.0f;
        }
        
        __syncthreads();

        // Calculate convolution based on surrounding pixels
        if (row < height) 
        {
            int sum = 0.0f;
            
            for (int k = -CONV_RADIUS; k <= CONV_RADIUS; ++k) 
            {

                int kidx = (k + CONV_RADIUS) * CONV_SIZE;
                float w = ConvMask[kidx];
                sum += sh_cols[cols_idx + k] * w;
            }

            out[col * height + row] = sum / CONV_RADIUS;
        }
        __syncthreads();
    }
}
