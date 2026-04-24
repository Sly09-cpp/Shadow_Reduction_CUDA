#include "ShadowReduction.cuh"

/**
* 2D grid of BLOCK_DIM×BLOCK_DIM thread blocks.
* The shared-memory tile is stored as a 1D array for fewer PTX load instructions.
* Grid-stride loop handles images larger than the grid.
**/
__global__ void ErosionSharedMem(const float* in, float* out, int width, int height)
{
    __shared__ float sh_tiles[EROSION_TILE_SIZE];

    for (int tile_row = blockIdx.y * blockDim.y + threadIdx.y; tile_row < height; tile_row += blockDim.y * gridDim.y)
    {
        for (int tile_col = blockIdx.x * blockDim.x + threadIdx.x; tile_col < width; tile_col += blockDim.x * gridDim.x)
        {
            // Local indeces
            int row = threadIdx.y + EROSION_RADIUS;
            int col = threadIdx.x + EROSION_RADIUS;

            auto safe_load = [&](int r, int c) -> float 
            {
                if (r < 0 || r >= height || c < 0 || c >= width) return 1.0f; // pad with 1 for erosion
                return in[r * width + c];
            };
            
            // Load center pixels into shared buffer 
            sh_tiles[row * EROSION_TILE_W + col] = safe_load(tile_row, tile_col);

            // Load halo rows (top/bottom)
            sh_tiles[(row - EROSION_RADIUS) * EROSION_TILE_W + col] = safe_load(tile_row - EROSION_RADIUS, tile_col);
            sh_tiles[(row + EROSION_RADIUS) * EROSION_TILE_W + col] = safe_load(tile_row + EROSION_RADIUS, tile_col);

            // Load halo cols (left/right)
            sh_tiles[row * EROSION_TILE_W + (col - EROSION_RADIUS)] = safe_load(tile_row, tile_col - EROSION_RADIUS);
            sh_tiles[row * EROSION_TILE_W + (col + EROSION_RADIUS)] = safe_load(tile_row, tile_col + EROSION_RADIUS);
            __syncthreads();

            // Pixel equals 1 only if all neighbors are 1 
            if (tile_row < height && tile_col < width) 
            {
                float result = 1.0f;
                for (int dr = -EROSION_RADIUS; dr <= EROSION_RADIUS; ++dr) 
                {    
                    for (int dc = -EROSION_RADIUS; dc <= EROSION_RADIUS; ++dc) 
                    {
                        int sr = row + dr;
                        int sc = col + dc;
                        result = fminf(result, sh_tiles[sr * EROSION_TILE_W + sc]);
                    }
                }
                
                out[tile_row * width + tile_col] = result;
            }
            __syncthreads();
        }
    }
}