/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "internal.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "device_utility.h"
#include "host_utility.h"

#if CUDA_VERSION >= 9000
#define SHFL_UP(mask, var, delta, w) __shfl_up_sync((mask), (var), (delta), (w))
#define SHFL_DOWN(mask, var, delta, w) __shfl_down_sync((mask), (var), (delta), (w))
#else
#define SHFL_UP(mask, var, delta, width) __shfl_up((var), (delta), (width))
#define SHFL_DOWN(mask, var, delta, width) __shfl_down((var), (delta), (width))
#endif

namespace sgm
{

    using COST_TYPE = cost_type;

    namespace cost_aggregation
    {

        template<typename T>
        __device__ inline int popcnt(T x)
        {
            return 0;
        }
        template<>
        __device__ inline int popcnt(uint32_t x)
        {
            return __popc(x);
        }
        template<>
        __device__ inline int popcnt(uint64_t x)
        {
            return __popcll(x);
        }

        template<unsigned int DP_BLOCK_SIZE, unsigned int SUBGROUP_SIZE>
        struct DynamicProgramming
        {
            static_assert(DP_BLOCK_SIZE >= 2, "DP_BLOCK_SIZE must be greater than or equal to 2");
            static_assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1)) == 0, "SUBGROUP_SIZE must be a power of 2");

            uint32_t last_min;
            uint32_t dp[DP_BLOCK_SIZE];

            __device__ DynamicProgramming() : last_min(0)
            {
                for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
                {
                    dp[i] = 0;
                }
            }

            __device__ void update(uint32_t *local_costs, uint32_t p1, uint32_t p2, uint32_t mask)
            {
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;

                const auto dp0 = dp[0];
                uint32_t lazy_out = 0, local_min = 0;
                {
                    const unsigned int k = 0;
                    const uint32_t prev = SHFL_UP(mask, dp[DP_BLOCK_SIZE - 1], 1, WARP_SIZE);
                    uint32_t out = min(dp[k] - last_min, p2);
                    if (lane_id != 0)
                    {
                        out = min(out, prev - last_min + p1);
                    }
                    out = min(out, dp[k + 1] - last_min + p1);
                    lazy_out = local_min = out + local_costs[k];
                }
                for (unsigned int k = 1; k + 1 < DP_BLOCK_SIZE; ++k)
                {
                    uint32_t out = min(dp[k] - last_min, p2);
                    out = min(out, dp[k - 1] - last_min + p1);
                    out = min(out, dp[k + 1] - last_min + p1);
                    dp[k - 1] = lazy_out;
                    lazy_out = out + local_costs[k];
                    local_min = min(local_min, lazy_out);
                }
                {
                    const unsigned int k = DP_BLOCK_SIZE - 1;
                    const uint32_t next = SHFL_DOWN(mask, dp0, 1, WARP_SIZE);
                    uint32_t out = min(dp[k] - last_min, p2);
                    out = min(out, dp[k - 1] - last_min + p1);
                    if (lane_id + 1 != SUBGROUP_SIZE)
                    {
                        out = min(out, next - last_min + p1);
                    }
                    dp[k - 1] = lazy_out;
                    dp[k] = out + local_costs[k];
                    local_min = min(local_min, dp[k]);
                }
                last_min = subgroup_min<SUBGROUP_SIZE>(local_min, mask);
            }
        };

        template<unsigned int SIZE>
        __device__ unsigned int generate_mask()
        {
            static_assert(SIZE <= 32, "SIZE must be less than or equal to 32");
            return static_cast<unsigned int>((1ull << SIZE) - 1u);
        }

        template<typename CENSUS_T>
        __device__ inline CENSUS_T load_census_with_check(const CENSUS_T *ptr, int x, int w)
        {
            return x >= 0 && x < w ? __ldg(ptr + x) : 0;
        }

        namespace vertical
        {

            static constexpr unsigned int DP_BLOCK_SIZE = 16u;
            static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

            template<typename CENSUS_TYPE, int DIRECTION, unsigned int MAX_DISPARITY>
            __global__ void aggregate_vertical_path_kernel(uint8_t *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1,
                                                           unsigned int p2, int min_disp)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
                static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

                static_assert(DIRECTION == 1 || DIRECTION == -1, "");
                if (width == 0 || height == 0)
                {
                    return;
                }

                __shared__ CENSUS_TYPE right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
                DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
                const unsigned int shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

                const unsigned int x = blockIdx.x * PATHS_PER_BLOCK + warp_id * PATHS_PER_WARP + group_id;
                const unsigned int right_x0 = blockIdx.x * PATHS_PER_BLOCK;
                const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

                const unsigned int right0_addr = (right_x0 + PATHS_PER_BLOCK - 1) - x + dp_offset;
                const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
                const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

                for (unsigned int iter = 0; iter < height; ++iter)
                {
                    const unsigned int y = (DIRECTION > 0 ? iter : height - 1 - iter);
                    // Load left to register
                    CENSUS_TYPE left_value;
                    if (x < width)
                    {
                        left_value = left[x + y * width];
                    }
                    // Load right to smem
                    for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE)
                    {
                        const unsigned int i = i0 + threadIdx.x;
                        if (i < RIGHT_BUFFER_SIZE)
                        {
                            const int right_x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
                            const CENSUS_TYPE right_value = load_census_with_check(&right[y * width], right_x, width);
                            const unsigned int lo = i % DP_BLOCK_SIZE;
                            const unsigned int hi = i / DP_BLOCK_SIZE;
                            right_buffer[lo][hi] = right_value;
                            if (hi > 0)
                            {
                                right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                            }
                        }
                    }
                    __syncthreads();
                    // Compute
                    if (x < width)
                    {
                        CENSUS_TYPE right_values[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
                        }
                        uint32_t local_costs[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            local_costs[j] = popcnt(left_value ^ right_values[j]);
                        }
                        dp.update(local_costs, p1, p2, shfl_mask);
                        store_uint8_vector<DP_BLOCK_SIZE>(&dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width], dp.dp);
                    }
                    __syncthreads();
                }
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_up2down(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2, int min_disp,
                                   cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_vertical_path_kernel<CENSUS_TYPE, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_down2up(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2, int min_disp,
                                   cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_vertical_path_kernel<CENSUS_TYPE, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

        } // namespace vertical

        namespace horizontal
        {

            static constexpr unsigned int DP_BLOCK_SIZE = 8u;
            static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;

            static constexpr unsigned int WARPS_PER_BLOCK = 4u;
            static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

            template<typename CENSUS_TYPE, int DIRECTION, unsigned int MAX_DISPARITY>
            __global__ void aggregate_horizontal_path_kernel(uint8_t *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1,
                                                             unsigned int p2, int min_disp)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int SUBGROUPS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_WARP = WARP_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

                static_assert(DIRECTION == 1 || DIRECTION == -1, "");
                if (width == 0 || height == 0)
                {
                    return;
                }

                CENSUS_TYPE right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
                DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp[DP_BLOCKS_PER_THREAD];

                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
                const unsigned int shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

                const unsigned int y0 = PATHS_PER_BLOCK * blockIdx.x + PATHS_PER_WARP * warp_id + group_id;
                const unsigned int feature_step = SUBGROUPS_PER_WARP * width;
                const unsigned int dest_step = SUBGROUPS_PER_WARP * MAX_DISPARITY * width;
                const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;
                left += y0 * width;
                right += y0 * width;
                dest += y0 * MAX_DISPARITY * width;

                if (y0 >= height)
                {
                    return;
                }

                // initialize census buffer
                {
                    const int x0 = (DIRECTION > 0 ? -1 : width) - (min_disp + static_cast<int>(dp_offset));
                    for (int dy = 0; dy < DP_BLOCKS_PER_THREAD; ++dy)
                        for (int dx = 0; dx < DP_BLOCK_SIZE; ++dx)
                            right_buffer[dy][dx] = load_census_with_check(&right[dy * feature_step], x0 - dx, width);
                }

                int x0 = (DIRECTION > 0) ? 0 : static_cast<int>((width - 1) & ~(DP_BLOCK_SIZE - 1));
                for (unsigned int iter = 0; iter < width; iter += DP_BLOCK_SIZE)
                {
                    for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
                    {
                        const unsigned int x = x0 + (DIRECTION > 0 ? i : (DP_BLOCK_SIZE - 1 - i));
                        if (x >= width)
                        {
                            continue;
                        }
                        for (unsigned int j = 0; j < DP_BLOCKS_PER_THREAD; ++j)
                        {
                            const unsigned int y = y0 + j * SUBGROUPS_PER_WARP;
                            if (y >= height)
                            {
                                continue;
                            }
                            const CENSUS_TYPE left_value = __ldg(&left[j * feature_step + x]);
                            if (DIRECTION > 0)
                            {
                                const CENSUS_TYPE t = right_buffer[j][DP_BLOCK_SIZE - 1];
                                for (unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k)
                                {
                                    right_buffer[j][k] = right_buffer[j][k - 1];
                                }
                                right_buffer[j][0] = SHFL_UP(shfl_mask, t, 1, SUBGROUP_SIZE);
                                if (lane_id == 0)
                                {
                                    right_buffer[j][0] = load_census_with_check(&right[j * feature_step], x - min_disp, width);
                                }
                            }
                            else
                            {
                                const CENSUS_TYPE t = right_buffer[j][0];
                                for (unsigned int k = 1; k < DP_BLOCK_SIZE; ++k)
                                {
                                    right_buffer[j][k - 1] = right_buffer[j][k];
                                }
                                right_buffer[j][DP_BLOCK_SIZE - 1] = SHFL_DOWN(shfl_mask, t, 1, SUBGROUP_SIZE);
                                if (lane_id + 1 == SUBGROUP_SIZE)
                                {
                                    right_buffer[j][DP_BLOCK_SIZE - 1] = load_census_with_check(&right[j * feature_step], x - (min_disp + dp_offset + DP_BLOCK_SIZE - 1), width);
                                }
                            }
                            uint32_t local_costs[DP_BLOCK_SIZE];
                            for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k)
                            {
                                local_costs[k] = popcnt(left_value ^ right_buffer[j][k]);
                            }
                            dp[j].update(local_costs, p1, p2, shfl_mask);
                            store_uint8_vector<DP_BLOCK_SIZE>(&dest[j * dest_step + x * MAX_DISPARITY + dp_offset], dp[j].dp);
                        }
                    }
                    x0 += static_cast<int>(DP_BLOCK_SIZE) * DIRECTION;
                }
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_left2right(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2, int min_disp,
                                      cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

                const int gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_horizontal_path_kernel<CENSUS_TYPE, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_right2left(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2, int min_disp,
                                      cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

                const int gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_horizontal_path_kernel<CENSUS_TYPE, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

        } // namespace horizontal

        namespace oblique
        {

            static constexpr unsigned int DP_BLOCK_SIZE = 16u;
            static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

            template<typename CENSUS_TYPE, int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
            __global__ void aggregate_oblique_path_kernel(uint8_t *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2,
                                                          int min_disp)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
                static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

                static_assert(X_DIRECTION == 1 || X_DIRECTION == -1, "");
                static_assert(Y_DIRECTION == 1 || Y_DIRECTION == -1, "");
                if (width == 0 || height == 0)
                {
                    return;
                }

                __shared__ CENSUS_TYPE right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS];
                DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
                const unsigned int shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

                const int x0 = blockIdx.x * PATHS_PER_BLOCK + warp_id * PATHS_PER_WARP + group_id + (X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
                const int right_x00 = blockIdx.x * PATHS_PER_BLOCK + (X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
                const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

                const unsigned int right0_addr = static_cast<unsigned int>(right_x00 + PATHS_PER_BLOCK - 1 - x0) + dp_offset;
                const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
                const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

                for (unsigned int iter = 0; iter < height; ++iter)
                {
                    const int y = static_cast<int>(Y_DIRECTION > 0 ? iter : height - 1 - iter);
                    const int x = x0 + static_cast<int>(iter) * X_DIRECTION;
                    const int right_x0 = right_x00 + static_cast<int>(iter) * X_DIRECTION;
                    // Load right to smem
                    for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE)
                    {
                        const unsigned int i = i0 + threadIdx.x;
                        if (i < RIGHT_BUFFER_SIZE)
                        {
                            const int right_x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
                            const CENSUS_TYPE right_value = load_census_with_check(&right[y * width], right_x, width);
                            const unsigned int lo = i % DP_BLOCK_SIZE;
                            const unsigned int hi = i / DP_BLOCK_SIZE;
                            right_buffer[lo][hi] = right_value;
                            if (hi > 0)
                            {
                                right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                            }
                        }
                    }
                    __syncthreads();
                    // Compute
                    if (0 <= x && x < static_cast<int>(width))
                    {
                        const CENSUS_TYPE left_value = __ldg(&left[x + y * width]);
                        CENSUS_TYPE right_values[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
                        }
                        uint32_t local_costs[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            local_costs[j] = popcnt(left_value ^ right_values[j]);
                        }
                        dp.update(local_costs, p1, p2, shfl_mask);
                        store_uint8_vector<DP_BLOCK_SIZE>(&dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width], dp.dp);
                    }
                    __syncthreads();
                }
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_upleft2downright(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2,
                                            int min_disp, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel<CENSUS_TYPE, 1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_upright2downleft(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2,
                                            int min_disp, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel<CENSUS_TYPE, -1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_downright2upleft(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2,
                                            int min_disp, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel<CENSUS_TYPE, -1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_downleft2upright(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right, int width, int height, unsigned int p1, unsigned int p2,
                                            int min_disp, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel<CENSUS_TYPE, 1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(dest, left, right, width, height, p1, p2, min_disp);
                CUDA_CHECK(cudaGetLastError());
            }

        } // namespace oblique

        namespace vertical_clipped
        {

            static constexpr unsigned int DP_BLOCK_SIZE = 16u;
            static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

            template<typename CENSUS_TYPE, int DIRECTION, unsigned int MAX_DISPARITY>
            __global__ void aggregate_vertical_path_kernel_clipped(
                uint8_t *dest,
                const CENSUS_TYPE *left,
                const CENSUS_TYPE *right,
                int width, int height,
                unsigned int p1, unsigned int p2,
                const int32_t *__restrict__ d_range_image,
                const uint32_t *__restrict__ d_range_offset,
                int range_length)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
                static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

                static_assert(DIRECTION == 1 || DIRECTION == -1, "");
                if (width == 0 || height == 0)
                {
                    return;
                }

                __shared__ CENSUS_TYPE right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
                DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
                const unsigned int shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

                const unsigned int x = blockIdx.x * PATHS_PER_BLOCK + warp_id * PATHS_PER_WARP + group_id;
                const unsigned int right_x0 = blockIdx.x * PATHS_PER_BLOCK;
                const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

                const unsigned int right0_addr = (right_x0 + PATHS_PER_BLOCK - 1) - x + dp_offset;
                const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
                const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

                for (unsigned int iter = 0; iter < height; ++iter)
                {
                    const unsigned int y = (DIRECTION > 0 ? iter : height - 1 - iter);
                    const unsigned int linearIdx = y * width + x;

                    // Load per-pixel range data
                    int minDisp = 1, maxDisp = 0; // invalid by default
                    uint32_t pixelOffset = 0;
                    if (x < width)
                    {
                        minDisp = d_range_image[linearIdx * 2];
                        maxDisp = d_range_image[linearIdx * 2 + 1];
                        pixelOffset = d_range_offset[linearIdx];
                    }
                    const bool isValid = (minDisp <= maxDisp) && (x < width);
                    const int rangeWidth = isValid ? (maxDisp - minDisp + 1) : 0;

                    // If pixel is invalid, reset DP state
                    if (!isValid)
                    {
                        dp.last_min = 0;
                        for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
                        {
                            dp.dp[i] = 0;
                        }
                    }

                    // Load left to register
                    CENSUS_TYPE left_value = 0;
                    if (isValid)
                    {
                        left_value = left[x + y * width];
                    }

                    // Load right to smem using per-pixel minDisp
                    // Use the first valid pixel's minDisp in the block for shared memory loading
                    // For vertical paths, all pixels in a column have different y but same x,
                    // so each iteration we reload with that pixel's minDisp
                    __shared__ int block_min_disp;
                    if (threadIdx.x == 0)
                    {
                        block_min_disp = minDisp;
                    }
                    __syncthreads();
                    // Use the first thread's minDisp as representative for right buffer loading
                    // This is a simplification: in vertical paths, all threads in a block
                    // process different x positions at the same y, so minDisp can vary per thread.
                    // We use per-thread minDisp for the census lookup.
                    const int load_min_disp = minDisp;

                    for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE)
                    {
                        const unsigned int i = i0 + threadIdx.x;
                        if (i < RIGHT_BUFFER_SIZE)
                        {
                            // Use the block_min_disp as baseline for shared memory loading
                            const int right_x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - block_min_disp);
                            const CENSUS_TYPE right_value = load_census_with_check(&right[y * width], right_x, width);
                            const unsigned int lo = i % DP_BLOCK_SIZE;
                            const unsigned int hi = i / DP_BLOCK_SIZE;
                            right_buffer[lo][hi] = right_value;
                            if (hi > 0)
                            {
                                right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                            }
                        }
                    }
                    __syncthreads();

                    // Compute
                    if (isValid)
                    {
                        // Adjust right buffer read address for per-pixel minDisp difference
                        const int disp_shift = block_min_disp - load_min_disp;
                        const unsigned int adj_right0_addr = right0_addr + disp_shift;
                        const unsigned int adj_right0_addr_lo = adj_right0_addr % DP_BLOCK_SIZE;
                        const unsigned int adj_right0_addr_hi = adj_right0_addr / DP_BLOCK_SIZE;

                        CENSUS_TYPE right_values[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            right_values[j] = right_buffer[adj_right0_addr_lo + j][adj_right0_addr_hi];
                        }
                        uint32_t local_costs[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            // For disparities beyond range width, set max cost
                            const int globalD = dp_offset + j;
                            if (globalD < rangeWidth)
                            {
                                local_costs[j] = popcnt(left_value ^ right_values[j]);
                            }
                            else
                            {
                                local_costs[j] = 255u;
                            }
                        }
                        dp.update(local_costs, p1, p2, shfl_mask);

                        // Store to flat cost buffer using per-pixel offset
                        // Only store valid disparity entries
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            const int globalD = dp_offset + j;
                            if (globalD < rangeWidth)
                            {
                                dest[pixelOffset + globalD] = static_cast<uint8_t>(dp.dp[j]);
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_up2down_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                           int width, int height, unsigned int p1, unsigned int p2,
                                           const int32_t *d_range_image, const uint32_t *d_range_offset,
                                           int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_vertical_path_kernel_clipped<CENSUS_TYPE, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_down2up_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                           int width, int height, unsigned int p1, unsigned int p2,
                                           const int32_t *d_range_image, const uint32_t *d_range_offset,
                                           int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_vertical_path_kernel_clipped<CENSUS_TYPE, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

        } // namespace vertical_clipped

        namespace horizontal_clipped
        {

            static constexpr unsigned int DP_BLOCK_SIZE = 8u;
            static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;

            static constexpr unsigned int WARPS_PER_BLOCK = 4u;
            static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

            template<typename CENSUS_TYPE, int DIRECTION, unsigned int MAX_DISPARITY>
            __global__ void aggregate_horizontal_path_kernel_clipped(
                uint8_t *dest,
                const CENSUS_TYPE *left,
                const CENSUS_TYPE *right,
                int width, int height,
                unsigned int p1, unsigned int p2,
                const int32_t *__restrict__ d_range_image,
                const uint32_t *__restrict__ d_range_offset,
                int range_length)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int SUBGROUPS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_WARP = WARP_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

                static_assert(DIRECTION == 1 || DIRECTION == -1, "");
                if (width == 0 || height == 0)
                {
                    return;
                }

                CENSUS_TYPE right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
                DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp[DP_BLOCKS_PER_THREAD];

                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
                const unsigned int shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

                const unsigned int y0 = PATHS_PER_BLOCK * blockIdx.x + PATHS_PER_WARP * warp_id + group_id;
                const unsigned int feature_step = SUBGROUPS_PER_WARP * width;
                const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

                if (y0 >= height)
                {
                    return;
                }

                // Initialize census buffer with first pixel's minDisp
                {
                    const unsigned int firstX = (DIRECTION > 0) ? 0 : (width - 1);
                    for (int dy = 0; dy < DP_BLOCKS_PER_THREAD; ++dy)
                    {
                        const unsigned int y = y0 + dy * SUBGROUPS_PER_WARP;
                        if (y < height)
                        {
                            const unsigned int linearIdx = y * width + firstX;
                            const int firstMinDisp = d_range_image[linearIdx * 2];
                            const int firstMaxDisp = d_range_image[linearIdx * 2 + 1];
                            const bool firstValid = (firstMinDisp <= firstMaxDisp);
                            const int initMinDisp = firstValid ? firstMinDisp : 0;

                            const int x0init = (DIRECTION > 0 ? -1 : width) - (initMinDisp + static_cast<int>(dp_offset));
                            for (int dx = 0; dx < DP_BLOCK_SIZE; ++dx)
                                right_buffer[dy][dx] = load_census_with_check(&right[y * width], x0init - dx, width);
                        }
                        else
                        {
                            for (int dx = 0; dx < DP_BLOCK_SIZE; ++dx)
                                right_buffer[dy][dx] = 0;
                        }
                    }
                }

                int prevMinDisp[DP_BLOCKS_PER_THREAD];
                for (int dy = 0; dy < DP_BLOCKS_PER_THREAD; ++dy)
                {
                    const unsigned int y = y0 + dy * SUBGROUPS_PER_WARP;
                    if (y < height)
                    {
                        const unsigned int firstX = (DIRECTION > 0) ? 0 : (width - 1);
                        const unsigned int linearIdx = y * width + firstX;
                        const int firstMinDisp = d_range_image[linearIdx * 2];
                        const int firstMaxDisp = d_range_image[linearIdx * 2 + 1];
                        prevMinDisp[dy] = (firstMinDisp <= firstMaxDisp) ? firstMinDisp : 0;
                    }
                    else
                    {
                        prevMinDisp[dy] = 0;
                    }
                }

                int x0 = (DIRECTION > 0) ? 0 : static_cast<int>((width - 1) & ~(DP_BLOCK_SIZE - 1));
                for (unsigned int iter = 0; iter < width; iter += DP_BLOCK_SIZE)
                {
                    for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
                    {
                        const unsigned int x = x0 + (DIRECTION > 0 ? i : (DP_BLOCK_SIZE - 1 - i));
                        if (x >= width)
                        {
                            continue;
                        }
                        for (unsigned int j = 0; j < DP_BLOCKS_PER_THREAD; ++j)
                        {
                            const unsigned int y = y0 + j * SUBGROUPS_PER_WARP;
                            if (y >= height)
                            {
                                continue;
                            }

                            const unsigned int linearIdx = y * width + x;
                            const int minDisp = d_range_image[linearIdx * 2];
                            const int maxDisp = d_range_image[linearIdx * 2 + 1];
                            const uint32_t pixelOffset = d_range_offset[linearIdx];
                            const bool isValid = (minDisp <= maxDisp);
                            const int rangeWidth = isValid ? (maxDisp - minDisp + 1) : 0;

                            if (!isValid)
                            {
                                dp[j].last_min = 0;
                                for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k)
                                {
                                    dp[j].dp[k] = 0;
                                }
                                // Re-initialize right buffer for next valid pixel
                                prevMinDisp[j] = 0;
                                continue;
                            }

                            const int curMinDisp = minDisp;

                            // Shift right buffer for minDisp change
                            const int dispShift = curMinDisp - prevMinDisp[j];
                            // For horizontal scanning, the buffer naturally shifts by 1 per step.
                            // We also need to account for minDisp change.

                            const CENSUS_TYPE left_value = __ldg(&left[y * width + x]);
                            if (DIRECTION > 0)
                            {
                                const CENSUS_TYPE t = right_buffer[j][DP_BLOCK_SIZE - 1];
                                for (unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k)
                                {
                                    right_buffer[j][k] = right_buffer[j][k - 1];
                                }
                                right_buffer[j][0] = SHFL_UP(shfl_mask, t, 1, SUBGROUP_SIZE);
                                if (lane_id == 0)
                                {
                                    right_buffer[j][0] = load_census_with_check(&right[y * width], x - curMinDisp, width);
                                }
                            }
                            else
                            {
                                const CENSUS_TYPE t = right_buffer[j][0];
                                for (unsigned int k = 1; k < DP_BLOCK_SIZE; ++k)
                                {
                                    right_buffer[j][k - 1] = right_buffer[j][k];
                                }
                                right_buffer[j][DP_BLOCK_SIZE - 1] = SHFL_DOWN(shfl_mask, t, 1, SUBGROUP_SIZE);
                                if (lane_id + 1 == SUBGROUP_SIZE)
                                {
                                    right_buffer[j][DP_BLOCK_SIZE - 1] = load_census_with_check(&right[y * width], x - (curMinDisp + dp_offset + DP_BLOCK_SIZE - 1), width);
                                }
                            }

                            uint32_t local_costs[DP_BLOCK_SIZE];
                            for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k)
                            {
                                const int globalD = dp_offset + k;
                                if (globalD < rangeWidth)
                                {
                                    local_costs[k] = popcnt(left_value ^ right_buffer[j][k]);
                                }
                                else
                                {
                                    local_costs[k] = 255u;
                                }
                            }
                            dp[j].update(local_costs, p1, p2, shfl_mask);

                            // Store to flat cost buffer
                            for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k)
                            {
                                const int globalD = dp_offset + k;
                                if (globalD < rangeWidth)
                                {
                                    dest[pixelOffset + globalD] = static_cast<uint8_t>(dp[j].dp[k]);
                                }
                            }

                            prevMinDisp[j] = curMinDisp;
                        }
                    }
                    x0 += static_cast<int>(DP_BLOCK_SIZE) * DIRECTION;
                }
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_left2right_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                              int width, int height, unsigned int p1, unsigned int p2,
                                              const int32_t *d_range_image, const uint32_t *d_range_offset,
                                              int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

                const int gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_horizontal_path_kernel_clipped<CENSUS_TYPE, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_right2left_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                              int width, int height, unsigned int p1, unsigned int p2,
                                              const int32_t *d_range_image, const uint32_t *d_range_offset,
                                              int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

                const int gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_horizontal_path_kernel_clipped<CENSUS_TYPE, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

        } // namespace horizontal_clipped

        namespace oblique_clipped
        {

            static constexpr unsigned int DP_BLOCK_SIZE = 16u;
            static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

            template<typename CENSUS_TYPE, int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
            __global__ void aggregate_oblique_path_kernel_clipped(
                uint8_t *dest,
                const CENSUS_TYPE *left,
                const CENSUS_TYPE *right,
                int width, int height,
                unsigned int p1, unsigned int p2,
                const int32_t *__restrict__ d_range_image,
                const uint32_t *__restrict__ d_range_offset,
                int range_length)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
                static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

                static_assert(X_DIRECTION == 1 || X_DIRECTION == -1, "");
                static_assert(Y_DIRECTION == 1 || Y_DIRECTION == -1, "");
                if (width == 0 || height == 0)
                {
                    return;
                }

                __shared__ CENSUS_TYPE right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS];
                DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
                const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
                const unsigned int shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

                const int x0 = blockIdx.x * PATHS_PER_BLOCK + warp_id * PATHS_PER_WARP + group_id + (X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
                const int right_x00 = blockIdx.x * PATHS_PER_BLOCK + (X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
                const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

                const unsigned int right0_addr = static_cast<unsigned int>(right_x00 + PATHS_PER_BLOCK - 1 - x0) + dp_offset;
                const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
                const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

                for (unsigned int iter = 0; iter < height; ++iter)
                {
                    const int y = static_cast<int>(Y_DIRECTION > 0 ? iter : height - 1 - iter);
                    const int x = x0 + static_cast<int>(iter) * X_DIRECTION;
                    const int right_x0_local = right_x00 + static_cast<int>(iter) * X_DIRECTION;

                    // Load per-pixel range data
                    int minDisp = 1, maxDisp = 0;
                    uint32_t pixelOffset = 0;
                    bool isValid = false;
                    if (0 <= x && x < static_cast<int>(width))
                    {
                        const unsigned int linearIdx = y * width + x;
                        minDisp = d_range_image[linearIdx * 2];
                        maxDisp = d_range_image[linearIdx * 2 + 1];
                        pixelOffset = d_range_offset[linearIdx];
                        isValid = (minDisp <= maxDisp);
                    }
                    const int rangeWidth = isValid ? (maxDisp - minDisp + 1) : 0;

                    if (!isValid)
                    {
                        dp.last_min = 0;
                        for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
                        {
                            dp.dp[i] = 0;
                        }
                    }

                    // Use a representative minDisp for shared memory right buffer loading
                    __shared__ int block_min_disp_oblique;
                    if (threadIdx.x == 0)
                    {
                        block_min_disp_oblique = minDisp;
                    }
                    __syncthreads();

                    // Load right to smem
                    for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE)
                    {
                        const unsigned int i = i0 + threadIdx.x;
                        if (i < RIGHT_BUFFER_SIZE)
                        {
                            const int right_x = static_cast<int>(right_x0_local + PATHS_PER_BLOCK - 1 - i - block_min_disp_oblique);
                            const CENSUS_TYPE right_value = load_census_with_check(&right[y * width], right_x, width);
                            const unsigned int lo = i % DP_BLOCK_SIZE;
                            const unsigned int hi = i / DP_BLOCK_SIZE;
                            right_buffer[lo][hi] = right_value;
                            if (hi > 0)
                            {
                                right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                            }
                        }
                    }
                    __syncthreads();

                    // Compute
                    if (isValid)
                    {
                        const CENSUS_TYPE left_value = __ldg(&left[x + y * width]);

                        const int disp_shift = block_min_disp_oblique - minDisp;
                        const unsigned int adj_right0_addr = right0_addr + disp_shift;
                        const unsigned int adj_right0_addr_lo = adj_right0_addr % DP_BLOCK_SIZE;
                        const unsigned int adj_right0_addr_hi = adj_right0_addr / DP_BLOCK_SIZE;

                        CENSUS_TYPE right_values[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            right_values[j] = right_buffer[adj_right0_addr_lo + j][adj_right0_addr_hi];
                        }
                        uint32_t local_costs[DP_BLOCK_SIZE];
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            const int globalD = dp_offset + j;
                            if (globalD < rangeWidth)
                            {
                                local_costs[j] = popcnt(left_value ^ right_values[j]);
                            }
                            else
                            {
                                local_costs[j] = 255u;
                            }
                        }
                        dp.update(local_costs, p1, p2, shfl_mask);

                        // Store to flat cost buffer
                        for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
                        {
                            const int globalD = dp_offset + j;
                            if (globalD < rangeWidth)
                            {
                                dest[pixelOffset + globalD] = static_cast<uint8_t>(dp.dp[j]);
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_upleft2downright_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                                    int width, int height, unsigned int p1, unsigned int p2,
                                                    const int32_t *d_range_image, const uint32_t *d_range_offset,
                                                    int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel_clipped<CENSUS_TYPE, 1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_upright2downleft_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                                    int width, int height, unsigned int p1, unsigned int p2,
                                                    const int32_t *d_range_image, const uint32_t *d_range_offset,
                                                    int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel_clipped<CENSUS_TYPE, -1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_downright2upleft_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                                    int width, int height, unsigned int p1, unsigned int p2,
                                                    const int32_t *d_range_image, const uint32_t *d_range_offset,
                                                    int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel_clipped<CENSUS_TYPE, -1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

            template<typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
            void aggregate_downleft2upright_clipped(COST_TYPE *dest, const CENSUS_TYPE *left, const CENSUS_TYPE *right,
                                                    int width, int height, unsigned int p1, unsigned int p2,
                                                    const int32_t *d_range_image, const uint32_t *d_range_offset,
                                                    int range_length, cudaStream_t stream)
            {
                static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
                static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

                const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
                const int bdim = BLOCK_SIZE;
                aggregate_oblique_path_kernel_clipped<CENSUS_TYPE, 1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
                    dest, left, right, width, height, p1, p2, d_range_image, d_range_offset, range_length);
                CUDA_CHECK(cudaGetLastError());
            }

        } // namespace oblique_clipped

    } // namespace cost_aggregation

    namespace details
    {

        template<typename CENSUS_TYPE, int MAX_DISPARITY>
        void cost_aggregation_(const DeviceImage &srcL, const DeviceImage &srcR, DeviceImage &dst, int P1, int P2, PathType path_type, int min_disp)
        {
            const int width = srcL.cols;
            const int height = srcL.rows;
            const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

            dst.create(num_paths, height * width * MAX_DISPARITY, SGM_8U);

            const CENSUS_TYPE *left = srcL.ptr<CENSUS_TYPE>();
            const CENSUS_TYPE *right = srcR.ptr<CENSUS_TYPE>();

            cudaStream_t streams[8];
            for (int i = 0; i < num_paths; i++)
                cudaStreamCreate(&streams[i]);

            cost_aggregation::vertical::aggregate_up2down<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(0), left, right, width, height, P1, P2, min_disp, streams[0]);
            cost_aggregation::vertical::aggregate_down2up<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(1), left, right, width, height, P1, P2, min_disp, streams[1]);
            cost_aggregation::horizontal::aggregate_left2right<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(2), left, right, width, height, P1, P2, min_disp, streams[2]);
            cost_aggregation::horizontal::aggregate_right2left<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(3), left, right, width, height, P1, P2, min_disp, streams[3]);

            if (path_type == PathType::SCAN_8PATH)
            {
                cost_aggregation::oblique::aggregate_upleft2downright<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(4), left, right, width, height, P1, P2, min_disp, streams[4]);
                cost_aggregation::oblique::aggregate_upright2downleft<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(5), left, right, width, height, P1, P2, min_disp, streams[5]);
                cost_aggregation::oblique::aggregate_downright2upleft<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(6), left, right, width, height, P1, P2, min_disp, streams[6]);
                cost_aggregation::oblique::aggregate_downleft2upright<CENSUS_TYPE, MAX_DISPARITY>(dst.ptr<COST_TYPE>(7), left, right, width, height, P1, P2, min_disp, streams[7]);
            }

            for (int i = 0; i < num_paths; i++)
                cudaStreamSynchronize(streams[i]);
            for (int i = 0; i < num_paths; i++)
                cudaStreamDestroy(streams[i]);
        }

        void cost_aggregation(const DeviceImage &srcL, const DeviceImage &srcR, DeviceImage &dst, int disp_size, int P1, int P2, PathType path_type, int min_disp)
        {
            SGM_ASSERT(srcL.type == srcR.type, "left and right image type must be same.");

            if (srcL.type == SGM_32U)
            {
                if (disp_size == 64)
                {
                    cost_aggregation_<uint32_t, 64>(srcL, srcR, dst, P1, P2, path_type, min_disp);
                }
                else if (disp_size == 128)
                {
                    cost_aggregation_<uint32_t, 128>(srcL, srcR, dst, P1, P2, path_type, min_disp);
                }
                else if (disp_size == 256)
                {
                    cost_aggregation_<uint32_t, 256>(srcL, srcR, dst, P1, P2, path_type, min_disp);
                }
            }
            else if (srcL.type == SGM_64U)
            {
                if (disp_size == 64)
                {
                    cost_aggregation_<uint64_t, 64>(srcL, srcR, dst, P1, P2, path_type, min_disp);
                }
                else if (disp_size == 128)
                {
                    cost_aggregation_<uint64_t, 128>(srcL, srcR, dst, P1, P2, path_type, min_disp);
                }
                else if (disp_size == 256)
                {
                    cost_aggregation_<uint64_t, 256>(srcL, srcR, dst, P1, P2, path_type, min_disp);
                }
            }
        }

        template<typename CENSUS_TYPE, int MAX_DISPARITY>
        void cost_aggregation_clipped_(const DeviceImage &srcL, const DeviceImage &srcR, DeviceImage &dst,
                                       const int32_t *d_range_image, const uint32_t *d_range_offset,
                                       int range_length, int P1, int P2, PathType path_type)
        {
            const int width = srcL.cols;
            const int height = srcL.rows;
            const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

            // Path-major layout: each path gets its own copy of the flat cost buffer
            dst.create(num_paths, range_length, SGM_8U);
            dst.fill_zero();

            const CENSUS_TYPE *left = srcL.ptr<CENSUS_TYPE>();
            const CENSUS_TYPE *right = srcR.ptr<CENSUS_TYPE>();

            cudaStream_t streams[8];
            for (int i = 0; i < num_paths; i++)
                cudaStreamCreate(&streams[i]);

            cost_aggregation::vertical_clipped::aggregate_up2down_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                dst.ptr<COST_TYPE>(0), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[0]);
            cost_aggregation::vertical_clipped::aggregate_down2up_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                dst.ptr<COST_TYPE>(1), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[1]);
            cost_aggregation::horizontal_clipped::aggregate_left2right_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                dst.ptr<COST_TYPE>(2), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[2]);
            cost_aggregation::horizontal_clipped::aggregate_right2left_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                dst.ptr<COST_TYPE>(3), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[3]);

            if (path_type == PathType::SCAN_8PATH)
            {
                cost_aggregation::oblique_clipped::aggregate_upleft2downright_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                    dst.ptr<COST_TYPE>(4), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[4]);
                cost_aggregation::oblique_clipped::aggregate_upright2downleft_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                    dst.ptr<COST_TYPE>(5), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[5]);
                cost_aggregation::oblique_clipped::aggregate_downright2upleft_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                    dst.ptr<COST_TYPE>(6), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[6]);
                cost_aggregation::oblique_clipped::aggregate_downleft2upright_clipped<CENSUS_TYPE, MAX_DISPARITY>(
                    dst.ptr<COST_TYPE>(7), left, right, width, height, P1, P2, d_range_image, d_range_offset, range_length, streams[7]);
            }

            for (int i = 0; i < num_paths; i++)
                cudaStreamSynchronize(streams[i]);
            for (int i = 0; i < num_paths; i++)
                cudaStreamDestroy(streams[i]);
        }

        void cost_aggregation(const DeviceImage &srcL, const DeviceImage &srcR, DeviceImage &dst,
                              const int32_t *d_range_image, const uint32_t *d_range_offset,
                              int range_length, int max_per_pixel_range,
                              int P1, int P2, PathType path_type)
        {
            SGM_ASSERT(srcL.type == srcR.type, "left and right image type must be same.");

            if (srcL.type == SGM_32U)
            {
                if (max_per_pixel_range <= 16)
                    cost_aggregation_clipped_<uint32_t, 16>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else if (max_per_pixel_range <= 32)
                    cost_aggregation_clipped_<uint32_t, 32>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else if (max_per_pixel_range <= 64)
                    cost_aggregation_clipped_<uint32_t, 64>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else if (max_per_pixel_range <= 128)
                    cost_aggregation_clipped_<uint32_t, 128>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else
                    cost_aggregation_clipped_<uint32_t, 256>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
            }
            else if (srcL.type == SGM_64U)
            {
                if (max_per_pixel_range <= 16)
                    cost_aggregation_clipped_<uint64_t, 16>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else if (max_per_pixel_range <= 32)
                    cost_aggregation_clipped_<uint64_t, 32>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else if (max_per_pixel_range <= 64)
                    cost_aggregation_clipped_<uint64_t, 64>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else if (max_per_pixel_range <= 128)
                    cost_aggregation_clipped_<uint64_t, 128>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
                else
                    cost_aggregation_clipped_<uint64_t, 256>(srcL, srcR, dst, d_range_image, d_range_offset, range_length, P1, P2, path_type);
            }
        }

    } // namespace details
} // namespace sgm
