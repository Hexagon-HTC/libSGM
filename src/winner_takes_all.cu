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

namespace sgm
{
    namespace
    {

        static constexpr unsigned int WARPS_PER_BLOCK = 8u;
        static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;

        __device__ inline uint32_t pack_cost_index(uint32_t cost, uint32_t index)
        {
            union
            {
                uint32_t uint32;
                ushort2 uint16x2;
            } u;
            u.uint16x2.x = static_cast<uint16_t>(index);
            u.uint16x2.y = static_cast<uint16_t>(cost);
            return u.uint32;
        }

        __device__ uint32_t unpack_cost(uint32_t packed)
        {
            return packed >> 16;
        }

        __device__ int unpack_index(uint32_t packed)
        {
            return packed & 0xffffu;
        }

        using ComputeDisparity = uint32_t (*)(uint32_t, uint32_t, uint16_t *);

        __device__ inline uint32_t compute_disparity_normal(uint32_t disp, uint32_t cost = 0, uint16_t *smem = nullptr)
        {
            return disp;
        }

        template<size_t MAX_DISPARITY>
        __device__ inline uint32_t compute_disparity_subpixel(uint32_t disp, uint32_t cost, uint16_t *smem)
        {
            int subp = disp;
            subp <<= sgm::StereoSGM::SUBPIXEL_SHIFT;
            if (disp > 0 && disp < MAX_DISPARITY - 1)
            {
                const int left = smem[disp - 1];
                const int right = smem[disp + 1];
                const int numer = left - right;
                const int denom = left - 2 * cost + right;
                subp += ((numer << sgm::StereoSGM::SUBPIXEL_SHIFT) + denom) / (2 * denom);
            }
            return subp;
        }

        template<unsigned int MAX_DISPARITY, unsigned int NUM_PATHS, ComputeDisparity compute_disparity = compute_disparity_normal>
        __global__ void winner_takes_all_kernel(output_type *left_dest, output_type *right_dest, const cost_type *src, int width, int height, int pitch, float uniqueness)
        {
            static const unsigned int ACCUMULATION_PER_THREAD = 16u;
            static const unsigned int REDUCTION_PER_THREAD = MAX_DISPARITY / WARP_SIZE;
            static const unsigned int ACCUMULATION_INTERVAL = ACCUMULATION_PER_THREAD / REDUCTION_PER_THREAD;
            static const unsigned int UNROLL_DEPTH = (REDUCTION_PER_THREAD > ACCUMULATION_INTERVAL) ? REDUCTION_PER_THREAD : ACCUMULATION_INTERVAL;

            const size_t cost_step = static_cast<size_t>(MAX_DISPARITY) * width * height;
            const unsigned int warp_id = threadIdx.x / WARP_SIZE;
            const unsigned int lane_id = threadIdx.x % WARP_SIZE;

            const unsigned int y = blockIdx.x * WARPS_PER_BLOCK + warp_id;
            src += y * MAX_DISPARITY * width;
            left_dest += y * pitch;
            right_dest += y * pitch;

            if (y >= height)
            {
                return;
            }

            __shared__ uint16_t smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY];

            uint32_t right_best[REDUCTION_PER_THREAD];
            for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
            {
                right_best[i] = 0xffffffffu;
            }

            for (unsigned int x0 = 0; x0 < width; x0 += UNROLL_DEPTH)
            {
#pragma unroll
                for (unsigned int x1 = 0; x1 < UNROLL_DEPTH; ++x1)
                {
                    if (x1 % ACCUMULATION_INTERVAL == 0)
                    {
                        const unsigned int k = lane_id * ACCUMULATION_PER_THREAD;
                        const unsigned int k_hi = k / MAX_DISPARITY;
                        const unsigned int k_lo = k % MAX_DISPARITY;
                        const unsigned int x = x0 + x1 + k_hi;
                        if (x < width)
                        {
                            const unsigned int offset = x * MAX_DISPARITY + k_lo;
                            uint32_t sum[ACCUMULATION_PER_THREAD];
                            for (unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i)
                            {
                                sum[i] = 0;
                            }
                            for (unsigned int p = 0; p < NUM_PATHS; ++p)
                            {
                                uint32_t load_buffer[ACCUMULATION_PER_THREAD];
                                load_uint8_vector<ACCUMULATION_PER_THREAD>(load_buffer, &src[p * cost_step + offset]);
                                for (unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i)
                                {
                                    sum[i] += load_buffer[i];
                                }
                            }
                            store_uint16_vector<ACCUMULATION_PER_THREAD>(&smem_cost_sum[warp_id][k_hi][k_lo], sum);
                        }
#if CUDA_VERSION >= 9000
                        __syncwarp();
#else
                        __threadfence_block();
#endif
                    }
                    const unsigned int x = x0 + x1;
                    if (x < width)
                    {
                        // Load sum of costs
                        const unsigned int smem_x = x1 % ACCUMULATION_INTERVAL;
                        const unsigned int k0 = lane_id * REDUCTION_PER_THREAD;
                        uint32_t local_cost_sum[REDUCTION_PER_THREAD];
                        load_uint16_vector<REDUCTION_PER_THREAD>(local_cost_sum, &smem_cost_sum[warp_id][smem_x][k0]);
                        // Pack sum of costs and dispairty
                        uint32_t local_packed_cost[REDUCTION_PER_THREAD];
                        for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                        {
                            local_packed_cost[i] = pack_cost_index(local_cost_sum[i], k0 + i);
                        }
                        // Update left
                        uint32_t best = 0xffffffffu;
                        for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                        {
                            best = min(best, local_packed_cost[i]);
                        }
                        best = subgroup_min<WARP_SIZE>(best, 0xffffffffu);
                        // Update right
#pragma unroll
                        for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                        {
                            const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
                            const int p = static_cast<int>(((x - k) & ~(MAX_DISPARITY - 1)) + k);
                            const unsigned int d = static_cast<unsigned int>(x - p);
#if CUDA_VERSION >= 9000
                            const uint32_t recv =
                                __shfl_sync(0xffffffffu, local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD], d / REDUCTION_PER_THREAD, WARP_SIZE);
#else
                            const uint32_t recv = __shfl(local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD], d / REDUCTION_PER_THREAD, WARP_SIZE);
#endif
                            right_best[i] = min(right_best[i], recv);
                            if (d == MAX_DISPARITY - 1)
                            {
                                if (0 <= p)
                                {
                                    right_dest[p] = compute_disparity_normal(unpack_index(right_best[i]));
                                }
                                right_best[i] = 0xffffffffu;
                            }
                        }
                        // Resume updating left to avoid execution dependency
                        const uint32_t bestCost = unpack_cost(best);
                        const int bestDisp = unpack_index(best);
                        bool uniq = true;
                        for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                        {
                            const uint32_t x = local_packed_cost[i];
                            const bool uniq1 = unpack_cost(x) * uniqueness >= bestCost;
                            const bool uniq2 = abs(unpack_index(x) - bestDisp) <= 1;
                            uniq &= uniq1 || uniq2;
                        }
                        uniq = subgroup_and<WARP_SIZE>(uniq, 0xffffffffu);
                        if (lane_id == 0)
                        {
                            left_dest[x] = uniq ? compute_disparity(bestDisp, bestCost, smem_cost_sum[warp_id][smem_x]) : INVALID_DISP;
                        }
                    }
                }
            }
            for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
            {
                const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
                const int p = static_cast<int>(((width - k) & ~(MAX_DISPARITY - 1)) + k);
                if (0 <= p && p < width)
                {
                    right_dest[p] = compute_disparity_normal(unpack_index(right_best[i]));
                }
            }
        }

    } // namespace

    namespace
    {
        // Clipped paths: WTA kernel for per-pixel variable disparity ranges
        // One thread per pixel, iterates over its disparity range
        static constexpr unsigned int WTA_CLIPPED_BLOCK_SIZE = 256u;

        template<unsigned int NUM_PATHS>
        __global__ void winner_takes_all_kernel_clipped(
            output_type *left_dest,
            const cost_type *src,
            int width, int height, int pitch,
            const int32_t *__restrict__ d_range_image,
            const uint32_t *__restrict__ d_range_offset,
            int range_length,
            float uniqueness,
            bool subpixel)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int x = idx % width;
            const int y = idx / width;

            if (y >= height)
            {
                return;
            }

            const int linearIdx = y * width + x;
            const int minDisp = d_range_image[linearIdx * 2];
            const int maxDisp = d_range_image[linearIdx * 2 + 1];

            if (minDisp > maxDisp)
            {
                left_dest[y * pitch + x] = INVALID_DISP;
                return;
            }

            const uint32_t pixelOffset = d_range_offset[linearIdx];
            const int rangeWidth = maxDisp - minDisp + 1;

            // First pass: accumulate costs and find best disparity
            uint32_t bestCost = 0xffffffffu;
            int bestLocalD = 0;

            for (int d = 0; d < rangeWidth; ++d)
            {
                uint32_t sum = 0;
                for (unsigned int p = 0; p < NUM_PATHS; ++p)
                {
                    sum += static_cast<uint32_t>(src[p * static_cast<size_t>(range_length) + pixelOffset + d]);
                }
                if (sum < bestCost)
                {
                    bestCost = sum;
                    bestLocalD = d;
                }
            }

            // Convert to global disparity
            int bestDisp = minDisp + bestLocalD;

            // Second pass: uniqueness test
            bool uniq = true;
            for (int d = 0; d < rangeWidth; ++d)
            {
                if (abs(d - bestLocalD) <= 1)
                    continue;
                uint32_t sum = 0;
                for (unsigned int p = 0; p < NUM_PATHS; ++p)
                {
                    sum += static_cast<uint32_t>(src[p * static_cast<size_t>(range_length) + pixelOffset + d]);
                }
                if (static_cast<float>(sum) * uniqueness < static_cast<float>(bestCost))
                {
                    uniq = false;
                    break;
                }
            }

            if (!uniq)
            {
                left_dest[y * pitch + x] = INVALID_DISP;
                return;
            }

            // Subpixel refinement
            if (subpixel)
            {
                int subp = bestDisp;
                subp <<= sgm::StereoSGM::SUBPIXEL_SHIFT;
                if (bestLocalD > 0 && bestLocalD < rangeWidth - 1)
                {
                    uint32_t costLeft = 0, costRight = 0;
                    for (unsigned int p = 0; p < NUM_PATHS; ++p)
                    {
                        costLeft += static_cast<uint32_t>(src[p * static_cast<size_t>(range_length) + pixelOffset + bestLocalD - 1]);
                        costRight += static_cast<uint32_t>(src[p * static_cast<size_t>(range_length) + pixelOffset + bestLocalD + 1]);
                    }
                    const int numer = static_cast<int>(costLeft) - static_cast<int>(costRight);
                    const int denom = static_cast<int>(costLeft) - 2 * static_cast<int>(bestCost) + static_cast<int>(costRight);
                    if (denom != 0)
                    {
                        subp += ((numer << sgm::StereoSGM::SUBPIXEL_SHIFT) + denom) / (2 * denom);
                    }
                }
                left_dest[y * pitch + x] = static_cast<output_type>(subp);
            }
            else
            {
                left_dest[y * pitch + x] = static_cast<output_type>(bestDisp);
            }
        }

        // Scatter kernel for right disparity with clipped paths
        __global__ void scatter_right_disparity_kernel(
            const output_type *left_disp,
            output_type *right_dest,
            int width, int height, int pitch,
            bool subpixel)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int x = idx % width;
            const int y = idx / width;

            if (y >= height)
            {
                return;
            }

            const output_type d = left_disp[y * pitch + x];
            if (d == INVALID_DISP)
            {
                return;
            }

            // Extract integer disparity for position calculation
            const int disp_int = subpixel ? (static_cast<int>(d) >> sgm::StereoSGM::SUBPIXEL_SHIFT) : static_cast<int>(d);
            const int right_x = x - disp_int;

            if (right_x >= 0 && right_x < width)
            {
                // Simple scatter - last writer wins
                right_dest[y * pitch + right_x] = d;
            }
        }

    } // namespace (clipped WTA)

    namespace details
    {

        template<int MAX_DISPARITY>
        void winner_takes_all_(const DeviceImage &src, DeviceImage &dstL, DeviceImage &dstR, float uniqueness, bool subpixel, PathType path_type)
        {
            const int width = dstL.cols;
            const int height = dstL.rows;
            const int pitch = dstL.step;

            const int gdim = divUp(height, WARPS_PER_BLOCK);
            const int bdim = BLOCK_SIZE;

            const cost_type *cost = src.ptr<cost_type>();
            output_type *dispL = dstL.ptr<output_type>();
            output_type *dispR = dstR.ptr<output_type>();

            if (subpixel && path_type == PathType::SCAN_8PATH)
            {
                winner_takes_all_kernel<MAX_DISPARITY, 8, compute_disparity_subpixel<MAX_DISPARITY>><<<gdim, bdim>>>(dispL, dispR, cost, width, height, pitch, uniqueness);
            }
            else if (subpixel && path_type == PathType::SCAN_4PATH)
            {
                winner_takes_all_kernel<MAX_DISPARITY, 4, compute_disparity_subpixel<MAX_DISPARITY>><<<gdim, bdim>>>(dispL, dispR, cost, width, height, pitch, uniqueness);
            }
            else if (!subpixel && path_type == PathType::SCAN_8PATH)
            {
                winner_takes_all_kernel<MAX_DISPARITY, 8, compute_disparity_normal><<<gdim, bdim>>>(dispL, dispR, cost, width, height, pitch, uniqueness);
            }
            else /* if (!subpixel && path_type == PathType::SCAN_4PATH) */
            {
                winner_takes_all_kernel<MAX_DISPARITY, 4, compute_disparity_normal><<<gdim, bdim>>>(dispL, dispR, cost, width, height, pitch, uniqueness);
            }

            CUDA_CHECK(cudaGetLastError());
        }

        void winner_takes_all(const DeviceImage &src, DeviceImage &dstL, DeviceImage &dstR, int disp_size, float uniqueness, bool subpixel, PathType path_type)
        {
            if (disp_size == 64)
            {
                winner_takes_all_<64>(src, dstL, dstR, uniqueness, subpixel, path_type);
            }
            else if (disp_size == 128)
            {
                winner_takes_all_<128>(src, dstL, dstR, uniqueness, subpixel, path_type);
            }
            else if (disp_size == 256)
            {
                winner_takes_all_<256>(src, dstL, dstR, uniqueness, subpixel, path_type);
            }
        }

        void winner_takes_all(const DeviceImage &src, DeviceImage &dstL, DeviceImage &dstR,
                              const int32_t *d_range_image, const uint32_t *d_range_offset,
                              int max_per_pixel_range,
                              float uniqueness, bool subpixel, PathType path_type)
        {
            const int width = dstL.cols;
            const int height = dstL.rows;
            const int pitch = dstL.step;

            const int num_pixels = width * height;
            const int range_length = src.cols; // cols stores range_length from cost_aggregation
            const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

            const cost_type *cost = src.ptr<cost_type>();
            output_type *dispL = dstL.ptr<output_type>();
            output_type *dispR = dstR.ptr<output_type>();

            // Initialize right disparity to INVALID_DISP
            CUDA_CHECK(cudaMemset(dispR, 0xFF, static_cast<size_t>(height) * pitch * sizeof(output_type)));

            const int gdim = divUp(num_pixels, WTA_CLIPPED_BLOCK_SIZE);
            const int bdim = WTA_CLIPPED_BLOCK_SIZE;

            if (num_paths == 8)
            {
                winner_takes_all_kernel_clipped<8><<<gdim, bdim>>>(
                    dispL, cost, width, height, pitch, d_range_image, d_range_offset, range_length, uniqueness, subpixel);
            }
            else
            {
                winner_takes_all_kernel_clipped<4><<<gdim, bdim>>>(
                    dispL, cost, width, height, pitch, d_range_image, d_range_offset, range_length, uniqueness, subpixel);
            }
            CUDA_CHECK(cudaGetLastError());

            // Scatter right disparity from left
            scatter_right_disparity_kernel<<<gdim, bdim>>>(dispL, dispR, width, height, pitch, subpixel);
            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace details
} // namespace sgm
