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

#include "constants.h"
#include "host_utility.h"

namespace
{

    __global__ void correct_disparity_range_kernel(uint16_t *d_disp, int width, int height, int pitch, int min_disp_scaled, int invalid_disp_scaled)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
        {
            return;
        }

        uint16_t d = d_disp[y * pitch + x];
        if (d == sgm::INVALID_DISP)
        {
            d = invalid_disp_scaled;
        }
        else
        {
            d += min_disp_scaled;
        }
        d_disp[y * pitch + x] = d;
    }

    __global__ void correct_disparity_range_kernel_clipped(
        uint16_t *d_disp,
        int width, int height, int pitch,
        const int32_t *__restrict__ d_range_image,
        int scale,
        int invalid_disp_scaled)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
        {
            return;
        }

        const int linearIdx = y * width + x;
        const int minDisp = d_range_image[linearIdx * 2];
        const int maxDisp = d_range_image[linearIdx * 2 + 1];
        const bool isValid = (minDisp <= maxDisp);

        uint16_t d = d_disp[y * pitch + x];
        if (d == sgm::INVALID_DISP || !isValid)
        {
            d_disp[y * pitch + x] = static_cast<uint16_t>(invalid_disp_scaled);
        }
        else
        {
            // WTA already wrote global disparity (minDisp + localD)
            // For subpixel mode, WTA already applied the subpixel shift
            // No additional scaling needed — the disparity is already correct
        }
    }

} // namespace

namespace sgm
{
    namespace details
    {

        void correct_disparity_range(DeviceImage &disp, bool subpixel, int min_disp)
        {
            if (!subpixel && min_disp == 0)
            {
                return;
            }

            const int w = disp.cols;
            const int h = disp.rows;
            constexpr int SIZE = 16;
            const dim3 blocks(divUp(w, SIZE), divUp(h, SIZE));
            const dim3 threads(SIZE, SIZE);

            const int scale = subpixel ? StereoSGM::SUBPIXEL_SCALE : 1;
            const int min_disp_scaled = min_disp * scale;
            const int invalid_disp_scaled = (min_disp - 1) * scale;

            correct_disparity_range_kernel<<<blocks, threads>>>(disp.ptr<uint16_t>(), w, h, disp.step, min_disp_scaled, invalid_disp_scaled);
            CUDA_CHECK(cudaGetLastError());
        }

        void correct_disparity_range(DeviceImage &disp, const int32_t *d_range_image, bool subpixel, int min_disp)
        {
            const int w = disp.cols;
            const int h = disp.rows;
            constexpr int SIZE = 16;
            const dim3 blocks(divUp(w, SIZE), divUp(h, SIZE));
            const dim3 threads(SIZE, SIZE);

            const int scale = subpixel ? StereoSGM::SUBPIXEL_SCALE : 1;
            const int invalid_disp_scaled = (min_disp - 1) * scale;

            correct_disparity_range_kernel_clipped<<<blocks, threads>>>(
                disp.ptr<uint16_t>(), w, h, disp.step, d_range_image, scale, invalid_disp_scaled);
            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace details
} // namespace sgm
