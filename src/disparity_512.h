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

#ifndef __DISPARITY_512_H__
#define __DISPARITY_512_H__

// Disparity 512 kernels use DP_BLOCK_SIZE=16 (required for SUBGROUP_SIZE<=32).
// They are compiled in separate translation units to allow 64/128/256 to use
// the optimal DP_BLOCK_SIZE=8 without performance degradation.

#include "device_image.h"
#include "libsgm.h"

namespace sgm
{
    namespace details
    {
        // Cost aggregation for disparity 512
        void cost_aggregation_512_32u(const DeviceImage &srcL, const DeviceImage &srcR, DeviceImage &dst, int P1, int P2, PathType path_type, int min_disp);
        void cost_aggregation_512_64u(const DeviceImage &srcL, const DeviceImage &srcR, DeviceImage &dst, int P1, int P2, PathType path_type, int min_disp);

        // Winner-takes-all for disparity 512
        void winner_takes_all_512(const DeviceImage &src, DeviceImage &dstL, DeviceImage &dstR, float uniqueness, bool subpixel, PathType path_type);

    } // namespace details
} // namespace sgm

#endif // !__DISPARITY_512_H__
