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

#ifndef __DEVICE_IMAGE_H__
#define __DEVICE_IMAGE_H__

#include "device_allocator.h"

#include <cuda_runtime.h>

namespace sgm
{

    enum ImageType
    {
        SGM_8U,  //! unsigned 8-bit integer representing input image in 8-bit grayscale format
        SGM_16U, //! unsigned 16-bit integer representing input image in 16-bit grayscale or cost/disparity format
        SGM_32U, //! unsigned 32-bit integer representing intermediate data (cost aggregation) or symmetric census output.
        SGM_64U, //! unsigned 64-bit integer representing standard census output.
    };

    class DeviceImage
    {
    public:
        DeviceImage();
        DeviceImage(int rows, int cols, ImageType type, int step = -1);
        DeviceImage(void *data, int rows, int cols, ImageType type, int step = -1);

        void create(int rows, int cols, ImageType type, int step = -1);
        void create(void *data, int rows, int cols, ImageType type, int step = -1);

        void upload(const void *data, cudaStream_t stream = 0);
        void download(void *data, cudaStream_t stream = 0) const;
        void fill_zero(cudaStream_t stream = 0);

        template<typename T>
        T *ptr(int y = 0)
        {
            return (T *)data + y * (size_t)step;
        }
        template<typename T>
        const T *ptr(int y = 0) const
        {
            return (T *)data + y * (size_t)step;
        }

        void *data;
        int rows, cols, step;
        ImageType type;

    private:
        DeviceAllocator allocator_;
    };

} // namespace sgm

#endif // !__DEVICE_IMAGE_H__
