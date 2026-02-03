#pragma once

#include "libsgm.h"

namespace sgm
{
    //! \brief CensusTransform class providing standalone access to the Census Transform algorithm.
    //!
    //! \details The census transform compares each pixel with its neighbors in a fixed 9x7 window.
    //! Three Census types are supported:
    //! - CENSUS_9x7: Standard census transform producing 64-bit descriptors.
    //! - CENSUS_9x7_WEIGHTED: Weighted variant producing 64-bit descriptors.
    //! - SYMMETRIC_CENSUS_9x7: Symmetric variant producing 32-bit descriptors.
    //!
    //! \note The output buffer must be allocated by the user with appropriate size:
    //! - For CENSUS_9x7 and CENSUS_9x7_WEIGHTED: width * height * sizeof(uint64_t) bytes
    //! - For SYMMETRIC_CENSUS_9x7: width * height * sizeof(uint32_t) bytes
    class CensusTransform
    {
    public:
        //! \param width Image width in pixels.
        //! \param height Image height in pixels.
        //! \param input_depth_bits Input image bits per pixel. Must be 8, 16, or 32.
        //! \param inout_type Specify input/output pointer type. See sgm::ExecuteInOut.
        //! \param census_type Type of census transform to apply.
        LIBSGM_API CensusTransform(const int width,               //
                                   const int height,              //
                                   const int input_depth_bits,    //
                                   const ExecuteInOut inout_type, //
                                   const CensusType census_type);

        //! \param width Image width in pixels.
        //! \param height Image height in pixels.
        //! \param input_depth_bits Input image bits per pixel. Must be 8, 16, or 32.
        //! \param src_pitch Source image pitch (pixels per row).
        //! \param dst_pitch Destination image pitch (pixels per row).
        //! \param inout_type Specify input/output pointer type. See sgm::ExecuteInOut.
        //! \param census_type Type of census transform to apply.
        LIBSGM_API CensusTransform(const int width,               //
                                   const int height,              //
                                   const int input_depth_bits,    //
                                   const int src_pitch,           //
                                   const int dst_pitch,           //
                                   const ExecuteInOut inout_type, //
                                   const CensusType census_type);

        LIBSGM_API virtual ~CensusTransform();

        //! \brief Execute census transform on the input image.
        //! \param src A pointer to the input image (host or device memory based on inout_type). 
        //! \param dst Output pointer for census-transformed image. User must allocate enough memory.
        //! \attention For CENSUS_9x7, allocate width * height * sizeof(uint64_t) bytes. 
        //! \attention For SYMMETRIC_CENSUS_9x7, allocate width * height * sizeof(uint32_t) bytes.
        LIBSGM_API void execute(const void *src, void *dst);

    private:
        CensusTransform(const CensusTransform &) = delete;
        CensusTransform &operator=(const CensusTransform &) = delete;

        class Impl;
        Impl *impl_;
    };

}
