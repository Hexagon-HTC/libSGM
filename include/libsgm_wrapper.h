#ifndef __LIBSGM_WRAPPER_H__
#define __LIBSGM_WRAPPER_H__

#include "libsgm.h"

#include <memory>
#ifdef BUILD_OPENCV_WRAPPER
#include <opencv2/core/cuda.hpp>
#endif

namespace sgm
{

    /**
     * @brief LibSGMWrapper class which is wrapper for sgm::StereoSGM.
     */
    class LibSGMWrapper
    {
    public:
        /**
         * @param numDisparity Maximum disparity minus minimum disparity.
         * @param censusType Type of census transform. Determines GPU buffer allocation and cannot change between executions.
         */
        LIBSGM_API LibSGMWrapper(int numDisparity = 128, CensusType censusType = CensusType::SYMMETRIC_CENSUS_9x7);
        LIBSGM_API ~LibSGMWrapper();

        LIBSGM_API int getNumDisparities() const;
        LIBSGM_API CensusType getCensusType() const;

        /**
         * Compute invalid disparity value for the given runtime parameters.
         */
        LIBSGM_API int getInvalidDisparity(const StereoSGM::RuntimeParameters &runtimeParam) const;

#ifdef BUILD_OPENCV_WRAPPER

        /**
         * Execute stereo semi global matching via wrapper class.
         * @param I1        Input left image.  Image's type is must be CV_8U, CV_16U or CV_32S
         * @param I2        Input right image.  Image's size and type must be same with I1.
         * @param disparity Output image.  Its memory will be allocated automatically dependent on input image size.
         * @attention
         * type of output image `disparity` is CV_16S.
         * Note that disparity element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
         */
        LIBSGM_API void execute(const cv::cuda::GpuMat &I1, const cv::cuda::GpuMat &I2, cv::cuda::GpuMat &disparity,
                                const StereoSGM::RuntimeParameters &runtimeParam = StereoSGM::RuntimeParameters());

        /**
         * Execute stereo semi global matching via wrapper class.
         * @param I1        Input left image.  Image's type is must be CV_8U, CV_16U or CV_32S.
         * @param I2        Input right image.  Image's size and type must be same with I1.
         * @param disparity Output image.  Its memory will be allocated automatically dependent on input image size.
         * @attention
         * type of output image `disparity` is CV_16S.
         * Note that disparity element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
         */
        LIBSGM_API void execute(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &disparity, const StereoSGM::RuntimeParameters &runtimeParam = StereoSGM::RuntimeParameters());

#endif // BUILD_OPRENCV_WRAPPER

    private:
        struct Creator;
        std::unique_ptr<sgm::StereoSGM> sgm_;
        int numDisparity_;
        sgm::StereoSGM::Parameters param_;
        std::unique_ptr<Creator> prev_;
    };

} // namespace sgm

#endif // __LIBSGM_WRAPPER_H__
