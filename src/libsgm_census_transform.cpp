#include <libsgm_census_transform.h>

#include "host_utility.h"
#include "internal.h"

#include <cuda_runtime.h>

namespace
{
    inline sgm::ImageType getSourceImageType(const int srcDepth)
    {
        using namespace sgm;

        switch (srcDepth)
        {
            case 8:
                return SGM_8U;
            case 16:
                return SGM_16U;
            case 32:
                return SGM_32U;
            default:
                throw std::invalid_argument("Source depth bits must be 8, 16 or 32");
        }
    }

    inline sgm::ImageType getDestinationImageType(const sgm::CensusType censusType)
    {
        switch (censusType)
        {
            case sgm::CensusType::CENSUS_9x7:
                return sgm::SGM_64U;
            case sgm::CensusType::CLASSIC_CENSUS_9x7:
                return sgm::SGM_64U;
            default:
                return sgm::SGM_32U;
        }
    }

    inline bool isSourceDevicePointer(const sgm::ExecuteInOut inoutType)
    {
        return (inoutType & 0x01) > 0;
    }

    inline bool isDestinationDevicePointer(const sgm::ExecuteInOut inoutType)
    {
        return (inoutType & 0x02) > 0;
    }

    inline size_t getElementSize(const sgm::ImageType imageType)
    {
        return imageType == sgm::SGM_64U ? sizeof(uint64_t) : sizeof(uint32_t);
    }
}

namespace sgm
{
    class CensusTransform::Impl
    {
    private:
        int m_width;
        int m_height;
        int m_srcPitch;
        int m_dstPitch;
        CensusType m_censusType;
        ImageType m_srcType;
        ImageType m_dstType;
        bool m_isSrcDevptr;
        bool m_isDstDevptr;
        DeviceImage m_dSrc;
        DeviceImage m_dCensus;
        cudaStream_t m_stream;

    public:
        Impl(const int width, const int height, const int srcDepth, const int srcPitch, const int dstPitch, ExecuteInOut inoutType, CensusType censusType) :
            m_width(width),
            m_height(height),
            m_srcPitch(srcPitch),
            m_dstPitch(dstPitch),
            m_censusType(censusType),
            m_stream(nullptr)
        {
            SGM_ASSERT(srcDepth == 8 || srcDepth == 16 || srcDepth == 32, "Source depth bits must be 8, 16 or 32");

            // Create dedicated CUDA stream for this instance
            CUDA_CHECK(cudaStreamCreate(&m_stream));

            m_srcType = getSourceImageType(srcDepth);
            m_dstType = getDestinationImageType(censusType);
            m_isSrcDevptr = isSourceDevicePointer(inoutType);
            m_isDstDevptr = isDestinationDevicePointer(inoutType);

            if (!m_isSrcDevptr)
            {
                m_dSrc.create(height, width, m_srcType, srcPitch);
            }

            prepareDeviceMemoryForCensusOutput();
        }

        ~Impl()
        {
            if (m_stream)
            {
                cudaStreamDestroy(m_stream);
                m_stream = nullptr;
            }
        }

        void execute(const void *src, void *dst)
        {
            processSourceData(src);
            performCensusTransform();
            copyResultToDestination(dst);
        }

    private:
        //! \brief Allocates memory for Census output on the device.
        void prepareDeviceMemoryForCensusOutput()
        {
            m_dCensus.create(m_height, m_width, m_dstType, m_dstPitch);
            m_dCensus.fill_zero(m_stream);
        }

        //! \brief Prepare source data for census transform. Upload to device if needed or wrap device pointer.\
        //! \param src Pointer to source data (host or device memory based on \p m_isSrcDevptr).
        void processSourceData(const void *src)
        {
            if (m_isSrcDevptr)
            {
                m_dSrc.create((void *)src, m_height, m_width, m_srcType, m_srcPitch);
            }
            else
            {
                m_dSrc.upload(src, m_stream);
            }
        }

        //! \brief Perform the census transform on the prepared source data.
        void performCensusTransform()
        {
            details::census_transform(m_dSrc, m_dCensus, m_censusType, m_stream);
        }

        //! \brief Copy the census transform result to the destination pointer.
        //! \param dst Pointer to destination data (host or device memory based on \p m_isDstDevptr).
        void copyResultToDestination(void *dst)
        {
            if (m_isDstDevptr)
            {
                copyResultToDeviceMemory(dst);
            }
            else
            {
                copyResultToHostMemory(dst);
            }
        }

        //! \brief Copy the census result to device memory.
        //! \param dst Pointer to device memory.
        void copyResultToDeviceMemory(void *dst)
        {
            DeviceImage d_dst(dst, m_height, m_width, m_dstType, m_dstPitch);
            const size_t elem_size = ::getElementSize(m_dstType);
            cudaMemcpy2DAsync(d_dst.data, d_dst.step * elem_size, m_dCensus.data, m_dCensus.step * elem_size, m_width * elem_size, m_height, cudaMemcpyDeviceToDevice, m_stream);
            cudaStreamSynchronize(m_stream);
        }

        //! \brief Copy the census result to host memory.
        //! \param dst Pointer to host memory.
        void copyResultToHostMemory(void *dst)
        {
            m_dCensus.download(dst, m_stream);
            cudaStreamSynchronize(m_stream);
        }
    };

    CensusTransform::CensusTransform(const int width, const int height, const int inputDepthBits, const ExecuteInOut inoutType, const CensusType censusType) :
        impl_(new Impl(width, height, inputDepthBits, width, width, inoutType, censusType))
    {
    }

    CensusTransform::CensusTransform(const int width, const int height, const int inputDepthBits, const int srcPitch, const int dstPitch, const ExecuteInOut inoutType,
                                     const CensusType censusType) :
        impl_(new Impl(width, height, inputDepthBits, srcPitch, dstPitch, inoutType, censusType))
    {
    }

    CensusTransform::~CensusTransform()
    {
        delete impl_;
    }

    void CensusTransform::execute(const void *src, void *dst)
    {
        impl_->execute(src, dst);
    }
}
