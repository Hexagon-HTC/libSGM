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

#include <gtest/gtest.h>

#include <libsgm.h>
#include <vector>
#include <cstring>
#include <stdexcept>

#include "host_image.h"
#include "test_utility.h"

namespace
{

// Helper to create random stereo pair with given dimensions
void create_random_stereo_pair(std::vector<uint8_t>& left, std::vector<uint8_t>& right, int width, int height)
{
    left.resize(width * height);
    right.resize(width * height);
    
    std::default_random_engine engine(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 255);
    
    for (size_t i = 0; i < left.size(); ++i)
    {
        left[i] = static_cast<uint8_t>(dist(engine));
        right[i] = static_cast<uint8_t>(dist(engine));
    }
}

} // anonymous namespace

// Test: Original execute() method works (backward compatibility)
TEST(StereoSGMTest, OriginalExecuteWorks)
{
    const int width = 64;
    const int height = 48;
    const int disp_size = 64;
    
    std::vector<uint8_t> left, right;
    create_random_stereo_pair(left, right, width, height);
    std::vector<uint16_t> disparity(width * height);
    
    sgm::StereoSGM sgm(width, height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    // Should not throw
    EXPECT_NO_THROW(sgm.execute(left.data(), right.data(), disparity.data()));
    
    // Disparity should have some non-zero values (random images will have matches)
    bool has_nonzero = false;
    for (const auto& d : disparity)
    {
        if (d != 0 && d != static_cast<uint16_t>(sgm.get_invalid_disparity()))
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test: New execute() with same dimensions as constructor
TEST(StereoSGMTest, ExecuteWithSameDimensions)
{
    const int width = 64;
    const int height = 48;
    const int disp_size = 64;
    
    std::vector<uint8_t> left, right;
    create_random_stereo_pair(left, right, width, height);
    std::vector<uint16_t> disparity1(width * height);
    std::vector<uint16_t> disparity2(width * height);
    
    sgm::StereoSGM sgm(width, height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    // Execute with original method
    sgm.execute(left.data(), right.data(), disparity1.data());
    
    // Execute with new method using same dimensions
    sgm.execute(left.data(), right.data(), disparity2.data(), width, height);
    
    // Results should be identical
    EXPECT_EQ(disparity1, disparity2);
}

// Test: New execute() with smaller dimensions than constructor
TEST(StereoSGMTest, ExecuteWithSmallerDimensions)
{
    const int max_width = 128;
    const int max_height = 96;
    const int actual_width = 64;
    const int actual_height = 48;
    const int disp_size = 64;
    
    // Create instance sized for max dimensions
    sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    // Create smaller images
    std::vector<uint8_t> left, right;
    create_random_stereo_pair(left, right, actual_width, actual_height);
    std::vector<uint16_t> disparity(actual_width * actual_height);
    
    // Execute with smaller dimensions - should work
    EXPECT_NO_THROW(sgm.execute(left.data(), right.data(), disparity.data(), actual_width, actual_height));
    
    // Verify disparity output makes sense
    bool has_valid = false;
    int invalid_disp = sgm.get_invalid_disparity();
    for (const auto& d : disparity)
    {
        if (static_cast<int16_t>(d) != invalid_disp)
        {
            has_valid = true;
            break;
        }
    }
    EXPECT_TRUE(has_valid);
}

// Test: Results match between dedicated instance and reused instance with smaller dimensions
TEST(StereoSGMTest, SmallerDimensionsMatchDedicatedInstance)
{
    const int max_width = 128;
    const int max_height = 96;
    const int actual_width = 64;
    const int actual_height = 48;
    const int disp_size = 64;
    
    // Create smaller images
    std::vector<uint8_t> left, right;
    create_random_stereo_pair(left, right, actual_width, actual_height);
    
    std::vector<uint16_t> disparity_dedicated(actual_width * actual_height);
    std::vector<uint16_t> disparity_reused(actual_width * actual_height);
    
    // Dedicated instance for exact size
    {
        sgm::StereoSGM sgm(actual_width, actual_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
        sgm.execute(left.data(), right.data(), disparity_dedicated.data());
    }
    
    // Reused instance with larger capacity
    {
        sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
        sgm.execute(left.data(), right.data(), disparity_reused.data(), actual_width, actual_height);
    }
    
    // Results should be identical
    EXPECT_EQ(disparity_dedicated, disparity_reused);
}

// Test: Multiple executes with varying sizes reuse memory
TEST(StereoSGMTest, MultipleExecutesWithVaryingSizes)
{
    const int max_width = 128;
    const int max_height = 96;
    const int disp_size = 64;
    
    sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    // Test various sizes smaller than max
    std::vector<std::pair<int, int>> sizes = {
        {128, 96},  // Full size
        {64, 48},   // Half size
        {96, 64},   // Different aspect
        {32, 32},   // Small square
        {128, 96},  // Back to full size
    };
    
    for (const auto& [w, h] : sizes)
    {
        std::vector<uint8_t> left, right;
        create_random_stereo_pair(left, right, w, h);
        std::vector<uint16_t> disparity(w * h);
        
        EXPECT_NO_THROW(sgm.execute(left.data(), right.data(), disparity.data(), w, h))
            << "Failed for size " << w << "x" << h;
    }
}

// Test: Execute with 8-bit output
TEST(StereoSGMTest, ExecuteWithSmallerDimensions8BitOutput)
{
    const int max_width = 128;
    const int max_height = 96;
    const int actual_width = 64;
    const int actual_height = 48;
    const int disp_size = 64;
    
    sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 8, sgm::EXECUTE_INOUT_HOST2HOST);
    
    std::vector<uint8_t> left, right;
    create_random_stereo_pair(left, right, actual_width, actual_height);
    std::vector<uint8_t> disparity(actual_width * actual_height);
    
    EXPECT_NO_THROW(sgm.execute(left.data(), right.data(), disparity.data(), actual_width, actual_height));
}

// Test: Execute with different disparity sizes
TEST(StereoSGMTest, ExecuteWithDifferentDisparitySizes)
{
    const int max_width = 128;
    const int max_height = 96;
    const int actual_width = 64;
    const int actual_height = 48;
    
    for (int disp_size : {64, 128, 256})
    {
        sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
        
        std::vector<uint8_t> left, right;
        create_random_stereo_pair(left, right, actual_width, actual_height);
        std::vector<uint16_t> disparity(actual_width * actual_height);
        
        EXPECT_NO_THROW(sgm.execute(left.data(), right.data(), disparity.data(), actual_width, actual_height))
            << "Failed for disparity size " << disp_size;
    }
}

// Test: Execute with 4-path vs 8-path
TEST(StereoSGMTest, ExecuteWithDifferentPathTypes)
{
    const int max_width = 128;
    const int max_height = 96;
    const int actual_width = 64;
    const int actual_height = 48;
    const int disp_size = 64;
    
    for (auto path_type : {sgm::PathType::SCAN_4PATH, sgm::PathType::SCAN_8PATH})
    {
        sgm::StereoSGM::Parameters params;
        params.path_type = path_type;
        
        sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST, params);
        
        std::vector<uint8_t> left, right;
        create_random_stereo_pair(left, right, actual_width, actual_height);
        std::vector<uint16_t> disparity(actual_width * actual_height);
        
        EXPECT_NO_THROW(sgm.execute(left.data(), right.data(), disparity.data(), actual_width, actual_height));
    }
}

// Test: Invalid dimensions should throw exception (width > max)
TEST(StereoSGMTest, ExecuteWithTooLargeWidthShouldFail)
{
    const int max_width = 64;
    const int max_height = 48;
    const int disp_size = 64;
    
    sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    // Create images larger than max
    const int bad_width = 128;  // Larger than max_width
    std::vector<uint8_t> left(bad_width * max_height);
    std::vector<uint8_t> right(bad_width * max_height);
    std::vector<uint16_t> disparity(bad_width * max_height);
    
    // SGM_ASSERT throws std::logic_error
    EXPECT_THROW(sgm.execute(left.data(), right.data(), disparity.data(), bad_width, max_height), std::logic_error);
}

// Test: Invalid dimensions should throw exception (height > max)
TEST(StereoSGMTest, ExecuteWithTooLargeHeightShouldFail)
{
    const int max_width = 64;
    const int max_height = 48;
    const int disp_size = 64;
    
    sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    // Create images larger than max
    const int bad_height = 96;  // Larger than max_height
    std::vector<uint8_t> left(max_width * bad_height);
    std::vector<uint8_t> right(max_width * bad_height);
    std::vector<uint16_t> disparity(max_width * bad_height);
    
    // SGM_ASSERT throws std::logic_error
    EXPECT_THROW(sgm.execute(left.data(), right.data(), disparity.data(), max_width, bad_height), std::logic_error);
}

// Test: Zero or negative dimensions should fail
TEST(StereoSGMTest, ExecuteWithZeroDimensionsShouldFail)
{
    const int max_width = 64;
    const int max_height = 48;
    const int disp_size = 64;
    
    sgm::StereoSGM sgm(max_width, max_height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST);
    
    std::vector<uint8_t> left(1), right(1);
    std::vector<uint16_t> disparity(1);
    
    // Zero width should throw
    EXPECT_THROW(sgm.execute(left.data(), right.data(), disparity.data(), 0, max_height), std::logic_error);
    
    // Zero height should throw
    EXPECT_THROW(sgm.execute(left.data(), right.data(), disparity.data(), max_width, 0), std::logic_error);
}
