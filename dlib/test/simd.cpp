// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/simd.h>
#include <dlib/array2d.h>
#include <dlib/image_transforms/interpolation.h>
#include <type_traits>

namespace
{
    using namespace test;
    using namespace dlib;

    template <typename float_vector, typename int_vector>
    void test_conversions()
    {
        const float input[] = {
            1.75f, -2.5f, 0.0f, 17.0f,
            0.6875f, -0.6875f, 1.0625f, -1.0625f,
            2147483520.0f, -2147483648.0f, 16777215.0f, -16777215.0f,
            0.0f, -0.0f, 123.875f, -123.875f
        };
        float_vector values;
        for (unsigned int offset = 0; offset < 16; offset += values.size())
        {
            values.load(input + offset);
            const int_vector converted(values);
            const float_vector restored(converted);
            const float_vector fractions = values - converted;
            for (unsigned int i = 0; i < values.size(); ++i)
            {
                const int32 expected = static_cast<int32>(input[offset+i]);
                DLIB_TEST(converted[i] == expected);
                DLIB_TEST(restored[i] == static_cast<float>(expected));
                DLIB_TEST(fractions[i] == input[offset+i] - static_cast<float>(expected));
            }
        }
    }

    void test_resize()
    {
        // Bilinear interpolation must preserve a linear ramp.  These dimensions
        // exercise fractional coordinates, complete SIMD groups, and the scalar tail.
        array2d<float> gray(17, 19), gray_out(12, 14);
        array2d<rgb_pixel> rgb(17, 19), rgb_out(12, 14);
        for (long r = 0; r < gray.nr(); ++r)
        {
            for (long c = 0; c < gray.nc(); ++c)
            {
                gray[r][c] = static_cast<float>(8*r + c);
                rgb[r][c] = rgb_pixel(
                    static_cast<unsigned char>(4*r + 2*c),
                    static_cast<unsigned char>(2*r + 4*c),
                    static_cast<unsigned char>(3*r + c));
            }
        }
        resize_image(gray, gray_out, interpolate_bilinear());
        resize_image(rgb, rgb_out, interpolate_bilinear());
        for (long r = 0; r < gray_out.nr(); ++r)
        {
            for (long c = 0; c < gray_out.nc(); ++c)
            {
                const double y = r*16.0/11;
                const double x = c*18.0/13;
                DLIB_TEST(std::abs(gray_out[r][c] - (8*y + x)) < 1e-4);
                DLIB_TEST(std::abs(rgb_out[r][c].red - (4*y + 2*x)) <= 1);
                DLIB_TEST(std::abs(rgb_out[r][c].green - (2*y + 4*x)) <= 1);
                DLIB_TEST(std::abs(rgb_out[r][c].blue - (3*y + x)) <= 1);
            }
        }
    }

    class simd_tester : public tester
    {
    public:
        simd_tester() : tester("test_simd", "Run tests on SIMD conversions.") {}

        void perform_test() override
        {
            test_conversions<simd4f, simd4i>();
            test_conversions<simd8f, simd8i>();
            test_resize();

            const simd4f values(1.75f, -2.5f, 0.0f, 17.0f);
            const simd4i converted(values);
            DLIB_TEST(converted[0] == 1);
            DLIB_TEST(converted[1] == -2);
            DLIB_TEST(converted[2] == 0);
            DLIB_TEST(converted[3] == 17);

            const simd4i integers(1, -2, 0, 17);
            for (unsigned int i = 0; i < 4; ++i)
                DLIB_TEST(integers[i] == converted[i]);

#ifdef DLIB_HAVE_NEON
            simd4i assigned;
            assigned = values;
            for (unsigned int i = 0; i < 4; ++i)
                DLIB_TEST(assigned[i] == converted[i]);

            // Both raw integer conversions remain available, even if their types alias.
            const int32x4_t signed_vector = integers;
            const uint32x4_t unsigned_vector = integers;
            int32 signed_lanes[4];
            uint32 unsigned_lanes[4];
            vst1q_s32(signed_lanes, signed_vector);
            vst1q_u32(unsigned_lanes, unsigned_vector);
            for (unsigned int i = 0; i < 4; ++i)
                DLIB_TEST(unsigned_lanes[i] == static_cast<uint32>(signed_lanes[i]));

            const float32x4_t float_vector = values;
            float float_lanes[4];
            vst1q_f32(float_lanes, float_vector);
            for (unsigned int i = 0; i < 4; ++i)
                DLIB_TEST(float_lanes[i] == values[i]);

            // A separate raw float-to-int conversion is possible only with distinct types.
            if (!std::is_same<float32x4_t, int32x4_t>::value)
            {
                const int32x4_t raw_converted = values;
                vst1q_s32(signed_lanes, raw_converted);
                for (unsigned int i = 0; i < 4; ++i)
                    DLIB_TEST(signed_lanes[i] == converted[i]);
            }
#endif
        }
    } a;
}
