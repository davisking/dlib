// Copyright (C) 2019  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.

#include <webp/encode.h>
#include <webp/decode.h>
#include <iostream>

// This code doesn't really make a lot of sense.  It's just calling all the libjpeg functions to make
// sure they can be compiled and linked.
int main()
{
    std::cerr << "This program is just for build system testing.  Don't actually run it." << std::endl;
    std::abort();

    uint8_t* data;
    size_t output_size = 0;
    int width, height, stride;
    float quality;
    output_size = WebPEncodeRGB(data, width, height, stride, quality, &data);
    WebPDecodeRGBInto(data, output_size, data, output_size, stride);
    WebPFree(data);
}
