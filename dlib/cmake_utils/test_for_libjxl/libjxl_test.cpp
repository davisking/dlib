// Copyright (C) 2023 Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.

#include <jxl/encode_cxx.h>
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner_cxx.h>
#include <iostream>
#include <memory>

// This code doesn't really make a lot of sense.  It's just calling all the libjpeg functions to make
// sure they can be compiled and linked.

int main()
{
    std::cerr << "This program is just for build system testing.  Don't actually run it." << std::endl;
    std::abort();
    auto enc = JxlEncoderMake(nullptr);
    auto dec = JxlDecoderMake(nullptr);
    auto runner = JxlResizableParallelRunnerMake(nullptr);
}
