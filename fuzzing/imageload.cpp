// Copyright (C) 2019  Paul Dreik (github@pauldreik.se)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/image_io.h>
#include <string>
#include <fstream>

/**
 * @brief The TempFile struct
 * A RAII temporary file, filled with the binary data given to it
 * during construction. When the object is destroyed, the file is removed.
 */
struct TempFile {
    TempFile(const uint8_t* Data, std::size_t Size, const char* suffix) {
        static unsigned int count=0;
        char buf[128];
        const auto nwritten=std::snprintf(buf,sizeof(buf),"/tmp/fuzz_imageload_%d_%d.%s",getpid(),count++,suffix);
        assert(nwritten<sizeof(buf));
        m_name=std::string(buf,buf+nwritten);

        std::ofstream os(buf);
        assert(os);
        os.write((const char*)Data,Size);
        assert(os.tellp()==Size);
    }
    ~TempFile() {
        unlink(m_name.c_str());
    }
    std::string m_name;
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, std::size_t Size) {

    // prevent large input, just to keep the fuzzing speed up.
    if(Size > 200) {
        return 0;
    }
    dlib::array2d<dlib::rgb_pixel> img;

    // the file suffix is irrelevant, hence the .bin for "binary data".
    TempFile tmpfile(Data,Size,"bin");

    try {
        dlib::load_image(img, tmpfile.m_name.c_str());
    } catch(...) {
    }

    return 0;
}
