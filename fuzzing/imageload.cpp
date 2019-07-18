#include <dlib/image_io.h>
#include <string>
#include <fstream>

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

    dlib::array2d<dlib::rgb_pixel> img;

    TempFile tmpfile(Data,Size,"bin");

    try {
        dlib::load_image(img, tmpfile.m_name.c_str());
    } catch(...) {
    }

    return 0;
}
