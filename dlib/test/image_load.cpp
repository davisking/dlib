// Copyright (C) 2019  Paul Dreik (github@pauldreik.se)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/image_io.h>


// Finding files recursively in the filesystem is easy if C++17 or boost
// can be used. However, none of those can be required, so instead the posix
// API is used (dirent.h). If that is not available, there is no recursive finding,
// instead the input argument is assumed to be a file.
#if defined(unix) || defined(__unix__) || defined(__unix)
# include <unistd.h>
#endif

#if _POSIX_VERSION >= 200112L
# include <dirent.h>
# define HAVE_DIRENT_H 1
#endif

namespace  
{
    dlib::logger dlog("test.image_load");

    using namespace test;

    class image_load_tester : public tester
    {
        /*!
             Tests the image loading utilities (parsing image files).
        !*/
    public:
        image_load_tester (
        ) :
            tester (
                "test_image_load",                // the command line argument name for this test
        #ifdef HAVE_DIRENT_H
                "Runs image files recursively found in arg through the image load routine.", // the command line argument description
        #else
                "Runs the image file found in arg through the image load routine.", // the command line argument description
        #endif
                1                                   // the number of command line arguments for this test
            )
        {}

        void perform_test (
            const std::string& arg
        )
        {
            dlog << dlib::LINFO << "the argument passed to this test was " << arg;

            m_arg=arg;

           print_spinner();
#ifdef HAVE_DIRENT_H
           // recursively load files
           invoke_on_file_or_dir(m_arg);
#else
           // load the single file given
           invoke_on_file((m_arg);
#endif
        }
     private:
#ifdef HAVE_DIRENT_H
        // switch to C++17 filesystem, when that is available
        void invoke_on_file_or_dir(const std::string& dir) {
            DIR* pDir=opendir(dir.c_str());
            if(pDir) {
                struct dirent *entry;
                   while ((entry = readdir(pDir))) {
                       const std::string bare_name(entry->d_name);
                           switch(entry->d_type) {
                           case DT_DIR:
                           if(bare_name=="." || bare_name=="..") {
                               continue;
                           }
                           invoke_on_file_or_dir(dir + "/" + bare_name);
                           break;
                           case DT_REG:
                           invoke_on_file(dir + "/" + bare_name);
                           break;
                           }
                   }
             closedir (pDir);
            } else {
             invoke_on_file(dir);
            }
        }
#endif
        void invoke_on_file(const std::string& filename) {
            print_spinner();
            dlog << dlib::LINFO << "attempting to parse image "<<filename;

                   dlib::array2d<dlib::rgb_pixel> img;
                   try {
                       dlib::load_image(img, filename);
                       dlog << dlib::LINFO << "successfully parsed image "<<filename;
                   } catch(std::exception& e) {
                       dlog << dlib::LERROR << "got an exception while loading image "<<filename<<": "<<e.what();
                   }
                   // get text output, so the user sees something happened
                   DLIB_TEST_MSG(true,"survived loading an image");
        }
        std::string m_arg;
    };

    // Create an instance of this object.  Doing this causes this test
    // to be automatically inserted into the testing framework whenever this cpp file
    // is linked into the project.  Note that since we are inside an unnamed-namespace 
    // we won't get any linker errors about the symbol a being defined multiple times. 
    image_load_tester a;
}



