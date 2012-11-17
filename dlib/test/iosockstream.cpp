// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/iosockstream.h>
#include <dlib/server.h>
#include <vector>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;


    logger dlog("test.iosockstream");

// ----------------------------------------------------------------------------------------

    class serv : public server_iostream
    {
        virtual void on_connect (
            std::istream& in,
            std::ostream& out,
            const std::string& ,
            const std::string& ,
            unsigned short ,
            unsigned short ,
            uint64 
        )
        {
            try
            {
                dlog << LINFO << "serving connection";

                std::string temp;
                in >> temp;
                DLIB_TEST(temp == "word");
                in >> temp;
                DLIB_TEST(temp == "another");
                out << "yay words ";
                in >> temp;
                DLIB_TEST(temp == "yep");
            }
            catch (error& e)
            {
                error_string = e.what();
            }

        }


    public:
        std::string error_string;

    };

// ----------------------------------------------------------------------------------------

    class test_iosockstream : public tester
    {
    public:
        test_iosockstream (
        ) :
            tester ("test_iosockstream",
                    "Runs tests on the iosockstream component.")
        {}

        void perform_test (
        )
        {
            serv theserv;
            theserv.set_listening_port(12345);
            theserv.start_async();

            for (int i = 0; i < 1001; ++i)
            {
                print_spinner();
                iosockstream stream("localhost:12345");

                stream << "word another ";
                std::string temp;
                stream >> temp;
                DLIB_TEST(temp == "yay");
                stream >> temp;
                DLIB_TEST(temp == "words");
                stream << "yep ";
            }

            // Just to make sure the server finishes processing the last connection before
            // we kill it and accidentally trigger a DLIB_TEST().
            dlib::sleep(500);

            if (theserv.error_string.size() != 0)
                throw error(theserv.error_string);
        }
    } a;

}


