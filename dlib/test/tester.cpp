// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <string>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include "tester.h"

namespace test
{
// -----------------------------------------------------------------------------

    bool be_verbose = true;

// -----------------------------------------------------------------------------

    static std::mutex spinner_mutex;
    static std::atomic<uint64_t> test_count(0);

// -----------------------------------------------------------------------------

    std::uint64_t number_of_testing_statements_executed (
    )
    {
        return test_count;
    }

    void increment_test_count (
    )
    {
        ++test_count;
    }

// -----------------------------------------------------------------------------

    void check_test (
        bool _exp,
        long line,
        const char* file,
        const char* _exp_str
    )
    {
        ++test_count;
        if ( !(_exp) )                                                         
        {                                                                       
            std::ostringstream dlib_o_out;                                       
            dlib_o_out << "\n\nError occurred at line " << line << ".\n";    
            dlib_o_out << "Error occurred in file " << file << ".\n";      
            dlib_o_out << "Failing expression was " << _exp_str << ".\n";           
            throw dlib::error(dlib_o_out.str());      
        }
    }                                                                      

// -----------------------------------------------------------------------------

    map_of_testers& testers (
    )
    {
        static map_of_testers t;
        return t;
    }

// -----------------------------------------------------------------------------

    tester::
    tester (
        const std::string& switch_name_x,
        const std::string& description_x,
        unsigned long num_of_args_x
    ) :
        switch_name(switch_name_x),
        description_(description_x),
        num_of_args_(num_of_args_x)
    {
        using namespace std;
        if (testers().find(switch_name) != testers().end())
        {
            cerr << "ERROR: More than one tester has been defined with the switch '" << switch_name << "'." << endl;
            exit(1);
        }

        testers()[switch_name] = this;
    }

// -----------------------------------------------------------------------------

    const std::string& tester::
    cmd_line_switch (
    ) const
    {
        return switch_name;
    }

// -----------------------------------------------------------------------------

    const std::string& tester::
    description (
    ) const
    {
        return description_;
    }

// -----------------------------------------------------------------------------

    unsigned long tester::
    num_of_args (
    ) const
    {
        return num_of_args_;
    }

// -----------------------------------------------------------------------------

    void tester::
    perform_test (
    )
    {
    }

// -----------------------------------------------------------------------------

    void tester::
    perform_test (
        const std::string&  
    )
    {
    }

// -----------------------------------------------------------------------------

    void tester::
    perform_test (
        const std::string&, 
        const std::string& 
    )
    {
    }

// -----------------------------------------------------------------------------

    void print_spinner (
    )
    {
        if (be_verbose)
        {
            using namespace std;
            std::unique_lock<std::mutex> M(spinner_mutex);
            static int i = 0;
            cout << "\b\b";
            switch (i)
            {
                case 0: cout << '|'; break;
                case 1: cout << '/'; break;
                case 2: cout << '-'; break;
                case 3: cout << '\\'; break;
            }
            cout << " " << flush;
            i = (i+1)%4;
        }
    }

// -----------------------------------------------------------------------------

}



