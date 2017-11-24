// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TIME_THIs_
#define DLIB_TIME_THIs_


#include <chrono>

// ----------------------------------------------------------------------------------------

#define TIME_THIS_TO(_tt_op,_tt_out)                                                                            \
    {                                                                                                           \
        auto _tt_start = std::chrono::high_resolution_clock::now();                                             \
        {_tt_op;}                                                                                               \
        auto _tt_stop = std::chrono::high_resolution_clock::now();                                              \
        auto _tt_thetime = _tt_stop-_tt_start;                                                                          \
        using std::chrono::duration_cast;                                                                       \
        using std::chrono::duration;                                                                            \
        if (_tt_thetime >= std::chrono::minutes(1))                                                             \
            _tt_out << "\ntime: " << duration_cast<duration<double,std::ratio<60>>>(_tt_thetime).count() << "min\n";           \
        else if (_tt_thetime >= std::chrono::seconds(1))                                                        \
            _tt_out << "\ntime: " << duration_cast<duration<double>>(_tt_thetime).count() << "sec\n";           \
        else if (_tt_thetime >= std::chrono::milliseconds(1))                                                   \
            _tt_out << "\ntime: " << duration_cast<duration<double,std::milli>>(_tt_thetime).count() << "ms\n"; \
        else if (_tt_thetime >= std::chrono::microseconds(1))                                                   \
            _tt_out << "\ntime: " << duration_cast<duration<double,std::micro>>(_tt_thetime).count() << "us\n"; \
        else                                                                                                    \
            _tt_out << "\ntime: " << duration_cast<duration<double,std::nano>>(_tt_thetime).count() << "ns\n";  \
    }

#define TIME_THIS(_tt_op)  TIME_THIS_TO(_tt_op,std::cout)

// ----------------------------------------------------------------------------------------

#endif // DLIB_TIME_THIs_

