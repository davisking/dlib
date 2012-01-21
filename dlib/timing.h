// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TImING_H__
#define DLIB_TImING_H__

#include "misc_api.h"

#include <iostream>

/*!A timing

    This set of functions is useful for determining how much time is spent
    executing blocks of code.  Consider the following example:

    int main()
    {
        using namespace dlib::timing;
        for (int i = 0; i < 10; ++i)
        {
            // timing block #1
            start(1);
            dlib::sleep(500);
            stop(1);

            // timing block #2
            start(2);
            dlib::sleep(1000);
            stop(2);
        }

        print();
    }

    This program would output:
        Timing report: 
            1: 5.0 seconds
            2: 10.0 seconds

    So we spent 5 seconds in block #1 and 10 seconds in block #2
!*/

namespace dlib
{
    namespace timing
    {
        const int TIME_SLOTS = 500;

        inline uint64* time_buf()
        {
            static uint64 buf[TIME_SLOTS] = {0};
            return buf;
        }

        inline timestamper& ts()
        {
            static timestamper ts_;
            return ts_;
        }

        inline void start(int i )
        {
            time_buf()[i] -= ts().get_timestamp();
        }

        inline void stop(int i)
        {
            time_buf()[i] += ts().get_timestamp();
        }

        inline void print()
        {
            using namespace std;
            cout << "Timing report: " << endl;
            for (int i = 0; i < TIME_SLOTS; ++i)
            {
                if (time_buf()[i] != 0)
                {
                    double time = time_buf()[i]/1000.0;
                    if (time < 1000)
                        cout << "  " << i << ": " << time << " milliseconds" << endl;
                    else if (time < 1000*1000)
                        cout << "  " << i << ": " << time/1000.0 << " seconds" << endl;
                    else if (time < 1000*1000*60)
                        cout << "  " << i << ": " << time/1000.0/60.0 << " minutes" << endl;
                    else
                        cout << "  " << i << ": " << time/1000.0/60.0/60.0 << " hours" << endl;
                }
            }
        }
    }
}


#endif // DLIB_TImING_H__

