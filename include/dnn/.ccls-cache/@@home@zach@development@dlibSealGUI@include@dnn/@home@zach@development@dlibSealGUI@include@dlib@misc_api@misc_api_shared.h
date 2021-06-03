// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MISC_API_ShARED_Hh_
#define DLIB_MISC_API_ShARED_Hh_

#include <string>
#include "../noncopyable.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class locally_change_current_dir : noncopyable
    {
    public:
        explicit locally_change_current_dir (
            const std::string& new_dir
        )
        {
            reverted = false;
            _old_dir = get_current_dir();
            set_current_dir(new_dir);
        }

        ~locally_change_current_dir()
        {
            revert();
        }

        const std::string& old_dir (
        ) const 
        {
            return _old_dir;
        }

        void revert (
        )
        {
            if (!reverted)
            {
                set_current_dir(_old_dir);
                reverted = true;
            }
        }

    private:
        bool reverted;
        std::string _old_dir;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MISC_API_ShARED_Hh_

