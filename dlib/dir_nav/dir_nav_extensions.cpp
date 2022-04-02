// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DIR_NAV_EXTENSIONs_CPP_
#define DLIB_DIR_NAV_EXTENSIONs_CPP_

#include "dir_nav_extensions.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace implementation_details
    {
        void get_all_sub_dirs (
            const directory& top_of_tree,
            unsigned long max_depth,
            std::vector<directory>& result,
            std::vector<directory>& temp
        )
        {
            if (max_depth > 0)
            {
                top_of_tree.get_dirs(temp);
                const unsigned long start = result.size();
                result.insert(result.end(), temp.begin(), temp.end());
                const unsigned long end = start + temp.size();

                for (unsigned long i = start; i < end; ++i)
                {
                    get_all_sub_dirs(result[i], max_depth-1, result, temp);
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    bool file_exists (
        const std::string& filename
    )
    {
        try
        {
            dlib::file temp(filename);
            return true;
        }
        catch (file::file_not_found&)
        {
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    bool directory_exists (
        const std::string& dirname
    )
    {
        try
        {
            dlib::directory temp(dirname);
            return true;
        }
        catch (directory::dir_not_found&)
        {
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    directory get_parent_directory (
        const directory& dir
    )
    {
        return dir.get_parent();
    }

// ----------------------------------------------------------------------------------------

    directory get_parent_directory (
        const file& f
    )
    {
        if (f.full_name().size() == 0)
            return directory();

        std::string::size_type pos = f.full_name().find_last_of("\\/");

        if (pos == std::string::npos)
            return directory();

        return directory(f.full_name().substr(0,pos));
    }

// ----------------------------------------------------------------------------------------

    std::string select_oldest_file (
        const std::string& filename1,
        const std::string& filename2
    )
    {
        file f1, f2;
        try{f1 = file(filename1);} catch(file::file_not_found&) { return filename1; }
        try{f2 = file(filename2);} catch(file::file_not_found&) { return filename2; }

        if (f1.last_modified() < f2.last_modified())
            return filename1;
        else
            return filename2;
    }

// ----------------------------------------------------------------------------------------

    std::string select_newest_file (
        const std::string& filename1,
        const std::string& filename2
    )
    {
        file f1, f2;
        try{f1 = file(filename1);} catch(file::file_not_found&) { return filename2; }
        try{f2 = file(filename2);} catch(file::file_not_found&) { return filename1; }

        if (f1.last_modified() > f2.last_modified())
            return filename1;
        else
            return filename2;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DIR_NAV_EXTENSIONs_CPP_
