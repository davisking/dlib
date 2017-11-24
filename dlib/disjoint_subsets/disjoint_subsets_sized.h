// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DISJOINT_SUBsETS_SIZED_Hh_
#define DLIB_DISJOINT_SUBsETS_SIZED_Hh_

#include "disjoint_subsets_sized_abstract.h"
#include "disjoint_subsets.h"
#include <vector>
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class disjoint_subsets_sized : public disjoint_subsets
    {
    public:

        virtual void clear (
        ) noexcept override
        {
            disjoint_subsets::clear();
            sets_size.clear();
            number_of_sets = 0;
        }

        virtual void set_size (
            unsigned long new_size
        ) override
        {
            disjoint_subsets::set_size(new_size);
            sets_size.assign(new_size, 1);
            number_of_sets = new_size;
        }

        virtual unsigned long merge_sets (
            unsigned long a,
            unsigned long b
        ) override
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(a != b &&
                        a < size() &&
                        b < size() &&
                        find_set(a) == a &&
                        find_set(b) == b,
                "\t unsigned long disjoint_subsets::merge_sets(a,b)"
                << "\n\t invalid arguments were given to this function"
                << "\n\t a: " << a
                << "\n\t b: " << b
                << "\n\t size(): " << size()
                << "\n\t find_set(a): " << find_set(a)
                << "\n\t find_set(b): " << find_set(b)
                << "\n\t this: " << this
                );

            disjoint_subsets::merge_sets(a, b);

            if (find_set(a) == a) sets_size[a] += sets_size[b];
            else sets_size[b] += sets_size[a];
            --number_of_sets;

            return find_set(a);
        }

        unsigned long get_number_of_sets (
        ) const noexcept
        {
            return number_of_sets;
        }

        unsigned long get_size_of_set(
                unsigned long item
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(item < size() &&
                        find_set(item) == item,
                        "\t unsigned long disjoint_subsets::get_size_of_set()"
                                << "\n\t invalid arguments were given to this function"
                                << "\n\t item: " << item
                                << "\n\t size(): " << size()
                                << "\n\t find_set(item): " << find_set(item)
                                << "\n\t this: " << this
            );

            return sets_size[item];
        }

    private:

        /*
            See the book Introduction to Algorithms by Cormen, Leiserson, Rivest and Stein
            for a discussion of how this algorithm works.
        */

        mutable std::vector<unsigned long> sets_size;
        unsigned long number_of_sets{0};

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DISJOINT_SUBsETS_SIZED_Hh_
