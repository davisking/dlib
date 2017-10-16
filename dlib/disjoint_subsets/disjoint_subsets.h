// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DISJOINT_SUBsETS_Hh_
#define DLIB_DISJOINT_SUBsETS_Hh_

#include "disjoint_subsets_abstract.h"
#include <vector>
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class disjoint_subsets
    {
    public:

        void clear (
        ) noexcept
        {
            items.clear();
            sets_size.clear();
            number_of_sets = 0;
        }

        void set_size (
            unsigned long new_size
        )
        {
            items.resize(new_size);
            sets_size.resize(new_size);
            for (unsigned long i = 0; i < items.size(); ++i)
            {
                items[i].parent = i;
                items[i].rank = 0;
                sets_size[i] = 1;
            }
            number_of_sets = new_size;
        }

        unsigned long size (
        ) const noexcept
        {
            return items.size();
        }

        unsigned long find_set (
            unsigned long item 
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(item < size(), 
                "\t unsigned long disjoint_subsets::find_set()"
                << "\n\t item must be less than size()"
                << "\n\t item: " << item 
                << "\n\t size(): " << size() 
                << "\n\t this: " << this
                );

            if (items[item].parent == item)
            {
                return item;
            }
            else
            {
                // find root of item
                unsigned long x = item;
                do
                {
                    x = items[x].parent;
                } while (items[x].parent != x);

                // do path compression
                const unsigned long root = x;
                x = item;
                while (items[x].parent != x)
                {
                    const unsigned long prev = x;
                    x = items[x].parent;
                    items[prev].parent = root;
                }

                return root;
            }
        }

        unsigned long merge_sets (
            unsigned long a,
            unsigned long b
        )
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

            if (items[a].rank > items[b].rank)
            {
                items[b].parent = a;
                sets_size[a] += sets_size[b];
                number_of_sets--;
                return a;
            }
            else
            {
                items[a].parent = b;
                sets_size[b] += sets_size[a];
                if (items[a].rank == items[b].rank)
                {
                    items[b].rank = items[b].rank + 1;
                }
                number_of_sets--;
                return b;
            }
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

        struct data
        {
            unsigned long rank;
            unsigned long parent;
        };

        mutable std::vector<data> items;
        mutable std::vector<unsigned long> sets_size;
        unsigned long number_of_sets{0};

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DISJOINT_SUBsETS_Hh_

