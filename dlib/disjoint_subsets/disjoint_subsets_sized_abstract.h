// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DISJOINT_SUBsETS_SIZED_ABSTRACT_Hh_
#ifdef DLIB_DISJOINT_SUBsETS_SIZED_ABSTRACT_Hh_

#include <vector>
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class disjoint_subsets_sized 
    {
        /*!
            INITIAL VALUE
                - size() == 0
                - get_number_of_sets() == 0

            WHAT THIS OBJECT REPRESENTS
                This object represents a set of integers which is partitioned into
                a number of disjoint subsets.  It supports the two fundamental operations
                of finding which subset a particular integer belongs to as well as
                merging subsets.  It also allows you to find out how big each subset is.  It 
                is therefore essentially the same thing as dlib::disjoint_subsets, except
                it also keeps track of of the size of each subset.
        !*/
    public:

        void clear (
        ) noexcept;
        /*!
            ensures
                - #size() == 0
                - #get_number_of_sets() == 0
                - returns this object to its initial value
        !*/

        void set_size (
            unsigned long new_size
        );
        /*!
            ensures
                - #size() == new_size
                - #get_number_of_sets() == new_size
                - for all valid i:
                    - #find_set(i) == i
                      (i.e. this object contains new_size subsets, each containing exactly one element)
                    - #get_size_of_set(i) == 1
        !*/

        unsigned long size (
        ) const noexcept;
        /*!
            ensures
                - returns the total number of integer elements represented
                  by this object.
        !*/

        unsigned long find_set (
            unsigned long item
        ) const;
        /*!
            requires
                - item < size()
            ensures
                - Each disjoint subset can be represented by any of its elements (since
                  the sets are all disjoint).  In particular, for each subset we define
                  a special "representative element" which is used to represent it.
                  Therefore, this function returns the representative element for the
                  set which contains item.
                - find_set(find_set(item)) == find_set(item)
                - Note that if A and B are both elements of the same subset then we always
                  have find_set(A) == find_set(B).
        !*/

        unsigned long merge_sets (
            unsigned long a,
            unsigned long b
        );
        /*!
            requires
                - a != b
                - a < size()
                - b < size()
                - find_set(a) == a
                  (i.e. a is the representative element of some set)
                - find_set(b) == b
                  (i.e. b is the representative element of some set)
            ensures
                - #find_set(a) == #find_set(b)
                  (i.e. merges the set's containing a and b)
                - #get_size_of_set(#find_set(a)) == get_size_of_set(a) + get_size_of_set(b)
                - #get_number_of_sets() == get_number_of_sets() - 1
                - returns #find_set(a)
        !*/

        unsigned long get_number_of_sets (
        ) const noexcept;
        /*!
            ensures
                - returns the current number of different subsets.
        !*/

        unsigned long get_size_of_set(
                unsigned long item
        ) const;
        /*!
            requires
                - item < size()
                - find_set(item) == item
                  (i.e. item is the representative element of some set)
            ensures
                - returns the number of elements which belongs to the set where item is the representative element.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DISJOINT_SUBsETS_ABSTRACT_Hh_
