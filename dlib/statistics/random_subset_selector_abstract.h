// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RANDOM_SUBSeT_SELECTOR_ABSTRACT_H_
#ifdef DLIB_RANDOM_SUBSeT_SELECTOR_ABSTRACT_H_

#include <vector>
#include "../rand/rand_kernel_abstract.h"
#include "../memory_manager.h"

namespace dlib
{
    template <
        typename T,
        typename Rand_type = dlib::rand::kernel_1a
        >
    class random_subset_selector
    {
        /*!
            REQUIREMENTS ON T
                T must be a copyable type

            REQUIREMENTS ON Rand_type
                must be an implementation of dlib/rand/rand_kernel_abstract.h

            INITIAL VALUE
                - size() == 0
                - max_size() == 0
                - next_add_accepts() == false

            WHAT THIS OBJECT REPRESENTS
                This object is a tool to help you select a random subset of a large body of data.  
                In particular, it is useful when the body of data is too large to fit into memory.

                So for example, suppose you have 1000000 data samples and you want to select a
                random subset of size 1000.   Then you could do that as follows:
                    
                    random_subset_selector<sample_type> rand_subset;
                    rand_subset.set_max_size(1000)
                    for (int i = 0; i < 1000000; ++i)
                        rand_subset.add( get_next_data_sample());

              
                At the end of the for loop you will have your random subset of 1000 samples.  And by
                random I mean that each of the 1000000 data samples has an equal change of ending
                up in the rand_subset object.


                Note that the above example calls get_next_data_sample() for each data sample.  This 
                may be inefficient since most of the data samples are just ignored.  An alternative 
                method that doesn't require you to load each sample can also be used.  Consider the 
                following:

                    random_subset_selector<sample_type> rand_subset;
                    rand_subset.set_max_size(1000)
                    for (int i = 0; i < 1000000; ++i)
                        if (rand_subset.next_add_accepts())
                            rand_subset.add(get_data_sample(i));
                        else
                            rand_subset.add() 

                In the above example we only actually fetch the data sample into memory if we
                know that the rand_subset would include it into the random subset.  Otherwise,
                we can just call the empty add().

        !*/
    public:
        typedef T type;
        typedef memory_manager<char>::kernel_1a mem_manager_type;
        typedef Rand_type rand_type;

        typedef typename std::vector<T>::iterator iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;

        random_subset_selector (
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        void set_seed(
            const std::string& value
        );
        /*!
            ensures
                - sets the seed of the random number generator that is embedded in
                  this object to the given value.
        !*/

        void make_empty (
        );
        /*!
            ensures
                - #size() == 0
        !*/

        unsigned long size (
        ) const;
        /*!
            ensures
                - returns the number of items of type T currently contained in this object
        !*/

        void set_max_size (
            unsigned long new_max_size
        );
        /*!
            ensures
                - #max_size() == new_max_size
                - #size() == 0
        !*/

        unsigned long max_size (
        ) const;
        /*!
            ensures
                - returns the maximum allowable size for this object
        !*/

        T& operator[] (
            unsigned long idx
        );
        /*!
            requires
                - idx < size()
            ensures
                - returns a non-const reference to the idx'th element of this object
        !*/

        const T& operator[] (
            unsigned long idx
        ) const;
        /*!
            requires
                - idx < size()
            ensures
                - returns a const reference to the idx'th element of this object
        !*/

        bool next_add_accepts (
        ) const;
        /*!
            ensures
                - if (the next call to add(item) will result in item being included
                  into *this) then
                    - returns true
                    - Note that the next item will always be accepted if size() < max_size().
                - else
                    - returns false
                    - Note that the next item will never be accepted if max_size() == 0.
        !*/

        void add (
            const T& new_item
        );
        /*!
            ensures
                - if (next_add_accepts()) then
                    - places new_item into *this object at a random location
                    - if (size() < max_size()) then
                        - #size() == size() + 1
                - #next_add_accepts() == The updated information about the acceptance
                  of the next call to add()
        !*/

        void add (
        );
        /*!
            requires
                - next_add_accepts() == false
            ensures
                - This function does nothing but update the value of #next_add_accepts()
        !*/

        iterator begin(
        );
        /*!
            ensures
                - if (size() > 0) then
                    - returns an iterator referring to the first element in 
                      this container.
                - else
                    - returns end()
        !*/
        
        const_iterator begin(
        ) const;
        /*!
            ensures
                - if (size() > 0) then
                    - returns a const_iterator referring to the first element in 
                      this container.
                - else
                    - returns end()
        !*/

        iterator end(
        ); 
        /*!
            ensures
                - returns an iterator that represents one past the end of
                  this container
        !*/

        const_iterator end(
        ) const;
        /*!
            ensures
                - returns an iterator that represents one past the end of
                  this container
        !*/

        void swap (
            random_subset_selector& item
        );
        /*!
            ensures
                - swaps *this and item
        !*/

    };

    template <
        typename T,
        typename rand_type 
        >
    void swap (
        random_subset_selector<T,rand_type>& a,
        random_subset_selector<T,rand_type>& b
    ) { a.swap(b); }
    /*!
        provides global swap support
    !*/

}

#endif // DLIB_RANDOM_SUBSeT_SELECTOR_ABSTRACT_H_

