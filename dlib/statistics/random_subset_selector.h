// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RANDOM_SUBSeT_SELECTOR_H_
#define DLIB_RANDOM_SUBSeT_SELECTOR_H_

#include "random_subset_selector_abstract.h"
#include "../rand.h"
#include <vector>
#include "../algs.h"
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
            INITIAL VALUE
                - _max_size == 0
                - items.size() == 0
                - count == 0

            CONVENTION
                - count == the number of times add() has been called since the last
                  time this object was empty.
                - items.size() == size()
                - max_size() == _max_size
        !*/
    public:
        typedef T type;
        typedef memory_manager<char>::kernel_1a mem_manager_type;
        typedef Rand_type rand_type;

        typedef typename std::vector<T>::iterator iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;


        random_subset_selector (
        )
        {
            _max_size = 0;
            make_empty();
        }

        void set_seed(const std::string& value)
        {
            rnd.set_seed(value);
        }

        void make_empty (
        )
        {
            items.resize(0);
            count = 0;
        }

        unsigned long size (
        ) const 
        {
            return items.size();
        }

        void set_max_size (
            unsigned long new_max_size
        )
        {
            items.reserve(new_max_size);
            make_empty();
            _max_size = new_max_size;
        }

        unsigned long max_size (
        ) const
        {
            return _max_size;
        }

        T& operator[] (
            unsigned long idx
        ) 
        {
            return items[idx];
        }

        const T& operator[] (
            unsigned long idx
        ) const
        {
            return items[idx];
        }

        iterator                begin()                         { return items.begin(); }
        const_iterator          begin() const                   { return items.begin(); }
        iterator                end()                           { return items.end(); }
        const_iterator          end() const                     { return items.end(); }

        void add (
            const T& new_item
        )
        {
            if (items.size() < _max_size)
            {
                items.push_back(new_item);
                // swap into a random place
                exchange(items[rnd.get_random_32bit_number()%items.size()], items.back());
            }
            else
            {
                // At this point each element of items has had an equal chance of being in this object.   
                // In particular, the probability that each arrived here is currently items.size()/count.    
                // We need to be able to say that, after this function ends, the probability of any 
                // particular object ending up in items is items.size()/(count+1).  So this means that 
                // we should decide to add new_item into items with this probability.  If we do so then 
                // we pick one of the current items and replace it at random with new_item.

                // Make me a random 64 bit number.   This might seem excessive but I want this object
                // to be able to handle an effectively infinite number of calls to add().  So count
                // might get very large and we need to deal with that properly.
                const unsigned long num1 = rnd.get_random_32bit_number();
                const unsigned long num2 = rnd.get_random_32bit_number();
                uint64 num = num1;
                num <<= 32;
                num |= num2;

                num %= (count+1);

                if (num < items.size())
                {
                    // pick a random element of items and replace it.
                    items[rnd.get_random_32bit_number()%items.size()] = new_item;
                }
            }

            ++count;
        }

        void swap (
            random_subset_selector& a
        )
        {
            a.swap(a.items);
            std::swap(_max_size, a._max_size);
            std::swap(count, a.count);
            rnd.swap(a.rnd);
        }

    private:

        std::vector<T> items;
        unsigned long _max_size;
        uint64 count; 

        rand_type rnd;

    };

    template <
        typename T,
        typename rand_type 
        >
    void swap (
        random_subset_selector<T,rand_type>& a,
        random_subset_selector<T,rand_type>& b
    ) { a.swap(b); }

}

#endif // DLIB_RANDOM_SUBSeT_SELECTOR_H_


