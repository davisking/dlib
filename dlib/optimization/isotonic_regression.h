// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ISOTONIC_ReGRESSION_H_
#define DLIB_ISOTONIC_ReGRESSION_H_

#include "isotonic_regression_abstract.h"
#include <vector>

namespace dlib
{
    class isotonic_regression
    {
    public:

        template <
            typename const_iterator, 
            typename iterator
            >
        void operator() (
            const_iterator begin,
            const_iterator end,
            iterator obegin
        )
        {
            for (auto i = begin; i != end; ++i)
            {
                blocks.emplace_back(*i);
                while (blocks.size() > 1 && prev_block().avg > current_block().avg)
                {
                    // merge the last two blocks.
                    prev_block() = prev_block() + current_block();
                    blocks.pop_back();
                }
            }

            // unpack results to output
            for (auto& block : blocks)
            {
                for (size_t k = 0; k < block.num; ++k)
                    *obegin++ = block.avg;
            }

            blocks.clear();
        }

        void operator() (
            std::vector<double>& vect
        ) { (*this)(vect.begin(), vect.end(), vect.begin()); }

    private:


        struct block_t
        {
            block_t(double val) : num(1), avg(val) {}
            block_t(size_t n, double val) : num(n), avg(val) {}

            size_t num;
            double avg;

            inline block_t operator+(const block_t& rhs) const
            {
                return block_t(num+rhs.num,
                    (num*avg + rhs.num*rhs.avg)/(num+rhs.num));
            }
        };

        inline block_t& prev_block() { return blocks[blocks.size()-2]; }
        inline block_t& current_block() { return blocks.back(); }

        std::vector<block_t> blocks;
    };
}

#endif // DLIB_ISOTONIC_ReGRESSION_H_


