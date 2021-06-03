// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ISOTONIC_ReGRESSION_H_
#define DLIB_ISOTONIC_ReGRESSION_H_

#include "isotonic_regression_abstract.h"
#include <vector>
#include <utility>
#include <cstddef>

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
            do_isotonic_regression(begin, end);

            // unpack blocks to output
            for (auto& block : blocks)
            {
                for (size_t k = 0; k < block.num; ++k)
                    set_val(*obegin++, block.avg);
            }

            blocks.clear();
        }

        void operator() (
            std::vector<double>& vect
        ) { (*this)(vect.begin(), vect.end(), vect.begin()); }

        template <typename T, typename U>
        void operator() (
            std::vector<std::pair<T,U>>& vect
        ) { (*this)(vect.begin(), vect.end(), vect.begin()); }


        template <
            typename const_iterator, 
            typename iterator
            >
        void fit_with_linear_output_interpolation (
            const_iterator begin,
            const_iterator end,
            iterator obegin
        )
        {
            do_isotonic_regression(begin, end);

            // Unpack blocks to output, but here instead of producing the step function
            // output we linearly interpolate.  Note that this actually fits the data less
            // than the step-function, but in many applications might be closer to what you
            // really want when using isotonic_regression than the step function.
            for (size_t i = 0; i < blocks.size(); ++i)
            {
                auto& block = blocks[i];

                double prev = (blocks.front().avg + block.avg)/2;
                if (i > 0)
                    prev = (blocks[i-1].avg+block.avg)/2;

                double next = (blocks.back().avg + block.avg)/2;
                if (i+1 < blocks.size())
                    next = (blocks[i+1].avg+block.avg)/2;

                for (size_t k = 0; k < block.num; ++k)
                {
                    const auto mid = block.num/2.0;
                    if (k < mid)
                    {
                        const double alpha = k/mid;
                        set_val(*obegin++, (1-alpha)*prev + alpha*block.avg);
                    }
                    else
                    {
                        const double alpha = k/mid-1;
                        set_val(*obegin++, alpha*next + (1-alpha)*block.avg);
                    }
                }
            }

            blocks.clear();
        }

        void fit_with_linear_output_interpolation (
            std::vector<double>& vect
        ) { fit_with_linear_output_interpolation(vect.begin(), vect.end(), vect.begin()); }

        template <typename T, typename U>
        void fit_with_linear_output_interpolation (
            std::vector<std::pair<T,U>>& vect
        ) { fit_with_linear_output_interpolation(vect.begin(), vect.end(), vect.begin()); }

    private:

        template <
            typename const_iterator
            >
        void do_isotonic_regression (
            const_iterator begin,
            const_iterator end
        )
        {
            blocks.clear();

            // Do the actual isotonic regression.  The output is a step-function and is
            // stored in the vector of blocks.
            for (auto i = begin; i != end; ++i)
            {
                blocks.emplace_back(get_val(*i));
                while (blocks.size() > 1 && prev_block().avg > current_block().avg)
                {
                    // merge the last two blocks.
                    prev_block() = prev_block() + current_block();
                    blocks.pop_back();
                }
            }
        }


        template <typename T>
        static double get_val(const T& v) { return v;}

        template <typename T, typename U>
        static double get_val(const std::pair<T,U>& v) { return v.second;}

        template <typename T>
        static void set_val(T& v, double val) { v = val;}

        template <typename T, typename U>
        static void set_val(std::pair<T,U>& v, double val) { v.second = val;}



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


