// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_INPUT_H_
#define DLIB_DNn_INPUT_H_

#include "input_abstract.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../pixel.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename T>
    class input
    {
        const static bool always_false = sizeof(T)!=sizeof(T); 
        static_assert(always_false, "Unsupported type given to input<>.  input<> only supports "
            "dlib::matrix and dlib::array2d objects."); 
    };

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename MM, typename L>
    class input<matrix<T,NR,NC,MM,L>> 
    {
    public:
        typedef matrix<T,NR,NC,MM,L> input_type;
        const static unsigned int sample_expansion_factor = 1;

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(begin,end) > 0,"");
            const auto nr = begin->nr();
            const auto nc = begin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = begin; i != end; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(begin,end), nr, nc, pixel_traits<T>::num);

            auto ptr = data.host();
            for (auto i = begin; i != end; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto temp = pixel_to_vector<float>((*i)(r,c));
                        for (long j = 0; j < temp.size(); ++j)
                            *ptr++ = temp(j);
                    }
                }
            }

        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename MM>
    class input<array2d<T,MM>> 
    {
    public:
        typedef array2d<T,MM> input_type;
        const static unsigned int sample_expansion_factor = 1;

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(begin,end) > 0,"");
            const auto nr = begin->nr();
            const auto nc = begin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = begin; i != end; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input::to_tensor()"
                    << "\n\t All array2d objects given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(begin,end), nr, nc, pixel_traits<T>::num);

            auto ptr = data.host();
            for (auto i = begin; i != end; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto temp = pixel_to_vector<float>((*i)[r][c]);
                        for (long j = 0; j < temp.size(); ++j)
                            *ptr++ = temp(j);
                    }
                }
            }

        }
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_H_

