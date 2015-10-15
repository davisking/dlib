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
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0,"");
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
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
            data.set_size(std::distance(ibegin,iend), nr, nc, pixel_traits<T>::num);

            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
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

        friend void serialize(const input& item, std::ostream& out)
        {
            serialize("input<matrix>", out);
        }

        friend void deserialize(input& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<matrix>")
                throw serialization_error("Unexpected version found while deserializing dlib::input.");
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
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0,"");
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
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
            data.set_size(std::distance(ibegin,iend), nr, nc, pixel_traits<T>::num);

            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
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

        friend void serialize(const input& item, std::ostream& out)
        {
            serialize("input<array2d>", out);
        }

        friend void deserialize(input& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<array2d>")
                throw serialization_error("Unexpected version found while deserializing dlib::input.");
        }

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_H_

