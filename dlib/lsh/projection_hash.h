// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PROJECTION_HASh_H__
#define DLIB_PROJECTION_HASh_H__

#include "projection_hash_abstract.h"
#include "../matrix.h"
#include "../rand.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class projection_hash
    {
    public:

        projection_hash() {}

        template <typename EXP1, typename EXP2>
        projection_hash(
            const matrix_exp<EXP1>& proj_,
            const matrix_exp<EXP2>& offset_
        ) : proj(proj_), offset(offset_) {}

        const matrix<double>& get_projection_matrix (
        ) const { return proj; }

        const matrix<double,0,1>& get_offset_matrix (
        ) const { return offset; }

        unsigned long size (
        ) const
        {
            return (unsigned long)std::pow(2, offset.size());
        }

        template <typename EXP>
        unsigned long operator() (
            const matrix_exp<EXP>& v
        ) const
        {
            return do_hash(proj*matrix_cast<double>(v) + offset);
        }

    private:

        template <typename EXP>
        unsigned long do_hash (
            const matrix_exp<EXP>& v
        ) const
        {
            unsigned long h = 0;
            for (long i = 0; i < v.size(); ++i)
            {
                h <<= 1;
                if (v(i) > 0)
                    h |= 1;
            }
            return h;
        }

        matrix<double> proj;
        matrix<double,0,1> offset;
    };

// ----------------------------------------------------------------------------------------

    inline void serialize (
        const projection_hash& item,
        std::ostream& out
    )
    {
        serialize(item.get_projection_matrix(), out);
        serialize(item.get_offset_matrix(), out);
    }

    inline void deserialize (
        projection_hash& item,
        std::istream& in 
    )
    {
        matrix<double> proj;
        matrix<double,0,1> offset;
        deserialize(proj, in);
        deserialize(offset, in);
        item = projection_hash(proj, offset);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PROJECTION_HASh_H__

