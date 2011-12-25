// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PROJECTION_HASh_ABSTRACT_H__
#ifdef DLIB_PROJECTION_HASh_ABSTRACT_H__

#include "../matrix.h"
#include "../rand.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class projection_hash
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/
    public:

        projection_hash(
        );

        template <typename EXP1, typename EXP2>
        projection_hash(
            const matrix_exp<EXP1>& proj,
            const matrix_exp<EXP2>& offset
        ); 

        const matrix<double>& get_projection_matrix (
        ) const;

        const matrix<double,0,1>& get_offset_matrix (
        ) const; 

        unsigned long size (
        ) const;

        template <typename EXP>
        unsigned long operator() (
            const matrix_exp<EXP>& v
        ) const;
    };

// ----------------------------------------------------------------------------------------

    void serialize (
        const projection_hash& item,
        std::ostream& out
    );

    void deserialize (
        projection_hash& item,
        std::istream& in 
    );

// ----------------------------------------------------------------------------------------

    template <typename vector_type>
    projection_hash create_random_projection_hash (
        const vector_type& v,
        const int bits
    );

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PROJECTION_HASh_ABSTRACT_H__

