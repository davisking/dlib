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

    template <typename vector_type>
    projection_hash create_random_projection_hash (
        const vector_type& v,
        const int bits
    ) 
    {

        // compute a whitening matrix
        matrix<double> whiten = trans(chol(pinv(covariance(vector_to_matrix(v)))));


        // hashes
        std::vector<unsigned long> h(v.size(),0);

        std::vector<double> vals(v.size(),0);

        // number of hits for each hash value
        std::vector<unsigned long> counts;

        std::vector<double> temp;

        // build a random projection matrix
        dlib::rand rnd;
        matrix<double> proj(bits, v[0].size());
        for (long r = 0; r < proj.nr(); ++r)
            for (long c = 0; c < proj.nc(); ++c)
                proj(r,c) = rnd.get_random_gaussian();

        // merge whitening matrix with projection matrix
        proj = proj*whiten;

        matrix<double,0,1> offset(bits);


        // figure out what the offset values should be
        for (int itr = 0; itr < offset.size(); ++itr)
        {
            counts.assign(std::pow(2,bits), 0);
            // count the popularity of each hash value
            for (unsigned long i = 0; i < h.size(); ++i)
            {
                h[i] <<= 1;
                counts[h[i]] += 1;
            }

            const unsigned long max_h = index_of_max(vector_to_matrix(counts));

            temp.clear();
            for (unsigned long i = 0; i < v.size(); ++i)
            {
                vals[i] = dot(rowm(proj,itr), v[i]);
                if (h[i] == max_h)
                    temp.push_back(vals[i]);
            }

            // split down the middle
            std::sort(temp.begin(), temp.end());
            const double split = temp[temp.size()/2];
            offset(itr) = -split;

            for (unsigned long i = 0; i < vals.size(); ++i)
            {
                if (vals[i] - split > 0)
                    h[i] |= 1;
            }
        }


        return projection_hash(proj, offset);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PROJECTION_HASh_H__

