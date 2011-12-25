// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CREATE_RANDOM_PROJECTION_HAsH_H__
#define DLIB_CREATE_RANDOM_PROJECTION_HAsH_H__

#include "create_random_projection_hash_abstract.h"
#include "projection_hash.h"
#include "../matrix.h"
#include "../rand.h"
#include <vector>

namespace dlib
{

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

#endif // DLIB_CREATE_RANDOM_PROJECTION_HAsH_H__

