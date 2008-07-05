// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KERNEL_FEATURE_RANKINg_H_
#define DLIB_KERNEL_FEATURE_RANKINg_H_

#include <vector>
#include <limits>

#include "feature_ranking_abstract.h"
#include "kcentroid.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename sample_matrix_type,
        typename label_matrix_type
        >
    matrix<typename kernel_type::scalar_type,0,2,typename kernel_type::mem_manager_type> rank_features (
        const kcentroid<kernel_type>& kc,
        const sample_matrix_type& samples,
        const label_matrix_type& labels,
        const long num_features
    )
    {
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mm;

        COMPILE_TIME_ASSERT(is_matrix<sample_matrix_type>::value);
        COMPILE_TIME_ASSERT(is_matrix<typename sample_matrix_type::type>::value);
        COMPILE_TIME_ASSERT(is_matrix<label_matrix_type>::value);
  
        // make sure requires clause is not broken
        DLIB_ASSERT(samples.nc() == 1 && labels.nc() == 1 && samples.size() == labels.size() &&
                    samples.size() > 0 && num_features > 0,
            "\tmatrix rank_features()"
            << "\n\t you have given invalid arguments to this function"
            << "\n\t samples.nc():   " << samples.nc() 
            << "\n\t labels.nc():    " << labels.nc()
            << "\n\t samples.size(): " << samples.size() 
            << "\n\t labels.size():  " << labels.size()
            << "\n\t num_features:   " << num_features 
            );

#ifdef ENABLE_ASSERTS
        for (long i = 0; i < samples.size(); ++i)
        {
            DLIB_ASSERT(samples(i).nc() == 1 && num_features <= samples(i).nr() &&
                        samples(0).nr() == samples(i).nr(),
                "\tmatrix rank_features()"
                << "\n\t you have given invalid arguments to this function"
                << "\n\t num_features:    " << num_features 
                << "\n\t samples(i).nc(): " << samples(i).nc() 
                << "\n\t samples(i).nr(): " << samples(i).nr() 
                << "\n\t samples(0).nr(): " << samples(0).nr() 
                );
        }
#endif


        matrix<scalar_type,0,2,mm> results(num_features, 2);
        matrix<scalar_type,sample_matrix_type::type::NR,1,mm> mask(samples(0).nr());
        set_all_elements(mask,0);

        using namespace std;

        for (long i = 0; i < results.nr(); ++i)
        {
            long worst_feature_idx = 0;
            scalar_type worst_feature_score = -std::numeric_limits<scalar_type>::infinity();

            // figure out which feature to remove next
            for (long j = 0; j < mask.size(); ++j)
            {
                // skip features we have already removed
                if (mask(j) == 1)
                    continue;

                kcentroid<kernel_type> c1(kc);
                kcentroid<kernel_type> c2(kc);

                // temporarily remove this feature from the working set of features
                mask(j) = 1;

                // find the centers of each class
                for (long s = 0; s < samples.size(); ++s)
                {
                    if (labels(s) < 0)
                    {
                        c1.train(pointwise_multiply(samples(s),mask));
                    }
                    else
                    {
                        c2.train(pointwise_multiply(samples(s),mask));
                    }

                }

                // find the distance between the two centroids and use that
                // as the score
                const double score = c1(c2);

                if (score > worst_feature_score)
                {
                    worst_feature_score = score;
                    worst_feature_idx = j;
                }

                // add this feature back to the working set of features
                mask(j) = 0;

            }

            // now that we know what the next worst feature is record it 
            mask(worst_feature_idx) = 1;
            results(i,0) = worst_feature_idx;
            results(i,1) = worst_feature_score; 
        }

        // now normalize the results 
        set_colm(results,1) = colm(results,1)/max(colm(results,1));
        for (long i = results.nr()-1; i > 0; --i)
        {
            results(i,1) -= results(i-1,1);
        }

        return results;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KERNEL_FEATURE_RANKINg_H_


