// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BOTTOM_uP_CLUSTER_ABSTRACT_Hh_
#ifdef DLIB_BOTTOM_uP_CLUSTER_ABSTRACT_Hh_

#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    unsigned long bottom_up_cluster (
        const matrix_exp<EXP>& dists,
        std::vector<unsigned long>& labels,
        unsigned long min_num_clusters,
        double max_dist = std::numeric_limits<double>::infinity()
    );
    /*!
        requires
            - dists.nr() == dists.nc()
            - min_num_clusters > 0
            - dists == trans(dists)
              (l.e. dists should be symmetric)
        ensures
            - Runs a bottom up agglomerative clustering algorithm.   
            - Interprets dists as a matrix that gives the distances between dists.nr()
              items.  In particular, we take dists(i,j) to be the distance between the ith
              and jth element of some set.  This function clusters the elements of this set
              into at least min_num_clusters (or dists.nr() if there aren't enough
              elements).  Additionally, within each cluster, the maximum pairwise distance
              between any two cluster elements is <= max_dist.
            - returns the number of clusters found.
            - #labels.size() == dists.nr()
            - for all valid i:
                - #labels[i] == the cluster ID of the node with index i (i.e. the node
                  corresponding to the distances dists(i,*)).  
                - 0 <= #labels[i] < the number of clusters found
                  (i.e. cluster IDs are assigned contiguously and start at 0) 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOTTOM_uP_CLUSTER_ABSTRACT_Hh_


