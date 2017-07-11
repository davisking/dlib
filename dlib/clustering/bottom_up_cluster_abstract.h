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
// ----------------------------------------------------------------------------------------

    struct snl_range
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents an interval on the real number line.  It is used
                to store the outputs of the segment_number_line() routine defined below.
        !*/

        snl_range(
        );
        /*!
            ensures
                - #lower == 0
                - #upper == 0
        !*/

        snl_range(
            double val
        );
        /*!
            ensures
                - #lower == val 
                - #upper == val 
        !*/

        snl_range(
            double l, 
            double u
        );
        /*!
            requires
                - l <= u
            ensures
                - #lower == l 
                - #upper == u 
        !*/

        double lower;
        double upper;

        double width(
        ) const { return upper-lower; }
        /*!
            ensures
                - returns the width of this interval on the number line.
        !*/

        bool operator<(const snl_range& item) const { return lower < item.lower; }
        /*!
            ensures
                - provides a total ordering of snl_range objects assuming they are
                  non-overlapping.
        !*/
    };

    std::ostream& operator<< (std::ostream& out, const snl_range& item );
    /*!
        ensures
            - prints item to out in the form [lower,upper].
    !*/

// ----------------------------------------------------------------------------------------

    std::vector<snl_range> segment_number_line (
        const std::vector<double>& x,
        const double max_range_width
    );
    /*!
        requires
            - max_range_width >= 0
        ensures
            - Finds a clustering of the values in x and returns the ranges that define the
              clustering.  This routine uses a combination of bottom up clustering and a
              simple greedy scan to try and find the most compact set of ranges that
              contain all the values in x.  
            - This routine has approximately linear runtime.
            - Every value in x will be contained inside one of the returned snl_range
              objects;
            - All returned snl_range object's will have a width() <= max_range_width and
              will also be non-overlapping.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOTTOM_uP_CLUSTER_ABSTRACT_Hh_


