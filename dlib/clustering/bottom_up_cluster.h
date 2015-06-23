// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BOTTOM_uP_CLUSTER_Hh_
#define DLIB_BOTTOM_uP_CLUSTER_Hh_

#include <queue>
#include <map>

#include "bottom_up_cluster_abstract.h"
#include "../algs.h"
#include "../matrix.h"
#include "../disjoint_subsets.h"
#include "../graph_utils.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace buc_impl
    {
        inline void merge_sets (
            matrix<double>& dists,
            unsigned long dest,
            unsigned long src
        )
        {
            for (long r = 0; r < dists.nr(); ++r)
                dists(dest,r) = dists(r,dest) = std::max(dists(r,dest), dists(r,src));
        }

        struct compare_dist
        {
            bool operator() (
                const sample_pair& a,
                const sample_pair& b
            ) const
            {
                return a.distance() > b.distance();
            }
        };
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    unsigned long bottom_up_cluster (
        const matrix_exp<EXP>& dists_,
        std::vector<unsigned long>& labels,
        unsigned long min_num_clusters,
        double max_dist = std::numeric_limits<double>::infinity()
    )
    {
        matrix<double> dists = matrix_cast<double>(dists_);
        // make sure requires clause is not broken
        DLIB_CASSERT(dists.nr() == dists.nc() && min_num_clusters > 0, 
            "\t unsigned long bottom_up_cluster()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t dists.nr(): " << dists.nr() 
            << "\n\t dists.nc(): " << dists.nc() 
            << "\n\t min_num_clusters: " << min_num_clusters 
            );

        using namespace buc_impl;

        labels.resize(dists.nr());
        disjoint_subsets sets;
        sets.set_size(dists.nr());
        if (labels.size() == 0)
            return 0;

        // push all the edges in the graph into a priority queue so the best edges to merge
        // come first.
        std::priority_queue<sample_pair, std::vector<sample_pair>, compare_dist> que;
        for (long r = 0; r < dists.nr(); ++r)
            for (long c = r+1; c < dists.nc(); ++c)
                que.push(sample_pair(r,c,dists(r,c)));

        // Now start merging nodes.
        for (unsigned long iter = min_num_clusters; iter < sets.size(); ++iter)
        {
            // find the next best thing to merge.
            double best_dist = que.top().distance();
            unsigned long a = sets.find_set(que.top().index1());
            unsigned long b = sets.find_set(que.top().index2());
            que.pop();
            // we have been merging and modifying the distances, so make sure this distance
            // is still valid and these guys haven't been merged already.
            while(a == b || best_dist < dists(a,b))
            {
                // Haven't merged it yet, so put it back in with updated distance for
                // reconsideration later.
                if (a != b)
                    que.push(sample_pair(a, b, dists(a, b)));

                best_dist = que.top().distance();
                a = sets.find_set(que.top().index1());
                b = sets.find_set(que.top().index2());
                que.pop();
            }


            // now merge these sets if the best distance is small enough
            if (best_dist > max_dist)
                break;
            unsigned long news = sets.merge_sets(a,b);
            unsigned long olds = (news==a)?b:a;
            merge_sets(dists, news, olds);
        }

        // figure out which cluster each element is in.  Also make sure the labels are
        // contiguous.
        std::map<unsigned long, unsigned long> relabel;
        for (unsigned long r = 0; r < labels.size(); ++r)
        {
            unsigned long l = sets.find_set(r);
            // relabel to make contiguous
            if (relabel.count(l) == 0)
            {
                unsigned long next = relabel.size();
                relabel[l] = next;
            }
            labels[r] = relabel[l];
        }


        return relabel.size();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOTTOM_uP_CLUSTER_Hh_

