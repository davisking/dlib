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
// ----------------------------------------------------------------------------------------

    struct snl_range
    {
        snl_range() = default;
        snl_range(double val) : lower(val), upper(val) {}
        snl_range(double l, double u) : lower(l), upper(u) { DLIB_ASSERT(lower <= upper)}

        double lower = 0;
        double upper = 0;

        double width() const { return upper-lower; }
        bool operator<(const snl_range& item) const { return lower < item.lower; }
    };

    inline snl_range merge(const snl_range& a, const snl_range& b)
    {
        return snl_range(std::min(a.lower, b.lower), std::max(a.upper, b.upper));
    }

    inline double distance (const snl_range& a, const snl_range& b)
    {
        return std::max(a.lower,b.lower) - std::min(a.upper,b.upper);
    }

    inline std::ostream& operator<< (std::ostream& out, const snl_range& item )
    {
        out << "["<<item.lower<<","<<item.upper<<"]";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<snl_range> segment_number_line (
        const std::vector<double>& x,
        const double max_range_width
    )
    {
        DLIB_CASSERT(max_range_width >= 0);

        // create initial ranges, one for each value in x.  So initially, all the ranges have
        // width of 0.
        std::vector<snl_range> ranges;
        for (auto v : x)
            ranges.push_back(v);
        std::sort(ranges.begin(), ranges.end());

        std::vector<snl_range> greedy_final_ranges;
        if (ranges.size() == 0)
            return greedy_final_ranges;
        // We will try two different clustering strategies.  One that does a simple greedy left
        // to right sweep and another that does a bottom up agglomerative clustering.  This
        // first loop runs the greedy left to right sweep.  Then at the end of this routine we
        // will return the results that produced the tightest clustering.
        greedy_final_ranges.push_back(ranges[0]);
        for (size_t i = 1; i < ranges.size(); ++i)
        {
            auto m = merge(greedy_final_ranges.back(), ranges[i]);
            if (m.width() <= max_range_width)
                greedy_final_ranges.back() = m;
            else
                greedy_final_ranges.push_back(ranges[i]);
        }


        // Here we do the bottom up clustering.  So compute the edges connecting our ranges.
        // We will simply say there are edges between ranges if and only if they are
        // immediately adjacent on the number line.
        std::vector<sample_pair> edges;
        for (size_t i = 1; i < ranges.size(); ++i)
            edges.push_back(sample_pair(i-1,i, distance(ranges[i-1],ranges[i])));
        std::sort(edges.begin(), edges.end(), order_by_distance<sample_pair>);

        disjoint_subsets sets;
        sets.set_size(ranges.size());

        // Now start merging nodes.
        for (auto edge : edges)
        {
            // find the next best thing to merge.
            unsigned long a = sets.find_set(edge.index1());
            unsigned long b = sets.find_set(edge.index2());

            // merge it if it doesn't result in an interval that's too big.
            auto m = merge(ranges[a], ranges[b]);
            if (m.width() <= max_range_width)
            {
                unsigned long news = sets.merge_sets(a,b);
                ranges[news] = m;
            }
        }

        // Now create a list of the final ranges.  We will do this by keeping track of which
        // range we already added to final_ranges.
        std::vector<snl_range> final_ranges;
        std::vector<bool> already_output(ranges.size(), false);
        for (unsigned long i = 0; i < sets.size(); ++i)
        {
            auto s = sets.find_set(i);
            if (!already_output[s])
            {
                final_ranges.push_back(ranges[s]);
                already_output[s] = true;
            }
        }

        // only use the greedy clusters if they found a clustering with fewer clusters.
        // Otherwise, the bottom up clustering probably produced a more sensible clustering.
        if (final_ranges.size() <= greedy_final_ranges.size())
            return final_ranges;
        else
            return greedy_final_ranges;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOTTOM_uP_CLUSTER_Hh_

