// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CHINESE_WHISPErS_ABSTRACT_Hh_
#ifdef DLIB_CHINESE_WHISPErS_ABSTRACT_Hh_

#include <vector>
#include "../rand.h"
#include "../graph_utils/ordered_sample_pair_abstract.h"
#include "../graph_utils/sample_pair_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    unsigned long chinese_whispers (
        const std::vector<ordered_sample_pair>& edges,
        std::vector<unsigned long>& labels,
        const unsigned long num_iterations,
        dlib::rand& rnd
    );
    /*!
        requires
            - is_ordered_by_index(edges) == true
        ensures
            - Partitions the graph nodes into clusters by label propagation, as
              described in: Chinese Whispers - an Efficient Graph Clustering
              Algorithm and its Application to Natural Language Processing
              Problems by Chris Biemann. Each node starts with its own label
              and repeatedly adopts the strongest label among its outgoing
              neighbors.
            - Interprets edges as a directed graph. edges[i] is an edge from
              node edges[i].index1() to node edges[i].index2() with weight
              edges[i].distance(). Larger weights indicate stronger connections.
              An edge with a distance() of infinity is a "must link" edge.
            - An edge with index1() == index2() declares that the node exists
              so it appears in labels, but it does not vote for any label.
              In particular, a node whose only edges are self-loops keeps its
              own label.
            - returns the number of clusters found.
            - #labels.size() == max_index_plus_one(edges)
            - for all valid i:
                - #labels[i] == the cluster ID of the node with index i in the graph.
                - 0 <= #labels[i] < the number of clusters found
                  (i.e. cluster IDs are assigned contiguously and start at 0)
            - Duplicate edges are interpreted as if there had been just one edge with a
              distance value equal to the sum of all the duplicate edge's distance values.
              Self-loops are excluded from this summation since they do not vote.
            - Performs labels.size()*num_iterations random node updates before
              terminating. Each update picks a node uniformly at random using rnd
              and sets its label to the label with the largest summed edge weight
              among its outgoing neighbors. Ties resolve to the smallest label.
    !*/

// ----------------------------------------------------------------------------------------

    unsigned long chinese_whispers (
        const std::vector<sample_pair>& edges,
        std::vector<unsigned long>& labels,
        const unsigned long num_iterations,
        dlib::rand& rnd
    );
    /*!
        ensures
            - Partitions the graph nodes into clusters with the label propagation
              defined above. Each sample_pair (i,j) is an undirected edge with
              weight distance(). It is interpreted as two directed edges, one
              from i to j and one from j to i, each carrying the same weight,
              except a self-loop (i == j) which is a single edge that only
              declares that the node exists and does not vote.
    !*/

// ----------------------------------------------------------------------------------------

    unsigned long chinese_whispers (
        const std::vector<ordered_sample_pair>& edges,
        std::vector<unsigned long>& labels,
        const unsigned long num_iterations = 100
    );
    /*!
        requires
            - is_ordered_by_index(edges) == true
        ensures
            - performs: return chinese_whispers(edges, labels, num_iterations, rnd)
              where rnd is a default initialized dlib::rand object.
    !*/

// ----------------------------------------------------------------------------------------

    unsigned long chinese_whispers (
        const std::vector<sample_pair>& edges,
        std::vector<unsigned long>& labels,
        const unsigned long num_iterations = 100
    );
    /*!
        ensures
            - performs: return chinese_whispers(edges, labels, num_iterations, rnd)
              where rnd is a default initialized dlib::rand object.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CHINESE_WHISPErS_ABSTRACT_Hh_

