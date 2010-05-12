// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GRAPH_CrEATION_ABSTRACT_H__
#ifdef DLIB_GRAPH_CrEATION_ABSTRACT_H__

#include <vector>
#include "../string.h"
#include "sample_pair_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type,
        typename distance_function_type,
        typename alloc,
        typename T
        >
    void find_percent_shortest_edges_randomly (
        const vector_type& samples,
        const distance_function_type& dist_funct,
        const double percent,
        const unsigned long num,
        const T& random_seed,
        std::vector<sample_pair, alloc>& out
    );
    /*!
        requires
            - samples.size() > 1
            - 0 < percent <= 1
            - num > 0
            - random_seed must be convertible to a string by dlib::cast_to_string()
            - dist_funct(samples[i], samples[j]) must be a valid expression that evaluates
              to a floating point number
        ensures
            - This function randomly samples the space of pairs of integers between
              0 and samples.size()-1 inclusive.  For each of these pairs, (i,j), a
              sample_pair is created as follows:    
                sample_pair(i, j, dist_funct(samples[i], samples[j]))
              num such sample_pair objects are generated, duplicates are removed, and 
              then the top percent of them with the smallest distance are stored into 
              out.  
            - #out.size() <= num*percent 
            - contains_duplicate_pairs(#out) == false
            - for all valid i:
                - #out[i].distance() == dist_funct(samples[#out[i].index1()], samples[#out[i].index2()])
            - random_seed is used to seed the random number generator used by this 
              function.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type,
        typename distance_function_type,
        typename alloc
        >
    void find_k_nearest_neighbors (
        const vector_type& samples,
        const distance_function_type& dist_funct,
        const unsigned long k,
        std::vector<sample_pair, alloc>& out
    );
    /*!
        requires
            - samples.size() > k
            - k > 0
            - dist_funct(samples[i], samples[j]) must be a valid expression that evaluates
              to a floating point number
        ensures
            - #out == a set of sample_pair objects that represent all the k nearest
              neighbors in samples according to the given distance function dist_funct.
            - for all valid i:
                - #out[i].distance() == dist_funct(samples[#out[i].index1()], samples[#out[i].index2()])
            - contains_duplicate_pairs(#out) == false
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type 
        >
    bool contains_duplicate_pairs (
        const vector_type& pairs
    );
    /*!
        requires
            - vector_type == a type with an interface compatible with std::vector and 
              it must in turn contain objects with an interface compatible with dlib::sample_pair
        ensures
            - if (pairs contains any elements that are equal according to operator==) then
                - returns true
            - else
                - returns false
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type 
        >
    unsigned long max_index_value_plus_one (
        const vector_type& pairs
    );
    /*!
        requires
            - vector_type == a type with an interface compatible with std::vector and 
              it must in turn contain objects with an interface compatible with dlib::sample_pair
        ensures
            - if (pairs.size() == 0) then
                - returns 0
            - else
                - returns a number N such that: 
                    - for all i:  pairs[i].index1()   <  N && pairs[i].index2()   <  N
                    - for some j: pairs[j].index1()+1 == N || pairs[j].index2()+1 == N
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GRAPH_CrEATION_ABSTRACT_H__

