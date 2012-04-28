// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FIND_MAX_FACTOR_GRAPH_PoTTS_ABSTRACT_H__
#ifdef DLIB_FIND_MAX_FACTOR_GRAPH_PoTTS_ABSTRACT_H__

#include "../matrix.h"
#include "min_cut_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class potts_problem 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a boolean valued factor graph or graphical model 
                that can be efficiently operated on using graph cuts.  In particular, this 
                object defines the interface a MAP problem on a factor graph must 
                implement if it is to be solved using the find_max_factor_graph_potts() 
                routine defined at the bottom of this file.  

                Note that there is no dlib::potts_problem object.  What you are looking 
                at here is simply the interface definition for a Potts problem.  You must 
                implement your own version of this object for the problem you wish to 
                solve and then pass it to the find_max_factor_graph_potts() routine.

                Note also that a factor graph should not have any nodes which are 
                neighbors with themselves.  Additionally, the graph is undirected. This
                mean that if A is a neighbor of B then B must be a neighbor of A for
                the MAP problem to be valid.
        !*/

    public:

        unsigned long number_of_nodes (
        ) const; 
        /*!
            ensures
                - returns the number of nodes in the factor graph.  Or in other words, 
                  returns the number of variables in the MAP problem/Potts model.
        !*/

        unsigned long number_of_neighbors (
            unsigned long idx
        ) const; 
        /*!
            requires
                - idx < number_of_nodes()
            ensures
                - returns the number of neighbors of node idx.
        !*/

        // This is an optional variable which specifies a number that is always
        // greater than or equal to number_of_neighbors(idx).  If you don't know
        // the value at compile time then either don't include max_number_of_neighbors 
        // in your potts_problem object or set it to 0.
        const static unsigned long max_number_of_neighbors = 0; 

        unsigned long get_neighbor (
            unsigned long idx,
            unsigned long n 
        ) const;
        /*!
            requires
                - idx < number_of_nodes()
                - n < number_of_neighbors(idx)
            ensures
                - returns the node index value of the n-th neighbor of 
                  the node with index value idx.
                - The neighbor relationship is reciprocal.  That is, if 
                  get_neighbor(A,i)==B then there is a value of j such 
                  that get_neighbor(B,j)==A.
                - A node is never its own neighbor.  That is, there is
                  no i such that get_neighbor(idx,i)==idx.
        !*/

        unsigned long get_neighbor_idx (
            unsigned long idx1,
            unsigned long idx2
        ) const;
        /*!
            requires
                - idx1 < number_of_nodes()
                - idx2 < number_of_nodes()
            ensures
                - This function is basically the inverse of get_neighbor().
                - returns a number IDX such that:
                    - get_neighbor(idx1,IDX) == idx2
                    - IDX < number_of_neighbors(node_idx1)
        !*/

        void set_label (
            const unsigned long& idx,
            node_label value
        );
        /*!
            requires
                - idx < number_of_nodes()
            ensures
                - #get_label(idx) == value
        !*/

        node_label get_label (
            const unsigned long& idx
        ) const;
        /*!
            requires
                - idx < number_of_nodes()
            ensures
                - returns the current label for the idx-th node.  This is a value which is
                  0 if the node's label is false and is any other value if it is true.  

                  Note that this value is not used by factor_value() or factor_value_disagreement().
                  It is simply here to provide a mechanism for find_max_factor_graph_potts()
                  to return its labeled result.  Additionally, the reason it returns a 
                  node_label rather than a bool is because doing it this way facilitates 
                  use of a graph cut algorithm for the solution of the MAP problem.  For 
                  more of an explanation you should read the paper referenced by the min_cut
                  object.
        !*/

        // This typedef should be for a type like int or double.  It
        // must also be capable of representing signed values.
        typedef an_integer_or_real_type value_type;

        value_type factor_value (
            unsigned long idx, 
            bool value
        ) const;
        /*!
            requires
                - idx < number_of_nodes()
            ensures
                - returns a value which indicates how "good" it is to assign the idx-node
                  a label equal to value.  The larger the value, the more desirable
                  the label contained by value. 
        !*/

        value_type factor_value_disagreement (
            unsigned long idx1, 
            unsigned long idx2
        ) const;
        /*!
            requires
                - idx1 < number_of_nodes()
                - idx2 < number_of_nodes()
                - idx1 != idx2
                - the idx1-th node and idx2-th node are neighbors in the graph.  That is, 
                  get_neighbor(idx1,i)==idx2 for some value of i.
            ensures
                - returns a number >= 0.  This is the penalty for giving node idx1 and idx2
                  different labels.  Larger values indicate a larger penalty.
                - this function is symmetric.  That is, it is true that: 
                  factor_value_disagreement(i,j) == factor_value_disagreement(j,i)
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename potts_problem
        >
    typename potts_problem::value_type potts_model_score (
        const potts_problem& prob 
    );
    /*!
        requires
            - potts_problem == an object with an interface compatible with the potts_problem 
              object defined at the top of this file.
            - for all valid i and j:
                - prob.factor_value_disagreement(i,j) == prob.factor_value_disagreement(j,i)
        ensures
            - computes the model score for the given potts_problem.  To define this
              precisely:
                - let L(i) == the boolean label of the ith variable in prob.  Or in other 
                  words, L(i) == (prob.get_label(i) != 0).
                - let F == the sum over valid i of prob.factor_value(i, L(i)).
                - Let D == the sum of all values of prob.factor_value_disagreement(i,j) 
                  whenever the following conditions are true about i and j:
                    - i and j are neighbors in the graph defined by prob, that is,
                      it is valid to call prob.factor_value_disagreement(i,j).
                    - L(i) != L(j)
                    - i < j
                      (i.e. We want to make sure to only count the edge between i and j once)

                - Then this function returns F - D
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename potts_problem
        >
    void find_max_factor_graph_potts (
        potts_problem& prob 
    )
    /*!
        requires
            - potts_problem == an object with an interface compatible with the potts_problem 
              object defined at the top of this file.
            - for all valid i and j:
                - prob.factor_value_disagreement(i,j) >= 0
                - prob.factor_value_disagreement(i,j) == prob.factor_value_disagreement(j,i)
        ensures
            - This function is a tool for exactly solving the MAP problem in a Potts
              model.  In particular, this means that this function finds the assignments 
              to all the labels in prob which maximizes potts_model_score(#prob).
            - Note that this routine is a little bit faster if all the values 
              returned by prob.factor_value() are negative.  So if you can arrange for that
              to be true without spending any extra CPU cycles then it might be a good idea 
              to do so.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FIND_MAX_FACTOR_GRAPH_PoTTS_ABSTRACT_H__


