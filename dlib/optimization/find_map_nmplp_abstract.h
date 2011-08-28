// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FIND_MAP_nMPLP_ABSTRACT_H__
#ifdef DLIB_FIND_MAP_nMPLP_ABSTRACT_H__

#include <vector>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class map_problem 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        class node_iterator
        {
        public:
            node_iterator();
            bool operator== (const node_iterator& item) const; 
            bool operator!= (const node_iterator& item) const;
            node_iterator& operator++();
        };

        class neighbor_iterator
        {
            neighbor_iterator(); 
            bool operator== (const neighbor_iterator& item) const; 
            bool operator!= (const neighbor_iterator& item) const;
            neighbor_iterator& operator++(); 
        };

        unsigned long number_of_nodes (
        ) const;
        /*!
            ensures
                - returns the number of nodes in the factor graph.  Or in other words, 
                  returns the number of variables in the MAP problem.
        !*/

        node_iterator begin(
        ) const;

        node_iterator end(
        ) const;

        neighbor_iterator begin(
            const node_iterator& it
        ) const;

        neighbor_iterator begin(
            const neighbor_iterator& it
        ) const;

        neighbor_iterator end(
            const node_iterator& it
        ) const;

        neighbor_iterator end(
            const neighbor_iterator& it
        ) const;

        unsigned long node_id (
            const node_iterator& it
        ) const;
        /*!
            requires
                - it == a valid iterator
            ensures
                - returns a number ID such that:
                - 0 <= ID < number_of_nodes()
                - ID == a number which uniquely identifies the node pointed to by it.
        !*/

        unsigned long node_id (
            const neighbor_iterator& it
        ) const;
        /*!
            requires
                - it == a valid iterator
            ensures
                - returns a number ID such that:
                - 0 <= ID < number_of_nodes()
                - ID == a number which uniquely identifies the node pointed to by it.
        !*/

        unsigned long num_states (
            const node_iterator& it
        ) const;

        unsigned long num_states (
            const neighbor_iterator& it
        ) const;

        double factor_value (const node_iterator& it1,     const node_iterator& it2,     unsigned long s1, unsigned long s2) const;
        double factor_value (const neighbor_iterator& it1, const node_iterator& it2,     unsigned long s1, unsigned long s2) const;
        double factor_value (const node_iterator& it1,     const neighbor_iterator& it2, unsigned long s1, unsigned long s2) const;
        double factor_value (const neighbor_iterator& it1, const neighbor_iterator& it2, unsigned long s1, unsigned long s2) const;

    };

// ----------------------------------------------------------------------------------------

    template <
        typename map_problem
        >
    void find_map_nmplp (
        const map_problem& prob,
        std::vector<unsigned long>& map_assignment,
        unsigned long max_iter,
        double eps
    );
    /*!
        requires
            - for all valid i: prob.num_states(i) >= 2
            - map_problem == an object with an interface compatible with the map_problem
              object defined at the top of this file.
        ensures
            - #map_assignment.size() == prob.number_of_nodes()
            - for all valid i:
                - #map_assignment[prob.node_id(i)] < prob.num_states(i)
                - #map_assignment[prob.node_id(i)] == approximate MAP assignment for node i.


            - This function is an implementation of the NMPLP algorithm introduced in the 
              following paper:
                Fixing Max-Product: Convergent Message Passing Algorithms for MAP LP-Relaxations 
                by Amir Globerson Tommi Jaakkola
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FIND_MAP_nMPLP_H__


