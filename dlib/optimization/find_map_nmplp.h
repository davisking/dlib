// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FIND_MAP_nMPLP_H__
#define DLIB_FIND_MAP_nMPLP_H__

#include "find_map_nmplp_abstract.h"
#include <vector>
#include <map>
#include "../matrix.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename map_problem
        >
    void find_map_nmplp (
        const map_problem& prob,
        std::vector<unsigned long>& map_assignment,
        unsigned long max_iter,
        double eps
    )
    {
        /*
            This function is an implementation of the NMPLP algorithm introduced in the 
            following paper:
                Fixing Max-Product: Convergent Message Passing Algorithms for MAP LP-Relaxations 
                by Amir Globerson Tommi Jaakkola

                In particular, see the pseudocode in Figure 1.  The code in this function
                follows what is described there.
        */

        typedef typename map_problem::node_iterator node_iterator;
        typedef typename map_problem::neighbor_iterator neighbor_iterator;

        map_assignment.resize(prob.number_of_nodes());


        if (prob.number_of_nodes() == 0)
            return;


        std::vector<double> gamma_elements;
        gamma_elements.reserve(prob.number_of_nodes()*prob.num_states(prob.begin())*3);

        std::map<std::pair<unsigned long, unsigned long>, unsigned long> gamma_idx;



        // initialize gamma according to the initialization instructions at top of Figure 1
        for (node_iterator i = prob.begin(); i != prob.end(); ++i)
        {
            const unsigned long id_i = prob.node_id(i);

            for (neighbor_iterator j = prob.begin(i); j != prob.end(i); ++j)
            {
                const unsigned long id_j = prob.node_id(j);

                gamma_idx[std::make_pair(id_i,id_j)] = gamma_elements.size();

                const unsigned long num_states_xj = prob.num_states(j);

                for (unsigned long xj = 0; xj < num_states_xj; ++xj)
                {
                    const unsigned long num_states_xi = prob.num_states(i);

                    double best_val = -std::numeric_limits<double>::infinity();
                    for (unsigned long xi = 0; xi < num_states_xi; ++xi)
                    {
                        double val = prob.factor_value(i,j,xi,xj); 

                        double sum_temp = 0;

                        for (neighbor_iterator k = prob.begin(i); k != prob.end(i); ++k)
                        {
                            if (j == k)
                                continue;

                            double max_val = -std::numeric_limits<double>::infinity();
                            for (unsigned long xk = 0; xk < prob.num_states(k); ++xk)
                            {
                                double temp = prob.factor_value(k,i,xk,xi);
                                if (temp > max_val)
                                    max_val = temp;
                            }

                            sum_temp += max_val;
                        }


                        val += 0.5*sum_temp;

                        if (val > best_val)
                            best_val = val;
                    }


                    gamma_elements.push_back(best_val);
                }
            }
        }




        double max_change = eps + 1; 
        // Now do the main body of the optimization. 
        for (unsigned long iter = 0; iter < max_iter && max_change > eps; ++iter)
        {
            max_change = -std::numeric_limits<double>::infinity();

            for (node_iterator i = prob.begin(); i != prob.end(); ++i)
            {
                const unsigned long id_i = prob.node_id(i);

                for (neighbor_iterator j = prob.begin(i); j != prob.end(i); ++j)
                {
                    const unsigned long id_j = prob.node_id(j);
                    double* const gamma_ji = &gamma_elements[gamma_idx[std::make_pair(id_j,id_i)]];
                    double* const gamma_ij = &gamma_elements[gamma_idx[std::make_pair(id_i,id_j)]];

                    const unsigned long num_states_xj = prob.num_states(j);

                    for (unsigned long xj = 0; xj < num_states_xj; ++xj)
                    {
                        const unsigned long num_states_xi = prob.num_states(i);

                        double best_val = -std::numeric_limits<double>::infinity();
                        for (unsigned long xi = 0; xi < num_states_xi; ++xi)
                        {
                            double val = prob.factor_value(i,j,xi,xj) - gamma_ji[xi];  

                            double sum_temp = 0;

                            int num_neighbors = 0;
                            for (neighbor_iterator k = prob.begin(i); k != prob.end(i); ++k)
                            {
                                const unsigned long id_k = prob.node_id(k);
                                ++num_neighbors;

                                const double* const gamma_ki = &gamma_elements[gamma_idx[std::make_pair(id_k,id_i)]];
                                sum_temp += gamma_ki[xi];
                            }


                            val += 2.0/(num_neighbors + 1.0)*sum_temp;

                            if (val > best_val)
                                best_val = val;
                        }


                        if (std::abs(gamma_ij[xj] - best_val) > max_change)
                            max_change = std::abs(gamma_ij[xj] - best_val);

                        gamma_ij[xj] = best_val;
                    }
                }
            }
        }


        // now decode the "beliefs"
        std::vector<double> b;
        for (node_iterator i = prob.begin(); i != prob.end(); ++i)
        {
            const unsigned long id_i = prob.node_id(i);
            b.assign(prob.num_states(i), 0);

            for (neighbor_iterator k = prob.begin(i); k != prob.end(i); ++k)
            {
                const unsigned long id_k = prob.node_id(k);

                for (unsigned long xi = 0; xi < b.size(); ++xi)
                {
                    const double* const gamma_ki = &gamma_elements[gamma_idx[std::make_pair(id_k,id_i)]];
                    b[xi] += gamma_ki[xi];
                }
            }

            map_assignment[id_i] = index_of_max(vector_to_matrix(b));
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FIND_MAP_nMPLP_H__

