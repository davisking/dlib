// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GRAPH_LaBELER_H__
#define DLIB_GRAPH_LaBELER_H__

#include "graph_labeler_abstract.h"
#include "../matrix.h"
#include "../string.h"
#include <vector>
#include "../graph_cuts.h"
#include "sparse_vector.h"
#include "../graph.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type 
        >
    class graph_labeler 
    {
    public:

        typedef std::vector<node_label> label_type;
        typedef label_type result_type;

        graph_labeler()
        {
        }

        graph_labeler(
            const vector_type& edge_weights_,
            const vector_type& node_weights_
        ) : 
            edge_weights(edge_weights_),
            node_weights(node_weights_)
        {
        }

        const vector_type& get_edge_weights (
        ) const { return edge_weights; }

        const vector_type& get_node_weights (
        ) const { return node_weights; }

        template <typename graph_type>
        void operator() (
            const graph_type& samp,
            result_type& labels 
        ) const
        {
            using dlib::sparse_vector::dot;
            using dlib::dot;

            labels.clear();

            graph<double,double>::kernel_1a g; 
            copy_graph_structure(samp, g);
            for (unsigned long i = 0; i < g.number_of_nodes(); ++i)
            {
                g.node(i).data = dot(node_weights, samp.node(i).data);

                for (unsigned long n = 0; n < g.node(i).number_of_neighbors(); ++n)
                {
                    const unsigned long j = g.node(i).neighbor(n).index();
                    // Don't compute an edge weight more than once. 
                    if (i < j)
                    {
                        g.node(i).edge(n) = dot(edge_weights, samp.node(i).edge(n));
                    }
                }

            }

            find_max_factor_graph_potts(g, labels);
        }

        template <typename graph_type>
        result_type operator() (
            const graph_type& sample 
        ) const
        {
            result_type temp;
            (*this)(sample, temp);
            return temp;
        }

    private:

        vector_type edge_weights;
        vector_type node_weights;
    };


// ----------------------------------------------------------------------------------------

    template <
        typename vector_type 
        >
    void serialize (
        const graph_labeler<vector_type>& item,
        std::ostream& out
    )
    {
        int version = 1;
        serialize(version, out);
        serialize(item.get_edge_weights(), out);
        serialize(item.get_node_weights(), out);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type 
        >
    void deserialize (
        graph_labeler<vector_type>& item,
        std::istream& in 
    )
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
        {
            throw dlib::serialization_error("While deserializing graph_labeler, found unexpected version number of " + 
                                            cast_to_string(version) + ".");
        }

        vector_type edge_weights, node_weights;
        deserialize(edge_weights, in);
        deserialize(node_weights, in);

        item = graph_labeler<vector_type>(edge_weights, node_weights);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GRAPH_LaBELER_H__


