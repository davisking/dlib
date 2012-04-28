// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FIND_MAX_FACTOR_GRAPH_PoTTS_H__
#define DLIB_FIND_MAX_FACTOR_GRAPH_PoTTS_H__

#include "find_max_factor_graph_potts_abstract.h"
#include "../matrix.h"
#include "min_cut.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {

        template <
            typename potts_problem,
            typename T = void
            >
        class flows_container
        {
            /*
                This object notionally represents a matrix of flow values.  It's
                overloaded to represent this matrix efficiently though.  In this case
                it represents the matrix using a sparse representation.
            */

            typedef typename potts_problem::value_type edge_type;
            std::vector<std::vector<edge_type> > flows;
        public:

            void setup(
                const potts_problem& p
            )
            {
                flows.resize(p.number_of_nodes());
                for (unsigned long i = 0; i < flows.size(); ++i)
                {
                    flows[i].resize(p.number_of_neighbors(i));
                }
            }

            edge_type& operator() (
                const long r,
                const long c
            ) { return flows[r][c]; }

            const edge_type& operator() (
                const long r,
                const long c
            ) const { return flows[r][c]; }
        };

// ----------------------------------------------------------------------------------------

        template <
            typename potts_problem
            >
        class flows_container<potts_problem, 
                              typename enable_if_c<potts_problem::max_number_of_neighbors!=0>::type>
        {
            /*
                This object notionally represents a matrix of flow values.  It's
                overloaded to represent this matrix efficiently though.  In this case
                it represents the matrix using a dense representation.

            */
            typedef typename potts_problem::value_type edge_type;
            const static unsigned long max_number_of_neighbors = potts_problem::max_number_of_neighbors;
            matrix<edge_type,0,max_number_of_neighbors> flows;
        public:

            void setup(
                const potts_problem& p
            )
            {
                flows.set_size(p.number_of_nodes(), max_number_of_neighbors);
            }

            edge_type& operator() (
                const long r,
                const long c
            ) { return flows(r,c); }

            const edge_type& operator() (
                const long r,
                const long c
            ) const { return flows(r,c); }
        };

// ----------------------------------------------------------------------------------------

        template <
            typename potts_problem 
            >
        class potts_flow_graph 
        {
        public:
            typedef typename potts_problem::value_type edge_type;
        private:
            /*!
                This is a utility class used by dlib::min_cut to convert a potts_problem 
                into the kind of flow graph expected by the min_cut object's main block
                of code.

                Within this object, we will use the convention that one past 
                potts_problem::number_of_nodes() is the source node and two past is 
                the sink node.
            !*/

            potts_problem& g;

            // flows(i,j) == the flow from node id i to it's jth neighbor
            flows_container<potts_problem> flows;
            // source_flows(i,0) == flow from source to node i, 
            // source_flows(i,1) == flow from node i to source
            matrix<edge_type,0,2> source_flows;

            // sink_flows(i,0) == flow from sink to node i, 
            // sink_flows(i,1) == flow from node i to sink
            matrix<edge_type,0,2> sink_flows;

            node_label source_label, sink_label;
        public:

            potts_flow_graph(
                potts_problem& g_
            ) : g(g_)
            {
                flows.setup(g);

                source_flows.set_size(g.number_of_nodes(), 2);
                sink_flows.set_size(g.number_of_nodes(), 2);
                source_flows = 0;
                sink_flows = 0;

                source_label = FREE_NODE;
                sink_label = FREE_NODE;

                // setup flows based on factor potentials
                edge_type min_value = std::numeric_limits<edge_type>::max();
                for (unsigned long i = 0; i < g.number_of_nodes(); ++i)
                {
                    source_flows(i,0) = -g.factor_value(i, true);
                    sink_flows(i,1) = -g.factor_value(i, false);

                    source_flows(i,1) = 0;
                    sink_flows(i,0) = 0;

                    if (source_flows(i,0) < min_value)
                        min_value = source_flows(i,0);
                    if (sink_flows(i,1) < min_value)
                        min_value = sink_flows(i,1);

                    for (unsigned long j = 0; j < g.number_of_neighbors(i); ++j)
                    {
                        flows(i,j) = g.factor_value_disagreement(i, g.get_neighbor(i,j));
                    }
                }

                // The flows can't be negative, so if they are then just adjust them
                // so they are positive.  Note that doing this doesn't change the optimal
                // labeling assignment.
                if (min_value < 0)
                {
                    source_flows -= min_value;
                    sink_flows -= min_value;
                }
            }

            class out_edge_iterator
            {
                friend class potts_flow_graph;
                unsigned long idx; // base node idx
                unsigned long cnt; // count over the neighbors of idx
            public:

                out_edge_iterator(
                ):idx(0),cnt(0){}

                out_edge_iterator(
                    unsigned long idx_,
                    unsigned long cnt_
                ):idx(idx_),cnt(cnt_)
                {}

                bool operator!= (
                    const out_edge_iterator& item
                ) const { return cnt != item.cnt; }

                out_edge_iterator& operator++(
                )
                {
                    ++cnt;
                    return *this;
                }
            };

            class in_edge_iterator
            {
                friend class potts_flow_graph;
                unsigned long idx; // base node idx
                unsigned long cnt; // count over the neighbors of idx
            public:

                in_edge_iterator(
                ):idx(0),cnt(0)  
                {}


                in_edge_iterator(
                    unsigned long idx_,
                    unsigned long cnt_
                ):idx(idx_),cnt(cnt_)
                {}

                bool operator!= (
                    const in_edge_iterator& item
                ) const { return cnt != item.cnt; }

                in_edge_iterator& operator++(
                )
                {
                    ++cnt;
                    return *this;
                }
            };

            unsigned long number_of_nodes (
            ) const { return g.number_of_nodes() + 2; }

            out_edge_iterator out_begin(
                const unsigned long& it
            ) const { return out_edge_iterator(it, 0); }

            in_edge_iterator in_begin(
                const unsigned long& it
            ) const { return in_edge_iterator(it, 0); }

            out_edge_iterator out_end(
                const unsigned long& it
            ) const 
            { 
                if (it >= g.number_of_nodes())
                    return out_edge_iterator(it, g.number_of_nodes()); 
                else
                    return out_edge_iterator(it, g.number_of_neighbors(it)+2); 
            }

            in_edge_iterator in_end(
                const unsigned long& it
            ) const 
            { 
                if (it >= g.number_of_nodes())
                    return in_edge_iterator(it, g.number_of_nodes()); 
                else
                    return in_edge_iterator(it, g.number_of_neighbors(it)+2); 
            }


            template <typename iterator_type>
            unsigned long node_id (
                const iterator_type& it
            ) const 
            { 
                // if this isn't an iterator over the source or sink nodes
                if (it.idx < g.number_of_nodes())
                {
                    const unsigned long num = g.number_of_neighbors(it.idx);
                    if (it.cnt < num)
                        return g.get_neighbor(it.idx, it.cnt); 
                    else if (it.cnt == num)
                        return g.number_of_nodes();
                    else
                        return g.number_of_nodes()+1;
                }
                else
                {
                    return it.cnt;
                }
            }


            edge_type get_flow (
                const unsigned long& it1,     
                const unsigned long& it2
            ) const
            {
                if (it1 >= g.number_of_nodes())
                {
                    // if it1 is the source
                    if (it1 == g.number_of_nodes())
                    {
                        return source_flows(it2,0);
                    }
                    else // if it1 is the sink
                    {
                        return sink_flows(it2,0);
                    }
                }
                else if (it2 >= g.number_of_nodes())
                {
                    // if it2 is the source
                    if (it2 == g.number_of_nodes())
                    {
                        return source_flows(it1,1);
                    }
                    else // if it2 is the sink
                    {
                        return sink_flows(it1,1);
                    }
                }
                else
                {
                    return flows(it1, g.get_neighbor_idx(it1, it2));
                }

            }

            edge_type get_flow (
                const out_edge_iterator& it
            ) const
            {
                if (it.idx < g.number_of_nodes())
                {
                    const unsigned long num = g.number_of_neighbors(it.idx);
                    if (it.cnt < num)
                        return flows(it.idx, it.cnt);
                    else if (it.cnt == num)
                        return source_flows(it.idx,1);
                    else
                        return sink_flows(it.idx,1);
                }
                else
                {
                    // if it.idx is the source
                    if (it.idx == g.number_of_nodes())
                    {
                        return source_flows(it.cnt,0);
                    }
                    else // if it.idx is the sink
                    {
                        return sink_flows(it.cnt,0);
                    }
                }
            }

            edge_type get_flow (
                const in_edge_iterator& it
            ) const
            {
                return get_flow(node_id(it), it.idx); 
            }

            void adjust_flow (
                const unsigned long& it1,     
                const unsigned long& it2,     
                const edge_type& value
            )
            {
                if (it1 >= g.number_of_nodes())
                {
                    // if it1 is the source
                    if (it1 == g.number_of_nodes())
                    {
                        source_flows(it2,0) += value;
                        source_flows(it2,1) -= value;
                    }
                    else // if it1 is the sink
                    {
                        sink_flows(it2,0) += value;
                        sink_flows(it2,1) -= value;
                    }
                }
                else if (it2 >= g.number_of_nodes())
                {
                    // if it2 is the source
                    if (it2 == g.number_of_nodes())
                    {
                        source_flows(it1,1) += value;
                        source_flows(it1,0) -= value;
                    }
                    else // if it2 is the sink
                    {
                        sink_flows(it1,1) += value;
                        sink_flows(it1,0) -= value;
                    }
                }
                else
                {
                    flows(it1, g.get_neighbor_idx(it1, it2)) += value;
                    flows(it2, g.get_neighbor_idx(it2, it1)) -= value;
                }

            }

            void set_label (
                const unsigned long& it,
                node_label value
            )
            {
                if (it < g.number_of_nodes())
                    g.set_label(it, value);
                else if (it == g.number_of_nodes())
                    source_label = value;
                else 
                    sink_label = value;
            }

            node_label get_label (
                const unsigned long& it
            ) const
            {
                if (it < g.number_of_nodes())
                    return g.get_label(it);
                if (it == g.number_of_nodes())
                    return source_label;
                else
                    return sink_label;
            }

        };
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename potts_model
        >
    typename potts_model::value_type potts_model_score (
        const potts_model& g
    )
    {
        typename potts_model::value_type score = 0;
        for (unsigned long i = 0; i < g.number_of_nodes(); ++i)
        {
            const bool label = (g.get_label(i)!=0);
            score += g.factor_value(i, label);
        }

        for (unsigned long i = 0; i < g.number_of_nodes(); ++i)
        {
            for (unsigned long n = 0; n < g.number_of_neighbors(i); ++n)
            {
                const unsigned long idx2 = g.get_neighbor(i,n);
                if (g.get_label(i) != g.get_label(idx2) && i < idx2)
                    score -= g.factor_value_disagreement(i, idx2);
            }
        }

        return score;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename potts_model
        >
    void find_max_factor_graph_potts (
        potts_model& m
    )
    {
        min_cut mc;
        dlib::impl::potts_flow_graph<potts_model> pfg(m);
        mc(pfg, m.number_of_nodes(), m.number_of_nodes()+1);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FIND_MAX_FACTOR_GRAPH_PoTTS_H__

