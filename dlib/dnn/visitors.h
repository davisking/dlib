// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_VISITORS_H_
#define DLIB_DNn_VISITORS_H_

#include "input.h"
#include "layers.h"
#include "loss.h"

namespace dlib
{
    namespace impl
    {
        class visitor_net_to_dot
        {
        public:

            visitor_net_to_dot(std::ostream& out) : out(out) {}

// ----------------------------------------------------------------------------------------

            template <typename input_layer_type>
            void operator()(size_t i, input_layer_type& l)
            {
                start_node(i, "input");
                end_node();
                from = i;
            }

// ----------------------------------------------------------------------------------------

            template <typename T, typename U> 
            void operator()(size_t i, const add_loss_layer<T, U>&)
            {
                start_node(i, "loss");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class... TAGS, typename U>
            void operator()(size_t i, const add_loss_layer<loss_yolo_<TAGS...>, U>&)
            {
                start_node(i, "loss_yolo");
                end_node();
                std::ostringstream sout;
                concat_helper_impl<TAGS...>::list_tags(sout);
                const auto tags = dlib::split(sout.str(), ",");
                for (const auto& tag : tags)
                    out << tag_to_layer.at(std::stoul(tag)) << " -> " << i << '\n';
            }

// ----------------------------------------------------------------------------------------

            template <unsigned long ID, typename U, typename E>
            void operator()(size_t i, const add_tag_layer<ID, U, E>&)
            {
                // check for consecutive tag layers
                tagged_layers.push_back(i);
                std::sort(tagged_layers.begin(), tagged_layers.end());
                std::vector<unsigned long> diffs;
                std::adjacent_difference(tagged_layers.begin(), tagged_layers.end(), std::back_inserter(diffs));
                from = i + 1;
                if (diffs.size() > 1 && diffs[1] == 1)
                {
                    for (size_t id = 1; id < diffs.size(); ++id)
                    {
                        if (diffs[id] == 1)
                            ++from;
                        else
                            break;
                    }
                }
                tag_to_layer[ID] = from;

                // In case we wanted to draw the tagged layers, instead:
                // tag_to_layer[ID] = i;
                // start_node(i, "tag", "Mrecord");
                // out << " | {id|{" << ID << "}}";
                // end_node();
                // update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class TAG, typename U>
            void operator()(size_t i, const add_skip_layer<TAG, U>&)
            {
                const auto t = tag_id<TAG>::id;
                from = tag_to_layer.at(t);
            }

// ----------------------------------------------------------------------------------------

            template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename U, typename E>
            void operator()(size_t i, const add_layer<con_<nf, nr, nc, sy, sx, py, px>, U, E>& l)
            {
                start_node(i, "con");
                out << " | {filters|{" << l.layer_details().num_filters() << "}}";
                out << " | {size|{" << nr << "," << nc << "}}";
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                if (py != 0 || px != 0)
                    out << " | {pad|{" << py<< "," << px << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename U, typename E>
            void operator()(size_t i, const add_layer<cont_<nf, nr, nc, sy, sx, py, px>, U, E>& l)
            {
                start_node(i, "cont");
                out << " | {filters|{" << l.layer_details().num_filters() << "}}";
                out << " | {size|{" << nr << "," << nc << "}}";
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                if (py != 0 || px != 0)
                    out << " | {pad|{" << py<< "," << px << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <int sy, int sx, typename U, typename E>
            void operator()(size_t i, const add_layer<upsample_<sy, sx>, U, E>&)
            {
                start_node(i, "upsample");
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <int NR, int NC, typename U, typename E>
            void operator()(size_t i, const add_layer<resize_to_<NR, NC>, U, E>&)
            {
                start_node(i, "resize_to");
                out << " | {size|{" << NR << "," << NC << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <long nr, long nc, int sy, int sx, int py, int px, typename U, typename E>
            void operator()(size_t i, const add_layer<max_pool_<nr, nc, sy, sx, py, px>, U, E>&)
            {
                start_node(i, "max_pool");
                out << " | {size|{" << nr << "," << nc << "}}";
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                if (py != 0 || px != 0)
                    out << " | {pad|{" << py<< "," << px << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <long nr, long nc, int sy, int sx, int py, int px, typename U, typename E>
            void operator()(size_t i, const add_layer<avg_pool_<nr, nc, sy, sx, py, px>, U, E>&)
            {
                start_node(i, "avg_pool");
                out << " | {size|{" << nr << "," << nc << "}}";
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                if (py != 0 || px != 0)
                    out << " | {pad|{" << py<< "," << px << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<layer_norm_, U, E>&)
            {
                start_node(i, "layer_norm");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <layer_mode MODE, typename U, typename E>
            void operator()(size_t i, const add_layer<bn_<MODE>, U, E>&)
            {
                start_node(i, "batch_norm");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <unsigned long no, fc_bias_mode bm, typename U, typename E>
            void operator()(size_t i, const add_layer<fc_<no, bm>, U, E>& l)
            {
                start_node(i, "fc");
                out << " | { outputs |{" << l.layer_details().get_num_outputs() << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<dropout_, U, E>&)
            {
                start_node(i, "dropout");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<multiply_, U, E>&)
            {
                start_node(i, "multiply");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<affine_, U, E>&)
            {
                start_node(i, "affine");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<add_prev_<TAG>, U, E>&)
            {
                start_node(i, "add");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<mult_prev_<TAG>, U, E>&)
            {
                start_node(i, "mult");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<resize_prev_to_tagged_<TAG>, U, E>&)
            {
                start_node(i, "resize_as");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << i << " -> " << tag_to_layer.at(t) << "[style=dashed]\n";
                update(i);
                from = i;
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<scale_<TAG>, U, E>&)
            {
                start_node(i, "scale");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<scale_prev_<TAG>, U, E>&)
            {
                start_node(i, "scale");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<relu_, U, E>&)
            {
                start_node(i, "relu");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<prelu_, U, E>&)
            {
                start_node(i, "prelu");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<leaky_relu_, U, E>&)
            {
                start_node(i, "leaky_relu");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<sig_, U, E>&)
            {
                start_node(i, "sigmoid");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<mish_, U, E>&)
            {
                start_node(i, "mish");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<htan_, U, E>&)
            {
                start_node(i, "htan");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<clipped_relu_, U, E>&)
            {
                start_node(i, "clipped_relu");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<elu_, U, E>&)
            {
                start_node(i, "elu");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<gelu_, U, E>&)
            {
                start_node(i, "gelu");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<softmax_, U, E>&)
            {
                start_node(i, "softmax");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<softmax_all_, U, E>&)
            {
                start_node(i, "softmax_all");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <template <typename> class... TAGS, typename U, typename E>
            void operator()(size_t i, const add_layer<concat_<TAGS...>, U, E>& l)
            {
                start_node(i, "concat");
                end_node();
                std::ostringstream sout;
                concat_helper_impl<TAGS...>::list_tags(sout);
                const auto tags = dlib::split(sout.str(), ",");
                for (const auto& tag : tags)
                    out << tag_to_layer.at(std::stoul(tag)) << " -> " << i << '\n';
                from = i;
            }

// ----------------------------------------------------------------------------------------

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<l2normalize_, U, E>&)
            {
                start_node(i, "l2normalize");
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <long offset, long k, int nr, int nc, typename U, typename E>
            void operator()(size_t i, const add_layer<extract_<offset, k, nr, nc>, U, E>&)
            {
                start_node(i, "extract");
                out << " | {offset|{" << offset << "}}";
                out << " | {k|{" << k << "}}";
                out << " | {nr|{" << nr << "}}";
                out << " | {nc|{" << nc << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <long long sy, long long sx, typename U, typename E>
            void operator()(size_t i, const add_layer<reorg_<sy, sx>, U, E>&)
            {
                start_node(i, "reorg");
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                end_node();
                update(i);
            }

// ----------------------------------------------------------------------------------------

            template <typename T, typename U, typename E>
            void operator()(size_t i, const add_layer<T, U, E>& l)
            {
                start_node(i, "unhandled layer");
                update(i);
            }

// ----------------------------------------------------------------------------------------

        private:
            size_t from;
            std::ostream& out;
            std::unordered_map<size_t, size_t> tag_to_layer;
            std::vector<size_t> tagged_layers;
            void update(const size_t i)
            {
                out << from << " -> " << i << '\n';
                from = i;
            }
            void start_node(const size_t i, const std::string& name, const std::string& shape = "record")
            {
                out << i << " [shape=" << shape << ", label=\"{layer|{" << i << "}} | " << name;
            }
            void end_node()
            {
                out << "\"]\n";
            }

        };
    }
    template <typename net_type>
    void net_to_dot (
        const net_type& net,
        std::ostream& out
    )
    {
        out << "digraph G {\n";
        out << "rankdir = BT\n";
        visit_layers_backwards(net, impl::visitor_net_to_dot(out));
        out << "}";
    }

    template <typename net_type>
    void net_to_dot (
        const net_type& net,
        const std::string& filename
    )
    {
        std::ofstream fout(filename);
        net_to_dot(net, fout);
    }
}

#endif // DLIB_DNn_VISITORS_H_

