// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_VISITORS_H_
#define DLIB_DNn_VISITORS_H_

#include "visitors_abstract.h"
#include "input.h"
#include "layers.h"
#include "loss.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {

        class visitor_net_map_input_to_output
        {
        public:

            visitor_net_map_input_to_output(dpoint& p_) : p(p_) {}

            dpoint& p;

            template<typename input_layer_type>
            void operator()(const input_layer_type& )
            {
            }

            template <typename T, typename U>
            void operator()(const add_loss_layer<T,U>& net)
            {
                (*this)(net.subnet());
            }

            template <typename T, typename U, typename E>
            void operator()(const add_layer<T,U,E>& net)
            {
                (*this)(net.subnet());
                p = net.layer_details().map_input_to_output(p);
            }
            template <bool B, typename T, typename U, typename E>
            void operator()(const dimpl::subnet_wrapper<add_layer<T,U,E>,B>& net)
            {
                (*this)(net.subnet());
                p = net.layer_details().map_input_to_output(p);
            }
            template <size_t N, template <typename> class R, typename U>
            void operator()(const repeat<N, R, U>& net)
            {
                (*this)(net.subnet());
                for (size_t i = 0; i < N; ++i)
                {
                    (*this)(net.get_repeated_layer(N-1-i).subnet());
                }
            }


            template <unsigned long ID, typename U, typename E>
            void operator()(const add_tag_layer<ID,U,E>& net)
            {
                // tag layers are an identity transform, so do nothing
                (*this)(net.subnet());
            }
            template <bool is_first, unsigned long ID, typename U, typename E>
            void operator()(const dimpl::subnet_wrapper<add_tag_layer<ID,U,E>,is_first>& net)
            {
                // tag layers are an identity transform, so do nothing
                (*this)(net.subnet());
            }


            template <template<typename> class TAG_TYPE, typename U>
            void operator()(const add_skip_layer<TAG_TYPE,U>& net)
            {
                (*this)(layer<TAG_TYPE>(net));
            }
            template <bool is_first, template<typename> class TAG_TYPE, typename SUBNET>
            void operator()(const dimpl::subnet_wrapper<add_skip_layer<TAG_TYPE,SUBNET>,is_first>& net)
            {
                // skip layers are an identity transform, so do nothing
                (*this)(layer<TAG_TYPE>(net));
            }

        };

        class visitor_net_map_output_to_input
        {
        public:
            visitor_net_map_output_to_input(dpoint& p_) : p(p_) {}

            dpoint& p;

            template<typename input_layer_type>
            void operator()(const input_layer_type& )
            {
            }

            template <typename T, typename U>
            void operator()(const add_loss_layer<T,U>& net)
            {
                (*this)(net.subnet());
            }

            template <typename T, typename U, typename E>
            void operator()(const add_layer<T,U,E>& net)
            {
                p = net.layer_details().map_output_to_input(p);
                (*this)(net.subnet());
            }
            template <bool B, typename T, typename U, typename E>
            void operator()(const dimpl::subnet_wrapper<add_layer<T,U,E>,B>& net)
            {
                p = net.layer_details().map_output_to_input(p);
                (*this)(net.subnet());
            }
            template <size_t N, template <typename> class R, typename U>
            void operator()(const repeat<N, R, U>& net)
            {
                for (size_t i = 0; i < N; ++i)
                {
                    (*this)(net.get_repeated_layer(i).subnet());
                }
                (*this)(net.subnet());
            }


            template <unsigned long ID, typename U, typename E>
            void operator()(const add_tag_layer<ID,U,E>& net)
            {
                // tag layers are an identity transform, so do nothing
                (*this)(net.subnet());
            }
            template <bool is_first, unsigned long ID, typename U, typename E>
            void operator()(const dimpl::subnet_wrapper<add_tag_layer<ID,U,E>,is_first>& net)
            {
                // tag layers are an identity transform, so do nothing
                (*this)(net.subnet());
            }


            template <template<typename> class TAG_TYPE, typename U>
            void operator()(const add_skip_layer<TAG_TYPE,U>& net)
            {
                (*this)(layer<TAG_TYPE>(net));
            }
            template <bool is_first, template<typename> class TAG_TYPE, typename SUBNET>
            void operator()(const dimpl::subnet_wrapper<add_skip_layer<TAG_TYPE,SUBNET>,is_first>& net)
            {
                // skip layers are an identity transform, so do nothing
                (*this)(layer<TAG_TYPE>(net));
            }

        };
    }

    template <typename net_type>
    inline dpoint input_tensor_to_output_tensor(
        const net_type& net,
        dpoint p
    )
    {
        impl::visitor_net_map_input_to_output temp(p);
        temp(net);
        return p;
    }

    template <typename net_type>
    inline dpoint output_tensor_to_input_tensor(
        const net_type& net,
        dpoint p
    )
    {
        impl::visitor_net_map_output_to_input temp(p);
        temp(net);
        return p;
    }

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    size_t count_parameters(
        const net_type& net
    )
    {
        size_t num_parameters = 0;
        visit_layer_parameters(net, [&](const tensor& t) { num_parameters += t.size(); });
        return num_parameters;
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_learning_rate_multiplier
        {
        public:
            visitor_learning_rate_multiplier(double new_learning_rate_multiplier_) :
                new_learning_rate_multiplier(new_learning_rate_multiplier_) {}

            template <typename layer>
            void operator()(layer& l) const
            {
                set_learning_rate_multiplier(l, new_learning_rate_multiplier);
            }

        private:

            double new_learning_rate_multiplier;
        };
    }

    template <typename net_type>
    void set_all_learning_rate_multipliers(
        net_type& net,
        double learning_rate_multiplier
    )
    {
        DLIB_CASSERT(learning_rate_multiplier >= 0);
        impl::visitor_learning_rate_multiplier temp(learning_rate_multiplier);
        visit_computational_layers(net, temp);
    }

    template <size_t begin, size_t end, typename net_type>
    void set_learning_rate_multipliers_range(
        net_type& net,
        double learning_rate_multiplier
    )
    {
        static_assert(begin <= end, "Invalid range");
        static_assert(end <= net_type::num_layers, "Invalid range");
        DLIB_CASSERT(learning_rate_multiplier >= 0);
        impl::visitor_learning_rate_multiplier temp(learning_rate_multiplier);
        visit_computational_layers_range<begin, end>(net, temp);
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_bn_running_stats_window_size
        {
        public:

            visitor_bn_running_stats_window_size(unsigned long new_window_size_) : new_window_size(new_window_size_) {}

            template <typename T>
            void set_window_size(T&) const
            {
                // ignore other layer detail types
            }

            template < layer_mode mode >
            void set_window_size(bn_<mode>& l) const
            {
                l.set_running_stats_window_size(new_window_size);
            }

            template<typename input_layer_type>
            void operator()(size_t , input_layer_type& )  const
            {
                // ignore other layers
            }

            template <typename T, typename U, typename E>
            void operator()(size_t , add_layer<T,U,E>& l)  const
            {
                set_window_size(l.layer_details());
            }

        private:

            unsigned long new_window_size;
        };
    }

    template <typename net_type>
    void set_all_bn_running_stats_window_sizes (
        net_type& net,
        unsigned long new_window_size
    )
    {
        visit_layers(net, impl::visitor_bn_running_stats_window_size(new_window_size));
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_disable_input_bias
        {
        public:

            template <typename T>
            void disable_input_bias(T&) const
            {
                // ignore other layer types
            }

            // handle the standard case
            template <typename U, typename E>
            void disable_input_bias(add_layer<layer_norm_, U, E>& l)
            {
                disable_bias(l.subnet().layer_details());
                set_bias_learning_rate_multiplier(l.subnet().layer_details(), 0);
                set_bias_weight_decay_multiplier(l.subnet().layer_details(), 0);
            }

            template <layer_mode mode, typename U, typename E>
            void disable_input_bias(add_layer<bn_<mode>, U, E>& l)
            {
                disable_bias(l.subnet().layer_details());
                set_bias_learning_rate_multiplier(l.subnet().layer_details(), 0);
                set_bias_weight_decay_multiplier(l.subnet().layer_details(), 0);
            }

            // handle input repeat layer case
            template <layer_mode mode, size_t N, template <typename> class R, typename U, typename E>
            void disable_input_bias(add_layer<bn_<mode>, repeat<N, R, U>, E>& l)
            {
                disable_bias(l.subnet().get_repeated_layer(0).layer_details());
                set_bias_learning_rate_multiplier(l.subnet().get_repeated_layer(0).layer_details(), 0);
                set_bias_weight_decay_multiplier(l.subnet().get_repeated_layer(0).layer_details(), 0);
            }

            template <size_t N, template <typename> class R, typename U, typename E>
            void disable_input_bias(add_layer<layer_norm_, repeat<N, R, U>, E>& l)
            {
                disable_bias(l.subnet().get_repeated_layer(0).layer_details());
                set_bias_learning_rate_multiplier(l.subnet().get_repeated_layer(0).layer_details(), 0);
                set_bias_weight_decay_multiplier(l.subnet().get_repeated_layer(0).layer_details(), 0);
            }

            // handle input repeat layer with tag case
            template <layer_mode mode, unsigned long ID, typename E>
            void disable_input_bias(add_layer<bn_<mode>, add_tag_layer<ID, impl::repeat_input_layer>, E>& )
            {
            }

            template <unsigned long ID, typename E>
            void disable_input_bias(add_layer<layer_norm_, add_tag_layer<ID, impl::repeat_input_layer>, E>& )
            {
            }

            // handle tag layer case
            template <layer_mode mode, unsigned long ID, typename U, typename E>
            void disable_input_bias(add_layer<bn_<mode>, add_tag_layer<ID, U>, E>& )
            {
            }

            template <unsigned long ID, typename U, typename E>
            void disable_input_bias(add_layer<layer_norm_, add_tag_layer<ID, U>, E>& )
            {
            }

            // handle skip layer case
            template <layer_mode mode, template <typename> class TAG, typename U, typename E>
            void disable_input_bias(add_layer<bn_<mode>, add_skip_layer<TAG, U>, E>& )
            {
            }

            template <template <typename> class TAG, typename U, typename E>
            void disable_input_bias(add_layer<layer_norm_, add_skip_layer<TAG, U>, E>& )
            {
            }

            template<typename input_layer_type>
            void operator()(size_t , input_layer_type& ) const
            {
                // ignore other layers
            }

            template <typename T, typename U, typename E>
            void operator()(size_t , add_layer<T,U,E>& l)
            {
                disable_input_bias(l);
            }
        };
    }


    template <typename net_type>
    void disable_duplicative_biases (
        net_type& net
    )
    {
        visit_layers(net, impl::visitor_disable_input_bias());
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_fuse_layers
        {
            public:
            template <typename T>
            void fuse_convolution(T&) const
            {
                // disable other layer types
            }

            // handle the case of convolutional layer followed by relu
            template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename U, typename R>
            void fuse_convolution(add_layer<relu_, add_layer<con_<nf, nr, nc, sy, sx, py, px>, U>, R>& l)
            {
                if (l.layer_details().is_disabled())
                    return;

                // get the convolution below the relu layer
                auto& conv = l.subnet().layer_details();

                conv.enable_relu();

                // disable the relu layer
                l.layer_details().disable();
            }

            // handle the case of convolutional layer followed by affine followed by relu
            template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename U, typename E, typename R>
            void fuse_convolution(add_layer<relu_, add_layer<affine_, add_layer<con_<nf, nr, nc, sy, sx, py, px>, U>, E>, R>& l)
            {
                if (l.layer_details().is_disabled())
                    return;

                // fuse the convolutional layer followed by affine
                fuse_convolution(l.subnet());

                // get the convolution below the affine layer
                auto& conv = l.subnet().subnet().layer_details();

                conv.enable_relu();

                // disable the relu layer
                l.layer_details().disable();
            }

            // handle the case of convolutional layer followed by affine
            template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename U, typename E>
            void fuse_convolution(add_layer<affine_, add_layer<con_<nf, nr, nc, sy, sx, py, px>, U>, E>& l)
            {
                if (l.layer_details().is_disabled())
                    return;

                // get the convolution below the affine layer
                auto& conv = l.subnet().layer_details();

                // get the parameters from the affine layer as alias_tensor_instance
                alias_tensor_instance gamma = l.layer_details().get_gamma();
                alias_tensor_instance beta = l.layer_details().get_beta();

                if (conv.bias_is_disabled())
                {
                    conv.enable_bias();
                }

                tensor& params = conv.get_layer_params();

                // update the biases
                auto biases = alias_tensor(1, conv.num_filters());
                biases(params, params.size() - conv.num_filters()) += mat(beta);

                // guess the number of input channels
                const long k_in = (params.size() - conv.num_filters()) / conv.num_filters() / conv.nr() / conv.nc();

                // rescale the filters
                DLIB_CASSERT(conv.num_filters() == gamma.k());
                alias_tensor filter(1, k_in, conv.nr(), conv.nc());
                const float* g = gamma.host();
                for (long n = 0; n < conv.num_filters(); ++n)
                {
                    filter(params, n * filter.size()) *= g[n];
                }

                // disable the affine layer
                l.layer_details().disable();
            }

            template <typename input_layer_type>
            void operator()(size_t , input_layer_type& ) const
            {
                // ignore other layers
            }

            template <typename T, typename U, typename E>
            void operator()(size_t , add_layer<T, U, E>& l)
            {
                fuse_convolution(l);
            }
        };
    }

    template <typename net_type>
    void fuse_layers (
        net_type& net
    )
    {
        DLIB_CASSERT(count_parameters(net) > 0, "The network has to be allocated before fusing the layers.");
        visit_layers(net, impl::visitor_fuse_layers());
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_net_to_xml
        {
        public:

            visitor_net_to_xml(std::ostream& out_) : out(out_) {}

            template<typename input_layer_type>
            void operator()(size_t idx, const input_layer_type& l)
            {
                out << "<layer idx='"<<idx<<"' type='input'>\n";
                to_xml(l,out);
                out << "</layer>\n";
            }

            template <typename T, typename U>
            void operator()(size_t idx, const add_loss_layer<T,U>& l)
            {
                out << "<layer idx='"<<idx<<"' type='loss'>\n";
                to_xml(l.loss_details(),out);
                out << "</layer>\n";
            }

            template <typename T, typename U, typename E>
            void operator()(size_t idx, const add_layer<T,U,E>& l)
            {
                out << "<layer idx='"<<idx<<"' type='comp'>\n";
                to_xml(l.layer_details(),out);
                out << "</layer>\n";
            }

            template <unsigned long ID, typename U, typename E>
            void operator()(size_t idx, const add_tag_layer<ID,U,E>& /*l*/)
            {
                out << "<layer idx='"<<idx<<"' type='tag' id='"<<ID<<"'/>\n";
            }

            template <template<typename> class T, typename U>
            void operator()(size_t idx, const add_skip_layer<T,U>& /*l*/)
            {
                out << "<layer idx='"<<idx<<"' type='skip' id='"<<(tag_id<T>::id)<<"'/>\n";
            }

        private:

            std::ostream& out;
        };
    }

    template <typename net_type>
    void net_to_xml (
        const net_type& net,
        std::ostream& out
    )
    {
        auto old_precision = out.precision(9);
        out << "<net>\n";
        visit_layers(net, impl::visitor_net_to_xml(out));
        out << "</net>\n";
        // restore the original stream precision.
        out.precision(old_precision);
    }

    template <typename net_type>
    void net_to_xml (
        const net_type& net,
        const std::string& filename
    )
    {
        std::ofstream fout(filename);
        net_to_xml(net, fout);
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_net_to_dot
        {
        public:

            visitor_net_to_dot(std::ostream& out) : out(out) {}

            template <typename input_layer_type>
            void operator()(size_t i, input_layer_type&)
            {
                start_node(i, "input");
                end_node();
                from = i;
            }

            template <typename T, typename U>
            void operator()(size_t i, const add_loss_layer<T, U>&)
            {
                start_node(i, "loss");
                end_node();
                update(i);
            }

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

            // Handle the special case when the tag layer is followed by a skip layer
            template <unsigned long ID, template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_tag_layer<ID, add_skip_layer<TAG, U>, E>&)
            {
                tagged_layers.push_back(i);
                const auto t = tag_id<TAG>::id;
                tag_to_layer.at(t) = from;
            }

            template <template <typename> class TAG, typename U>
            void operator()(size_t, const add_skip_layer<TAG, U>&)
            {
                const auto t = tag_id<TAG>::id;
                from = tag_to_layer.at(t);
            }

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

            template <int sy, int sx, typename U, typename E>
            void operator()(size_t i, const add_layer<upsample_<sy, sx>, U, E>&)
            {
                start_node(i, "upsample");
                if (sy != 1 || sx != 1)
                    out << " | {scale|{" << sy<< "," << sx << "}}";
                end_node();
                update(i);
            }

            template <int NR, int NC, typename U, typename E>
            void operator()(size_t i, const add_layer<resize_to_<NR, NC>, U, E>&)
            {
                start_node(i, "resize_to");
                out << " | {size|{" << NR << "," << NC << "}}";
                end_node();
                update(i);
            }

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

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<layer_norm_, U, E>&)
            {
                start_node(i, "layer_norm");
                end_node();
                update(i);
            }

            template <layer_mode MODE, typename U, typename E>
            void operator()(size_t i, const add_layer<bn_<MODE>, U, E>&)
            {
                start_node(i, "batch_norm");
                end_node();
                update(i);
            }

            template <unsigned long no, fc_bias_mode bm, typename U, typename E>
            void operator()(size_t i, const add_layer<fc_<no, bm>, U, E>& l)
            {
                start_node(i, "fc");
                out << " | { outputs |{" << l.layer_details().get_num_outputs() << "}}";
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<dropout_, U, E>&)
            {
                start_node(i, "dropout");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<multiply_, U, E>&)
            {
                start_node(i, "multiply");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<affine_, U, E>&)
            {
                start_node(i, "affine");
                end_node();
                update(i);
            }

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<add_prev_<TAG>, U, E>&)
            {
                start_node(i, "add");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<mult_prev_<TAG>, U, E>&)
            {
                start_node(i, "mult");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

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

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<scale_<TAG>, U, E>&)
            {
                start_node(i, "scale");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

            template <template <typename> class TAG, typename U, typename E>
            void operator()(size_t i, const add_layer<scale_prev_<TAG>, U, E>&)
            {
                start_node(i, "scale");
                end_node();
                const auto t = tag_id<TAG>::id;
                out << tag_to_layer.at(t) << " -> " << i << '\n';
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<relu_, U, E>&)
            {
                start_node(i, "relu");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<prelu_, U, E>&)
            {
                start_node(i, "prelu");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<leaky_relu_, U, E>& l)
            {
                start_node(i, "leaky_relu");
                out << " | { alpha |{" << l.layer_details().get_alpha() << "}}";
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<sig_, U, E>&)
            {
                start_node(i, "sigmoid");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<mish_, U, E>&)
            {
                start_node(i, "mish");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<htan_, U, E>&)
            {
                start_node(i, "htan");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<clipped_relu_, U, E>&)
            {
                start_node(i, "clipped_relu");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<elu_, U, E>&)
            {
                start_node(i, "elu");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<gelu_, U, E>&)
            {
                start_node(i, "gelu");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<smelu_, U, E>& l)
            {
                start_node(i, "smelu");
                out << " | { beta |{" << l.layer_details().get_beta() << "}}";
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<silu_, U, E>&)
            {
                start_node(i, "silu");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<softmax_, U, E>&)
            {
                start_node(i, "softmax");
                end_node();
                update(i);
            }

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<softmax_all_, U, E>&)
            {
                start_node(i, "softmax_all");
                end_node();
                update(i);
            }

            template <template <typename> class... TAGS, typename U, typename E>
            void operator()(size_t i, const add_layer<concat_<TAGS...>, U, E>&)
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

            template <typename U, typename E>
            void operator()(size_t i, const add_layer<l2normalize_, U, E>&)
            {
                start_node(i, "l2normalize");
                end_node();
                update(i);
            }

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

            template <long long sy, long long sx, typename U, typename E>
            void operator()(size_t i, const add_layer<reorg_<sy, sx>, U, E>&)
            {
                start_node(i, "reorg");
                if (sy != 1 || sx != 1)
                    out << " | {stride|{" << sy<< "," << sx << "}}";
                end_node();
                update(i);
            }

            template <typename T, typename U, typename E>
            void operator()(size_t i, const add_layer<T, U, E>&)
            {
                start_node(i, "unhandled layer");
                update(i);
            }

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

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_VISITORS_H_

