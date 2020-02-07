// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_UTILITIES_H_
#define DLIB_DNn_UTILITIES_H_

#include "core.h"
#include "utilities_abstract.h"
#include "../geometry.h"
#include <fstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline double log1pexp(double x)
    {
        using std::exp;
        using namespace std; // Do this instead of using std::log1p because some compilers
                             // error out otherwise (E.g. gcc 4.9 in cygwin)
        if (x <= -37)
            return exp(x);
        else if (-37 < x && x <= 18)
            return log1p(exp(x));
        else if (18 < x && x <= 33.3)
            return x + exp(-x);
        else
            return x;
    }
    
// ----------------------------------------------------------------------------------------

    inline void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    )
    {
        for (auto& val : params)
        {
            // Draw a random number to initialize the layer according to formula (16)
            // from Understanding the difficulty of training deep feedforward neural
            // networks by Xavier Glorot and Yoshua Bengio.
            val = 2*rnd.get_random_float()-1;
            val *= std::sqrt(6.0/(num_inputs_and_outputs));
        }
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
            void operator()(size_t idx, const add_tag_layer<ID,U,E>& l) 
            {
                out << "<layer idx='"<<idx<<"' type='tag' id='"<<ID<<"'/>\n";
            }

            template <template<typename> class T, typename U>
            void operator()(size_t idx, const add_skip_layer<T,U>& l) 
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

    namespace impl
    {
        class visitor_count_parameters
        {
        public:
            visitor_count_parameters(size_t& num_parameters_) : num_parameters(num_parameters_) {}

            void operator()(size_t, const tensor& t)
            {
                num_parameters += t.size();
            }

        private:
            size_t& num_parameters;
        };
    }

    template <typename net_type>
    inline size_t count_parameters(
        const net_type& net
    )
    {
        size_t num_parameters = 0;
        impl::visitor_count_parameters temp(num_parameters);
        visit_layer_parameters(net, temp);
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

            template <typename input_layer_type>
            void operator()(size_t , input_layer_type& ) const
            {
                // ignore other layers
            }

            template <typename T, typename U, typename E>
            void operator()(size_t , add_layer<T,U,E>& l) const
            {
                set_learning_rate_multiplier(l.layer_details(), new_learning_rate_multiplier);
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
        visit_layers(net, temp);
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
        visit_layers_range<begin, end>(net, temp);
    }

// ----------------------------------------------------------------------------------------
}

#endif // DLIB_DNn_UTILITIES_H_ 



