// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_CORE_H_
#define DLIB_DNn_CORE_H_

#include "core_abstract.h"
#include "tensor.h"
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include "../statistics.h"
#include "../rand.h"
#include "../algs.h"
#include <utility>
#include <tuple>
#include <cmath>


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

    // Tell us if T is one of the special layer types (i.e. add_layer, add_tag_layer, or
    // add_skip_layer).
    template <typename T> struct is_nonloss_layer_type : std::false_type {};
    // Tell us if T is an instance of add_loss_layer.
    template <typename T> struct is_loss_layer_type : std::false_type {};

    namespace impl
    {
        template <size_t... n>
        struct ct_integers_list {
            template <size_t m>
            struct push_back
            {
                typedef ct_integers_list<n..., m> type;
            };
        };

        template <size_t max>
        struct ct_make_integer_range
        {
            // recursively call push_back on ct_integers_list to build a range from 0 to max
            // inclusive.
            typedef typename ct_make_integer_range<max-1>::type::template push_back<max>::type type;
        };

        template <>
        struct ct_make_integer_range<0>
        {
            typedef ct_integers_list<> type;
        };

        template <size_t... indices, typename Tuple>
        auto tuple_subset(
            const Tuple& item, 
            ct_integers_list<indices...>
        ) -> decltype(std::make_tuple(std::get<indices>(item)...))
        {
            return std::make_tuple(std::get<indices>(item)...);
        }

        template <typename T> struct alwaysbool { typedef bool type; };

        resizable_tensor& rt();

        // The significance of a layer's backward method requiring forward's outputs is
        // that such as layer can't have an in-place layer stacked on top of it because
        // in-place layers overwrite the output of the layer they sit on top of.
        template <typename layer_type, typename SUBNET>
        constexpr auto backward_requires_forward_output(
            layer_type& layer,
            SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.backward(rt(),rt(),sub,rt()))>::type
        {
            return true;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto backward_requires_forward_output(
            layer_type& layer,
            SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.backward(rt(),sub,rt()))>::type
        {
            return false;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto backward_requires_forward_output(
            layer_type& layer,
            SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.backward_inplace(rt(),rt(),sub.get_gradient_input(),rt()))>::type
        {
            return true;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto has_inplace_backward(
            layer_type& layer,
            SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.backward(rt(),rt(),sub,rt()))>::type
        {
            return false;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto has_inplace_backward(
            layer_type& layer,
            SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.backward(rt(),sub,rt()))>::type
        {
            return false;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto has_inplace_backward(
            layer_type& layer,
            SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.backward_inplace(rt(),rt(),sub.get_gradient_input(),rt()))>::type
        {
            return true;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto is_inplace_layer(
            layer_type& layer,
            const SUBNET& sub 
        ) -> typename alwaysbool<decltype(layer.forward(sub,rt()))>::type
        {
            return false;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto is_inplace_layer(
            layer_type& layer,
            const SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.forward_inplace(sub.get_output(),rt()))>::type
        {
            return true;
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_backward(
            layer_type& layer,
            const tensor& computed_output, 
            const tensor& gradient_input, 
            SUBNET& sub, 
            tensor& params_grad
        ) -> decltype(layer.backward(computed_output,gradient_input,sub,params_grad))
        {
            layer.backward(computed_output,gradient_input,sub,params_grad);
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_backward(
            layer_type& layer,
            const tensor& , 
            const tensor& gradient_input, 
            SUBNET& sub, 
            tensor& params_grad
        ) -> decltype(layer.backward(gradient_input,sub,params_grad))
        {
            layer.backward(gradient_input,sub,params_grad);
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_backward(
            layer_type& layer,
            const tensor& computed_output, 
            const tensor& gradient_input, 
            SUBNET& sub, 
            tensor& params_grad
        ) -> decltype(layer.backward_inplace(computed_output,gradient_input,sub.get_gradient_input(),params_grad))
        {
            layer.backward_inplace(computed_output,gradient_input,sub.get_gradient_input(),params_grad);
        }


        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            tensor& data_output
        ) -> decltype(layer.forward(sub,rt()))
        {
            // This overload of call_layer_forward() is here because this template
            // naturally gets instantiated but only on code paths that never get executed.
            // So rather than writing a bunch of hard to read template magic around call
            // sites we just have this overload that doesn't do anything (and an assert to
            // make sure that's the case).
            DLIB_CASSERT(false, "This should never happen");
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            resizable_tensor& data_output
        ) -> decltype(layer.forward(sub,data_output))
        {
            layer.forward(sub,data_output);
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            tensor& data_output
        ) -> decltype(layer.forward_inplace(sub.get_output(),data_output))
        {
            layer.forward_inplace(sub.get_output(),data_output);
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            resizable_tensor& data_output
        ) -> decltype(layer.forward_inplace(sub.get_output(),data_output))
        {
            if (!have_same_dimensions(data_output, sub.get_output()))
                data_output.copy_size(sub.get_output());
            layer.forward_inplace(sub.get_output(),data_output);
        }


    } // end namespace impl

    template <typename Head, typename... Tail>
    std::tuple<Tail...> tuple_tail(
        const std::tuple<Head, Tail...>& item
    )
    {
        return impl::tuple_subset(item, typename impl::ct_make_integer_range<sizeof...(Tail)>::type());
    }

// ----------------------------------------------------------------------------------------

    inline void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    )
    {
        float* data = params.host();
        for (size_t i = 0; i < params.size(); ++i)
        {
            // Draw a random number to initialize the layer according to formula (16)
            // from Understanding the difficulty of training deep feedforward neural
            // networks by Xavier Glorot and Yoshua Bengio.
            float val = 2*rnd.get_random_float()-1;
            val *= std::sqrt(6.0/(num_inputs_and_outputs));

            data[i] = val;
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename T, size_t N>
    class sstack
    {
    public:
        static_assert(N > 0, "You can't create an empty sstack.");
        typedef T value_type;
        const static size_t num_elements = N;

        sstack() {}
        sstack(const T& item_) : item(item_), data(item_) {}

        const T& top() const { return item; }
        T& top() { return item; }

        size_t size() const { return N; }

        const sstack<T,N-1>& pop() const { return data; }
        sstack<T,N-1>& pop() { return data; }

        friend void serialize(const sstack& item, std::ostream& out)
        {
            serialize(item.top(), out);
            serialize(item.pop(), out);
        }

        friend void deserialize(sstack& item, std::istream& in)
        {
            deserialize(item.top(), in);
            deserialize(item.pop(), in);
        }

    private:
        T item;
        sstack<T,N-1> data;
    };

    template <typename T>
    class sstack<T,1> // base case of recursive definition.
    {
    public:
        sstack() {}
        sstack(const T& item_) : item(item_) {}

        const T& top() const { return item; }
        T& top() { return item; }

        size_t size() const { return 1; }

        friend void serialize(const sstack& item, std::ostream& out)
        {
            serialize(item.top(), out);
        }

        friend void deserialize(sstack& item, std::istream& in)
        {
            deserialize(item.top(), in);
        }

    private:
        T item;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace dimpl
    {
        template <typename T, bool is_first = true, typename enabled=void>
        class subnet_wrapper
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a tool that makes an add_layer or add_loss_layer object
                    expose only the part of its interface defined by the SUBNET
                    type in layers_abstract.h.  This way, when we pass subnetwork
                    objects to the layer callbacks those callbacks won't be able to 
                    interact with the subnetworks in a way other than specified 
                    by the SUBNET interface spec.

                    We also allow the top layer of a subnet_wrapper stack to call the
                    private_get_output() and private_get_gradient_input() functions.  This
                    way, layers that have had their output/gradient overwritten by in-place
                    layers can only be accessed from the in-place layers that sit directly
                    on top of them since those in-place layers are the only layers that
                    know how to interact with them properly.
            !*/

        public:
            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            subnet_wrapper(T& l_) {}
            // Nothing here because in this case T is one of the input layer types 
            // that doesn't have anything in it.
        };

        template <typename T>
        class subnet_wrapper<T,true, typename std::enable_if<is_nonloss_layer_type<T>::value>::type>
        {

        public:
            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            typedef T wrapped_type;
            const static size_t num_layers = T::num_layers;

            subnet_wrapper(T& l_) : l(l_),subnetwork(l.subnet()) {}

            const tensor& get_output() const { return l.private_get_output(); }
            tensor& get_gradient_input() { return l.private_get_gradient_input(); }

            const subnet_wrapper<typename T::subnet_type,false>& subnet() const { return subnetwork; }
            subnet_wrapper<typename T::subnet_type,false>& subnet() { return subnetwork; }

        private:
            T& l;
            subnet_wrapper<typename T::subnet_type,false> subnetwork;
        };

        template <typename T>
        class subnet_wrapper<T,false, typename std::enable_if<is_nonloss_layer_type<T>::value>::type>
        {

        public:
            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            typedef T wrapped_type;
            const static size_t num_layers = T::num_layers;

            subnet_wrapper(T& l_) : l(l_),subnetwork(l.subnet()) {}

            const tensor& get_output() const { return l.get_output(); }
            tensor& get_gradient_input() { return l.get_gradient_input(); }

            const subnet_wrapper<typename T::subnet_type,false>& subnet() const { return subnetwork; }
            subnet_wrapper<typename T::subnet_type,false>& subnet() { return subnetwork; }

        private:
            T& l;
            subnet_wrapper<typename T::subnet_type,false> subnetwork;
        };
    }

// ----------------------------------------------------------------------------------------

    template <typename LAYER_DETAILS, typename SUBNET, typename enabled = void>
    class add_layer;

    template <typename T, typename U>
    struct is_nonloss_layer_type<add_layer<T,U>> : std::true_type {};

    template <typename LAYER_DETAILS, typename SUBNET>
    class add_layer<LAYER_DETAILS,SUBNET, 
            typename std::enable_if<is_nonloss_layer_type<SUBNET>::value>::type>
    {
    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        const static size_t num_layers = subnet_type::num_layers + 1;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;

        add_layer(
        ):
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork.disable_output_and_gradient_getters();
        }

        add_layer(const add_layer&) = default;
        add_layer& operator=(const add_layer&) = default;
        add_layer(add_layer&& item) : add_layer() { swap(item); }
        add_layer& operator=(add_layer&& item) { swap(item); return *this; }

        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;

        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U, typename E>
        add_layer(
            const add_layer<T,U,E>& item
        ) :
            subnetwork(item.subnet()),
            details(item.layer_details()), 
            this_layer_setup_called(item.this_layer_setup_called),
            gradient_input_is_stale(item.gradient_input_is_stale),
            get_output_and_gradient_input_disabled(item.get_output_and_gradient_input_disabled),
            x_grad(item.x_grad),
            cached_output(item.cached_output)
        {
            if (this_layer_operates_inplace())
                subnetwork.disable_output_and_gradient_getters();
        }

        template <typename ...T>
        add_layer(
            const LAYER_DETAILS& layer_det, 
            T&& ...args
        ) : 
            details(layer_det), 
            subnetwork(std::forward<T>(args)...),
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork.disable_output_and_gradient_getters();
        }

        template <typename ...T>
        add_layer(
            LAYER_DETAILS&& layer_det, 
            T&& ...args
        ) : 
            details(std::move(layer_det)), 
            subnetwork(std::forward<T>(args)...),
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork.disable_output_and_gradient_getters();
        }

        template <typename ...T, typename ...U>
        add_layer(
            const std::tuple<LAYER_DETAILS,U...>& layer_det, 
            T&& ...args
        ) : 
            details(std::get<0>(layer_det)), 
            subnetwork(tuple_tail(layer_det),std::forward<T>(args)...),
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork.disable_output_and_gradient_getters();
        }

        template <typename ...T, typename ...U>
        add_layer(
            std::tuple<>,
            const std::tuple<LAYER_DETAILS,U...>& layer_det, 
            T&& ...args
        ) : add_layer(layer_det,args...) { }

        template <typename ...T>
        add_layer(
            std::tuple<>, 
            LAYER_DETAILS&& layer_det, 
            T&& ...args
        ) : add_layer(layer_det, args...) { }

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            subnetwork.to_tensor(ibegin,iend,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            return forward(temp_tensor);
        }


        const tensor& operator() (const input_type& x)
        {
            return (*this)(&x, &x+1);
        }

        const tensor& forward(const tensor& x)
        {
            subnetwork.forward(x);
            const dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            if (!this_layer_setup_called)
            {
                details.setup(wsub);
                this_layer_setup_called = true;
            }
            if (this_layer_operates_inplace())
                impl::call_layer_forward(details, wsub, private_get_output());
            else
                impl::call_layer_forward(details, wsub, cached_output);

            gradient_input_is_stale = true;
            return private_get_output();
        }

    private:
        tensor& private_get_output() const
        { 
            if (const_cast<add_layer&>(*this).this_layer_operates_inplace())
                return subnetwork.private_get_output();
            else
                return const_cast<resizable_tensor&>(cached_output); 
        }
        tensor& private_get_gradient_input() 
        { 
            if (this_layer_operates_inplace())
            {
                return subnetwork.private_get_gradient_input();
            }
            else
            {
                if (gradient_input_is_stale)
                {
                    gradient_input_is_stale = false;
                    x_grad.copy_size(private_get_output());
                    x_grad = 0;
                }
                return x_grad; 
            }
        }
        void disable_output_and_gradient_getters (
        ) { get_output_and_gradient_input_disabled = true; }
    public:
        const tensor& get_output() const 
        { 
            if (get_output_and_gradient_input_disabled)
                throw dlib::error("Accessing this layer's get_output() is disabled because an in-place layer has been stacked on top of it.");
            return private_get_output(); 
        }
        tensor& get_gradient_input() 
        { 
            if (get_output_and_gradient_input_disabled)
                throw dlib::error("Accessing this layer's get_gradient_input() is disabled because an in-place layer has been stacked on top of it.");
            return private_get_gradient_input();
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        {
            dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            params_grad.copy_size(details.get_layer_params());
            impl::call_layer_backward(details, private_get_output(),
                private_get_gradient_input(), wsub, static_cast<tensor&>(params_grad));
            // Don't try to adjust the parameters if this layer doesn't have any.
            if (params_grad.size() != 0)
                solvers.top()(details, static_cast<const tensor&>(params_grad));
            subnetwork.update(x, solvers.pop());
            gradient_input_is_stale = true;
        }

        const subnet_type& subnet() const { return subnetwork; }
        subnet_type& subnet() { return subnetwork; }

        const layer_details_type& layer_details() const { return details; } 
        layer_details_type& layer_details() { return details; } 

        void clean()
        {
            x_grad.clear();
            cached_output.clear();
            params_grad.clear();
            temp_tensor.clear();
            gradient_input_is_stale = true;
            subnetwork.clean();
        }

        friend void serialize(const add_layer& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.subnetwork, out);
            serialize(item.details, out);
            serialize(item.this_layer_setup_called, out);
            serialize(item.gradient_input_is_stale, out);
            serialize(item.get_output_and_gradient_input_disabled, out);
            serialize(item.x_grad, out);
            serialize(item.cached_output, out);
        }

        friend void deserialize(add_layer& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::add_layer.");
            deserialize(item.subnetwork, in);
            deserialize(item.details, in);
            deserialize(item.this_layer_setup_called, in);
            deserialize(item.gradient_input_is_stale, in);
            deserialize(item.get_output_and_gradient_input_disabled, in);
            deserialize(item.x_grad, in);
            deserialize(item.cached_output, in);
        }

    private:

        bool this_layer_operates_inplace(
        ) 
        {
            // This layer can run in-place if it's an in-place capable layer and also if
            // the layer it's on top of doesn't need it's own output tensor (since in-place
            // layers overwrite that tensor)
            return impl::is_inplace_layer(details, subnetwork) && !subnetwork.this_layer_requires_forward_output();
        }
        bool this_layer_requires_forward_output(
        ) 
        {
            return impl::backward_requires_forward_output(details, subnetwork);
        }

        void swap(add_layer& item)
        {
            std::swap(subnetwork,item.subnetwork);
            std::swap(details, item.details);
            std::swap(this_layer_setup_called, item.this_layer_setup_called);
            std::swap(gradient_input_is_stale, item.gradient_input_is_stale);
            std::swap(get_output_and_gradient_input_disabled, item.get_output_and_gradient_input_disabled);
            std::swap(x_grad, item.x_grad);
            std::swap(cached_output, item.cached_output);
        }


        subnet_type subnetwork;
        LAYER_DETAILS details;
        bool this_layer_setup_called;
        bool gradient_input_is_stale;
        bool get_output_and_gradient_input_disabled;
        // Note that if this_layer_operates_inplace()==true then x_grad and cached_output
        // are not used at all.  Instead, this layer uses these variables from the lower
        // layer.
        resizable_tensor x_grad;
        resizable_tensor cached_output; 

        // The following 2 objects don't logically contribute to the state of this class.
        // They are only here to prevent them from being reallocated over and over in
        // member functions.
        resizable_tensor params_grad; 
        resizable_tensor temp_tensor;

    };

// ----------------------------------------------------------------------------------------

// This version of add_layer handles the special case where the subnetwork being given is
// just an input layer object.
    template <typename LAYER_DETAILS, typename INPUT_LAYER, typename enabled>
    class add_layer
    {
    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef INPUT_LAYER subnet_type;
        typedef typename INPUT_LAYER::input_type input_type;
        const static unsigned int sample_expansion_factor = INPUT_LAYER::sample_expansion_factor;
        const static size_t num_layers = 1;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_layer(
        ): 
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(const add_layer&) = default;
        add_layer(add_layer&& item) : add_layer() { swap(item); }
        add_layer& operator=(const add_layer&) = default;
        add_layer& operator=(add_layer&& item) { swap(item); return *this; }

        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;

        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U, typename E>
        add_layer(
            const add_layer<T,U,E>& item
        ):
            input_layer(item.subnet()),
            details(item.layer_details()),
            this_layer_setup_called(item.this_layer_setup_called),
            gradient_input_is_stale(item.gradient_input_is_stale),
            get_output_and_gradient_input_disabled(false),
            x_grad(item.x_grad),
            cached_output(item.cached_output)
        {
        }

        add_layer(
            const LAYER_DETAILS& layer_det
        ) : 
            details(layer_det), 
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            LAYER_DETAILS&& layer_det
        ) : 
            details(std::move(layer_det)), 
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            LAYER_DETAILS layer_det, 
            INPUT_LAYER il
        ) : 
            details(std::move(layer_det)),
            input_layer(std::move(il)),
            this_layer_setup_called(false),
            gradient_input_is_stale(true),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            std::tuple<>,
            const LAYER_DETAILS& layer_det
        ) : add_layer(layer_det) {}

        add_layer(
            std::tuple<>,
            LAYER_DETAILS&& layer_det
        ) : add_layer(layer_det) {}

        add_layer(
            std::tuple<>,
            LAYER_DETAILS layer_det, 
            INPUT_LAYER il
        ) : add_layer(layer_det,il) {}

        add_layer(
            const std::tuple<LAYER_DETAILS>& layer_det
        ) : add_layer(std::get<0>(layer_det)) {}

        add_layer(
            const std::tuple<LAYER_DETAILS>& layer_det,
            INPUT_LAYER il
        ) : add_layer(std::get<0>(layer_det),il) {}

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            input_layer.to_tensor(ibegin, iend, data);
            // make sure the input layer's to_tensor() function is implemented properly.
            DLIB_CASSERT(std::distance(ibegin,iend)*sample_expansion_factor == data.num_samples(),"");
            data.async_copy_to_device();
        }


        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            return forward(temp_tensor);
        }


        const tensor& operator() (const input_type& x)
        {
            return (*this)(&x, &x+1);
        }

        const tensor& forward (const tensor& x)
        {
            DLIB_CASSERT(x.num_samples()%sample_expansion_factor == 0,"");
            subnet_wrapper wsub(x, grad_final_ignored);
            if (!this_layer_setup_called)
            {
                details.setup(wsub);
                this_layer_setup_called = true;
            }
            impl::call_layer_forward(details, wsub, cached_output);
            gradient_input_is_stale = true;
            return private_get_output();
        }

    private:
        tensor& private_get_output() const { return const_cast<resizable_tensor&>(cached_output); }
        tensor& private_get_gradient_input() 
        { 
            if (gradient_input_is_stale)
            {
                gradient_input_is_stale = false;
                x_grad.copy_size(private_get_output());
                x_grad = 0;
            }
            return x_grad; 
        }
        void disable_output_and_gradient_getters (
        ) { get_output_and_gradient_input_disabled = true; }
    public:
        const tensor& get_output() const 
        { 
            if (get_output_and_gradient_input_disabled)
                throw dlib::error("Accessing this layer's get_output() is disabled because an in-place layer has been stacked on top of it.");
            return private_get_output(); 
        }
        tensor& get_gradient_input() 
        { 
            if (get_output_and_gradient_input_disabled)
                throw dlib::error("Accessing this layer's get_gradient_input() is disabled because an in-place layer has been stacked on top of it.");
            return private_get_gradient_input();
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        {
            subnet_wrapper wsub(x, grad_final_ignored);
            params_grad.copy_size(details.get_layer_params());
            impl::call_layer_backward(details, private_get_output(),
                private_get_gradient_input(), wsub, static_cast<tensor&>(params_grad));
            // Don't try to adjust the parameters if this layer doesn't have any.
            if (params_grad.size() != 0)
                solvers.top()(details, static_cast<const tensor&>(params_grad));
            gradient_input_is_stale = true;
        }

        const subnet_type& subnet() const { return input_layer; } 
        subnet_type& subnet() { return input_layer; } 

        const layer_details_type& layer_details() const { return details; } 
        layer_details_type& layer_details() { return details; } 

        void clean()
        {
            x_grad.clear();
            grad_final_ignored.clear();
            cached_output.clear();
            params_grad.clear();
            temp_tensor.clear();
            gradient_input_is_stale = true;
        }

        friend void serialize(const add_layer& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.input_layer, out);
            serialize(item.details, out);
            serialize(item.this_layer_setup_called, out);
            serialize(item.gradient_input_is_stale, out);
            serialize(item.get_output_and_gradient_input_disabled, out);
            serialize(item.x_grad, out);
            serialize(item.cached_output, out);
        }

        friend void deserialize(add_layer& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::add_layer.");
            deserialize(item.input_layer, in);
            deserialize(item.details, in);
            deserialize(item.this_layer_setup_called, in);
            deserialize(item.gradient_input_is_stale, in);
            deserialize(item.get_output_and_gradient_input_disabled, in);
            deserialize(item.x_grad, in);
            deserialize(item.cached_output, in);
        }

    private:

        bool this_layer_requires_forward_output(
        ) 
        {
            subnet_wrapper wsub(grad_final_ignored, grad_final_ignored);
            return impl::backward_requires_forward_output(details, wsub);
        }

        class subnet_wrapper
        {
        public:
            subnet_wrapper(const tensor& x_, resizable_tensor& grad_final_ignored_) :
                x(x_), grad_final_ignored(grad_final_ignored_) {}

            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            const tensor& get_output() const { return x; }
            tensor& get_gradient_input() 
            { 
                // It doesn't matter what values are in this tensor but client code will
                // always assume it's the same dimension as the output so make sure that is
                // the case.  Note that we do set it to a non-crazy value though to avoid
                // it being full of NaN and slowing the processing down.
                if (!have_same_dimensions(x, grad_final_ignored))
                {
                    grad_final_ignored.copy_size(x);
                    grad_final_ignored = 0;  
                }
                return grad_final_ignored; 
            }

        private:
            const tensor& x;
            resizable_tensor& grad_final_ignored;
        };

        void swap(add_layer& item)
        {
            std::swap(input_layer, item.input_layer);
            std::swap(details, item.details);
            std::swap(this_layer_setup_called, item.this_layer_setup_called);
            std::swap(gradient_input_is_stale, item.gradient_input_is_stale);
            std::swap(get_output_and_gradient_input_disabled, item.get_output_and_gradient_input_disabled);
            std::swap(x_grad, item.x_grad); 
            std::swap(cached_output, item.cached_output); 
        }

        subnet_type input_layer;
        LAYER_DETAILS details;
        bool this_layer_setup_called;
        bool gradient_input_is_stale;
        bool get_output_and_gradient_input_disabled;
        resizable_tensor x_grad; 
        resizable_tensor cached_output; 

        // The following 3 objects don't logically contribute to the state of this class.
        // They are only here to prevent them from being reallocated over and over in
        // member functions.
        resizable_tensor params_grad; 
        resizable_tensor temp_tensor; 
        resizable_tensor grad_final_ignored;
    };

// ----------------------------------------------------------------------------------------

    template <unsigned long ID, typename SUBNET, typename enabled=void>
    class add_tag_layer;

    template <unsigned long ID, typename SUBNET>
    class add_tag_layer<ID,SUBNET,
            typename std::enable_if<is_nonloss_layer_type<SUBNET>::value>::type>
    {
    public:
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        const static size_t num_layers = subnet_type::num_layers + 1;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_tag_layer() = default;
        add_tag_layer(const add_tag_layer&) = default;
        add_tag_layer(add_tag_layer&&) = default;
        add_tag_layer& operator=(add_tag_layer&&) = default;
        add_tag_layer& operator=(const add_tag_layer&) = default;

        template <typename T>
        add_tag_layer(
            const add_tag_layer<ID,T>& item
        ) : subnetwork(item.subnet())
        {}

        template <typename ...T>
        add_tag_layer(
            T ...args
        ) : 
            subnetwork(std::move(args)...) 
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            subnetwork.to_tensor(ibegin,iend,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            return subnetwork(ibegin,iend);
        }

        const tensor& operator() (const input_type& x)
        {
            return subnetwork(x);
        }

        const tensor& forward(const tensor& x)
        {
            return subnetwork.forward(x);
        }

        const tensor& get_output() const { return subnetwork.get_output(); }

        tensor& get_gradient_input() 
        { 
            return subnetwork.get_gradient_input();
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        {
            subnetwork.update(x,solvers.pop());
        }

        const subnet_type& subnet() const { return subnetwork; }
        subnet_type& subnet() { return subnetwork; }

        void clean()
        {
            subnetwork.clean();
        }

        friend void serialize(const add_tag_layer& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.subnetwork, out);
        }

        friend void deserialize(add_tag_layer& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::add_tag_layer.");
            deserialize(item.subnetwork, in);
        }

    private:

        subnet_type subnetwork;
    };

// ----------------------------------------------------------------------------------------

// This version of add_tag_layer handles the special case where the subnetwork being given
// is just an input layer object.
    template <unsigned long ID, typename INPUT_LAYER, typename enabled>
    class add_tag_layer
    {
    public:
        typedef INPUT_LAYER subnet_type;
        typedef typename subnet_type::input_type input_type;
        const static size_t num_layers = 1;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_tag_layer() = default;
        add_tag_layer(const add_tag_layer&) = default;
        add_tag_layer& operator=(const add_tag_layer&) = default;
        add_tag_layer(add_tag_layer&& item) : add_tag_layer() { swap(item); }
        add_tag_layer& operator=(add_tag_layer&& item) { swap(item); return *this; }

        template <typename T, typename E>
        add_tag_layer(
            const add_tag_layer<ID,T,E>& item
        ) : input_layer(item.subnet())
        {}

        template <typename ...T>
        add_tag_layer(
            T ...args
        ) : 
            input_layer(std::move(args)...) 
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            input_layer.to_tensor(ibegin,iend,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin, 
            input_iterator iend
        )
        {
            input_layer.to_tensor(ibegin,iend,cached_output);
            return get_output();
        }

        const tensor& operator() (const input_type& x)
        {
            return (*this)(&x, &x+1);
        }

        const tensor& forward(const tensor& x)
        {
            cached_output = x;
            return get_output();
        }

        const tensor& get_output() const 
        { 
            return cached_output; 
        }

        tensor& get_gradient_input() 
        { 
            if (!have_same_dimensions(cached_output, grad_final_ignored))
            {
                grad_final_ignored.copy_size(get_output());
                grad_final_ignored = 0;
            }
            return grad_final_ignored; 
        }

        template <typename solver_type>
        void update(const tensor& /*x*/, sstack<solver_type,num_layers>& /*solvers*/)
        {
            // nothing to update
        }

        const subnet_type& subnet() const { return input_layer; }
        subnet_type& subnet() { return input_layer; }

        void clean()
        {
            grad_final_ignored.clear();
            cached_output.clear();
        }

        friend void serialize(const add_tag_layer& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.input_layer, out);
            serialize(item.cached_output, out);
            serialize(item.grad_final_ignored, out);
        }

        friend void deserialize(add_tag_layer& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::add_tag_layer.");
            deserialize(item.input_layer, in);
            deserialize(item.cached_output, in);
            deserialize(item.grad_final_ignored, in);
        }

    private:

        void swap(add_tag_layer& item)
        {
            std::swap(input_layer, item.input_layer);
            std::swap(cached_output, item.cached_output);
            std::swap(grad_final_ignored, item.grad_final_ignored);
        }

        subnet_type input_layer;
        resizable_tensor cached_output;
        resizable_tensor grad_final_ignored;
    };

    template <unsigned long ID, typename U, typename E>
    struct is_nonloss_layer_type<add_tag_layer<ID,U,E>> : std::true_type {};


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename LOSS_DETAILS, typename SUBNET>
    class add_loss_layer;

    class no_label_type
    {
    private:
        // We don't want anyone making these no_label_type objects.  They are here only to
        // allow add_loss_layer::label_type and dnn_trainer::label_type to exist which avoids
        // needing to overload add_loss_layer and dnn_trainer for supervised an unsupervised
        // losses.  It also can be a type to use in template metaprogramming to indicate
        // "no label".  So here we make the constructor private with the exception that
        // add_loss_layer objects can make it (again, just to simplify add_loss_layer's
        // implementation).
        no_label_type()=default;
        template <typename LOSS_DETAILS, typename SUBNET> friend class add_loss_layer;
    };

// ----------------------------------------------------------------------------------------

    template <typename LOSS_DETAILS, typename SUBNET>
    class add_loss_layer
    {
        template <typename T, typename enabled=void>
        struct get_loss_layer_label_type
        {
            typedef no_label_type type;
        };
        template <typename T>
        struct get_loss_layer_label_type<T,typename std::enable_if<sizeof(typename T::label_type)!=0>::type>
        {
            typedef typename T::label_type type;
        };

    public:
        typedef LOSS_DETAILS loss_details_type;
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        // Note that the loss layer doesn't count as an additional layer.
        const static size_t num_layers = subnet_type::num_layers;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;
        typedef typename get_loss_layer_label_type<LOSS_DETAILS>::type label_type;

        static_assert(is_nonloss_layer_type<SUBNET>::value, 
            "SUBNET must be of type add_layer, add_skip_layer, or add_tag_layer."); 
        static_assert(sample_expansion_factor == LOSS_DETAILS::sample_expansion_factor,
            "The loss layer and input layer must agree on the sample_expansion_factor.");


        add_loss_layer() {};
        add_loss_layer(const add_loss_layer&) = default;
        add_loss_layer& operator=(const add_loss_layer&) = default;
        add_loss_layer(add_loss_layer&& item) : add_loss_layer() { swap(item); }
        add_loss_layer& operator=(add_loss_layer&& item) { swap(item); return *this; }

        template <typename T, typename U>
        add_loss_layer(
            const add_loss_layer<T,U>& item
        ) : 
            loss(item.loss_details()),
            subnetwork(item.subnet())
        {}

        template <typename ...T>
        add_loss_layer(
            const LOSS_DETAILS& layer_det, 
            T&& ...args
        ) : 
            loss(layer_det), 
            subnetwork(std::forward<T>(args)...)
        {
        }

        template <typename ...T>
        add_loss_layer(
            LOSS_DETAILS&& layer_det, 
            T&& ...args
        ) : 
            loss(std::move(layer_det)), 
            subnetwork(std::forward<T>(args)...)
        {
        }

        template <typename ...T>
        add_loss_layer(
            T ...args
        ) : 
            subnetwork(std::move(args)...)
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            subnetwork.to_tensor(ibegin,iend,data);
        }

        template <typename output_iterator>
        void operator() (
            const tensor& x, 
            output_iterator obegin
        )
        {
            subnetwork.forward(x);
            const dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            loss.to_label(x, wsub, obegin);
        }

        template <typename input_iterator, typename output_iterator>
        void operator() (
            input_iterator ibegin,
            input_iterator iend,
            output_iterator obegin
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            (*this)(temp_tensor, obegin);
        }

        const label_type& operator() (const input_type& x)
        {
            (*this)(&x, &x+1, &temp_label);
            return temp_label;
        }

        template <typename label_iterator>
        double compute_loss (
            const tensor& x,
            label_iterator lbegin 
        )
        {
            subnetwork.forward(x);
            dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            return loss.compute_loss(x, lbegin, wsub);
        }

        template <typename input_iterator, typename label_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin 
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            return compute_loss(temp_tensor, lbegin);
        }

        double compute_loss (
            const tensor& x
        )
        {
            subnetwork.forward(x);
            dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            return loss.compute_loss(x, wsub);
        }

        template <typename input_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            return compute_loss(temp_tensor);
        }

        template <typename label_iterator, typename solver_type>
        double update (
            const tensor& x,
            label_iterator lbegin,
            sstack<solver_type,num_layers>& solvers
        )
        {
            subnetwork.forward(x);
            dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            double l = loss.compute_loss(x, lbegin, wsub);
            subnetwork.update(x, solvers);
            return l;
        }

        template <typename input_iterator, typename label_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin,
            sstack<solver_type,num_layers>& solvers
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            return update(temp_tensor, lbegin, solvers);
        }

        template <typename solver_type>
        double update (
            const tensor& x,
            sstack<solver_type,num_layers>& solvers
        )
        {
            subnetwork.forward(x);
            dimpl::subnet_wrapper<subnet_type> wsub(subnetwork);
            double l = loss.compute_loss(x, wsub);
            subnetwork.update(x, solvers);
            return l;
        }

        template <typename input_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            sstack<solver_type,num_layers>& solvers
        )
        {
            to_tensor(ibegin,iend,temp_tensor);
            return update(temp_tensor, solvers);
        }

        const subnet_type& subnet() const { return subnetwork; }
        subnet_type& subnet() { return subnetwork; }
        const loss_details_type& loss_details() const { return loss; }
        loss_details_type& loss_details() { return loss; }

        void clean (
        )
        {
            temp_tensor.clear();
            subnetwork.clean();
        }

        friend void serialize(const add_loss_layer& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.loss, out);
            serialize(item.subnetwork, out);
        }

        friend void deserialize(add_loss_layer& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::add_loss_layer.");
            deserialize(item.loss, in);
            deserialize(item.subnetwork, in);
        }

    private:

        void swap(add_loss_layer& item)
        {
            std::swap(loss, item.loss);
            std::swap(subnetwork, item.subnetwork);
        }

        loss_details_type loss;
        subnet_type subnetwork;

        // These two objects don't logically contribute to the state of this object.  They
        // are here to prevent them from being reallocated over and over.
        label_type temp_label;
        resizable_tensor temp_tensor;
    };


    template <typename T, typename U>
    struct is_loss_layer_type<add_loss_layer<T,U>> : std::true_type {};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <unsigned int i, typename T>
        struct layer_helper
        {
            static T& makeT();
            using next_type = typename std::remove_reference<decltype(makeT().subnet())>::type;
            using type = typename layer_helper<i-1,next_type>::type;
            static type& layer(T& n)
            {
                return layer_helper<i-1,next_type>::layer(n.subnet());
            }
        };
        template <typename T>
        struct layer_helper<0,T>
        {
            using type = T;
            static type& layer(T& n)
            {
                return n;
            }
        };

        template <template<typename> class Match, typename T, unsigned int i, typename enabled = void>
        struct layer_helper_match
        {
            static T& makeT();
            using next_type = typename std::remove_reference<decltype(makeT().subnet())>::type;
            using type = typename layer_helper_match<Match,next_type,i>::type;
            static type& layer(T& n)
            {
                return layer_helper_match<Match,next_type,i>::layer(n.subnet());
            }
        };
        // This overload catches add_layer and add_loss_layer templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const T,const  Match<typename T::subnet_type>>::value>::type>
        {
            using type = typename layer_helper<i,T>::type;
            static type& layer(T& n)
            {
                return layer_helper<i,T>::layer(n);
            }
        };
        // This overload catches input templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const T,const  Match<typename T::input_type>>::value>::type>
        {
            using type = typename layer_helper<i,T>::type;
            static type& layer(T& n)
            {
                return layer_helper<i,T>::layer(n);
            }
        };
        // This overload catches subnet_wrapper templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const typename T::wrapped_type, 
                                                 const Match<typename T::wrapped_type::subnet_type>>::value>::type>
        {
            using type = typename layer_helper<i,T>::type;
            static type& layer(T& n)
            {
                return layer_helper<i,T>::layer(n);
            }
        };
    }

    template <unsigned int i, typename T>
    typename impl::layer_helper<i,T>::type& layer (T& n) 
    {
        return impl::layer_helper<i,T>::layer(n);
    }

    template <template<typename> class Match, typename T>
    typename impl::layer_helper_match<Match,T,0>::type& layer (T& n) 
    {
        return impl::layer_helper_match<Match,T,0>::layer(n);
    }

    template <template<typename> class Match, unsigned int i, typename T>
    typename impl::layer_helper_match<Match,T,i>::type& layer (T& n) 
    {
        return impl::layer_helper_match<Match,T,i>::layer(n);
    }

// ----------------------------------------------------------------------------------------

    template <template<typename> class TAG_TYPE, typename SUBNET>
    class add_skip_layer
    {
    public:
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        const static size_t num_layers = subnet_type::num_layers + 1;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_skip_layer() = default;
        add_skip_layer(const add_skip_layer&) = default;
        add_skip_layer(add_skip_layer&&) = default;
        add_skip_layer& operator=(add_skip_layer&&) = default;
        add_skip_layer& operator=(const add_skip_layer&) = default;

        template <typename T>
        add_skip_layer(
            const add_skip_layer<TAG_TYPE,T>& item
        ) : subnetwork(item.subnet())
        {}

        template <typename ...T>
        add_skip_layer(
            T ...args
        ) : 
            subnetwork(std::move(args)...) 
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const
        {
            subnetwork.to_tensor(ibegin,iend,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            subnetwork(ibegin,iend);
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const tensor& operator() (const input_type& x)
        {
            subnetwork(x);
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const tensor& forward(const tensor& x)
        {
            subnetwork.forward(x);
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const tensor& get_output() const 
        { 
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        tensor& get_gradient_input() 
        { 
            return layer<TAG_TYPE>(subnetwork).get_gradient_input();
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        {
            subnetwork.update(x,solvers.pop());
        }

        const subnet_type& subnet() const 
        { 
            return subnetwork; 
        }

        subnet_type& subnet() 
        { 
            return subnetwork; 
        }

        void clean()
        {
            subnetwork.clean();
        }

        friend void serialize(const add_skip_layer& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.subnetwork, out);
        }

        friend void deserialize(add_skip_layer& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::add_skip_layer.");
            deserialize(item.subnetwork, in);
        }

    private:

        subnet_type subnetwork;
    };
    template <template<typename> class T, typename U>
    struct is_nonloss_layer_type<add_skip_layer<T,U>> : std::true_type {};

    template <typename SUBNET> using tag1  = add_tag_layer< 1, SUBNET>;
    template <typename SUBNET> using tag2  = add_tag_layer< 2, SUBNET>;
    template <typename SUBNET> using tag3  = add_tag_layer< 3, SUBNET>;
    template <typename SUBNET> using tag4  = add_tag_layer< 4, SUBNET>;
    template <typename SUBNET> using tag5  = add_tag_layer< 5, SUBNET>;
    template <typename SUBNET> using tag6  = add_tag_layer< 6, SUBNET>;
    template <typename SUBNET> using tag7  = add_tag_layer< 7, SUBNET>;
    template <typename SUBNET> using tag8  = add_tag_layer< 8, SUBNET>;
    template <typename SUBNET> using tag9  = add_tag_layer< 9, SUBNET>;
    template <typename SUBNET> using tag10 = add_tag_layer<10, SUBNET>;

    template <typename SUBNET> using skip1  = add_skip_layer< tag1, SUBNET>;
    template <typename SUBNET> using skip2  = add_skip_layer< tag2, SUBNET>;
    template <typename SUBNET> using skip3  = add_skip_layer< tag3, SUBNET>;
    template <typename SUBNET> using skip4  = add_skip_layer< tag4, SUBNET>;
    template <typename SUBNET> using skip5  = add_skip_layer< tag5, SUBNET>;
    template <typename SUBNET> using skip6  = add_skip_layer< tag6, SUBNET>;
    template <typename SUBNET> using skip7  = add_skip_layer< tag7, SUBNET>;
    template <typename SUBNET> using skip8  = add_skip_layer< tag8, SUBNET>;
    template <typename SUBNET> using skip9  = add_skip_layer< tag9, SUBNET>;
    template <typename SUBNET> using skip10 = add_skip_layer<tag10, SUBNET>;

// ----------------------------------------------------------------------------------------

    namespace timpl
    {
        inline void fill_with_gassuan_random_numbers (
            tensor& t,
            dlib::rand& rnd,
            double sigma = 1
        )
        {
            float* data = t.host();
            for (size_t i = 0; i < t.size(); ++i)
                data[i] = rnd.get_random_gaussian()*sigma;
        }

        class test_layer_subnet 
        {
        public:
            test_layer_subnet (
                dlib::rand& rnd_
            ) : rnd(rnd_) 
            {
                // Output and gradient_input have to have the same dimensions in each
                // layer.
                const long num_samples = rnd.get_random_32bit_number()%4+3;
                const long k  = rnd.get_random_32bit_number()%4+2;
                const long nr = rnd.get_random_32bit_number()%4+2;
                const long nc = rnd.get_random_32bit_number()%4+2;

                output.set_size(num_samples, k, nr, nc);
                gradient_input.set_size(num_samples, k, nr, nc);

                // Use a non-zero initial gradient to make sure the layers add to it
                // rather than assign and blow away the initial value.
                fill_with_gassuan_random_numbers(gradient_input, rnd, 0.01);

                fill_with_gassuan_random_numbers(output, rnd);
            }


            tensor& get_mutable_output() { return output; }
            const tensor& get_output() const { return output; }
            const tensor& private_get_output() const { return get_output(); }
            const test_layer_subnet& subnet() const { init_sub(); return *subnetwork; }

            tensor& get_gradient_input() { return gradient_input; }
            tensor& private_get_gradient_input() { return get_gradient_input(); }
            test_layer_subnet& subnet() { init_sub(); return *subnetwork; }



            unsigned long count_outputs() const
            {
                if (subnetwork)
                    return subnetwork->count_outputs() + output.size();
                else
                    return output.size();
            }

            float& get_output_element(unsigned long i)
            {
                if (i < output.size())
                    return output.host()[i];
                else
                    return subnet().get_output_element(i-output.size());
            }

            float get_gradient_input_element(unsigned long i) const
            {
                if (i < gradient_input.size())
                    return gradient_input.host()[i];
                else
                    return subnet().get_gradient_input_element(i-gradient_input.size());
            }


        private:
            // We lazily initialize sub-layers as needed when someone tries to call
            // subnet()
            void init_sub() const
            {
                if (!subnetwork)
                    subnetwork.reset(new test_layer_subnet(rnd));
            }

            dlib::rand& rnd;
            mutable std::unique_ptr<test_layer_subnet> subnetwork;
            resizable_tensor output;
            resizable_tensor gradient_input;
        };

    }

    struct layer_test_results
    {
        layer_test_results() : was_good(true) {}
        explicit layer_test_results(const std::string& l) : log(l),was_good(false) {}

        std::string log;
        bool was_good;

        operator bool() const { return was_good; }
    };

    inline std::ostream& operator<< (std::ostream& out, const layer_test_results& item)
    {
        out << item.log;
        return out;
    }

    template <
        typename layer_details_type
        >
    layer_test_results test_layer (
        layer_details_type l
    )
    {
        const float base_eps = 0.01;
        using namespace timpl;
        // Do some setup
        dlib::rand rnd;
        test_layer_subnet subnetwork(rnd);
        resizable_tensor output, out2, out3;
        // Run setup() and forward() as well to make sure any calls to subnet() have
        // happened before we start assuming we know how many data elements there are
        // (since we do a lazy layer creation thing based on calls to subnet() inside
        // test_layer_subnet).
        l.setup(subnetwork);
        impl::call_layer_forward(l, subnetwork, output);

        resizable_tensor input_grad;
        input_grad.copy_size(output);
        fill_with_gassuan_random_numbers(input_grad, rnd);

        std::ostringstream sout;

        // The f() we are computing gradients of is this thing.  It's value at the current
        // parameter and data values is:
        //sout << "f(data,params): " << dot(output, input_grad) << std::endl;

        // We are going to save a copy of the subnetwork.get_gradient_input() data before we do
        // backpropagation since the backward() function is supposed to *add* to the
        // gradients rather than overwrite them.  We will use this saved data to check if
        // that is the case.
        const unsigned long num_data_inputs = subnetwork.count_outputs();
        std::vector<float> initial_gradient_input(num_data_inputs);
        for (unsigned long i = 0; i < num_data_inputs; ++i)
            initial_gradient_input[i] = subnetwork.get_gradient_input_element(i);


        // Now tell the layer to compute all the gradients.  In the rest of this function
        // we will just be checking that these gradients were computed correctly by
        // comparing them to a central differences approximation.
        resizable_tensor params_grad;
        params_grad.copy_size(l.get_layer_params());
        // But first, set the params grad to something crazy so that it's very obvious if
        // it doesn't get fully assigned.
        params_grad = std::numeric_limits<float>::infinity();
        impl::call_layer_backward(l, output, input_grad, subnetwork, params_grad);

        static_assert(impl::is_inplace_layer(l, subnetwork) == impl::has_inplace_backward(l, subnetwork),
            "Layer not defined correctly.  forward and backward methods must either both be in-place or both out-of-place. ");

        // Make sure the outputs of forward() and backward() are the same when they are run
        // in in-place mode.
        if (impl::is_inplace_layer(l, subnetwork))
        {
            test_layer_subnet subnetwork2(rnd);
            layer_details_type ll(l);
            ll.setup(subnetwork2);
            resizable_tensor ip_out;
            impl::call_layer_forward(ll, subnetwork2, ip_out);
            impl::call_layer_forward(ll, subnetwork2, subnetwork2.get_mutable_output());
            const auto forward_error = max(abs(mat(ip_out) - mat(subnetwork2.get_output())));
            if (forward_error > 0.00001)
            {
                using namespace std;
                sout << "This layer is supposed to support in-place computations but the output of forward_inplace()\n";
                sout << "changes when invoked in-place vs. out-of-place. The error was: " << forward_error << endl;
                return layer_test_results(sout.str()); 
            }

            resizable_tensor params_grad;
            params_grad.copy_size(ll.get_layer_params());
            params_grad = std::numeric_limits<float>::infinity();

            resizable_tensor input_grad;
            input_grad.copy_size(ip_out);
            fill_with_gassuan_random_numbers(input_grad, rnd);
            resizable_tensor params_grad1, params_grad2, data_grad1, data_grad2;
            params_grad1 = params_grad;
            params_grad2 = params_grad;
            // Now call backward() and make sure it works as well.
            subnetwork2.get_gradient_input() = 9999;
            impl::call_layer_backward(ll, ip_out, input_grad, subnetwork2, params_grad1);
            data_grad1 = subnetwork2.get_gradient_input();

            subnetwork2.get_gradient_input() = mat(input_grad);
            impl::call_layer_backward(ll, ip_out, subnetwork2.get_gradient_input(), subnetwork2, params_grad2);
            data_grad2 = subnetwork2.get_gradient_input();
            if (params_grad.size() != 0)
            {
                const auto backward_param_error = max(abs(mat(params_grad1) - mat(params_grad2)));
                if (backward_param_error > 0.00001)
                {
                    using namespace std;
                    sout << "This layer is supposed to support in-place computations but the output of backward_inplace()\n";
                    sout << "changes when invoked in-place vs. out-of-place. The error was: " << backward_param_error << endl;
                    return layer_test_results(sout.str()); 
                }
            }
            const auto backward_data_error = max(abs(mat(data_grad1) - mat(data_grad2)));
            if (backward_data_error > 0.00001)
            {
                using namespace std;
                sout << "This layer is supposed to support in-place computations but the output of backward_inplace()\n";
                sout << "changes when invoked in-place vs. out-of-place. The error was: " << backward_data_error << endl;
                return layer_test_results(sout.str()); 
            }
        }

        // ==================================================================
        // first validate the way the parameter gradients are computed
        for (unsigned long i = 0; i < params_grad.size(); ++i)
        {
            layer_details_type l1(l);

            float eps = l1.get_layer_params().host()[i]*base_eps;
            if (eps == 0)
                eps = base_eps;
            const float oldval = l1.get_layer_params().host()[i];
            l1.get_layer_params().host()[i] = oldval+eps;
            impl::call_layer_forward(l1, subnetwork, out2);
            l1.get_layer_params().host()[i] = oldval-eps;
            impl::call_layer_forward(l1, subnetwork, out3);
            l1.get_layer_params().host()[i] = oldval;

            // Compute a reference derivative via a central differences approximation and
            // compare it to the one output by the layer and make sure they match.
            double reference_derivative = (dot(out2,input_grad)-dot(out3, input_grad))/(2*eps);
            double output_derivative = params_grad.host()[i];
            double relative_error = (reference_derivative - output_derivative)/(reference_derivative + 1e-100);
            if (std::abs(relative_error) > 0.01)
            {
                using namespace std;
                sout << "Gradient error in parameter #" << i <<".  Relative error: "<< relative_error << endl;
                sout << "expected derivative: " << reference_derivative << endl;
                sout << "output derivative:   " << output_derivative << endl;
                return layer_test_results(sout.str()); 
            }

        }

        // ==================================================================
        // now validate the data gradients
        for (unsigned long i = 0; i < num_data_inputs; ++i)
        {
            const float oldval = subnetwork.get_output_element(i);
            float eps = oldval*base_eps;
            if (eps == 0)
                eps = base_eps;
            subnetwork.get_output_element(i) = oldval+eps;
            impl::call_layer_forward(l, subnetwork, out2);
            subnetwork.get_output_element(i) = oldval-eps;
            impl::call_layer_forward(l, subnetwork, out3);
            subnetwork.get_output_element(i) = oldval;

            // Compute a reference derivative via a central differences approximation and
            // compare it to the one output by the layer and make sure they match.
            double reference_derivative = (dot(out2,input_grad)-dot(out3, input_grad))/(2*eps);
            double output_derivative = subnetwork.get_gradient_input_element(i);
            if (!impl::is_inplace_layer(l,subnetwork))
                output_derivative -= initial_gradient_input[i];
            double relative_error = (reference_derivative - output_derivative)/(reference_derivative + 1e-100);
            if (std::abs(relative_error) > 0.01)
            {
                using namespace std;
                sout << "Gradient error in data variable #" << i <<".  Relative error: "<< relative_error << endl;
                sout << "expected derivative: " << reference_derivative << endl;
                sout << "output derivative:   " << output_derivative << endl;
                return layer_test_results(sout.str()); 
            }
        }

        return layer_test_results();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_CORE_H_


