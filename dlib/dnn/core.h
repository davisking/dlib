// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_CORE_H_
#define DLIB_DNn_CORE_H_

#include "core_abstract.h"
#include "tensor.h"
#include "solvers.h"
#include <iterator>
#include <memory>
#include <type_traits>
#include "../statistics.h"
#include "../rand.h"
#include <utility>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    // Tell us if T is one of the special layer types (i.e. add_layer, add_loss, add_tag,
    // or add_skip).
    template <typename T> struct is_layer_type : std::false_type {};
    template <typename T> struct is_loss_layer_type : std::false_type {};

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

        private:
            T item;
            sstack<T,N-1> data;
    };

    template <typename T>
    class sstack<T,1> // base case of recursive definition.
    {
    public:
        sstack() {}
        explicit sstack(const T& item_) : item(item_) {}

        const T& top() const { return item; }
        T& top() { return item; }

        size_t size() const { return 1; }
    private:
        T item;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace dimpl
    {
        template <typename T, typename enabled=void>
        class sub_net_wrapper
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a tool that makes an add_layer or add_loss object
                    expose only the part of its interface defined by the SUB_NET
                    type in layers_abstract.h.  This way, when we pass sub network
                    objects to the layer callbacks those callbacks won't be able to 
                    interact with the sub networks in a way other than specified 
                    by the SUB_NET interface spec.
            !*/

        public:
            sub_net_wrapper(T& l_) {}
            // Nothing here because in this case T is one of the input layer types 
            // that doesn't have anything in it.
        };

        template <typename T>
        class sub_net_wrapper<T,typename std::enable_if<is_layer_type<T>::value>::type>
        {

        public:
            typedef T wrapped_type;
            const static size_t num_layers = T::num_layers;

            sub_net_wrapper(T& l_) : l(l_),sub(l.sub_net()) {}

            const tensor& get_output() const { return l.get_output(); }
            tensor& get_gradient_input() { return l.get_gradient_input(); }

            const sub_net_wrapper<typename T::sub_net_type>& sub_net() const { sub; }
            sub_net_wrapper<typename T::sub_net_type>& sub_net() { sub; }

        private:
            T& l;
            sub_net_wrapper<typename T::sub_net_type> sub;
        };
    }

    template <typename LAYER_DETAILS, typename SUB_NET, typename enabled = void>
    class add_layer;

    template <typename T, typename U>
    struct is_layer_type<add_layer<T,U>> : std::true_type {};

    template <typename LAYER_DETAILS, typename SUB_NET>
    class add_layer<LAYER_DETAILS,SUB_NET, 
            typename std::enable_if<is_layer_type<SUB_NET>::value>::type>
    {
    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef SUB_NET sub_net_type;
        typedef typename sub_net_type::input_type input_type;
        const static size_t num_layers = sub_net_type::num_layers + 1;
        const static unsigned int sample_expansion_factor = sub_net_type::sample_expansion_factor;

        add_layer(
        ):
            this_layer_setup_called(false),
            gradient_input_is_stale(true)
        {
        }

        add_layer(const add_layer&) = default;
        add_layer(add_layer&&) = default;
        add_layer& operator=(add_layer&&) = default;
        add_layer& operator=(const add_layer&) = default;

        template <typename T, typename U, typename E>
        friend class add_layer;

        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U, typename E>
        add_layer(
            const add_layer<T,U,E>& item
        ) :
            sub_network(item.sub_net()),
            details(item.layer_details()), 
            this_layer_setup_called(item.this_layer_setup_called),
            gradient_input_is_stale(item.gradient_input_is_stale),
            x_grad(item.x_grad),
            cached_output(item.cached_output)
        {
        }

        template <typename ...T>
        add_layer(
            const LAYER_DETAILS& layer_det, 
            T&& ...args
        ) : 
            details(layer_det), 
            sub_network(std::forward<T>(args)...),
            this_layer_setup_called(false),
            gradient_input_is_stale(true)
        {
        }

        template <typename ...T>
        add_layer(
            LAYER_DETAILS&& layer_det, 
            T&& ...args
        ) : 
            details(std::move(layer_det)), 
            sub_network(std::forward<T>(args)...),
            this_layer_setup_called(false),
            gradient_input_is_stale(true)
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        {
            sub_network.to_tensor(begin,end,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        /*!
            ensures
                - runs [ibegin,iend) through the network and returns the results 
        !*/
        {
            to_tensor(ibegin,iend,temp_tensor);
            return forward(temp_tensor);
        }


        const tensor& operator() (const input_type& x)
        /*!
            ensures
                - runs a single x through the network and returns the output.
        !*/
        {
            return (*this)(&x, &x+1);
        }

        const tensor& forward(const tensor& x)
        {
            sub_network.forward(x);
            const dimpl::sub_net_wrapper<sub_net_type> wsub(sub_network);
            if (!this_layer_setup_called)
            {
                details.setup(wsub);
                this_layer_setup_called = true;
            }
            details.forward(wsub, cached_output);
            gradient_input_is_stale = true;
            return get_output();
        }

        const tensor& get_output() const { return cached_output; }
        tensor& get_gradient_input() 
        { 
            if (gradient_input_is_stale)
            {
                gradient_input_is_stale = false;
                x_grad.copy_size(get_output());
                x_grad = 0;
            }
            return x_grad; 
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        /*!
            requires
                - forward(x) was called to forward propagate x though the network.
                - x.num_samples() == get_gradient_input().num_samples()
                - get_gradient_input() == the gradient of the network with respect
                  to some loss.
        !*/
        {
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub_network);
            params_grad.copy_size(details.get_layer_params());
            params_grad = 0;
            details.backward(get_gradient_input(), wsub, static_cast<tensor&>(params_grad));
            // Don't try to adjust the parameters if this layer doesn't have any.
            if (params_grad.size() != 0)
                solvers.top()(details, static_cast<const tensor&>(params_grad));
            sub_network.update(x, solvers.pop());
        }

        const sub_net_type& sub_net() const { return sub_network; }
        sub_net_type& sub_net() { return sub_network; }

        const layer_details_type& layer_details() const { return details; } 
        layer_details_type& layer_details() { return details; } 

        void clean()
        {
            x_grad.clear();
            cached_output.clear();
            params_grad.clear();
            temp_tensor.clear();
            gradient_input_is_stale = true;
            sub_network.clean();
        }

    private:


        sub_net_type sub_network;
        LAYER_DETAILS details;
        bool this_layer_setup_called;
        bool gradient_input_is_stale;
        resizable_tensor x_grad;
        resizable_tensor cached_output; 

        // The following 2 objects don't logically contribute to the state of this class.
        // They are only here to prevent them from being reallocated over and over in
        // member functions.
        resizable_tensor params_grad; 
        resizable_tensor temp_tensor;

    };

// ----------------------------------------------------------------------------------------

    template <typename LAYER_DETAILS, typename INPUT_LAYER, typename enabled>
    class add_layer
    {
    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef INPUT_LAYER sub_net_type;
        typedef typename INPUT_LAYER::input_type input_type;
        const static unsigned int sample_expansion_factor = INPUT_LAYER::sample_expansion_factor;
        const static size_t num_layers = 1;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_layer(
        ): 
            this_layer_setup_called(false),
            gradient_input_is_stale(true) 
        {}

        add_layer(const add_layer&) = default;
        add_layer(add_layer&&) = default;
        add_layer& operator=(add_layer&&) = default;
        add_layer& operator=(const add_layer&) = default;

        template <typename T, typename U, typename E>
        friend class add_layer;

        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U, typename E>
        add_layer(
            const add_layer<T,U,E>& item
        ):
            input_layer(item.sub_net()),
            details(item.layer_details()),
            this_layer_setup_called(item.this_layer_setup_called),
            gradient_input_is_stale(item.gradient_input_is_stale),
            x_grad(item.x_grad),
            cached_output(item.cached_output)
        {
        }

        add_layer(
            const LAYER_DETAILS& layer_det
        ) : 
            details(layer_det), 
            this_layer_setup_called(false),
            gradient_input_is_stale(true) 
        {}

        add_layer(
            LAYER_DETAILS&& layer_det
        ) : 
            details(std::move(layer_det)), 
            this_layer_setup_called(false),
            gradient_input_is_stale(true) 
        {}

        add_layer(
            LAYER_DETAILS layer_det, 
            INPUT_LAYER il
        ) : 
            details(layer_det),
            input_layer(il),
            this_layer_setup_called(false),
            gradient_input_is_stale(true)
        {}

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        {
            input_layer.to_tensor(begin, end, data);
            // make sure the input layer's to_tensor() function is implemented properly.
            DLIB_CASSERT(std::distance(begin,end)*sample_expansion_factor == data.num_samples(),"");
            data.async_copy_to_device();
        }


        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        /*!
            ensures
                - runs [ibegin,iend) through the network and returns the results 
        !*/
        {
            to_tensor(ibegin,iend,temp_tensor);
            return forward(temp_tensor);
        }


        const tensor& operator() (const input_type& x)
        /*!
            ensures
                - runs a single x through the network and returns the output.
        !*/
        {
            return (*this)(&x, &x+1);
        }

        const tensor& forward (const tensor& x)
        /*!
            requires
                - x.num_samples() is a multiple of sample_expansion_factor.
        !*/
        {
            DLIB_CASSERT(x.num_samples()%sample_expansion_factor == 0,"");
            sub_net_wrapper wsub(x, grad_final_ignored);
            if (!this_layer_setup_called)
            {
                details.setup(wsub);
                this_layer_setup_called = true;
            }
            details.forward(wsub, cached_output);
            gradient_input_is_stale = true;
            return get_output();
        }

        const tensor& get_output() const { return cached_output; }
        tensor& get_gradient_input() 
        { 
            if (gradient_input_is_stale)
            {
                gradient_input_is_stale = false;
                x_grad.copy_size(get_output());
                x_grad = 0;
            }
            return x_grad; 
        }


        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        /*!
            requires
                - x.num_samples() is a multiple of sample_expansion_factor.
                - forward(x) was called to forward propagate x though the network.
                - x.num_samples() == get_gradient_input().num_samples()
        !*/
        {
            sub_net_wrapper wsub(x, grad_final_ignored);
            params_grad.copy_size(details.get_layer_params());
            params_grad = 0;
            details.backward(get_gradient_input(), wsub, static_cast<tensor&>(params_grad));
            // Don't try to adjust the parameters if this layer doesn't have any.
            if (params_grad.size() != 0)
                solvers.top()(details, static_cast<const tensor&>(params_grad));
        }

        const sub_net_type& sub_net() const { return input_layer; } 
        sub_net_type& sub_net() { return input_layer; } 

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

    private:

        class sub_net_wrapper
        {
        public:
            sub_net_wrapper(const tensor& x_, resizable_tensor& grad_final_ignored_) :
                x(x_), grad_final_ignored(grad_final_ignored_) {}

            const tensor& get_output() const { return x; }
            tensor& get_gradient_input() 
            { 
                // It doesn't matter what values are in this tensor but client code will
                // always assume it's the same dimension as the output so make sure that is
                // the case.
                grad_final_ignored.copy_size(x);
                return grad_final_ignored; 
            }

        private:
            const tensor& x;
            resizable_tensor& grad_final_ignored;
        };

        sub_net_type input_layer;
        LAYER_DETAILS details;
        bool this_layer_setup_called;
        bool gradient_input_is_stale;
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

    template <unsigned long ID, typename SUB_NET>
    class add_tag
    {
    public:
        typedef SUB_NET sub_net_type;
        typedef typename sub_net_type::input_type input_type;
        const static size_t num_layers = sub_net_type::num_layers + 1;
        const static unsigned int sample_expansion_factor = sub_net_type::sample_expansion_factor;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_tag() = default;
        add_tag(const add_tag&) = default;
        add_tag(add_tag&&) = default;
        add_tag& operator=(add_tag&&) = default;
        add_tag& operator=(const add_tag&) = default;

        template <typename T>
        add_tag(
            const add_tag<ID,T>& item
        ) : sub_network(item.sub_net())
        {}

        template <typename ...T>
        add_tag(
            T ...args
        ) : 
            sub_network(std::move(args)...) 
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        {
            sub_network.to_tensor(begin,end,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            return sub_network(ibegin,iend);
        }

        const tensor& operator() (const input_type& x)
        {
            return sub_network(x);
        }

        const tensor& forward(const tensor& x)
        {
            return sub_network.forward(x);
        }

        const tensor& get_output() const { return sub_network.get_output(); }

        tensor& get_gradient_input() 
        { 
            return sub_network.get_gradient_input();
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        {
            sub_network.update(x,solvers.pop());
        }

        const sub_net_type& sub_net() const { return sub_network; }
        sub_net_type& sub_net() { return sub_network; }

        void clean()
        {
            sub_network.clean();
        }

    private:

        sub_net_type sub_network;
    };

    template <unsigned long ID, typename U>
    struct is_layer_type<add_tag<ID,U>> : std::true_type {};


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename LOSS_DETAILS, typename SUB_NET>
    class add_loss;

    class no_label_type
    {
    private:
        // We don't want anyone making these no_label_type objects.  They are here only to
        // allow add_loss::label_type and dnn_trainer::label_type to exist which voids
        // needing to overload add_loss and dnn_trainer for supervised an unsupervised
        // losses.  It also can be a type to use in template metaprogramming to indicate
        // "no label".  So here we make the constructor private with the exception that
        // add_loss objects can make it (again, just to simplify add_loss's
        // implementation).
        no_label_type()=default;
        template <typename LOSS_DETAILS, typename SUB_NET> friend class add_loss;
    };

// ----------------------------------------------------------------------------------------

    template <typename LOSS_DETAILS, typename SUB_NET>
    class add_loss
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
        typedef SUB_NET sub_net_type;
        typedef typename sub_net_type::input_type input_type;
        // Note that the loss layer doesn't count as an additional layer.
        const static size_t num_layers = sub_net_type::num_layers;
        const static unsigned int sample_expansion_factor = sub_net_type::sample_expansion_factor;
        typedef typename get_loss_layer_label_type<LOSS_DETAILS>::type label_type;

        static_assert(is_layer_type<SUB_NET>::value, "SUB_NET must be of type add_layer, add_skip, or add_tag."); 
        static_assert(sample_expansion_factor == LOSS_DETAILS::sample_expansion_factor,
            "The loss layer and input layer must agree on the sample_expansion_factor.");


        add_loss() = default;
        add_loss(const add_loss&) = default;
        add_loss(add_loss&&) = default;
        add_loss& operator=(add_loss&&) = default;
        add_loss& operator=(const add_loss&) = default;

        template <typename T, typename U>
        add_loss(
            const add_loss<T,U>& item
        ) : 
            loss(item.loss_details()),
            sub(item.sub_net())
        {}

        template <typename ...T>
        add_loss(
            const LOSS_DETAILS& layer_det, 
            T&& ...args
        ) : 
            loss(layer_det), 
            sub(std::forward<T>(args)...)
        {
        }

        template <typename ...T>
        add_loss(
            LOSS_DETAILS&& layer_det, 
            T&& ...args
        ) : 
            loss(std::move(layer_det)), 
            sub(std::forward<T>(args)...)
        {
        }

        template <typename ...T>
        add_loss(
            T ...args
        ) : 
            sub(std::move(args)...)
        {
        }

        template <typename input_iterator, typename output_iterator>
        void operator() (
            input_iterator ibegin,
            input_iterator iend,
            output_iterator obegin
        )
        /*!
            requires
                - obegin == iterator pointing to the start of a range of distance(ibegin,iend)
                  elements.
            ensures
                - runs [ibegin,iend) through the network and writes the output to the range at obegin.
        !*/
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            loss.to_label(sub, obegin);
        }


        const label_type& operator() (const input_type& x)
        /*!
            ensures
                - runs a single x through the network and returns the output.
        !*/
        {
            (*this)(&x, &x+1, &temp_label);
            return temp_label;
        }


        template <typename input_iterator, typename label_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin 
        )
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub);
            return loss.compute_loss(temp_tensor, lbegin, wsub);
        }

        template <typename input_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub);
            return loss.compute_loss(temp_tensor, wsub);
        }

        template <typename input_iterator, typename label_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin,
            sstack<solver_type,num_layers>& solvers
        )
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub);
            double l = loss.compute_loss(temp_tensor, lbegin, wsub);
            sub.update(temp_tensor, solvers);
            return l;
        }

        template <typename input_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            sstack<solver_type,num_layers>& solvers
        )
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub);
            double l = loss.compute_loss(temp_tensor, wsub);
            sub.update(temp_tensor, solvers);
            return l;
        }

        const sub_net_type& sub_net() const { return sub; }
        sub_net_type& sub_net() { return sub; }
        const loss_details_type& loss_details() const { return loss; }
        loss_details_type& loss_details() { return loss; }

        void clean (
        )
        /*!
            ensures
                - Causes the network to forget about everything but its parameters.  
                  That is, for each layer we will have:
                    - get_output().num_samples() == 0
                    - get_gradient_input().num_samples() == 0
                  However, running new input data though this network will still have the
                  same output it would have had regardless of any calls to clean().
                  Finally, the purpose of clean() is to compact the network object prior to
                  saving it to disk so that it takes up less space and the IO is quicker.
        !*/
        {
            temp_tensor.clear();
            sub.clear();
        }

    private:

        loss_details_type loss;
        sub_net_type sub;

        // These two objects don't logically contribute to the state of this object.  They
        // are here to prevent them from being reallocated over and over.
        label_type temp_label;
        resizable_tensor temp_tensor;
    };


    template <typename T, typename U>
    struct is_loss_layer_type<add_loss<T,U>> : std::true_type {};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <unsigned int i, typename T>
        struct layer_helper
        {
            static T& makeT();
            using next_type = typename std::remove_reference<decltype(makeT().sub_net())>::type;
            using type = typename layer_helper<i-1,next_type>::type;
            static type& layer(T& n)
            {
                return layer_helper<i-1,next_type>::layer(n.sub_net());
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
            using next_type = typename std::remove_reference<decltype(makeT().sub_net())>::type;
            using type = typename layer_helper_match<Match,next_type,i>::type;
            static type& layer(T& n)
            {
                return layer_helper_match<Match,next_type,i>::layer(n.sub_net());
            }
        };
        // This overload catches add_layer and add_loss templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const T,const  Match<typename T::sub_net_type>>::value>::type>
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
        // This overload catches sub_net_wrapper templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const typename T::wrapped_type, 
                                                 const Match<typename T::wrapped_type::sub_net_type>>::value>::type>
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

    template <template<typename> class TAG_TYPE, typename SUB_NET>
    class add_skip
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object draws its inputs from layer<TAG_TYPE>(SUB_NET())
                and performs the identity transform.
        !*/
    public:
        typedef SUB_NET sub_net_type;
        typedef typename sub_net_type::input_type input_type;
        const static size_t num_layers = sub_net_type::num_layers + 1;
        const static unsigned int sample_expansion_factor = sub_net_type::sample_expansion_factor;
        static_assert(sample_expansion_factor >= 1,
            "The input layer can't produce fewer output tensors than there are inputs.");

        add_skip() = default;
        add_skip(const add_skip&) = default;
        add_skip(add_skip&&) = default;
        add_skip& operator=(add_skip&&) = default;
        add_skip& operator=(const add_skip&) = default;

        template <typename T>
        add_skip(
            const add_skip<TAG_TYPE,T>& item
        ) : sub_network(item.sub_net())
        {}

        template <typename ...T>
        add_skip(
            T ...args
        ) : 
            sub_network(std::move(args)...) 
        {
        }

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        {
            sub_network.to_tensor(begin,end,data);
        }

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        )
        {
            sub_network(ibegin,iend);
            return layer<TAG_TYPE>(sub_network).get_output();
        }

        const tensor& operator() (const input_type& x)
        {
            sub_network(x);
            return layer<TAG_TYPE>(sub_network).get_output();
        }

        const tensor& forward(const tensor& x)
        {
            sub_network.forward(x);
            return layer<TAG_TYPE>(sub_network).get_output();
        }

        const tensor& get_output() const 
        { 
            return layer<TAG_TYPE>(sub_network).get_output();
        }

        tensor& get_gradient_input() 
        { 
            return layer<TAG_TYPE>(sub_network).get_gradient_input();
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        {
            sub_network.update(x,solvers.pop());
        }

        const sub_net_type& sub_net() const 
        { 
            return sub_network; 
        }

        sub_net_type& sub_net() 
        { 
            return sub_network; 
        }

        void clean()
        {
            sub_network.clean();
        }

    private:

        sub_net_type sub_network;
    };
    template <template<typename> class T, typename U>
    struct is_layer_type<add_skip<T,U>> : std::true_type {};

    template <typename SUB_NET> using tag1  = add_tag< 1, SUB_NET>;
    template <typename SUB_NET> using tag2  = add_tag< 2, SUB_NET>;
    template <typename SUB_NET> using tag3  = add_tag< 3, SUB_NET>;
    template <typename SUB_NET> using tag4  = add_tag< 4, SUB_NET>;
    template <typename SUB_NET> using tag5  = add_tag< 5, SUB_NET>;
    template <typename SUB_NET> using tag6  = add_tag< 6, SUB_NET>;
    template <typename SUB_NET> using tag7  = add_tag< 7, SUB_NET>;
    template <typename SUB_NET> using tag8  = add_tag< 8, SUB_NET>;
    template <typename SUB_NET> using tag9  = add_tag< 9, SUB_NET>;
    template <typename SUB_NET> using tag10 = add_tag<10, SUB_NET>;

    template <typename SUB_NET> using skip1  = add_skip< tag1, SUB_NET>;
    template <typename SUB_NET> using skip2  = add_skip< tag2, SUB_NET>;
    template <typename SUB_NET> using skip3  = add_skip< tag3, SUB_NET>;
    template <typename SUB_NET> using skip4  = add_skip< tag4, SUB_NET>;
    template <typename SUB_NET> using skip5  = add_skip< tag5, SUB_NET>;
    template <typename SUB_NET> using skip6  = add_skip< tag6, SUB_NET>;
    template <typename SUB_NET> using skip7  = add_skip< tag7, SUB_NET>;
    template <typename SUB_NET> using skip8  = add_skip< tag8, SUB_NET>;
    template <typename SUB_NET> using skip9  = add_skip< tag9, SUB_NET>;
    template <typename SUB_NET> using skip10 = add_skip<tag10, SUB_NET>;

// ----------------------------------------------------------------------------------------

    namespace timpl
    {
        void fill_with_gassuan_random_numbers (
            tensor& t,
            dlib::rand& rnd,
            double sigma = 1
        )
        {
            float* data = t.host();
            for (size_t i = 0; i < t.size(); ++i)
                data[i] = rnd.get_random_gaussian()*sigma;
        }

        class test_layer_sub_net 
        {
        public:
            test_layer_sub_net (
                dlib::rand& rnd_
            ) : rnd(rnd_) 
            {
                // Output and gradient_input have to have the same dimensions in each
                // layer.
                const long num_samples = rnd.get_random_32bit_number()%4+3;
                const long nr = rnd.get_random_32bit_number()%4+2;
                const long nc = rnd.get_random_32bit_number()%4+2;
                const long k  = rnd.get_random_32bit_number()%4+2;

                output.set_size(num_samples, nr, nc, k);
                gradient_input.set_size(num_samples, nr, nc, k);

                // Use a non-zero initial gradient to make sure the layers add to it
                // rather than assign and blow away the initial value.
                fill_with_gassuan_random_numbers(gradient_input, rnd, 0.01);

                fill_with_gassuan_random_numbers(output, rnd);
            }


            const tensor& get_output() const { return output; }
            const test_layer_sub_net& sub_net() const { init_sub(); return *sub; }

            tensor& get_gradient_input() { return gradient_input; }
            test_layer_sub_net& sub_net() { init_sub(); return *sub; }



            unsigned long count_outputs() const
            {
                if (sub)
                    return sub->count_outputs() + output.size();
                else
                    return output.size();
            }

            float& get_output_element(unsigned long i)
            {
                if (i < output.size())
                    return output.host()[i];
                else
                    return sub_net().get_output_element(i-output.size());
            }

            float get_gradient_input_element(unsigned long i) const
            {
                if (i < gradient_input.size())
                    return gradient_input.host()[i];
                else
                    return sub_net().get_gradient_input_element(i-gradient_input.size());
            }


        private:
            // We lazily initialize sub-layers as needed when someone tries to call
            // sub_net()
            void init_sub() const
            {
                if (!sub)
                    sub.reset(new test_layer_sub_net(rnd));
            }

            dlib::rand& rnd;
            mutable std::unique_ptr<test_layer_sub_net> sub;
            resizable_tensor output;
            resizable_tensor gradient_input;
        };


        void print_tensor(
            const tensor& a
        )
        {
            auto data = a.host();
            for (size_t i = 0; i < a.size(); ++i)
                std::cout << data[i] << " ";
            std::cout << std::endl;
        }
    }

    template <
        typename layer_details_type
        >
    void test_layer (
        layer_details_type l
    )
    {
        const float base_eps = 0.01;
        using namespace timpl;
        // Do some setup
        dlib::rand rnd;
        test_layer_sub_net sub(rnd);
        resizable_tensor output, out2, out3;
        // Run setup() and forward() as well to make sure any calls to sub_net() have
        // happened before we start assuming we know how many data elements there are
        // (since we do a lazy layer creation thing based on calls to sub_net() inside
        // test_layer_sub_net).
        l.setup(sub);
        l.forward(sub, output);

        resizable_tensor input_grad;
        input_grad.copy_size(output);
        std::cout << "output.num_samples(): "<< output.num_samples() << std::endl;
        fill_with_gassuan_random_numbers(input_grad, rnd);

        // The f() we are computing gradients of is this thing.  It's value at the current
        // parameter and data values is:
        std::cout << "f(data,params): " << dot(output, input_grad) << std::endl;

        // We are going to save a copy of the sub.get_gradient_input() data before we do
        // backpropagation since the backward() function is supposed to *add* to the
        // gradients rather than overwrite them.  We will use this saved data to check if
        // that is the case.
        const unsigned long num_data_inputs = sub.count_outputs();
        std::vector<float> initial_gradient_input(num_data_inputs);
        for (unsigned long i = 0; i < num_data_inputs; ++i)
            initial_gradient_input[i] = sub.get_gradient_input_element(i);


        // Now tell the layer to compute all the gradients.  In the rest of this function
        // we will just be checking that these gradients were computed correctly by
        // comparing them to a central differences approximation.
        resizable_tensor params_grad, random_noise;
        params_grad.copy_size(l.get_layer_params());
        random_noise.copy_size(l.get_layer_params());
        randomize_parameters(random_noise, 5, rnd);
        params_grad = random_noise;
        l.backward(input_grad, sub, params_grad);

        running_stats<double> rs_param, rs_data;

        // ==================================================================
        // first validate the way the parameter gradients are computed
        for (long i = 0; i < params_grad.size(); ++i)
        {
            layer_details_type l1(l);

            float eps = l1.get_layer_params().host()[i]*base_eps;
            if (eps == 0)
                eps = base_eps;
            const float oldval = l1.get_layer_params().host()[i];
            l1.get_layer_params().host()[i] = oldval+eps;
            l1.forward(sub, out2);
            l1.get_layer_params().host()[i] = oldval-eps;
            l1.forward(sub, out3);

            // Compute a reference derivative via a central differences approximation and
            // compare it to the one output by the layer and make sure they match.
            double reference_derivative = (dot(out2,input_grad)-dot(out3, input_grad))/(2*eps);
            double output_derivative = params_grad.host()[i]-random_noise.host()[i];
            double relative_error = (reference_derivative - output_derivative)/(reference_derivative + 1e-100);
            if (std::abs(relative_error) > 0.01)
            {
                using namespace std;
                cout << "PARAM ERROR: "<< relative_error << endl;
                cout << "   reference_derivative:   " << reference_derivative << endl;
                cout << "   output_derivative: " << output_derivative << endl;
            }

            rs_param.add(std::abs(relative_error));
        }

        // ==================================================================
        // now validate the data gradients
        for (unsigned long i = 0; i < num_data_inputs; ++i)
        {
            const float oldval = sub.get_output_element(i);
            float eps = oldval*base_eps;
            if (eps == 0)
                eps = base_eps;
            sub.get_output_element(i) = oldval+eps;
            l.forward(sub, out2);
            sub.get_output_element(i) = oldval-eps;
            l.forward(sub, out3);

            // Compute a reference derivative via a central differences approximation and
            // compare it to the one output by the layer and make sure they match.
            double reference_derivative = (dot(out2,input_grad)-dot(out3, input_grad))/(2*eps);
            double output_derivative = sub.get_gradient_input_element(i)-initial_gradient_input[i];
            double relative_error = (reference_derivative - output_derivative)/(reference_derivative + 1e-100);
            if (std::abs(relative_error) > 0.01)
            {
                using namespace std;
                cout << "DATA ERROR: "<< relative_error << endl;
                cout << "   reference_derivative:   " << reference_derivative << endl;
                cout << "   output_derivative: " << output_derivative << endl;
            }
            rs_data.add(std::abs(relative_error));
        }

        using namespace std;
        if (rs_param.current_n() > 1)
        {
            cout << "rs_param.mean():   " << rs_param.mean() << endl;
            cout << "rs_param.stddev(): " << rs_param.stddev() << endl;
            cout << "rs_param.max():    " << rs_param.max() << endl;
        }
        if (rs_data.current_n() > 1)
        {
            cout << "rs_data.mean():    " << rs_data.mean() << endl;
            cout << "rs_data.stddev():  " << rs_data.stddev() << endl;
            cout << "rs_data.max():     " << rs_data.max() << endl;
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename net_type, 
        typename solver_type = sgd
        >
    class dnn_trainer
    {
    public:

        static_assert(is_loss_layer_type<net_type>::value, 
            "The last layer in a network must be a loss layer.");

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;

        dnn_trainer()
        {}

        explicit dnn_trainer(const net_type& net_) : net(net_) {}

        dnn_trainer(
            const net_type& net_, 
            const solver_type& solver_
        ) : net(net_), solvers(solver_) {}

        const net_type& get_net (
        ) const { return net; }

        void set_net (
            const net_type& net_
        ) 
        { 
            return net = net_; 
        }

        void set_solver (
            const solver_type& solver_
        ) 
        { 
            solvers = solver_; 
        }

        const sstack<solver_type,net_type::num_layers>& get_solvers (
        ) const { return solvers; }

        sstack<solver_type,net_type::num_layers>& get_solvers (
        ) { return solvers; }

        const net_type& train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ) 
        /*!
            requires
                - data.size() == labels.size()
        !*/
        {
            const int batch_size = 11;
            for (int iter = 0; iter < 300; ++iter)
            {
                for (unsigned long i = 0; i < data.size(); i+=batch_size)
                {
                    // TODO, move the contents of update() here and do the alternating tensor
                    // loading thing to hide GPU transfer latency.
                    std::cout << "loss: "<<net.update(data.begin()+i, 
                        data.begin()+std::min(i+batch_size,i+data.size()-1), 
                        labels.begin()+i,
                        solvers) << std::endl;
                }
            }
            return net;
        }

        const net_type& train (
            const std::vector<input_type>& data
        ) 
        /*!
            ensures
                - trains an auto-encoder
        !*/
        {
            const bool has_unsupervised_loss = std::is_same<no_label_type, label_type>::value; 
            static_assert(has_unsupervised_loss, 
                "You can only call this version of train() when using an unsupervised loss.");

            const int batch_size = 10;
            for (int iter = 0; iter < 300; ++iter)
            {
                for (unsigned long i = 0; i < data.size(); i+=batch_size)
                {
                    // TODO, move the contents of update() here and do the alternating tensor
                    // loading thing to hide GPU transfer latency.
                    std::cout << "loss: "<<net.update(data.begin()+i, 
                        data.begin()+std::min(i+batch_size,i+data.size()-1), 
                        solvers) << std::endl;
                }
            }
            return net;
        }

    private:

        net_type net;
        sstack<solver_type,net_type::num_layers> solvers;
    };

    // TODO, make dnn_trainer serializable. 

// ----------------------------------------------------------------------------------------

}

#endif // #define DLIB_DNn_CORE_H_


