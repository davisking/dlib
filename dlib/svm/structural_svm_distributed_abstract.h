// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STRUCTURAL_SVM_DISTRIBUTeD_ABSTRACT_H__
#ifdef DLIB_STRUCTURAL_SVM_DISTRIBUTeD_ABSTRACT_H__


#include "structural_svm_problem_abstract.h"
#include "../optimization/optimization_oca_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class svm_struct_processing_node : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        template <
            typename T,
            typename U 
            >
        svm_struct_processing_node (
            const structural_svm_problem<T,U>& problem,
            unsigned short port,
            unsigned short num_threads
        );
        /*!
        !*/
    };

// ----------------------------------------------------------------------------------------

    class svm_struct_controller_node : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        svm_struct_controller_node (
        );
        /*!
        !*/

        void set_epsilon (
            double eps
        );
        /*!
        !*/

        double get_epsilon (
        ) const;
        /*!
        !*/

        void be_verbose (
        );
        /*!
        !*/

        void be_quiet(
        );
        /*!
        !*/

        double get_c (
        ) const;
        /*!
        !*/

        void set_c (
            double C
        );
        /*!
        !*/

        void add_processing_node (
            const std::string& ip,
            unsigned short port
        );
        /*!
        !*/

        unsigned long get_num_processing_nodes (
        ) const;
        /*!
        !*/

        void remove_processing_nodes (
        );
        /*!
        !*/

        template <typename matrix_type>
        double operator() (
            const oca& solver,
            matrix_type& w
        ) const;
        /*!
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_DISTRIBUTeD_ABSTRACT_H__


