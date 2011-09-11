// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STRUCTURAL_OBJECT_DETECTION_TRAiNER_H_ABSTRACT__
#ifdef DLIB_STRUCTURAL_OBJECT_DETECTION_TRAiNER_H_ABSTRACT__

#include "structural_svm_object_detection_problem_abstract.h"
#include "../image_processing/object_detector_abstract.h"
#include "../image_processing/box_overlap_testing_abstract.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type = test_box_overlap
        >
    class structural_object_detection_trainer : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:
        typedef double scalar_type;
        typedef default_memory_manager mem_manager_type;
        typedef object_detector<image_scanner_type,overlap_tester_type> trained_function_type;


        explicit structural_object_detection_trainer (
            const image_scanner_type& scanner
        );
        /*!
            requires
                - scanner.get_num_detection_templates() > 0
            ensures
                - #get_c() == 1
                - this object isn't verbose
                - #get_epsilon() == 0.3
                - #get_num_threads() == 2
                - #get_max_cache_size() == 40
                - #get_overlap_eps() == 0.5
                - #get_loss_per_missed_target() == 1
                - #get_loss_per_false_alarm() == 1
                - This object will attempt to learn a model for the given
                  scanner object when train() is called.
        !*/

        void set_overlap_tester (
            const overlap_tester_type& tester
        );

        overlap_tester_type get_overlap_tester (
        ) const;

        void set_num_threads (
            unsigned long num
        );

        unsigned long get_num_threads (
        ) const;

        void set_epsilon (
            scalar_type eps
        );
        /*!
            requires
                - eps > 0
            ensures
                - #get_epsilon() == eps
        !*/

        const scalar_type get_epsilon (
        ) const;

        void set_max_cache_size (
            unsigned long max_size
        );

        unsigned long get_max_cache_size (
        ) const;

        void be_verbose (
        );

        void be_quiet (
        );

        void set_oca (
            const oca& item
        );

        const oca get_oca (
        ) const;

        void set_c (
            scalar_type C
        );
        /*!
            requires
                - C > 0
            ensures
                - #get_c() = C
        !*/

        const scalar_type get_c (
        ) const;

        void set_overlap_eps (
            double eps
        );
        /*!
            requires
                - 0 < eps < 1
            ensures
                - #get_overlap_eps() == eps
        !*/

        double get_overlap_eps (
        ) const;

        double get_loss_per_missed_target (
        ) const;

        void set_loss_per_missed_target (
            double loss
        );
        /*!
            requires
                - loss > 0
            ensures
                - #get_loss_per_missed_target() == loss
        !*/

        double get_loss_per_false_alarm (
        ) const;

        void set_loss_per_false_alarm (
            double loss
        );
        /*!
            requires
                - loss > 0
            ensures
                - #get_loss_per_false_alarm() == loss
        !*/

        template <
            typename image_array_type
            >
        const trained_function_type train (
            const image_array_type& images,
            const std::vector<std::vector<rectangle> >& truth_rects
        ) const;
        /*!
            requires
                - is_learning_problem(images, truth_rects) == true
                - it must be valid to pass images[0] into the image_scanner_type::load() method.
            ensures
                - 
        !*/
    }; 

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_OBJECT_DETECTION_TRAiNER_H_ABSTRACT__


