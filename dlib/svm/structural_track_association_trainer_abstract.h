// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STRUCTURAL_TRACK_ASSOCIATION_TRAnER_ABSTRACT_H__
#ifdef DLIB_STRUCTURAL_TRACK_ASSOCIATION_TRAnER_ABSTRACT_H__

#include "track_association_function_abstract.h"
#include "structural_assignment_trainer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename detection_type_,
        typename detection_id_type_ = unsigned long 
        >
    class structural_track_association_trainer
    {
    public:
        typedef detection_type_ detection_type;
        typedef typename detection_type::track_type track_type;
        typedef detection_id_type_ detection_id_type;
        typedef std::pair<detection_type, detection_id_type> labeled_detection;
        typedef std::vector<labeled_detection> detections_at_single_time_step;
        // This type logically represents an entire track history
        typedef std::vector<detections_at_single_time_step> sample_type;

        typedef track_association_function<detection_type> trained_function_type;

        structural_track_association_trainer (
        );  
        /*!
            C = 100;
            verbose = false;
            eps = 0.1;
            num_threads = 2;
            max_cache_size = 5;
            learn_nonnegative_weights = false;
        !*/

        void set_num_threads (
            unsigned long num
        );

        unsigned long get_num_threads (
        ) const;

        void set_epsilon (
            double eps
        );

        double get_epsilon (
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
            double C
        );

        double get_c (
        ) const;

        bool learns_nonnegative_weights (
        ) const; 
       
        void set_learns_nonnegative_weights (
            bool value
        );

        const track_association_function<detection_type> train (  
            const std::vector<sample_type>& samples
        ) const;
        /*!
            requires
                - is_track_association_problem(samples) == true
            ensures
                - 
        !*/

        const track_association_function<detection_type> train (  
            const sample_type& sample
        ) const;
        /*!
            requires
                - is_track_association_problem(samples) == true
            ensures
                - 
        !*/
    };

}

#endif // DLIB_STRUCTURAL_TRACK_ASSOCIATION_TRAnER_ABSTRACT_H__


