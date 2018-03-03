// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_KALMAN_FiLTER_ABSTRACT_Hh_
#ifdef DLIB_KALMAN_FiLTER_ABSTRACT_Hh_

#include "../serialize.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        long states,
        long measurements
        >
    class kalman_filter
    {
        /*!
            REQUIREMENTS ON states
                states > 0

            REQUIREMENTS ON measurements 
                measurements > 0

            WHAT THIS OBJECT REPRESENTS
                This object implements the Kalman filter, which is a tool for 
                recursively estimating the state of a process given measurements
                related to that process.  To use this tool you will have to 
                be familiar with the workings of the Kalman filter.  An excellent
                introduction can be found in the paper:
                    An Introduction to the Kalman Filter
                    by Greg Welch and Gary Bishop

        !*/

    public:

        kalman_filter(
        );
        /*!
            - #get_observation_model()    == 0
            - #get_transition_model()     == 0
            - #get_process_noise()        == 0
            - #get_measurement_noise()    == 0
            - #get_current_state()        == 0
            - #get_predicted_next_state() == 0
            - #get_current_estimation_error_covariance() == the identity matrix
        !*/

        void set_observation_model ( 
            const matrix<double,measurements,states>& H
        );
        /*!
            ensures
                - #get_observation_model() == H
        !*/

        void set_transition_model  ( 
            const matrix<double,states,states>& A
        );
        /*!
            ensures
                - #get_transition_model() == A
        !*/

        void set_process_noise     ( 
            const matrix<double,states,states>& Q
        );
        /*!
            ensures
                - #get_process_noise() == Q
        !*/

        void set_measurement_noise ( 
            const matrix<double,measurements,measurements>& R
        );
        /*!
            ensures
                - #get_measurement_noise() == R
        !*/

        void set_estimation_error_covariance ( 
            const matrix<double,states,states>& P
        ); 
        /*!
            ensures
                - #get_current_estimation_error_covariance() == P
                  (Note that you should only set this before you start filtering
                  since the Kalman filter will maintain the value of P on its own.
                  So only set this during initialization unless you are sure you
                  understand what you are doing.)
        !*/

        void set_state ( 
            const matrix<double,states,1>& xb
        ); 
        /*!
            ensures
                - This function can be used when the initial state is known, or if the
                  state needs to be corrected before the next update().
                - #get_predicted_next_state() == xb
                - If (update() hasn't been called yet) then 
                    - #get_current_state() == xb 
        !*/

        const matrix<double,measurements,states>& get_observation_model (
        ) const;
        /*!
            ensures
                - Returns the matrix "H" which relates process states x to measurements z.
                  The relation is linear, therefore, z = H*x.  That is, multiplying a
                  state by H gives the measurement you expect to observe for that state.
        !*/

        const matrix<double,states,states>& get_transition_model (
        ) const;
        /*!
            ensures
                - Returns the matrix "A" which determines how process states change over time.
                  The relation is linear, therefore, given a state vector x, the value you
                  expect it to have at the next time step is A*x.
        !*/

        const matrix<double,states,states>& get_process_noise (
        ) const;
        /*!
            ensures
                - returns the process noise covariance matrix.  You can think of this
                  covariance matrix as a measure of how wrong the assumption of
                  linear state transitions is. 
        !*/

        const matrix<double,measurements,measurements>& get_measurement_noise (
        ) const;
        /*!
            ensures
                - returns the measurement noise covariance matrix.  That is, when we
                  measure a state x we only obtain H*x corrupted by Gaussian noise.
                  The measurement noise is the covariance matrix of this Gaussian
                  noise which corrupts our measurements.
        !*/

        void update (
        );
        /*!
            ensures
                - propagates the current state estimate forward in time one
                  time step.  In particular:
                    - #get_current_state() == get_predicted_next_state()
                    - #get_predicted_next_state() == get_transition_model()*get_current_state()
                    - #get_current_estimation_error_covariance() == the propagated value of this covariance matrix
        !*/

        void update (
            const matrix<double,measurements,1>& z
        );
        /*!
            ensures
                - propagates the current state estimate forward in time one time step.  
                  Also applies a correction based on the given measurement z.  In particular:
                    - #get_current_state(), #get_predicted_next_state(), and
                      #get_current_estimation_error_covariance() are updated using the
                      Kalman filter method based on the new measurement in z.
        !*/

        const matrix<double,states,1>& get_current_state(
        ) const;
        /*!
            ensures
                - returns the current estimate of the state of the process.  This
                  estimate is based on all the measurements supplied to the update()
                  method.
        !*/

        const matrix<double,states,1>& get_predicted_next_state(
        ) const;
        /*!
            ensures
                - returns the next expected value of the process state.  
                - Specifically, returns get_transition_model()*get_current_state()
                  
        !*/

        const matrix<double,states,states>& get_current_estimation_error_covariance(
        ) const;
        /*!
            ensures
                - returns the current state error estimation covariance matrix.  
                  This matrix captures our uncertainty about the value of get_current_state().
        !*/

    };

// ----------------------------------------------------------------------------------------

    void serialize (
        const kalman_filter& item, 
        std::ostream& out 
    );   
    /*!
        provides serialization support 
    !*/

    void deserialize (
        kalman_filter& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class momentum_filter
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/
    public:

        momentum_filter(
            double meas_noise,
            double acc,
            double max_meas_dev
        ); 

        momentum_filter() = default; 

        double get_measurement_noise (
        ) const; 

        double get_typical_acceleration (
        ) const;

        double get_max_measurement_deviation (
        ) const;

        void reset(
        );

        double get_predicted_next_state(
        ) const;

        double operator()(
            const double val
        );

    };

    std::ostream& operator << (std::ostream& out, const momentum_filter& item);
    void serialize(const momentum_filter& item, std::ostream& out);
    void deserialize(momentum_filter& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    momentum_filter find_optimal_momentum_filter (
        const std::vector<std::vector<double>>& sequences,
        const double smoothness = 1
    );
    /*!
        requires
            - sequences.size() != 0
            - for all valid i: sequences[i].size() > 4
            - smoothness >= 0
    !*/

// ----------------------------------------------------------------------------------------

    momentum_filter find_optimal_momentum_filter (
        const std::vector<double>& sequence,
        const double smoothness = 1
    );
    /*!
        requires
            - sequence.size() > 4
            - smoothness >= 0
        ensures
            - performs: find_optimal_momentum_filter({1,sequence}, smoothness);
    !*/

// ----------------------------------------------------------------------------------------

    class rect_filter
    {
    public:
        rect_filter() = default;

        rect_filter(
            const momentum_filter& filt
        );

        drectangle operator()(
            const drectangle& r
        );

        drectangle operator()(
            const rectangle& r
        ); 

        const momentum_filter& get_left() const; 
        momentum_filter&       get_left();
        const momentum_filter& get_top() const; 
        momentum_filter&       get_top();
        const momentum_filter& get_right() const; 
        momentum_filter&       get_right();
        const momentum_filter& get_bottom() const;
        momentum_filter&       get_bottom(); 
    };

    void serialize(const rect_filter& item, std::ostream& out);
    void deserialize(rect_filter& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    rect_filter find_optimal_rect_filter (
        const std::vector<rectangle>& rects,
        const double smoothness = 1
    );
    /*!
        requires
            - rects.size() > 4
            - smoothness >= 0
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KALMAN_FiLTER_ABSTRACT_Hh_


