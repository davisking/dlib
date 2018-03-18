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
                This object is a simple tool for filtering a single scalar value that
                measures the location of a moving object that has some non-trivial
                momentum.  Importantly, the measurements are noisy and the object can
                experience sudden unpredictable accelerations.  To accomplish this
                filtering we use a simple Kalman filter with a state transition model of:

                    position_{i+1} = position_{i} + velocity_{i} 
                    velocity_{i+1} = velocity_{i} + some_unpredictable_acceleration

                and a measurement model of:
                    
                    measured_position_{i} = position_{i} + measurement_noise

                Where some_unpredictable_acceleration and measurement_noise are 0 mean Gaussian 
                noise sources with standard deviations of get_typical_acceleration() and
                get_measurement_noise() respectively.

                To allow for really sudden and large but infrequent accelerations, at each
                step we check if the current measured position deviates from the predicted
                filtered position by more than get_max_measurement_deviation()*get_measurement_noise() 
                and if so we adjust the filter's state to keep it within these bounds.
                This allows the moving object to undergo large unmodeled accelerations, far
                in excess of what would be suggested by get_typical_acceleration(), without
                then experiencing a long lag time where the Kalman filter has to "catch
                up" to the new position.
        !*/

    public:

        momentum_filter(
        ) = default; 
        /*!
            ensures
                - #get_measurement_noise() == 2
                - #get_typical_acceleration() == 0.1
                - #get_max_measurement_deviation() == 3
        !*/

        momentum_filter(
            double meas_noise,
            double acc,
            double max_meas_dev
        ); 
        /*!
            requires
                - meas_noise >= 0
                - acc >= 0
                - max_meas_dev >= 0
            ensures
                - #get_measurement_noise() == meas_noise
                - #get_typical_acceleration() == acc
                - #get_max_measurement_deviation() == max_meas_dev
        !*/


        double get_measurement_noise (
        ) const; 
        /*!
            ensures
                - Returns the standard deviation of the 0 mean Gaussian noise that corrupts
                  measurements of the moving object.
        !*/

        double get_typical_acceleration (
        ) const;
        /*!
            ensures
                - We assume that the moving object experiences random accelerations that
                  are distributed by 0 mean Gaussian noise with get_typical_acceleration()
                  standard deviation.
        !*/

        double get_max_measurement_deviation (
        ) const;
        /*!
            ensures
                - This object will never let the filtered location of the object deviate
                  from the measured location by much more than
                  get_max_measurement_deviation()*get_measurement_noise().
        !*/

        void reset(
        );
        /*!
            ensures
                - Returns this object to the state immediately after construction. To be precise, we do:
                   *this = momentum_filter(get_measurement_noise(), get_typical_acceleration(), get_max_measurement_deviation());
        !*/

        double operator()(
            const double measured_position
        );
        /*!
            ensures
                - Updates the Kalman filter with the new measured position of the object
                  and returns the new filtered estimate of the object's position, now that
                  we have seen the latest measured position.
                - #get_predicted_next_position() == the prediction for the *next* place we
                  will see the object. That is, where we think it will be in the future
                  rather than where it is now.
        !*/

        double get_predicted_next_position (
        ) const;
        /*!
            ensures
                - Returns the Kalman filter's estimate of the next position we will see the object. 
        !*/
    };

    std::ostream& operator << (std::ostream& out, const momentum_filter& item);
    void serialize(const momentum_filter& item, std::ostream& out);
    void deserialize(momentum_filter& item, std::istream& in);
    /*!
        Provide printing and serialization support.
    !*/

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
        ensures
            - This function finds the "optimal" settings of a momentum_filter based on
              recorded measurement data stored in sequences.  Here we assume that each
              vector in sequences is a complete track history of some object's measured
              positions.  What we do is find the momentum_filter that minimizes the
              following objective function:
                 sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1])
                 Where i is a time index.
              The sum runs over all the data in sequences.  So what we do is find the
              filter settings that produce smooth filtered trajectories but also produce
              filtered outputs that are as close to the measured positions as possible.
              The larger the value of smoothness the less jittery the filter outputs will
              be, but they might become biased or laggy if smoothness is set really high. 
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
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object simply contains four momentum_filters and applies them to the
                4 components of a dlib::rectangle's position.  It therefore allows you to
                easily filter a sequence of rectangles.  For instance, it can be used to
                smooth the output of an object detector running on a video.
        !*/

    public:
        rect_filter(
        ) = default;
        /*!
            ensures
                - The four momentum_filters in this object are default initialized.
        !*/

        rect_filter(
            const momentum_filter& filt
        );
        /*!
            ensures
                - #get_left() == filt
                - #get_top() == filt
                - #get_right() == filt
                - #get_bottom() == filt
        !*/

        rect_filter(
            double meas_noise,
            double acc,
            double max_meas_dev
        ) : rect_filter(momentum_filter(meas_noise, acc, max_meas_dev)) {}
        /*!
            requires
                - meas_noise >= 0
                - acc >= 0
                - max_meas_dev >= 0
            ensures
                - Initializes this object with momentum_filter(meas_noise, acc, max_meas_dev)
        !*/

        drectangle operator()(
            const drectangle& r
        );
        /*!
            ensures
                - Runs the given rectangle through the momentum_filters and returns the
                  filtered rectangle location.  That is, performs:
                  return drectangle(get_left()(r.left()),
                                    get_top()(r.top()),
                                    get_right()(r.right()),
                                    get_bottom()(r.bottom()));
        !*/

        drectangle operator()(
            const rectangle& r
        ); 
        /*!
            ensures
                - Runs the given rectangle through the momentum_filters and returns the
                  filtered rectangle location.  That is, performs:
                  return drectangle(get_left()(r.left()),
                                    get_top()(r.top()),
                                    get_right()(r.right()),
                                    get_bottom()(r.bottom()));
        !*/

        const momentum_filter& get_left() const; 
        momentum_filter&       get_left();
        const momentum_filter& get_top() const; 
        momentum_filter&       get_top();
        const momentum_filter& get_right() const; 
        momentum_filter&       get_right();
        const momentum_filter& get_bottom() const;
        momentum_filter&       get_bottom(); 
        /*!
            Provides access to the 4 momentum_filters used to filter the 4 coordinates that define a rectangle.
        !*/
    };

    void serialize(const rect_filter& item, std::ostream& out);
    void deserialize(rect_filter& item, std::istream& in);
    /*!
        Provide serialization support.
    !*/

// ----------------------------------------------------------------------------------------

    rect_filter find_optimal_rect_filter (
        const std::vector<rectangle>& rects,
        const double smoothness = 1
    );
    /*!
        requires
            - rects.size() > 4
            - smoothness >= 0
        ensures
            - This routine simply invokes find_optimal_momentum_filter() to find the
              momentum_filter that works best on the provided sequence of rectangles.  It
              then constructs a rect_filter using that momentum_filter and returns it.
              Therefore, this routine finds the rect_filter that is "optimal" for filtering
              the given sequence of rectangles.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KALMAN_FiLTER_ABSTRACT_Hh_


