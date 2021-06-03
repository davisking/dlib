// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KALMAN_FiLTER_Hh_
#define DLIB_KALMAN_FiLTER_Hh_

#include "kalman_filter_abstract.h"
#include "../matrix.h"
#include "../geometry.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        long states,
        long measurements
        >
    class kalman_filter
    {
    public:

        kalman_filter()
        {
            H = 0;
            A = 0;
            Q = 0;
            R = 0;
            x = 0;
            xb = 0;
            P = identity_matrix<double>(states);
            got_first_meas = false;
        }

        void set_observation_model ( const matrix<double,measurements,states>& H_) { H = H_; }
        void set_transition_model  ( const matrix<double,states,states>& A_) { A = A_; }
        void set_process_noise     ( const matrix<double,states,states>& Q_) { Q = Q_; }
        void set_measurement_noise ( const matrix<double,measurements,measurements>& R_) { R = R_; }
        void set_estimation_error_covariance( const matrix<double,states,states>& P_) { P = P_; }
        void set_state             ( const matrix<double,states,1>& xb_) 
        {
            xb = xb_;
            if (!got_first_meas) 
            {
                x = xb_;
                got_first_meas = true;
            }
        }

        const matrix<double,measurements,states>& get_observation_model (
        ) const { return H; }

        const matrix<double,states,states>& get_transition_model (
        ) const { return A; }

        const matrix<double,states,states>& get_process_noise (
        ) const { return Q; }

        const matrix<double,measurements,measurements>& get_measurement_noise (
        ) const { return R; }

        void update (
        )
        {
            // propagate estimation error covariance forward
            P = A*P*trans(A) + Q;

            // propagate state forward
            x = xb;
            xb = A*x;
        }

        void update (const matrix<double,measurements,1>& z)
        {
            // propagate estimation error covariance forward
            P = A*P*trans(A) + Q;

            // compute Kalman gain matrix
            const matrix<double,states,measurements> K = P*trans(H)*pinv(H*P*trans(H) + R);

            if (got_first_meas)
            {
                const matrix<double,measurements,1> res = z - H*xb;
                // correct the current state estimate
                x = xb + K*res;
            }
            else
            {
                // Since we don't have a previous state estimate at the start of filtering,
                // we will just set the current state to whatever is indicated by the measurement
                x = pinv(H)*z; 
                got_first_meas = true;
            }

            // propagate state forward in time
            xb = A*x;

            // update estimation error covariance since we got a measurement.
            P = (identity_matrix<double,states>() - K*H)*P;
        }

        const matrix<double,states,1>& get_current_state(
        ) const
        {
            return x;
        }

        const matrix<double,states,1>& get_predicted_next_state(
        ) const
        {
            return xb;
        }

        const matrix<double,states,states>& get_current_estimation_error_covariance(
        ) const
        {
            return P;
        }

        friend inline void serialize(const kalman_filter& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.got_first_meas, out);
            serialize(item.x, out);
            serialize(item.xb, out);
            serialize(item.P, out);
            serialize(item.H, out);
            serialize(item.A, out);
            serialize(item.Q, out);
            serialize(item.R, out);
        }

        friend inline void deserialize(kalman_filter& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw dlib::serialization_error("Unknown version number found while deserializing kalman_filter object.");

            deserialize(item.got_first_meas, in);
            deserialize(item.x, in);
            deserialize(item.xb, in);
            deserialize(item.P, in);
            deserialize(item.H, in);
            deserialize(item.A, in);
            deserialize(item.Q, in);
            deserialize(item.R, in);
        }

    private:

        bool got_first_meas;
        matrix<double,states,1> x, xb;
        matrix<double,states,states> P;

        matrix<double,measurements,states> H;
        matrix<double,states,states> A;
        matrix<double,states,states> Q;
        matrix<double,measurements,measurements> R;


    };

// ----------------------------------------------------------------------------------------

    class momentum_filter
    {
    public:

        momentum_filter(
            double meas_noise,
            double acc,
            double max_meas_dev
        ) : 
            measurement_noise(meas_noise),
            typical_acceleration(acc),
            max_measurement_deviation(max_meas_dev)
        {
            DLIB_CASSERT(meas_noise >= 0);
            DLIB_CASSERT(acc >= 0);
            DLIB_CASSERT(max_meas_dev >= 0);

            kal.set_observation_model({1, 0});
            kal.set_transition_model( {1, 1,
                0, 1});
            kal.set_process_noise({0, 0,
                0, typical_acceleration*typical_acceleration});

            kal.set_measurement_noise({measurement_noise*measurement_noise});
        }

        momentum_filter() = default; 

        double get_measurement_noise (
        ) const { return measurement_noise; }

        double get_typical_acceleration (
        ) const { return typical_acceleration; }

        double get_max_measurement_deviation (
        ) const { return max_measurement_deviation; }

        void reset()
        {
            *this = momentum_filter(measurement_noise, typical_acceleration, max_measurement_deviation);
        }

        double get_predicted_next_position(
        ) const
        {
            return kal.get_predicted_next_state()(0);
        }

        double operator()(
            const double measured_position
        )
        {
            auto x = kal.get_predicted_next_state();
            const auto max_deviation = max_measurement_deviation*measurement_noise;
            // Check if measured_position has suddenly jumped in value by a whole lot. This
            // could happen if the velocity term experiences a much larger than normal
            // acceleration, e.g.  because the underlying object is doing a maneuver.  If
            // this happens then we clamp the state so that the predicted next value is no
            // more than max_deviation away from measured_position at all times.
            if (x(0) > measured_position + max_deviation)
            {
                x(0) = measured_position + max_deviation;
                kal.set_state(x);
            }
            else if (x(0) < measured_position - max_deviation)
            {
                x(0) = measured_position - max_deviation;
                kal.set_state(x);
            }

            kal.update({measured_position});

            return kal.get_current_state()(0);
        }

        friend std::ostream& operator << (std::ostream& out, const momentum_filter& item)
        {
            out << "measurement_noise:         " << item.measurement_noise << "\n";
            out << "typical_acceleration:      " << item.typical_acceleration << "\n";
            out << "max_measurement_deviation: " << item.max_measurement_deviation;
            return out;
        }

        friend void serialize(const momentum_filter& item, std::ostream& out)
        {
            int version = 15;
            serialize(version, out);
            serialize(item.measurement_noise, out);
            serialize(item.typical_acceleration, out);
            serialize(item.max_measurement_deviation, out);
            serialize(item.kal, out);
        }

        friend void deserialize(momentum_filter& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 15)
                throw serialization_error("Unexpected version found while deserializing momentum_filter.");
            deserialize(item.measurement_noise, in);
            deserialize(item.typical_acceleration, in);
            deserialize(item.max_measurement_deviation, in);
            deserialize(item.kal, in);
        }

    private:

        double measurement_noise = 2;
        double typical_acceleration = 0.1;
        double max_measurement_deviation = 3; // nominally number of standard deviations

        kalman_filter<2,1> kal;
    };

// ----------------------------------------------------------------------------------------

    momentum_filter find_optimal_momentum_filter (
        const std::vector<std::vector<double>>& sequences,
        const double smoothness = 1
    );

// ----------------------------------------------------------------------------------------

    momentum_filter find_optimal_momentum_filter (
        const std::vector<double>& sequence,
        const double smoothness = 1
    );

// ----------------------------------------------------------------------------------------

    class rect_filter
    {
    public:
        rect_filter() = default;

        rect_filter(
            double meas_noise,
            double acc,
            double max_meas_dev
        ) : rect_filter(momentum_filter(meas_noise, acc, max_meas_dev)) {}

        rect_filter(
            const momentum_filter& filt
        ) : 
            left(filt),
            top(filt),
            right(filt),
            bottom(filt)
        {
        }

        drectangle operator()(const drectangle& r) 
        {
            return drectangle(left(r.left()),
                            top(r.top()),
                            right(r.right()),
                            bottom(r.bottom()));
        }

        drectangle operator()(const rectangle& r) 
        {
            return drectangle(left(r.left()),
                            top(r.top()),
                            right(r.right()),
                            bottom(r.bottom()));
        }

        const momentum_filter& get_left   () const { return left; }
        momentum_filter&       get_left   ()       { return left; }
        const momentum_filter& get_top    () const { return top; }
        momentum_filter&       get_top    ()       { return top; }
        const momentum_filter& get_right  () const { return right; }
        momentum_filter&       get_right  ()       { return right; }
        const momentum_filter& get_bottom () const { return bottom; }
        momentum_filter&       get_bottom ()       { return bottom; }

        friend void serialize(const rect_filter& item, std::ostream& out)
        {
            int version = 123;
            serialize(version, out);
            serialize(item.left, out);
            serialize(item.top, out);
            serialize(item.right, out);
            serialize(item.bottom, out);
        }

        friend void deserialize(rect_filter& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 123)
                throw dlib::serialization_error("Unknown version number found while deserializing rect_filter object.");
            deserialize(item.left, in);
            deserialize(item.top, in);
            deserialize(item.right, in);
            deserialize(item.bottom, in);
        }

    private:

        momentum_filter left, top, right, bottom;
    };

// ----------------------------------------------------------------------------------------

    rect_filter find_optimal_rect_filter (
        const std::vector<rectangle>& rects,
        const double smoothness = 1
    );

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KALMAN_FiLTER_Hh_

