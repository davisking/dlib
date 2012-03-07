// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KALMAN_FiLTER_H__
#define DLIB_KALMAN_FiLTER_H__

#include "kalman_filter_abstract.h"
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
        void set_transitoin_model  ( const matrix<double,states,states>& A_) { A = A_; }
        void set_process_noise     ( const matrix<double,states,states>& Q_) { Q = Q_; }
        void set_measurement_noise ( const matrix<double,measurements,measurements>& R_) { R = R_; }

        void update (
        )
        {
            P = A*P*trans(A) + Q;
            const matrix<double,states,measurements> K = P*trans(H)*pinv(H*P*trans(H) + R);

            x = xb;
            xb = A*x;

            P = (identity_matrix<double,states>() - K*H)*P;
        }

        void update (const matrix<double,measurements,1>& z)
        {
            P = A*P*trans(A) + Q;
            const matrix<double,states,measurements> K = P*trans(H)*pinv(H*P*trans(H) + R);

            if (got_first_meas)
            {
                const matrix<double,measurements,1> res = z - H*xb;
                x = xb + K*res;
            }
            else
            {
                // Since we don't have a previous state estimate at the start of filtering,
                // we will just set the current state to whatever is indicated by the measurement
                x = pinv(H)*z; 
                got_first_meas = true;
            }
            xb = A*x;

            P = (identity_matrix<double,states>() - K*H)*P;

        }


        const matrix<double,states,1>& get_current_state()
        {
            return x;
        }

        const matrix<double,states,1>& get_predicted_next_state()
        {
            return xb;
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

}

#endif // DLIB_KALMAN_FiLTER_H__

