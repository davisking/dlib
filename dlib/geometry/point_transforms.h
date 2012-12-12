// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_POINT_TrANSFORMS_H_
#define DLIB_POINT_TrANSFORMS_H_

#include "point_transforms_abstract.h"
#include "../algs.h"
#include "vector.h"
#include "../matrix.h"
#include "../matrix/matrix_la.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class point_rotator
    {
    public:
        point_rotator (
            const double& angle
        )
        {
            sin_angle = std::sin(angle);
            cos_angle = std::cos(angle);
        }

        template <typename T>
        const dlib::vector<T,2> operator() (
            const dlib::vector<T,2>& p
        ) const
        {
            double x = cos_angle*p.x() - sin_angle*p.y();
            double y = sin_angle*p.x() + cos_angle*p.y();

            return dlib::vector<double,2>(x,y);
        }

    private:
        double sin_angle;
        double cos_angle;
    };

// ----------------------------------------------------------------------------------------

    class point_transform
    {
    public:
        point_transform (
            const double& angle,
            const dlib::vector<double,2>& translate_
        )
        {
            sin_angle = std::sin(angle);
            cos_angle = std::cos(angle);
            translate = translate_;
        }

        template <typename T>
        const dlib::vector<T,2> operator() (
            const dlib::vector<T,2>& p
        ) const
        {
            double x = cos_angle*p.x() - sin_angle*p.y();
            double y = sin_angle*p.x() + cos_angle*p.y();

            return dlib::vector<double,2>(x,y) + translate;
        }

    private:
        double sin_angle;
        double cos_angle;
        dlib::vector<double,2> translate;
    };

// ----------------------------------------------------------------------------------------

    class point_transform_affine
    {
    public:
        point_transform_affine (
            const matrix<double,2,2>& m_,
            const dlib::vector<double,2>& b_
        ) :m(m_), b(b_)
        {
        }

        const dlib::vector<double,2> operator() (
            const dlib::vector<double,2>& p
        ) const
        {
            return m*p + b;
        }

    private:
        matrix<double,2,2> m;
        dlib::vector<double,2> b;
    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    point_transform_affine find_affine_transform (
        const std::vector<dlib::vector<T,2> >& from_points,
        const std::vector<dlib::vector<T,2> >& to_points
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(from_points.size() == to_points.size() &&
                    from_points.size() >= 3,
            "\t point_transform_affine find_affine_transform(from_points, to_points)"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t from_points.size(): " << from_points.size()
            << "\n\t to_points.size():   " << to_points.size()
            );

        matrix<double,3,0> P(3, from_points.size());
        matrix<double,2,0> Q(2, from_points.size());

        for (unsigned long i = 0; i < from_points.size(); ++i)
        {
            P(0,i) = from_points[i].x();
            P(1,i) = from_points[i].y();
            P(2,i) = 1;

            Q(0,i) = to_points[i].x();
            Q(1,i) = to_points[i].y();
        }

        const matrix<double,2,3> m = Q*pinv(P);
        return point_transform_affine(subm(m,0,0,2,2), colm(m,2));
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    const dlib::vector<T,2> rotate_point (
        const dlib::vector<T,2>& center,
        const dlib::vector<T,2>& p,
        double angle
    )
    {
        point_rotator rot(angle);
        return rot(p-center)+center;
    }

// ----------------------------------------------------------------------------------------

    inline matrix<double,2,2> rotation_matrix (
         double angle
    )
    {
        const double ca = std::cos(angle);
        const double sa = std::sin(angle);

        matrix<double,2,2> m;
        m = ca, -sa,
            sa, ca;
        return m;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_POINT_TrANSFORMS_H_

