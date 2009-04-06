// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SURf_ABSTRACT_H_
#ifdef DLIB_SURf_ABSTRACT_H_

#include "hessian_pyramid_abstract.h"
#include "../geometry/vector_abstract.h"
#include "../matrix/matrix_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
    
    double gaussian (
        double x, 
        double y,
        double sig
    );
    /*!
    !*/
    {
        const double pi = 3.1415926535898;
        return 1.0/(sig*std::sqrt(2*pi)) * std::exp( -(x*x + y*y)/(2*sig*sig));
    }

// ----------------------------------------------------------------------------------------

    template <typename integral_image_type, typename T>
    double compute_dominant_angle (
        const integral_image_type& img,
        const dlib::vector<T,2>& center,
        const double& scale
    );
    /*!
        requires
            - integral_image_type == an object such as dlib::integral_image or another
              type that implements the interface defined in image_transforms/integral_image_abstract.h
            - scale > 0
            - get_rect(img).contains(centered_rect(center, 17*scale, 17*scale)) == true
              (i.e. center can't be within 17*scale pixels of the edge of the image)
    !*/

// ----------------------------------------------------------------------------------------

    template <typename integral_image_type, typename T, typename MM, typename L>
    void compute_surf_descriptor (
        const integral_image_type& img,
        const dlib::vector<T,2>& center,
        const double scale,
        const double angle,
        matrix<double,64,1,MM,L>& des
    )
    /*!
        requires
            - integral_image_type == an object such as dlib::integral_image or another
              type that implements the interface defined in image_transforms/integral_image_abstract.h
            - scale > 0
            - get_rect(img).contains(centered_rect(center, 31*scale, 31*scale)) == true
              (i.e. center can't be within 31*scale pixels of the edge of the image)
    !*/

// ----------------------------------------------------------------------------------------

    struct surf_point
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a detected SURF point.
        !*/

        interest_point p;
        matrix<double,64,1> des;
        double angle;

        double match_score;

        bool operator < (const surf_point& p) const { return match_score < p.match_score; }
    };

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    const std::vector<surf_point> get_surf_points (
        const image_type& img,
        long max_points
    );
    /*!
        requires
            - max_points > 0
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SURf_ABSTRACT_H_


