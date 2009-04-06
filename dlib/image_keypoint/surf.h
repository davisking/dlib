// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SURf_H_
#define DLIB_SURf_H_

#include "surf_abstract.h"
#include "hessian_pyramid.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
    
    struct surf_point
    {
        interest_point p;
        matrix<double,64,1> des;
        double angle;

        double match_score;

        bool operator < (const surf_point& p) const { return match_score < p.match_score; }
    };

// ----------------------------------------------------------------------------------------

    inline double gaussian (double x, double y, double sig)
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
    )
    {
        DLIB_ASSERT(get_rect(img).contains(centered_rect(center, 17*scale,17*scale)) == true,
            "\tdouble compute_dominant_angle(img, center, scale)"
            << "\n\tAll arguments to this function must be > 0"
            << "\n\t get_rect(img): " << get_rect(img) 
            << "\n\t center:        " << center 
            << "\n\t scale:         " << scale 
        );

        const double pi = 3.1415926535898;

        std::vector<double> ang;
        std::vector<dlib::vector<double,2> > samples;

        // accumulate a bunch of angle and vector samples
        dlib::vector<double,2> vect;
        for (long r = -6; r <= 6; ++r)
        {
            for (long c = -6; c <= 6; ++c)
            {
                if (r*r + c*c < 36)
                {
                    // compute a Gaussian weighted gradient and the gradient's angle.
                    const double gauss = gaussian(c,r, 2.5);
                    vect.x() = gauss*haar_x(img, scale*point(c,r)+center, static_cast<long>(4*scale+0.5));
                    vect.y() = gauss*haar_y(img, scale*point(c,r)+center, static_cast<long>(4*scale+0.5));
                    samples.push_back(vect);
                    ang.push_back(atan2(vect.y(), vect.x()));
                }
            }
        }

        //cout << "ang size: " << ang.size() << endl;

        // now find the dominant direction
        double max_length = 0;
        double best_ang = 0;
        // look at a bunch of pie shaped slices of a circle 
        const long slices = 45;
        const double ang_step = (2*pi)/slices;
        for (long ang_i = 0; ang_i < slices; ++ang_i)
        {
            // compute the bounding angles
            double ang1 = ang_step*ang_i - pi;
            double ang2 = ang1 + pi/3;


            // compute sum of all vectors that are within the above two angles
            vect.x() = 0;
            vect.y() = 0;
            for (unsigned long i = 0; i < ang.size(); ++i)
            {
                if (ang1 <= ang[i] && ang[i] <= ang2)
                {
                    vect += samples[i];
                    //cout << ".";
                }
                else if (ang2 > pi && (ang[i] >= ang1 || ang[i] <= (-2*pi+ang2)))
                {
                    vect += samples[i];
                    //cout << ".";
                }
            }
            //cout << "$";


            // record the angle of the best vectors
            if (length_squared(vect) > max_length)
            {
                max_length = length_squared(vect);
                best_ang = atan2(vect.y(), vect.x());
            }
        }

        return best_ang;
    }

// ----------------------------------------------------------------------------------------

    template <typename integral_image_type, typename T, typename MM, typename L>
    void compute_surf_descriptor (
        const integral_image_type& img,
        const dlib::vector<T,2>& center,
        const double scale,
        const double angle,
        matrix<double,64,1,MM,L>& des
    )
    {
        DLIB_ASSERT(get_rect(img).contains(centered_rect(center, 31*scale,31*scale)) == true,
            "\tvoid compute_surf_descriptor(img, center, scale, angle)"
            << "\n\tAll arguments to this function must be > 0"
            << "\n\t get_rect(img): " << get_rect(img) 
            << "\n\t center:        " << center 
            << "\n\t scale:         " << scale 
        );

        point_rotator rot(angle);
        point_rotator inv_rot(-angle);

        long count = 0;

        // loop over the 4x4 grid of histogram buckets 
        for (long r = -10; r < 10; r += 5)
        {
            for (long c = -10; c < 10; c += 5)
            {
                dlib::vector<double,2> vect, abs_vect, temp;

                // now loop over 25 points in this bucket and sum their features 
                for (long y = r; y < r+5; ++y)
                {
                    for (long x = c; x < c+5; ++x)
                    {
                        // get the rotated point for this extraction point
                        point p(rot(point(x,y)*scale) + center); 

                        const double gauss = gaussian(x,y, 3.3);
                        temp.x() = gauss*haar_x(img, p, static_cast<long>(2*scale+0.5));
                        temp.y() = gauss*haar_y(img, p, static_cast<long>(2*scale+0.5));

                        // rotate this vector into alignment with the surf descriptor box 
                        temp = inv_rot(temp);

                        vect += temp;
                        abs_vect += abs(temp);
                    }
                }

                des(count++) = vect.x();
                des(count++) = vect.y();
                des(count++) = abs_vect.x();
                des(count++) = abs_vect.y();
            }
        }

        // return the length normalized descriptor
        des = des/length(des);
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    const std::vector<surf_point> get_surf_points (
        const image_type& img,
        long max_points
    )
    {
        integral_image int_img;

        int_img.load(img);
        hessian_pyramid pyr;
        pyr.build_pyramid(int_img, 4, 6, 2);

        std::vector<interest_point> points; 
        get_interest_points(pyr, 0.10, points);
        std::vector<surf_point> spoints;

        // sort all the points by how strong their detect is
        std::sort(points.rbegin(), points.rend());


        // now extract SURF descriptors for the points
        for (unsigned long i = 0; i < std::min((size_t)max_points,points.size()); ++i)
        {
            // ignore points that are close to the edge of the image
            const double border = 31;
            //const double border = std::sqrt(22.0*22 + 22*22);
            if (get_rect(int_img).contains(centered_rect(points[i].center, border*points[i].scale, border*points[i].scale)))
            {
                surf_point sp;
                sp.angle = compute_dominant_angle(int_img, points[i].center, points[i].scale);
                compute_surf_descriptor(int_img, points[i].center, points[i].scale, sp.angle, sp.des);
                sp.p = points[i];

                spoints.push_back(sp);

            }
        }

        return spoints;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SURf_H_

