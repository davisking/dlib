// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_HOUGH_tRANSFORM_Hh_
#define DLIB_HOUGH_tRANSFORM_Hh_

#include "hough_transform_abstract.h"
#include "../image_processing/generic_image.h"
#include "../geometry.h"
#include "../algs.h"
#include "assign_image.h"
#include <limits>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class hough_transform
    {

    public:
        explicit hough_transform (
            unsigned long size_
        ) : _size(size_) 
        {
            DLIB_CASSERT(size_ > 0,
                "\t hough_transform::hough_transform(size_)"
                << "\n\t Invalid arguments given to this function."
                );

            even_size = _size - (_size%2);

            const point cent = center(rectangle(0,0,size_-1,size_-1));
            xcos_theta.set_size(size_, size_);
            ysin_theta.set_size(size_, size_);

            std::vector<double> cos_theta(size_), sin_theta(size_);
            const double scale = 1<<16;
            for (unsigned long t = 0; t < size_; ++t)
            {
                double theta = t*pi/even_size;

                cos_theta[t] = scale*std::cos(theta)/sqrt_2;
                sin_theta[t] = scale*std::sin(theta)/sqrt_2;
            }
            const double offset = scale*even_size/4.0 + 0.5;

            for (unsigned long c = 0; c < size_; ++c)
            {
                const long x = c - cent.x();
                for (unsigned long t = 0; t < size_; ++t)
                    xcos_theta(c,t) = static_cast<int32>(x*cos_theta[t] + offset);
            }
            for (unsigned long r = 0; r < size_; ++r)
            {
                const long y = r - cent.y();
                for (unsigned long t = 0; t < size_; ++t)
                    ysin_theta(r,t) = static_cast<int32>(y*sin_theta[t] + offset);
            }
        }

        inline unsigned long size(
        ) const { return _size; }

        long nr(
        ) const { return _size; }

        long nc(
        ) const { return _size; }

        std::pair<dpoint, dpoint> get_line (
            const dpoint& p
        ) const
        {
            DLIB_ASSERT(rectangle(0,0,size()-1,size()-1).contains(p) == true,
                "\t pair<dpoint,dpoint> hough_transform::get_line(dpoint)"
                << "\n\t Invalid arguments given to this function."
                << "\n\t p:      " << p 
                << "\n\t size(): " << size()
                );

            // First we compute the radius measured in pixels from the center and the theta
            // angle in radians.
            double theta, radius;
            get_line_properties(p, theta, radius);
            theta *= pi/180;

            // now make a line segment on the line.
            const rectangle box = get_rect(*this);
            const dpoint cent = center(box);
            dpoint v1 = cent + dpoint(size()+1000,0) + dpoint(0,radius);
            dpoint v2 = cent - dpoint(size()+1000,0) + dpoint(0,radius);
            dpoint p1 = rotate_point(cent, v1, theta);
            dpoint p2 = rotate_point(cent, v2, theta);

            clip_line_to_rectangle(box, p1, p2);

            return std::make_pair(p1,p2);
        }

        double get_line_angle_in_degrees (
            const dpoint& p 
        ) const
        {
            double angle, radius;
            get_line_properties(p, angle, radius);
            return angle;
        }

        void get_line_properties (
            const dpoint& p,
            double& angle_in_degrees,
            double& radius
        ) const
        {
            const dpoint cent = center(get_rect(*this));
            double theta = p.x()-cent.x();
            radius = p.y()-cent.y();
            angle_in_degrees = 180*theta/even_size;
            radius = radius*sqrt_2 + 0.5;
        }

        template <
            typename image_type
            >
        point get_best_hough_point (
            const point& p,
            const image_type& himg_
        )
        {
            const const_image_view<image_type> himg(himg_);

            DLIB_ASSERT(himg.nr() == size() && himg.nc() == size() &&
                rectangle(0,0,size()-1,size()-1).contains(p) == true,
                "\t point hough_transform::get_best_hough_point()"
                << "\n\t Invalid arguments given to this function."
                << "\n\t himg.nr(): " << himg.nr()
                << "\n\t himg.nc(): " << himg.nc()
                << "\n\t size():    " << size()
                << "\n\t p:         " << p 
                );


            typedef typename image_traits<image_type>::pixel_type pixel_type;
            COMPILE_TIME_ASSERT(pixel_traits<pixel_type>::grayscale == true);
            pixel_type best_val = std::numeric_limits<pixel_type>::min();
            point best_point;


            const long max_n8 = (himg.nc()/8)*8;
            const long max_n4 = (himg.nc()/4)*4;
            const long r = p.y();
            const long c = p.x();

            const int32* ysin = &ysin_theta(r,0);
            const int32* xcos = &xcos_theta(c,0);
            long t = 0;
            while(t < max_n8)
            {
                long rr0 = (*xcos++ + *ysin++)>>16;
                long rr1 = (*xcos++ + *ysin++)>>16;
                long rr2 = (*xcos++ + *ysin++)>>16;
                long rr3 = (*xcos++ + *ysin++)>>16;
                long rr4 = (*xcos++ + *ysin++)>>16;
                long rr5 = (*xcos++ + *ysin++)>>16;
                long rr6 = (*xcos++ + *ysin++)>>16;
                long rr7 = (*xcos++ + *ysin++)>>16;

                if (himg[rr0][t++] > best_val)
                {
                    best_val = himg[rr0][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr0;
                }
                if (himg[rr1][t++] > best_val)
                {
                    best_val = himg[rr1][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr1;
                }
                if (himg[rr2][t++] > best_val)
                {
                    best_val = himg[rr2][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr2;
                }
                if (himg[rr3][t++] > best_val)
                {
                    best_val = himg[rr3][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr3;
                }
                if (himg[rr4][t++] > best_val)
                {
                    best_val = himg[rr4][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr4;
                }
                if (himg[rr5][t++] > best_val)
                {
                    best_val = himg[rr5][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr5;
                }
                if (himg[rr6][t++] > best_val)
                {
                    best_val = himg[rr6][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr6;
                }
                if (himg[rr7][t++] > best_val)
                {
                    best_val = himg[rr7][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr7;
                }
            }
            while(t < max_n4)
            {
                long rr0 = (*xcos++ + *ysin++)>>16;
                long rr1 = (*xcos++ + *ysin++)>>16;
                long rr2 = (*xcos++ + *ysin++)>>16;
                long rr3 = (*xcos++ + *ysin++)>>16;
                if (himg[rr0][t++] > best_val)
                {
                    best_val = himg[rr0][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr0;
                }
                if (himg[rr1][t++] > best_val)
                {
                    best_val = himg[rr1][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr1;
                }
                if (himg[rr2][t++] > best_val)
                {
                    best_val = himg[rr2][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr2;
                }
                if (himg[rr3][t++] > best_val)
                {
                    best_val = himg[rr3][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr3;
                }
            }
            while(t < himg.nc())
            {
                long rr0 = (*xcos++ + *ysin++)>>16;
                if (himg[rr0][t++] > best_val)
                {
                    best_val = himg[rr0][t-1];
                    best_point.x() = t-1;
                    best_point.y() = rr0;
                }
            }

            return best_point;
        }

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& img_,
            const rectangle& box,
            out_image_type& himg_
        ) const
        {
            typedef typename image_traits<in_image_type>::pixel_type in_pixel_type;
            typedef typename image_traits<out_image_type>::pixel_type out_pixel_type;

            DLIB_CASSERT(box.width() == size() && box.height() == size(),
                "\t void hough_transform::operator()"
                << "\n\t Invalid arguments given to this function."
                << "\n\t box.width():  " << box.width()
                << "\n\t box.height(): " << box.height()
                << "\n\t size():       " << size()
                );

            COMPILE_TIME_ASSERT(pixel_traits<in_pixel_type>::grayscale == true);
            COMPILE_TIME_ASSERT(pixel_traits<out_pixel_type>::grayscale == true);

            const_image_view<in_image_type> img(img_);
            image_view<out_image_type> himg(himg_);

            himg.set_size(size(), size());
            assign_all_pixels(himg, 0);

            auto record_hit = [&](const point& hough_point, const point& /*img_point*/, const in_pixel_type& val)
            {
                himg[hough_point.y()][hough_point.x()] += val;
            };
            perform_generic_hough_transform(img_, box, record_hit);
        }

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& img_,
            out_image_type& himg_
        ) const
        {
            rectangle box(0,0, num_columns(img_)-1, num_rows(img_)-1);
            (*this)(img_, box, himg_);
        }

        template <
            typename in_image_type
            >
        std::vector<std::vector<point>> find_pixels_voting_for_lines (
            const in_image_type& img,
            const rectangle& box,
            const std::vector<point>& hough_points,
            const unsigned long angle_window_size = 1,
            const unsigned long radius_window_size = 1
        ) const
        {

            typedef typename image_traits<in_image_type>::pixel_type in_pixel_type;

            DLIB_CASSERT(angle_window_size >= 1);
            DLIB_CASSERT(radius_window_size >= 1);
            DLIB_CASSERT(box.width() == size() && box.height() == size(),
                "\t std::vector<std::vector<point>> hough_transform::find_pixels_voting_for_lines()"
                << "\n\t Invalid arguments given to this function."
                << "\n\t box.width():  " << box.width()
                << "\n\t box.height(): " << box.height()
                << "\n\t size():       " << size()
                );
#ifdef ENABLE_ASSERTS
            for (auto& p : hough_points)
                DLIB_CASSERT(get_rect(*this).contains(p), 
                    "You gave a hough_points that isn't actually in the Hough space of this object."
                    << "\n\t get_rect(*this): "<< get_rect(*this) 
                    << "\n\t p: "<< p 
                    );
#endif

            std::vector<std::vector<point>> constituent_points(hough_points.size());

            // make a map that lets us look up in constant time if a hough point is in the
            // constituent_points output and if so where.
            matrix<uint32> hmap(size(),size());
            hmap = hough_points.size();
            for (size_t i = 0; i < hough_points.size(); ++i)
            {
                rectangle area = centered_rect(hough_points[i],angle_window_size,radius_window_size).intersect(get_rect(hmap));
                for (long r = area.top(); r <= area.bottom(); ++r)
                {
                    for (long c = area.left(); c <= area.right(); ++c)
                    {
                        hmap(r,c) = i;
                    }
                }
            }

            // record that this image point voted for this Hough point
            auto record_hit = [&](const point& hough_point, const point& img_point, in_pixel_type)
            {
                auto idx = hmap(hough_point.y(), hough_point.x());
                if (idx < constituent_points.size())
                {
                    // don't add img_point if it's already in the list.
                    if (constituent_points[idx].size() == 0 || constituent_points[idx].back() != img_point)
                        constituent_points[idx].push_back(img_point);
                }
            };

            perform_generic_hough_transform(img, box, record_hit);

            return constituent_points;
        }

        template <
            typename in_image_type
            >
        std::vector<std::vector<point>> find_pixels_voting_for_lines (
            const in_image_type& img,
            const std::vector<point>& hough_points,
            const unsigned long angle_window_size = 1,
            const unsigned long radius_window_size = 1
        ) const
        {
            rectangle box(0,0, num_columns(img)-1, num_rows(img)-1);
            return find_pixels_voting_for_lines(img, box, hough_points, angle_window_size, radius_window_size);
        }

        template <
            typename image_type,
            typename thresh_type
            >
        std::vector<point> find_strong_hough_points(
            const image_type& himg_,
            const thresh_type hough_count_threshold,
            const double angle_nms_thresh,
            const double radius_nms_thresh
        )
        {
            const_image_view<image_type> himg(himg_);

            DLIB_CASSERT(himg.nr() == size());
            DLIB_CASSERT(himg.nc() == size());
            DLIB_CASSERT(angle_nms_thresh >= 0)
            DLIB_CASSERT(radius_nms_thresh >= 0)

            std::vector<std::pair<double,point>> initial_lines;
            for (long r = 0; r < himg.nr(); ++r)
            {
                for (long c = 0; c < himg.nc(); ++c)
                {
                    if (himg[r][c] >= hough_count_threshold)
                        initial_lines.emplace_back(himg[r][c], point(c,r));
                }
            }


            std::vector<point> final_lines;
            std::vector<std::pair<double,double>> final_angle_and_radius;

            // Now do non-max suppression.  First, sort the initial_lines so the best lines come first.
            std::sort(initial_lines.rbegin(), initial_lines.rend(), 
                [](const std::pair<double,point>& a, const std::pair<double,point>& b){ return a.first<b.first;});
            for (auto& r : initial_lines)
            {
                double angle, radius;
                get_line_properties(r.second, angle, radius);

                // check if anything in final_lines is too close to r.second.  If
                // something is found then discard r.second.
                auto too_close = false;
                for (auto& ref : final_angle_and_radius)
                {
                    auto& ref_angle = ref.first;
                    auto& ref_radius = ref.second;

                    // We need to check for wrap around in angle since, for instance, a
                    // line with angle and radius of 90 and 10 is the same line as one with
                    // angle -90 and radius -10.
                    if ((std::abs(ref_angle - angle) < angle_nms_thresh && std::abs(ref_radius-radius) < radius_nms_thresh) ||
                        (180 - std::abs(ref_angle - angle) < angle_nms_thresh && std::abs(ref_radius+radius) < radius_nms_thresh))
                    {
                        too_close = true;
                        break;
                    }
                }

                if (!too_close)
                {
                    final_lines.emplace_back(r.second);
                    final_angle_and_radius.emplace_back(angle, radius);
                }
            }

            return final_lines;
        }


        template <
            typename in_image_type,
            typename record_hit_function_type
            >
        void perform_generic_hough_transform (
            const in_image_type& img_,
            const rectangle& box,
            record_hit_function_type record_hit
        ) const
        {

            typedef typename image_traits<in_image_type>::pixel_type in_pixel_type;

            DLIB_ASSERT(box.width() == size() && box.height() == size(),
                "\t void hough_transform::perform_generic_hough_transform()"
                << "\n\t Invalid arguments given to this function."
                << "\n\t box.width():  " << box.width()
                << "\n\t box.height(): " << box.height()
                << "\n\t size():       " << size()
                );

            COMPILE_TIME_ASSERT(pixel_traits<in_pixel_type>::grayscale == true);


            const_image_view<in_image_type> img(img_);


            const rectangle area = box.intersect(get_rect(img));

            const long max_n8 = (size()/8)*8;
            const long max_n4 = (size()/4)*4;
            for (long r = area.top(); r <= area.bottom(); ++r)
            {
                const int32* ysin_base = &ysin_theta(r-box.top(),0);
                for (long c = area.left(); c <= area.right(); ++c)
                {
                    const auto val = img[r][c];
                    if (val != 0)
                    {
                        /*
                        // The code in this comment is equivalent to the more complex but
                        // faster code below.  We keep this simple version of the Hough
                        // transform implementation here just to document what it's doing
                        // more clearly.
                        const point cent = center(box);
                        const long x = c - cent.x();
                        const long y = r - cent.y();
                        for (long t = 0; t < size(); ++t)
                        {
                            double theta = t*pi/even_size;
                            double radius = (x*std::cos(theta) + y*std::sin(theta))/sqrt_2 + even_size/2 + 0.5;
                            long rr = static_cast<long>(radius);

                            record_hit(point(t,rr), point(c,r), val);
                        }
                        continue;
                        */

                        // Run the speed optimized version of the code in the above
                        // comment.
                        const int32* ysin = ysin_base;
                        const int32* xcos = &xcos_theta(c-box.left(),0);
                        long t = 0;
                        while(t < max_n8)
                        {
                            long rr0 = (*xcos++ + *ysin++)>>16;
                            long rr1 = (*xcos++ + *ysin++)>>16;
                            long rr2 = (*xcos++ + *ysin++)>>16;
                            long rr3 = (*xcos++ + *ysin++)>>16;
                            long rr4 = (*xcos++ + *ysin++)>>16;
                            long rr5 = (*xcos++ + *ysin++)>>16;
                            long rr6 = (*xcos++ + *ysin++)>>16;
                            long rr7 = (*xcos++ + *ysin++)>>16;

                            record_hit(point(t++,rr0), point(c,r), val);
                            record_hit(point(t++,rr1), point(c,r), val);
                            record_hit(point(t++,rr2), point(c,r), val);
                            record_hit(point(t++,rr3), point(c,r), val);
                            record_hit(point(t++,rr4), point(c,r), val);
                            record_hit(point(t++,rr5), point(c,r), val);
                            record_hit(point(t++,rr6), point(c,r), val);
                            record_hit(point(t++,rr7), point(c,r), val);
                        }
                        while(t < max_n4)
                        {
                            long rr0 = (*xcos++ + *ysin++)>>16;
                            long rr1 = (*xcos++ + *ysin++)>>16;
                            long rr2 = (*xcos++ + *ysin++)>>16;
                            long rr3 = (*xcos++ + *ysin++)>>16;
                            record_hit(point(t++,rr0), point(c,r), val);
                            record_hit(point(t++,rr1), point(c,r), val);
                            record_hit(point(t++,rr2), point(c,r), val);
                            record_hit(point(t++,rr3), point(c,r), val);
                        }
                        while(t < (long)size())
                        {
                            long rr0 = (*xcos++ + *ysin++)>>16;
                            record_hit(point(t++,rr0), point(c,r), val);
                        }
                    }
                }
            }
        }

        template <
            typename in_image_type,
            typename record_hit_function_type
            >
        void perform_generic_hough_transform (
            const in_image_type& img_,
            record_hit_function_type record_hit
        ) const
        {
            rectangle box(0,0, num_columns(img_)-1, num_rows(img_)-1);
            perform_generic_hough_transform(img_, box, record_hit);
        }
        
    private:

        unsigned long _size;
        unsigned long even_size; // equal to _size if _size is even, otherwise equal to _size-1.
        matrix<int32> xcos_theta, ysin_theta;
    };
}

#endif // DLIB_HOUGH_tRANSFORM_Hh_

