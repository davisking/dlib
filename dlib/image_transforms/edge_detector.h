// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_EDGE_DETECTOr_
#define DLIB_EDGE_DETECTOr_

#include "edge_detector_abstract.h"
#include "../pixel.h"
#include "../array2d.h"
#include "../geometry.h"
#include <vector>
#include "../image_keypoint/build_separable_poly_filters.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    inline char edge_orientation (
        const T& x_,
        const T& y_
    )
    {

        // if this is a perfectly horizontal gradient then return right away
        if (x_ == 0)
        {
            return '|';
        }
        else if (y_ == 0) // if this is a perfectly vertical gradient then return right away
        {
            return '-';
        }

        // Promote x so that when we multiply by 128 later we know overflow won't happen.
        typedef typename promote<T>::type type;
        type x = x_;
        type y = y_;

        if (x < 0)
        {
            x = -x;
            if (y < 0)
            {
                y = -y;
                x *= 128;
                const type temp = x/y;
                if (temp > 309)
                    return '-';
                else if (temp > 53)
                    return '/';
                else
                    return '|';
            }
            else
            {
                x *= 128;
                const type temp = x/y;
                if (temp > 309)
                    return '-';
                else if (temp > 53)
                    return '\\';
                else
                    return '|';
            }
        }
        else
        {
            if (y < 0)
            {
                y = -y;
                x *= 128;

                const type temp = x/y;
                if (temp > 309)
                    return '-';
                else if (temp > 53)
                    return '\\';
                else
                    return '|';
            }
            else
            {
                x *= 128;

                const type temp = x/y;
                if (temp > 309)
                    return '-';
                else if (temp > 53)
                    return '/';
                else
                    return '|';
            }
        }

    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void sobel_edge_detector (
        const in_image_type& in_img_,
        out_image_type& horz_,
        out_image_type& vert_
    )
    {
        typedef typename image_traits<out_image_type>::pixel_type pixel_type;
        COMPILE_TIME_ASSERT(pixel_traits<pixel_type>::is_unsigned == false);
        DLIB_ASSERT( !is_same_object(in_img_,horz_) && !is_same_object(in_img_,vert_) &&
                     !is_same_object(horz_,vert_),
            "\tvoid sobel_edge_detector(in_img_, horz_, vert_)"
            << "\n\t You can't give the same image as more than one argument"
            << "\n\t is_same_object(in_img_,horz_): " << is_same_object(in_img_,horz_)
            << "\n\t is_same_object(in_img_,vert_): " << is_same_object(in_img_,vert_)
            << "\n\t is_same_object(horz_,vert_):    " << is_same_object(horz_,vert_)
            );


        const int vert_filter[3][3] = {{-1,-2,-1}, 
        {0,0,0}, 
        {1,2,1}};
        const int horz_filter[3][3] = { {-1,0,1}, 
        {-2,0,2}, 
        {-1,0,1}};

        const long M = 3;
        const long N = 3;


        const_image_view<in_image_type> in_img(in_img_);
        image_view<out_image_type> horz(horz_);
        image_view<out_image_type> vert(vert_);

        horz.set_size(in_img.nr(),in_img.nc());
        vert.set_size(in_img.nr(),in_img.nc());

        assign_border_pixels(horz,1,1,0);
        assign_border_pixels(vert,1,1,0);

        // figure out the range that we should apply the filter to
        const long first_row = M/2;
        const long first_col = N/2;
        const long last_row = in_img.nr() - M/2;
        const long last_col = in_img.nc() - N/2;


        // apply the filter to the image
        for (long r = first_row; r < last_row; ++r)
        {
            for (long c = first_col; c < last_col; ++c)
            {
                typedef typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type bp_type;

                typename promote<bp_type>::type p, horz_temp, vert_temp;
                horz_temp = 0;
                vert_temp = 0;
                for (long m = 0; m < M; ++m)
                {
                    for (long n = 0; n < N; ++n)
                    {
                        // pull out the current pixel and put it into p
                        p = get_pixel_intensity(in_img[r-M/2+m][c-N/2+n]);

                        horz_temp += p*horz_filter[m][n];
                        vert_temp += p*vert_filter[m][n];
                    }
                }

                assign_pixel(horz[r][c] , horz_temp);
                assign_pixel(vert[r][c] , vert_temp);

            }
        }
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename T>
        typename promote<T>::type square (const T& a)
        { 
            return static_cast<T>(a)*static_cast<T>(a); 
        }
    }

    template <
        typename in_image_type,
        typename out_image_type
        >
    void suppress_non_maximum_edges (
        const in_image_type& horz_,
        const in_image_type& vert_,
        out_image_type& out_img_
    )
    {
        const_image_view<in_image_type> horz(horz_);
        const_image_view<in_image_type> vert(vert_);
        image_view<out_image_type> out_img(out_img_);

        COMPILE_TIME_ASSERT(is_signed_type<typename image_traits<in_image_type>::pixel_type>::value);
        DLIB_ASSERT( horz.nr() == vert.nr() && horz.nc() == vert.nc(),
            "\tvoid suppress_non_maximum_edges(horz, vert, out_img)"
            << "\n\tYou have to give horz and vert gradient images that are the same size"
            << "\n\thorz.nr():   " << horz.nr() 
            << "\n\thorz.nc():   " << horz.nc() 
            << "\n\tvert.nr():   " << vert.nr() 
            << "\n\tvert.nc():   " << vert.nc() 
            );
        DLIB_ASSERT( !is_same_object(out_img_,horz_) && !is_same_object(out_img_,vert_),
            "\tvoid suppress_non_maximum_edges(horz_, vert_, out_img_)"
            << "\n\t out_img can't be the same as one of the input images."
            << "\n\t is_same_object(out_img_,horz_):    " << is_same_object(out_img_,horz_)
            << "\n\t is_same_object(out_img_,vert_):    " << is_same_object(out_img_,vert_)
            );

        using std::min;
        using std::abs;


        // if there isn't any input image then don't do anything
        if (horz.size() == 0)
        {
            out_img.clear();
            return;
        }

        out_img.set_size(horz.nr(),horz.nc());

        zero_border_pixels(out_img,1,1);

        // now do non maximum suppression while we copy the 
        const long M = 3;
        const long N = 3;

        // figure out the range that we should apply the filter to
        const long first_row = M/2;
        const long first_col = N/2;
        const long last_row = horz.nr() - M/2;
        const long last_col = horz.nc() - N/2;


        // apply the filter to the image
        for (long r = first_row; r < last_row; ++r)
        {
            for (long c = first_col; c < last_col; ++c)
            {
                typedef typename promote<typename image_traits<in_image_type>::pixel_type>::type T;
                const T y = horz[r][c];
                const T x = vert[r][c];

                using impl::square;

                const T val = square(horz[r][c]) + square(vert[r][c]); 

                const char ori = edge_orientation(x,y);
                const unsigned char zero = 0;
                switch (ori)
                {
                    case '-':
                        if (square(horz[r-1][c])+square(vert[r-1][c]) > val || square(horz[r+1][c]) + square(vert[r+1][c]) > val)
                            assign_pixel(out_img[r][c] , zero);
                        else
                            assign_pixel(out_img[r][c] , std::sqrt((double)val));
                        break;

                    case '|':
                        if (square(horz[r][c-1]) + square(vert[r][c-1]) > val || square(horz[r][c+1]) + square(vert[r][c+1]) > val)
                            assign_pixel(out_img[r][c] , zero);
                        else
                            assign_pixel(out_img[r][c] , std::sqrt((double)val));
                        break;

                    case '/':
                        if (square(horz[r-1][c-1]) + square(vert[r-1][c-1]) > val || square(horz[r+1][c+1]) + square(vert[r+1][c+1]) > val)
                            assign_pixel(out_img[r][c] , zero);
                        else
                            assign_pixel(out_img[r][c] , std::sqrt((double)val));
                        break;

                    case '\\':
                        if (square(horz[r+1][c-1]) + square(vert[r+1][c-1]) > val || square(horz[r-1][c+1]) + square(vert[r-1][c+1]) > val)
                            assign_pixel(out_img[r][c] , zero);
                        else
                            assign_pixel(out_img[r][c] , std::sqrt((double)val));
                        break;

                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void normalize_image_gradients (
        image_type& img1_,
        image_type& img2_
    )
    {
        image_view<image_type> img1(img1_);
        image_view<image_type> img2(img2_);

        using pixel_type = typename image_traits<image_type>::pixel_type;
        static_assert(std::is_same<pixel_type,float>::value || 
            std::is_same<pixel_type,double>::value ||
            std::is_same<pixel_type,long double>::value, 
            "normalize_image_gradients() requires the input images to use floating point pixel types.");

        DLIB_CASSERT(img1.nr() == img2.nr());
        DLIB_CASSERT(img1.nc() == img2.nc());

        // normalize all the gradients
        for (long r = 0; r < img1.nr(); ++r)
        {
            for (long c = 0; c < img1.nc(); ++c)
            {
                if (img1[r][c] != 0 || img2[r][c] != 0)
                {
                    double len = std::sqrt(img1[r][c]*img1[r][c] + img2[r][c]*img2[r][c]);
                    img1[r][c] /= len;
                    img2[r][c] /= len;
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    std::vector<point> remove_incoherent_edge_pixels (
        const std::vector<point>& line,
        const image_type& horz_gradient_,
        const image_type& vert_gradient_,
        double angle_threshold
    )
    {
        const_image_view<image_type> horz_gradient(horz_gradient_);
        const_image_view<image_type> vert_gradient(vert_gradient_);

        DLIB_CASSERT(horz_gradient.nr() == vert_gradient.nr());
        DLIB_CASSERT(horz_gradient.nc() == vert_gradient.nc());
        DLIB_CASSERT(angle_threshold >= 0);
#ifdef ENABLE_ASSERTS
        for (auto& p : line)
            DLIB_ASSERT(get_rect(horz_gradient).contains(p), "All line points must be inside the given images.");
#endif

        // We make sure that each vector is within this threshold of the mean vector.  So
        // to make sure they are pairwise within the user supplied angel threshold we need
        // to divide by 2 before we proceed.
        angle_threshold /= 2;

        const double dotthresh = std::cos(angle_threshold*pi/180);
        // find the average gradient on this line
        dpoint avg;
        for (auto p : line)
            avg += dpoint(horz_gradient[p.y()][p.x()], vert_gradient[p.y()][p.x()]);
        dpoint ref = avg.normalize();

        // now iterate a few times and find the most common average gradient.
        for (int i = 0; i < 10; ++i)
        {
            avg = dpoint();
            for (auto p : line)
            {
                const dpoint v(horz_gradient[p.y()][p.x()], vert_gradient[p.y()][p.x()]);
                const double dp = ref.dot(v);
                if (dp > dotthresh)
                    avg += v;
                else if (-dp > dotthresh)
                    avg -= v;
            }
            ref = avg.normalize();
        }

        // now remove all the points that deviate from the average gradient too much.
        std::vector<point> newpixels;
        for (auto p : line)
        {
            dpoint v(horz_gradient[p.y()][p.x()], vert_gradient[p.y()][p.x()]);
            if (std::abs(ref.dot(v)) > dotthresh)
                newpixels.push_back(p);
        }
        return newpixels;
    }

    template <
        typename image_type
        >
    std::vector<std::vector<point>> remove_incoherent_edge_pixels (
        const std::vector<std::vector<point>>& line_pixels,
        const image_type& horz_gradient_,
        const image_type& vert_gradient_,
        const double angle_threshold
    )
    {
        std::vector<std::vector<point>> temp;
        temp.reserve(line_pixels.size());
        for (auto& line : line_pixels)
            temp.emplace_back(remove_incoherent_edge_pixels(line, horz_gradient_, vert_gradient_, angle_threshold));
        return temp;
    }

// ----------------------------------------------------------------------------------------

    class image_gradients
    {

    public:

        image_gradients (
        ) : image_gradients(1) {}

        image_gradients (
            long scale
        ) : the_scale(scale)
        {
            DLIB_CASSERT(scale >= 1);

            scale = 2*scale+1;

            auto dfilters = build_separable_poly_filters(2,scale); 
            DLIB_CASSERT(dfilters[1].size() == 1);
            DLIB_CASSERT(dfilters[2].size() == 1);
            DLIB_CASSERT(dfilters[3].size() == 1);
            DLIB_CASSERT(dfilters[4].size() == 1);
            DLIB_CASSERT(dfilters[5].size() == 1);

            filter_x.first = matrix_cast<float>(dfilters[1][0].first);
            filter_x.second = matrix_cast<float>(dfilters[1][0].second);

            filter_y.first = matrix_cast<float>(dfilters[2][0].first);
            filter_y.second = matrix_cast<float>(dfilters[2][0].second);

            // We multiply by 2 so that the filter gives the gradient rather than the x^2
            // polynomial coefficient.
            filter_xx.first = 2*matrix_cast<float>(dfilters[3][0].first);
            filter_xx.second = matrix_cast<float>(dfilters[3][0].second);

            filter_xy.first = matrix_cast<float>(dfilters[4][0].first);
            filter_xy.second = matrix_cast<float>(dfilters[4][0].second);

            // We multiply by 2 so that the filter gives the gradient rather than the y^2
            // polynomial coefficient.
            filter_yy.first = 2*matrix_cast<float>(dfilters[5][0].first);
            filter_yy.second = matrix_cast<float>(dfilters[5][0].second);
        }


        long get_scale() const { return the_scale; }

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_x(
            const in_image_type& img,
            out_image_type& out
        ) const 
        { 
            return spatially_filter_image_separable(img, out, filter_x.second, filter_x.first); 
        }

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_y(
            const in_image_type& img,
            out_image_type& out
        ) const 
        { 
            return spatially_filter_image_separable(img, out, filter_y.second, filter_y.first); 
        }


        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_xx(
            const in_image_type& img,
            out_image_type& out
        )  const
        { 
            return spatially_filter_image_separable(img, out, filter_xx.second, filter_xx.first); 
        }

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_xy(
            const in_image_type& img,
            out_image_type& out
        ) const 
        { 
            return spatially_filter_image_separable(img, out, filter_xy.second, filter_xy.first); 
        }

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_yy(
            const in_image_type& img,
            out_image_type& out
        ) const
        { 
            return spatially_filter_image_separable(img, out, filter_yy.second, filter_yy.first); 
        }

        matrix<float> get_x_filter()  const { return filter_x.first*trans(filter_x.second); }
        matrix<float> get_y_filter()  const { return filter_y.first*trans(filter_y.second); }
        matrix<float> get_xx_filter() const { return filter_xx.first*trans(filter_xx.second); }
        matrix<float> get_xy_filter() const { return filter_xy.first*trans(filter_xy.second); }
        matrix<float> get_yy_filter() const { return filter_yy.first*trans(filter_yy.second); }

    private:
        std::pair<matrix<float,0,1>,matrix<float,0,1>> filter_x;
        std::pair<matrix<float,0,1>,matrix<float,0,1>> filter_y;
        std::pair<matrix<float,0,1>,matrix<float,0,1>> filter_xx;
        std::pair<matrix<float,0,1>,matrix<float,0,1>> filter_xy;
        std::pair<matrix<float,0,1>,matrix<float,0,1>> filter_yy;

        long the_scale;
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <
            typename in_image_type,
            typename out_image_type
            >
        void find_lines(
            const in_image_type& xx_,
            const in_image_type& xy_,
            const in_image_type& yy_,
            out_image_type& horz_,
            out_image_type& vert_,
            double positive_if_should_find_dark_lines
        )
        {
            typedef typename image_traits<out_image_type>::pixel_type out_pixel_type;
            static_assert(std::is_same<float,out_pixel_type>::value || std::is_same<double,out_pixel_type>::value,
                "Output images must contain either float or double valued pixels");

            const_image_view<in_image_type> xx(xx_);
            const_image_view<in_image_type> xy(xy_);
            const_image_view<in_image_type> yy(yy_);

            DLIB_CASSERT(xx.nr() == xy.nr());
            DLIB_CASSERT(xx.nr() == yy.nr());
            DLIB_CASSERT(xx.nc() == xy.nc());
            DLIB_CASSERT(xx.nc() == yy.nc());


            image_view<out_image_type> x(horz_);
            image_view<out_image_type> y(vert_);

            x.set_size(xx.nr(), xx.nc());
            y.set_size(xx.nr(), xx.nc());


            // store the max eigenvalue into xy and then the associated eigen vector into [xx,yy]
            for (long r = 0; r < xx.nr(); ++r)
            {
                for (long c = 0; c < xx.nc(); ++c)
                {
                    // negate to that lambda will be the *minimum* eigenvalue
                    double w1 = positive_if_should_find_dark_lines*xx[r][c]/2.0;
                    double w2 = positive_if_should_find_dark_lines*yy[r][c]/2.0;
                    double w3 = positive_if_should_find_dark_lines*xy[r][c];


                    auto lambda = w1 + w2 + std::sqrt((w1-w2)*(w1-w2) + w3*w3);
                    if (lambda < 0)
                        lambda = 0;

                    if (2*w1!=lambda)
                    {
                        x[r][c] = -w3/(2*w1-lambda);
                        y[r][c] = 1;

                        double norm = std::sqrt(x[r][c]*x[r][c] + y[r][c]*y[r][c]);
                        x[r][c] *= lambda/norm;
                        y[r][c] *= lambda/norm;
                    }
                    else
                    {
                        x[r][c] = lambda;
                        y[r][c] = 0;
                    }
                }
            }
        }
    }

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_bright_lines(
        const in_image_type& xx,
        const in_image_type& xy,
        const in_image_type& yy,
        out_image_type& horz,
        out_image_type& vert
    )
    {
        impl::find_lines(xx,xy,yy,horz,vert,-1);
    }

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_dark_lines(
        const in_image_type& xx,
        const in_image_type& xy,
        const in_image_type& yy,
        out_image_type& horz,
        out_image_type& vert
    )
    {
        impl::find_lines(xx,xy,yy,horz,vert,+1);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_bright_keypoints(
        const in_image_type& xx_,
        const in_image_type& xy_,
        const in_image_type& yy_,
        out_image_type& saliency_
    )
    {
        typedef typename image_traits<out_image_type>::pixel_type out_pixel_type;
        static_assert(std::is_same<float,out_pixel_type>::value || std::is_same<double,out_pixel_type>::value,
            "Output images must contain either float or double valued pixels");

        const_image_view<in_image_type> xx(xx_);
        const_image_view<in_image_type> xy(xy_);
        const_image_view<in_image_type> yy(yy_);

        DLIB_CASSERT(xx.nr() == xy.nr());
        DLIB_CASSERT(xx.nr() == yy.nr());
        DLIB_CASSERT(xx.nc() == xy.nc());
        DLIB_CASSERT(xx.nc() == yy.nc());


        image_view<out_image_type> saliency(saliency_);
        saliency.set_size(xx.nr(), xx.nc());


        for (long r = 0; r < xx.nr(); ++r)
        {
            for (long c = 0; c < xx.nc(); ++c)
            {
                matrix<double,2,2> tmp;
                tmp = xx[r][c], xy[r][c],
                      xy[r][c], yy[r][c];

                matrix<double,2,1> e = real_eigenvalues(tmp);
                saliency[r][c] = prod(upperbound(e,0));
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_dark_keypoints(
        const in_image_type& xx_,
        const in_image_type& xy_,
        const in_image_type& yy_,
        out_image_type& saliency_
    )
    {
        typedef typename image_traits<out_image_type>::pixel_type out_pixel_type;
        static_assert(std::is_same<float,out_pixel_type>::value || std::is_same<double,out_pixel_type>::value,
            "Output images must contain either float or double valued pixels");

        const_image_view<in_image_type> xx(xx_);
        const_image_view<in_image_type> xy(xy_);
        const_image_view<in_image_type> yy(yy_);

        DLIB_CASSERT(xx.nr() == xy.nr());
        DLIB_CASSERT(xx.nr() == yy.nr());
        DLIB_CASSERT(xx.nc() == xy.nc());
        DLIB_CASSERT(xx.nc() == yy.nc());


        image_view<out_image_type> saliency(saliency_);
        saliency.set_size(xx.nr(), xx.nc());


        for (long r = 0; r < xx.nr(); ++r)
        {
            for (long c = 0; c < xx.nc(); ++c)
            {
                matrix<double,2,2> tmp;
                tmp = xx[r][c], xy[r][c],
                      xy[r][c], yy[r][c];

                matrix<double,2,1> e = real_eigenvalues(tmp);
                saliency[r][c] = prod(lowerbound(e,0));
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EDGE_DETECTOr_



