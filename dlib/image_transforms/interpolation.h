// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_INTERPOlATION__
#define DLIB_INTERPOlATION__ 

#include "interpolation_abstract.h"
#include "../pixel.h"
#include "../matrix.h"
#include "assign_image.h"
#include "../simd.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class interpolate_nearest_neighbor
    {
    public:

        template <typename image_type, typename pixel_type>
        bool operator() (
            const image_type& img,
            const dlib::point& p,
            pixel_type& result
        ) const
        {
            COMPILE_TIME_ASSERT(pixel_traits<typename image_type::type>::has_alpha == false);

            if (get_rect(img).contains(p))
            {
                assign_pixel(result, img[p.y()][p.x()]);
                return true;
            }
            else
            {
                return false;
            }
        }

    };

// ----------------------------------------------------------------------------------------

    class interpolate_bilinear
    {
        template <typename T>
        struct is_rgb_image 
        {
            const static bool value = pixel_traits<typename T::type>::rgb;
        };

    public:

        template <typename T, typename image_type, typename pixel_type>
        typename disable_if<is_rgb_image<image_type>,bool>::type operator() (
            const image_type& img,
            const dlib::vector<T,2>& p,
            pixel_type& result
        ) const
        {
            COMPILE_TIME_ASSERT(pixel_traits<typename image_type::type>::has_alpha == false);

            const long left   = static_cast<long>(std::floor(p.x()));
            const long top    = static_cast<long>(std::floor(p.y()));
            const long right  = left+1;
            const long bottom = top+1;


            // if the interpolation goes outside img 
            if (!get_rect(img).contains(rectangle(left,top,right,bottom))) 
                return false;

            const double lr_frac = p.x() - left;
            const double tb_frac = p.y() - top;

            double tl = 0, tr = 0, bl = 0, br = 0;

            assign_pixel(tl, img[top][left]);
            assign_pixel(tr, img[top][right]);
            assign_pixel(bl, img[bottom][left]);
            assign_pixel(br, img[bottom][right]);
            
            double temp = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                              tb_frac*((1-lr_frac)*bl + lr_frac*br);
                            
            assign_pixel(result, temp);
            return true;
        }

        template <typename T, typename image_type, typename pixel_type>
        typename enable_if<is_rgb_image<image_type>,bool>::type operator() (
            const image_type& img,
            const dlib::vector<T,2>& p,
            pixel_type& result
        ) const
        {
            COMPILE_TIME_ASSERT(pixel_traits<typename image_type::type>::has_alpha == false);

            const long left   = static_cast<long>(std::floor(p.x()));
            const long top    = static_cast<long>(std::floor(p.y()));
            const long right  = left+1;
            const long bottom = top+1;


            // if the interpolation goes outside img 
            if (!get_rect(img).contains(rectangle(left,top,right,bottom))) 
                return false;

            const double lr_frac = p.x() - left;
            const double tb_frac = p.y() - top;

            double tl, tr, bl, br;

            tl = img[top][left].red;
            tr = img[top][right].red;
            bl = img[bottom][left].red;
            br = img[bottom][right].red;
            const double red = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                                   tb_frac*((1-lr_frac)*bl + lr_frac*br);

            tl = img[top][left].green;
            tr = img[top][right].green;
            bl = img[bottom][left].green;
            br = img[bottom][right].green;
            const double green = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                                   tb_frac*((1-lr_frac)*bl + lr_frac*br);

            tl = img[top][left].blue;
            tr = img[top][right].blue;
            bl = img[bottom][left].blue;
            br = img[bottom][right].blue;
            const double blue = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                                   tb_frac*((1-lr_frac)*bl + lr_frac*br);
                            
            rgb_pixel temp;
            assign_pixel(temp.red, red);
            assign_pixel(temp.green, green);
            assign_pixel(temp.blue, blue);
            assign_pixel(result, temp);
            return true;
        }
    };

// ----------------------------------------------------------------------------------------

    class interpolate_quadratic
    {
        template <typename T>
        struct is_rgb_image 
        {
            const static bool value = pixel_traits<typename T::type>::rgb;
        };

    public:

        template <typename T, typename image_type, typename pixel_type>
        typename disable_if<is_rgb_image<image_type>,bool>::type operator() (
            const image_type& img,
            const dlib::vector<T,2>& p,
            pixel_type& result
        ) const
        {
            COMPILE_TIME_ASSERT(pixel_traits<typename image_type::type>::has_alpha == false);

            const point pp(p);

            // if the interpolation goes outside img 
            if (!get_rect(img).contains(grow_rect(pp,1))) 
                return false;

            const long r = pp.y();
            const long c = pp.x();

            const double temp = interpolate(p-pp, 
                                    img[r-1][c-1],
                                    img[r-1][c  ],
                                    img[r-1][c+1],
                                    img[r  ][c-1],
                                    img[r  ][c  ],
                                    img[r  ][c+1],
                                    img[r+1][c-1],
                                    img[r+1][c  ],
                                    img[r+1][c+1]);

            assign_pixel(result, temp);
            return true;
        }

        template <typename T, typename image_type, typename pixel_type>
        typename enable_if<is_rgb_image<image_type>,bool>::type operator() (
            const image_type& img,
            const dlib::vector<T,2>& p,
            pixel_type& result
        ) const
        {
            COMPILE_TIME_ASSERT(pixel_traits<typename image_type::type>::has_alpha == false);

            const point pp(p);

            // if the interpolation goes outside img 
            if (!get_rect(img).contains(grow_rect(pp,1))) 
                return false;

            const long r = pp.y();
            const long c = pp.x();

            const double red = interpolate(p-pp, 
                            img[r-1][c-1].red,
                            img[r-1][c  ].red,
                            img[r-1][c+1].red,
                            img[r  ][c-1].red,
                            img[r  ][c  ].red,
                            img[r  ][c+1].red,
                            img[r+1][c-1].red,
                            img[r+1][c  ].red,
                            img[r+1][c+1].red);
            const double green = interpolate(p-pp, 
                            img[r-1][c-1].green,
                            img[r-1][c  ].green,
                            img[r-1][c+1].green,
                            img[r  ][c-1].green,
                            img[r  ][c  ].green,
                            img[r  ][c+1].green,
                            img[r+1][c-1].green,
                            img[r+1][c  ].green,
                            img[r+1][c+1].green);
            const double blue = interpolate(p-pp, 
                            img[r-1][c-1].blue,
                            img[r-1][c  ].blue,
                            img[r-1][c+1].blue,
                            img[r  ][c-1].blue,
                            img[r  ][c  ].blue,
                            img[r  ][c+1].blue,
                            img[r+1][c-1].blue,
                            img[r+1][c  ].blue,
                            img[r+1][c+1].blue);


            rgb_pixel temp;
            assign_pixel(temp.red, red);
            assign_pixel(temp.green, green);
            assign_pixel(temp.blue, blue);
            assign_pixel(result, temp);

            return true;
        }

    private:

        /*  tl tm tr
            ml mm mr
            bl bm br
        */
        // The above is the pixel layout in our little 3x3 neighborhood.  interpolate() will 
        // fit a quadratic to these 9 pixels and then use that quadratic to find the interpolated 
        // value at point p.
        inline double interpolate(
            const dlib::vector<double,2>& p,
            double tl, double tm, double tr, 
            double ml, double mm, double mr, 
            double bl, double bm, double br
        ) const
        {
            matrix<double,6,1> w;
            // x
            w(0) = (tr + mr + br - tl - ml - bl)*0.16666666666;
            // y
            w(1) = (bl + bm + br - tl - tm - tr)*0.16666666666;
            // x^2
            w(2) = (tl + tr + ml + mr + bl + br)*0.16666666666 - (tm + mm + bm)*0.333333333;
            // x*y
            w(3) = (tl - tr - bl + br)*0.25;
            // y^2
            w(4) = (tl + tm + tr + bl + bm + br)*0.16666666666 - (ml + mm + mr)*0.333333333;
            // 1 (constant term)
            w(5) = (tm + ml + mr + bm)*0.222222222 - (tl + tr + bl + br)*0.11111111 + (mm)*0.55555556;

            const double x = p.x();
            const double y = p.y();

            matrix<double,6,1> z;
            z = x, y, x*x, x*y, y*y, 1.0;
                            
            return dot(w,z);
        }
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class black_background
    {
    public:
        template <typename pixel_type>
        void operator() ( pixel_type& p) const { assign_pixel(p, 0); }
    };

    class white_background
    {
    public:
        template <typename pixel_type>
        void operator() ( pixel_type& p) const { assign_pixel(p, 255); }
    };

    class no_background
    {
    public:
        template <typename pixel_type>
        void operator() ( pixel_type& ) const { }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2,
        typename interpolation_type,
        typename point_mapping_type,
        typename background_type
        >
    void transform_image (
        const image_type1& in_img,
        image_type2& out_img,
        const interpolation_type& interp,
        const point_mapping_type& map_point,
        const background_type& set_background,
        const rectangle& area
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( get_rect(out_img).contains(area) == true &&
                     is_same_object(in_img, out_img) == false ,
            "\t void transform_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t get_rect(out_img).contains(area): " << get_rect(out_img).contains(area)
            << "\n\t get_rect(out_img): " << get_rect(out_img)
            << "\n\t area:              " << area
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );


        for (long r = area.top(); r <= area.bottom(); ++r)
        {
            for (long c = area.left(); c <= area.right(); ++c)
            {
                if (!interp(in_img, map_point(dlib::vector<double,2>(c,r)), out_img[r][c]))
                    set_background(out_img[r][c]);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2,
        typename interpolation_type,
        typename point_mapping_type,
        typename background_type
        >
    void transform_image (
        const image_type1& in_img,
        image_type2& out_img,
        const interpolation_type& interp,
        const point_mapping_type& map_point,
        const background_type& set_background
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void transform_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        transform_image(in_img, out_img, interp, map_point, set_background, get_rect(out_img));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2,
        typename interpolation_type,
        typename point_mapping_type
        >
    void transform_image (
        const image_type1& in_img,
        image_type2& out_img,
        const interpolation_type& interp,
        const point_mapping_type& map_point
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void transform_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );


        transform_image(in_img, out_img, interp, map_point, black_background(), get_rect(out_img));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2,
        typename interpolation_type
        >
    void rotate_image (
        const image_type1& in_img,
        image_type2& out_img,
        double angle,
        const interpolation_type& interp
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void rotate_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        const rectangle rimg = get_rect(in_img);


        // figure out bounding box for rotated rectangle
        rectangle rect;
        rect += rotate_point(center(rimg), rimg.tl_corner(), -angle);
        rect += rotate_point(center(rimg), rimg.tr_corner(), -angle);
        rect += rotate_point(center(rimg), rimg.bl_corner(), -angle);
        rect += rotate_point(center(rimg), rimg.br_corner(), -angle);
        out_img.set_size(rect.height(), rect.width());

        const matrix<double,2,2> R = rotation_matrix(angle);

        transform_image(in_img, out_img, interp, 
                        point_transform_affine(R, -R*dcenter(get_rect(out_img)) + dcenter(rimg)));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2
        >
    void rotate_image (
        const image_type1& in_img,
        image_type2& out_img,
        double angle
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void rotate_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        rotate_image(in_img, out_img, angle, interpolate_quadratic());
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class helper_resize_image 
        {
        public:
            helper_resize_image(
                double x_scale_,
                double y_scale_
            ):
                x_scale(x_scale_),
                y_scale(y_scale_)
            {}

            dlib::vector<double,2> operator() (
                const dlib::vector<double,2>& p
            ) const
            {
                return dlib::vector<double,2>(p.x()*x_scale, p.y()*y_scale);
            }

        private:
            const double x_scale;
            const double y_scale;
        };
    }

    template <
        typename image_type1,
        typename image_type2,
        typename interpolation_type
        >
    void resize_image (
        const image_type1& in_img,
        image_type2& out_img,
        const interpolation_type& interp
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void resize_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        const double x_scale = (in_img.nc()-1)/(double)std::max<long>((out_img.nc()-1),1);
        const double y_scale = (in_img.nr()-1)/(double)std::max<long>((out_img.nr()-1),1);
        transform_image(in_img, out_img, interp, 
                        dlib::impl::helper_resize_image(x_scale,y_scale));
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    struct is_rgb_image { const static bool value = pixel_traits<typename image_type::type>::rgb; };
    template <typename image_type>
    struct is_grayscale_image { const static bool value = pixel_traits<typename image_type::type>::grayscale; };

    // This is an optimized version of resize_image for the case where bilinear
    // interpolation is used.
    template <
        typename image_type1,
        typename image_type2
        >
    typename disable_if_c<(is_rgb_image<image_type1>::value&&is_rgb_image<image_type2>::value) || 
                          (is_grayscale_image<image_type1>::value&&is_grayscale_image<image_type2>::value)>::type 
    resize_image (
        const image_type1& in_img,
        image_type2& out_img,
        interpolate_bilinear
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void resize_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );
        if (out_img.nr() <= 1 || out_img.nc() <= 1)
        {
            assign_all_pixels(out_img, 0);
            return;
        }


        typedef typename image_type1::type T;
        typedef typename image_type2::type U;
        const double x_scale = (in_img.nc()-1)/(double)std::max<long>((out_img.nc()-1),1);
        const double y_scale = (in_img.nr()-1)/(double)std::max<long>((out_img.nr()-1),1);
        double y = -y_scale;
        for (long r = 0; r < out_img.nr(); ++r)
        {
            y += y_scale;
            const long top    = static_cast<long>(std::floor(y));
            const long bottom = std::min(top+1, in_img.nr()-1);
            const double tb_frac = y - top;
            double x = -x_scale;
            if (!pixel_traits<U>::rgb)
            {
                for (long c = 0; c < out_img.nc(); ++c)
                {
                    x += x_scale;
                    const long left   = static_cast<long>(std::floor(x));
                    const long right  = std::min(left+1, in_img.nc()-1);
                    const double lr_frac = x - left;

                    double tl = 0, tr = 0, bl = 0, br = 0;

                    assign_pixel(tl, in_img[top][left]);
                    assign_pixel(tr, in_img[top][right]);
                    assign_pixel(bl, in_img[bottom][left]);
                    assign_pixel(br, in_img[bottom][right]);

                    double temp = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                        tb_frac*((1-lr_frac)*bl + lr_frac*br);

                    assign_pixel(out_img[r][c], temp);
                }
            }
            else
            {
                for (long c = 0; c < out_img.nc(); ++c)
                {
                    x += x_scale;
                    const long left   = static_cast<long>(std::floor(x));
                    const long right  = std::min(left+1, in_img.nc()-1);
                    const double lr_frac = x - left;

                    const T tl = in_img[top][left];
                    const T tr = in_img[top][right];
                    const T bl = in_img[bottom][left];
                    const T br = in_img[bottom][right];

                    T temp;
                    assign_pixel(temp, 0);
                    vector_to_pixel(temp, 
                        (1-tb_frac)*((1-lr_frac)*pixel_to_vector<double>(tl) + lr_frac*pixel_to_vector<double>(tr)) + 
                            tb_frac*((1-lr_frac)*pixel_to_vector<double>(bl) + lr_frac*pixel_to_vector<double>(br)));
                    assign_pixel(out_img[r][c], temp);
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename enable_if<is_grayscale_image<image_type> >::type resize_image (
        const image_type& in_img,
        image_type& out_img,
        interpolate_bilinear
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void resize_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );
        if (out_img.nr() <= 1 || out_img.nc() <= 1)
        {
            assign_all_pixels(out_img, 0);
            return;
        }

        typedef typename image_type::type T;
        const double x_scale = (in_img.nc()-1)/(double)std::max<long>((out_img.nc()-1),1);
        const double y_scale = (in_img.nr()-1)/(double)std::max<long>((out_img.nr()-1),1);
        double y = -y_scale;
        for (long r = 0; r < out_img.nr(); ++r)
        {
            y += y_scale;
            const long top    = static_cast<long>(std::floor(y));
            const long bottom = std::min(top+1, in_img.nr()-1);
            const double tb_frac = y - top;
            double x = -4*x_scale;

            const simd4f _tb_frac = tb_frac;
            const simd4f _inv_tb_frac = 1-tb_frac;
            const simd4f _x_scale = 4*x_scale;
            simd4f _x(x, x+x_scale, x+2*x_scale, x+3*x_scale);
            long c = 0;
            const long num_simd_blocks = static_cast<long>(out_img.nc()-3*x_scale)-1;
            for (; c < num_simd_blocks; c+=4)
            {
                _x += _x_scale;
                simd4f _left = floor(_x);
                simd4i left = simd4i(_left);

                simd4f _lr_frac = _x-_left;
                simd4f _inv_lr_frac = 1-_lr_frac; 
                simd4i right = left+1;

                simd4f tlf = _inv_tb_frac*_inv_lr_frac;
                simd4f trf = _inv_tb_frac*_lr_frac;
                simd4f blf = _tb_frac*_inv_lr_frac;
                simd4f brf = _tb_frac*_lr_frac;

                int32 fleft[4];
                int32 fright[4];
                left.store(fleft);
                right.store(fright);

                simd4f tl(in_img[top][fleft[0]],     in_img[top][fleft[1]],     in_img[top][fleft[2]],     in_img[top][fleft[3]]);
                simd4f tr(in_img[top][fright[0]],    in_img[top][fright[1]],    in_img[top][fright[2]],    in_img[top][fright[3]]);
                simd4f bl(in_img[bottom][fleft[0]],  in_img[bottom][fleft[1]],  in_img[bottom][fleft[2]],  in_img[bottom][fleft[3]]);
                simd4f br(in_img[bottom][fright[0]], in_img[bottom][fright[1]], in_img[bottom][fright[2]], in_img[bottom][fright[3]]);

                simd4i out = simd4i(tlf*tl + trf*tr + blf*bl + brf*br);
                int32 fout[4];
                out.store(fout);

                out_img[r][c]   = static_cast<T>(fout[0]);
                out_img[r][c+1] = static_cast<T>(fout[1]);
                out_img[r][c+2] = static_cast<T>(fout[2]);
                out_img[r][c+3] = static_cast<T>(fout[3]);
            }
            x = -x_scale + c*x_scale;
            for (; c < out_img.nc(); ++c)
            {
                x += x_scale;
                const long left   = static_cast<long>(std::floor(x));
                const long right  = std::min(left+1, in_img.nc()-1);
                const float lr_frac = x - left;

                float tl = 0, tr = 0, bl = 0, br = 0;

                assign_pixel(tl, in_img[top][left]);
                assign_pixel(tr, in_img[top][right]);
                assign_pixel(bl, in_img[bottom][left]);
                assign_pixel(br, in_img[bottom][right]);

                float temp = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                    tb_frac*((1-lr_frac)*bl + lr_frac*br);

                assign_pixel(out_img[r][c], temp);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename enable_if<is_rgb_image<image_type> >::type resize_image (
        const image_type& in_img,
        image_type& out_img,
        interpolate_bilinear
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void resize_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );
        if (out_img.nr() <= 1 || out_img.nc() <= 1)
        {
            assign_all_pixels(out_img, 0);
            return;
        }


        typedef typename image_type::type T;
        const double x_scale = (in_img.nc()-1)/(double)std::max<long>((out_img.nc()-1),1);
        const double y_scale = (in_img.nr()-1)/(double)std::max<long>((out_img.nr()-1),1);
        double y = -y_scale;
        for (long r = 0; r < out_img.nr(); ++r)
        {
            y += y_scale;
            const long top    = static_cast<long>(std::floor(y));
            const long bottom = std::min(top+1, in_img.nr()-1);
            const double tb_frac = y - top;
            double x = -4*x_scale;

            const simd4f _tb_frac = tb_frac;
            const simd4f _inv_tb_frac = 1-tb_frac;
            const simd4f _x_scale = 4*x_scale;
            simd4f _x(x, x+x_scale, x+2*x_scale, x+3*x_scale);
            long c = 0;
            const long num_simd_blocks = static_cast<long>(out_img.nc()-3*x_scale)-1;
            for (; c < num_simd_blocks; c+=4)
            {
                _x += _x_scale;
                simd4f _left = floor(_x);
                simd4i left = simd4i(_left);
                simd4f lr_frac = _x-_left;
                simd4f _inv_lr_frac = 1-lr_frac; 
                simd4i right = left+1;

                simd4f tlf = _inv_tb_frac*_inv_lr_frac;
                simd4f trf = _inv_tb_frac*lr_frac;
                simd4f blf = _tb_frac*_inv_lr_frac;
                simd4f brf = _tb_frac*lr_frac;

                int32 fleft[4];
                int32 fright[4];
                left.store(fleft);
                right.store(fright);

                simd4f tl(in_img[top][fleft[0]].red,     in_img[top][fleft[1]].red,     in_img[top][fleft[2]].red,     in_img[top][fleft[3]].red);
                simd4f tr(in_img[top][fright[0]].red,    in_img[top][fright[1]].red,    in_img[top][fright[2]].red,    in_img[top][fright[3]].red);
                simd4f bl(in_img[bottom][fleft[0]].red,  in_img[bottom][fleft[1]].red,  in_img[bottom][fleft[2]].red,  in_img[bottom][fleft[3]].red);
                simd4f br(in_img[bottom][fright[0]].red, in_img[bottom][fright[1]].red, in_img[bottom][fright[2]].red, in_img[bottom][fright[3]].red);

                simd4i out = simd4i(tlf*tl + trf*tr + blf*bl + brf*br);
                int32 fout[4];
                out.store(fout);

                out_img[r][c].red   = static_cast<unsigned char>(fout[0]);
                out_img[r][c+1].red = static_cast<unsigned char>(fout[1]);
                out_img[r][c+2].red = static_cast<unsigned char>(fout[2]);
                out_img[r][c+3].red = static_cast<unsigned char>(fout[3]);


                tl = simd4f(in_img[top][fleft[0]].green,    in_img[top][fleft[1]].green,    in_img[top][fleft[2]].green,    in_img[top][fleft[3]].green);
                tr = simd4f(in_img[top][fright[0]].green,   in_img[top][fright[1]].green,   in_img[top][fright[2]].green,   in_img[top][fright[3]].green);
                bl = simd4f(in_img[bottom][fleft[0]].green, in_img[bottom][fleft[1]].green, in_img[bottom][fleft[2]].green, in_img[bottom][fleft[3]].green);
                br = simd4f(in_img[bottom][fright[0]].green, in_img[bottom][fright[1]].green, in_img[bottom][fright[2]].green, in_img[bottom][fright[3]].green);
                out = simd4i(tlf*tl + trf*tr + blf*bl + brf*br);
                out.store(fout);
                out_img[r][c].green   = static_cast<unsigned char>(fout[0]);
                out_img[r][c+1].green = static_cast<unsigned char>(fout[1]);
                out_img[r][c+2].green = static_cast<unsigned char>(fout[2]);
                out_img[r][c+3].green = static_cast<unsigned char>(fout[3]);


                tl = simd4f(in_img[top][fleft[0]].blue,     in_img[top][fleft[1]].blue,     in_img[top][fleft[2]].blue,     in_img[top][fleft[3]].blue);
                tr = simd4f(in_img[top][fright[0]].blue,    in_img[top][fright[1]].blue,    in_img[top][fright[2]].blue,    in_img[top][fright[3]].blue);
                bl = simd4f(in_img[bottom][fleft[0]].blue,  in_img[bottom][fleft[1]].blue,  in_img[bottom][fleft[2]].blue,  in_img[bottom][fleft[3]].blue);
                br = simd4f(in_img[bottom][fright[0]].blue, in_img[bottom][fright[1]].blue, in_img[bottom][fright[2]].blue, in_img[bottom][fright[3]].blue);
                out = simd4i(tlf*tl + trf*tr + blf*bl + brf*br);
                out.store(fout);
                out_img[r][c].blue   = static_cast<unsigned char>(fout[0]);
                out_img[r][c+1].blue = static_cast<unsigned char>(fout[1]);
                out_img[r][c+2].blue = static_cast<unsigned char>(fout[2]);
                out_img[r][c+3].blue = static_cast<unsigned char>(fout[3]);
            }
            x = -x_scale + c*x_scale;
            for (; c < out_img.nc(); ++c)
            {
                x += x_scale;
                const long left   = static_cast<long>(std::floor(x));
                const long right  = std::min(left+1, in_img.nc()-1);
                const double lr_frac = x - left;

                const T tl = in_img[top][left];
                const T tr = in_img[top][right];
                const T bl = in_img[bottom][left];
                const T br = in_img[bottom][right];

                T temp;
                assign_pixel(temp, 0);
                vector_to_pixel(temp, 
                    (1-tb_frac)*((1-lr_frac)*pixel_to_vector<double>(tl) + lr_frac*pixel_to_vector<double>(tr)) + 
                    tb_frac*((1-lr_frac)*pixel_to_vector<double>(bl) + lr_frac*pixel_to_vector<double>(br)));
                assign_pixel(out_img[r][c], temp);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2
        >
    void resize_image (
        const image_type1& in_img,
        image_type2& out_img
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void resize_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        resize_image(in_img, out_img, interpolate_bilinear());
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2
        >
    void flip_image_left_right (
        const image_type1& in_img,
        image_type2& out_img
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void rotate_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        assign_image(out_img, fliplr(mat(in_img)));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2
        >
    void flip_image_up_down (
        const image_type1& in_img,
        image_type2& out_img
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void rotate_image()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        assign_image(out_img, flipud(mat(in_img)));
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline rectangle flip_rect_left_right (
            const rectangle& rect,
            const rectangle& window 
        )
        {
            rectangle temp;
            temp.top() = rect.top();
            temp.bottom() = rect.bottom();

            const long left_dist = rect.left()-window.left();

            temp.right() = window.right()-left_dist; 
            temp.left()  = temp.right()-rect.width()+1; 
            return temp;
        }
    }

    template <
        typename image_type
        >
    void add_image_left_right_flips (
        dlib::array<image_type>& images,
        std::vector<std::vector<rectangle> >& objects
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( images.size() == objects.size(),
            "\t void add_image_left_right_flips()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t images.size():  " << images.size() 
            << "\n\t objects.size(): " << objects.size() 
            );

        image_type temp;
        std::vector<rectangle> rects;

        const unsigned long num = images.size();
        for (unsigned long j = 0; j < num; ++j)
        {
            flip_image_left_right(images[j], temp);

            rects.clear();
            for (unsigned long i = 0; i < objects[j].size(); ++i)
                rects.push_back(impl::flip_rect_left_right(objects[j][i], get_rect(images[j])));

            images.push_back(temp);
            objects.push_back(rects);
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2,
        typename pyramid_type,
        typename interpolation_type
        >
    void pyramid_up (
        const image_type1& in_img,
        image_type2& out_img,
        const pyramid_type& pyr,
        const interpolation_type& interp
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void pyramid_up()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        if (in_img.size() == 0)
        {
            out_img.clear();
            return;
        }

        rectangle rect = get_rect(in_img);
        rectangle uprect = pyr.rect_up(rect);
        if (uprect.is_empty())
        {
            out_img.clear();
            return;
        }
        out_img.set_size(uprect.bottom()+1, uprect.right()+1);

        resize_image(in_img, out_img, interp);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type1,
        typename image_type2,
        typename pyramid_type
        >
    void pyramid_up (
        const image_type1& in_img,
        image_type2& out_img,
        const pyramid_type& pyr
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT( is_same_object(in_img, out_img) == false ,
            "\t void pyramid_up()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t is_same_object(in_img, out_img):  " << is_same_object(in_img, out_img)
            );

        pyramid_up(in_img, out_img, pyr, interpolate_bilinear());
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_INTERPOlATION__

