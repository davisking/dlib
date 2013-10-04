// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_fHOG_H__
#define DLIB_fHOG_H__

#include "fhog_abstract.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../array.h"
#include "../geometry.h"
#include "assign_image.h"
#include "draw.h"
#include "interpolation.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl_fhog
    {
        template <typename image_type>
        inline typename dlib::enable_if_c<pixel_traits<typename image_type::type>::rgb>::type get_gradient (
            const int r,
            const int c,
            const image_type& img,
            matrix<double,2,1>& grad,
            double& len
        )
        {
            matrix<double,2,1> grad2, grad3;
            // get the red gradient
            grad = (int)img[r][c+1].red-(int)img[r][c-1].red, 
                 (int)img[r+1][c].red-(int)img[r-1][c].red;
            len = length_squared(grad);

            // get the green gradient
            grad2 = (int)img[r][c+1].green-(int)img[r][c-1].green, 
                  (int)img[r+1][c].green-(int)img[r-1][c].green;
            double v2 = length_squared(grad2);

            // get the blue gradient
            grad3 = (int)img[r][c+1].blue-(int)img[r][c-1].blue, 
                  (int)img[r+1][c].blue-(int)img[r-1][c].blue;
            double v3 = length_squared(grad3);

            // pick color with strongest gradient
            if (v2 > len) 
            {
                len = v2;
                grad = grad2;
            } 
            if (v3 > len) 
            {
                len = v3;
                grad = grad3;
            }
        }

    // ------------------------------------------------------------------------------------

        template <typename image_type>
        inline typename dlib::disable_if_c<pixel_traits<typename image_type::type>::rgb>::type get_gradient (
            const int r,
            const int c,
            const image_type& img,
            matrix<double,2,1>& grad,
            double& len
        )
        {
            grad = (int)get_pixel_intensity(img[r][c+1])-(int)get_pixel_intensity(img[r][c-1]), 
            (int)get_pixel_intensity(img[r+1][c])-(int)get_pixel_intensity(img[r-1][c]);
            len = length_squared(grad);
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename mm1, typename mm2>
        void set_hog (
            dlib::array<array2d<T,mm1>,mm2>& hog,
            int o,
            int x, 
            int y,
            const double& value
        )
        {
            hog[o][y][x] = value;
        }

        template <typename T, typename mm1, typename mm2>
        void init_hog (
            dlib::array<array2d<T,mm1>,mm2>& hog,
            int hog_nr,
            int hog_nc
        )
        {
            const int num_hog_bands = 27+4;
            hog.resize(num_hog_bands);
            for (int i = 0; i < num_hog_bands; ++i)
            {
                hog[i].set_size(hog_nr, hog_nc);
            }
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename mm>
        void set_hog (
            array2d<matrix<T,31,1>,mm>& hog,
            int o,
            int x, 
            int y,
            const double& value
        )
        {
            hog[y][x](o) = value;
        }

        template <typename T, typename mm>
        void init_hog (
            array2d<matrix<T,31,1>,mm>& hog,
            int hog_nr,
            int hog_nc
        )
        {
            hog.set_size(hog_nr, hog_nc);
        }

    // ------------------------------------------------------------------------------------

        template <
            typename image_type, 
            typename out_type
            >
        void impl_extract_fhog_features(
            const image_type& img, 
            out_type& hog, 
            int cell_size
        ) 
        {
            /*
                This function implements the HOG feature extraction method described in 
                the paper:
                    P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
                    Object Detection with Discriminatively Trained Part Based Models
                    IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010

                Moreover, this function is derived from the HOG feature extraction code
                from the features.cc file in the voc-releaseX code (see
                http://people.cs.uchicago.edu/~rbg/latent/) which is has the following
                license (note that the code has been modified to work with grayscale and
                color as well as planar and interlaced input and output formats):

                Copyright (C) 2011, 2012 Ross Girshick, Pedro Felzenszwalb
                Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
                Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan

                Permission is hereby granted, free of charge, to any person obtaining
                a copy of this software and associated documentation files (the
                "Software"), to deal in the Software without restriction, including
                without limitation the rights to use, copy, modify, merge, publish,
                distribute, sublicense, and/or sell copies of the Software, and to
                permit persons to whom the Software is furnished to do so, subject to
                the following conditions:

                The above copyright notice and this permission notice shall be
                included in all copies or substantial portions of the Software.

                THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
                EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
                MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
                NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
                LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
                OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
                WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
            */

            // unit vectors used to compute gradient orientation
            matrix<double,2,1> directions[9];
            directions[0] =  1.0000, 0.0000; 
            directions[1] =  0.9397, 0.3420;
            directions[2] =  0.7660, 0.6428;
            directions[3] =  0.500,  0.8660;
            directions[4] =  0.1736, 0.9848;
            directions[5] = -0.1736, 0.9848;
            directions[6] = -0.5000, 0.8660;
            directions[7] = -0.7660, 0.6428;
            directions[8] = -0.9397, 0.3420;



            // First we allocate memory for caching orientation histograms & their norms.
            const int cells_nr = (int)((double)img.nr()/(double)cell_size + 0.5);
            const int cells_nc = (int)((double)img.nc()/(double)cell_size + 0.5);

            if (cells_nr == 0 || cells_nc == 0)
            {
                hog.clear();
                return;
            }

            array2d<matrix<float,18,1> > hist(cells_nr, cells_nc);
            for (long r = 0; r < hist.nr(); ++r)
            {
                for (long c = 0; c < hist.nc(); ++c)
                {
                    hist[r][c] = 0;
                }
            }

            array2d<float> norm(cells_nr, cells_nc);
            assign_all_pixels(norm, 0);

            // memory for HOG features
            const int hog_nr = std::max(cells_nr-2, 0);
            const int hog_nc = std::max(cells_nc-2, 0);
            if (hog_nr == 0 || hog_nc == 0)
            {
                hog.clear();
                return;
            }
            init_hog(hog, hog_nr, hog_nc);

            const int visible_nr = cells_nr*cell_size;
            const int visible_nc = cells_nc*cell_size;

            // First populate the gradient histograms
            for (int y = 1; y < visible_nr-1; y++) 
            {
                for (int x = 1; x < visible_nc-1; x++) 
                {
                    const int r = std::min<int>(y, img.nr()-2);
                    const int c = std::min<int>(x, img.nc()-2);

                    matrix<double,2,1> grad;
                    double v;
                    get_gradient(r,c,img,grad,v);
                    v = std::sqrt(v);

                    // snap to one of 18 orientations
                    double best_dot = 0;
                    int best_o = 0;
                    for (int o = 0; o < 9; o++) 
                    {
                        const double dot = dlib::dot(directions[o], grad); 
                        if (dot > best_dot) 
                        {
                            best_dot = dot;
                            best_o = o;
                        } 
                        else if (-dot > best_dot) 
                        {
                            best_dot = -dot;
                            best_o = o+9;
                        }
                    }

                    // add to 4 histograms around pixel using bilinear interpolation
                    double xp = ((double)x+0.5)/(double)cell_size - 0.5;
                    double yp = ((double)y+0.5)/(double)cell_size - 0.5;
                    int ixp = (int)std::floor(xp);
                    int iyp = (int)std::floor(yp);
                    double vx0 = xp-ixp;
                    double vy0 = yp-iyp;
                    double vx1 = 1.0-vx0;
                    double vy1 = 1.0-vy0;

                    if (ixp >= 0 && iyp >= 0) 
                        hist[iyp][ixp](best_o) += vy1*vx1*v;

                    if (iyp+1 < cells_nr && ixp >= 0) 
                        hist[iyp+1][ixp](best_o) += vy0*vx1*v;

                    if (iyp >= 0 && ixp+1 < cells_nc) 
                        hist[iyp][ixp+1](best_o) += vy1*vx0*v;

                    if (ixp+1 < cells_nc && iyp+1 < cells_nr) 
                        hist[iyp+1][ixp+1](best_o) += vy0*vx0*v;
                }
            }

            // compute energy in each block by summing over orientations
            for (int r = 0; r < cells_nr; ++r)
            {
                for (int c = 0; c < cells_nc; ++c)
                {
                    for (int o = 0; o < 9; o++) 
                    {
                        norm[r][c] += (hist[r][c](o) + hist[r][c](o+9)) * (hist[r][c](o) + hist[r][c](o+9));
                    }
                }
            }

            const double eps = 0.0001;
            // compute features
            for (int y = 0; y < hog_nr; y++) 
            {
                for (int x = 0; x < hog_nc; x++) 
                {
                    double n1, n2, n3, n4;

                    n1 = 1.0 / std::sqrt(norm[y+1][x+1] + norm[y+1][x+2] + norm[y+2][x+1] + norm[y+2][x+2] + eps);
                    n2 = 1.0 / std::sqrt(norm[y][x+1]   + norm[y][x+2]   + norm[y+1][x+1] + norm[y+1][x+2] + eps);
                    n3 = 1.0 / std::sqrt(norm[y+1][x]   + norm[y+1][x+1] + norm[y+2][x]   + norm[y+2][x+1] + eps);
                    n4 = 1.0 / std::sqrt(norm[y][x]     + norm[y][x+1]   + norm[y+1][x]   + norm[y+1][x+1] + eps);

                    double t1 = 0;
                    double t2 = 0;
                    double t3 = 0;
                    double t4 = 0;

                    // contrast-sensitive features
                    for (int o = 0; o < 18; o++) 
                    {
                        double h1 = std::min(hist[y+1][x+1](o) * n1, 0.2);
                        double h2 = std::min(hist[y+1][x+1](o) * n2, 0.2);
                        double h3 = std::min(hist[y+1][x+1](o) * n3, 0.2);
                        double h4 = std::min(hist[y+1][x+1](o) * n4, 0.2);
                        set_hog(hog,o,x,y,0.5 * (h1 + h2 + h3 + h4));
                        t1 += h1;
                        t2 += h2;
                        t3 += h3;
                        t4 += h4;
                    }

                    // contrast-insensitive features
                    for (int o = 0; o < 9; o++) 
                    {
                        double sum = hist[y+1][x+1](o) + hist[y+1][x+1](o+9);
                        double h1 = std::min(sum * n1, 0.2);
                        double h2 = std::min(sum * n2, 0.2);
                        double h3 = std::min(sum * n3, 0.2);
                        double h4 = std::min(sum * n4, 0.2);
                        set_hog(hog,o+18,x,y, 0.5 * (h1 + h2 + h3 + h4));
                    }

                    // texture features
                    set_hog(hog,27,x,y, 0.2357 * t1);
                    set_hog(hog,28,x,y, 0.2357 * t2);
                    set_hog(hog,29,x,y, 0.2357 * t3);
                    set_hog(hog,30,x,y, 0.2357 * t4);
                }
            }
        }

    // ------------------------------------------------------------------------------------

        inline void create_fhog_bar_images (
            dlib::array<matrix<float> >& mbars,
            const long w
        )
        {
            const long bdims = 9;
            // Make the oriented lines we use to draw on each HOG cell.
            mbars.resize(bdims);
            dlib::array<array2d<unsigned char> > bars(bdims);
            array2d<unsigned char> temp(w,w);
            for (unsigned long i = 0; i < bars.size(); ++i)
            {
                assign_all_pixels(temp, 0);
                draw_line(temp, point(w/2,0), point(w/2,w-1), 255);
                rotate_image(temp, bars[i], i*-pi/bars.size());

                mbars[i] = subm(matrix_cast<float>(mat(bars[i])), centered_rect(get_rect(bars[i]),w,w) );
            }
        }

    } // end namespace impl_fhog

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename image_type, 
        typename T, 
        typename mm1, 
        typename mm2
        >
    void extract_fhog_features(
        const image_type& img, 
        dlib::array<array2d<T,mm1>,mm2>& hog, 
        int cell_size = 8
    ) 
    {
        return impl_fhog::impl_extract_fhog_features(img, hog, cell_size);
    }

    template <
        typename image_type, 
        typename T, 
        typename mm
        >
    void extract_fhog_features(
        const image_type& img, 
        array2d<matrix<T,31,1>,mm>& hog, 
        int cell_size = 8
    ) 
    {
        return impl_fhog::impl_extract_fhog_features(img, hog, cell_size);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    inline point image_to_fhog (
        point p,
        int cell_size = 8
    )
    {
        // There is a one pixel border around the imag.
        p -= point(1,1);
        // There is also a 1 "cell" border around the HOG image formation.
        return p/cell_size - point(1,1);
    }

// ----------------------------------------------------------------------------------------

    inline point fhog_to_image (
        point p,
        int cell_size = 8
    )
    {
        // Convert to image space and then set to the center of the cell.
        return (p+point(1,1))*cell_size + point(1,1) + point(cell_size/2,cell_size/2);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename mm1, 
        typename mm2
        >
    matrix<unsigned char> draw_fhog(
        const dlib::array<array2d<T,mm1>,mm2>& hog,
        const long w = 15
    )
    {
        dlib::array<matrix<float> > mbars;
        impl_fhog::create_fhog_bar_images(mbars,w);

        // now draw the bars onto the HOG cells
        matrix<float> himg(hog[0].nr()*w, hog[0].nc()*w);
        himg = 0;
        for (unsigned long d = 0; d < mbars.size(); ++d)
        {
            for (long r = 0; r < himg.nr(); r+=w)
            {
                for (long c = 0; c < himg.nc(); c+=w)
                {
                    const float val = hog[d][r/w][c/w] + hog[d+mbars.size()][r/w][c/w] + hog[d+mbars.size()*2][r/w][c/w];
                    if (val > 0)
                    {
                        set_subm(himg, r, c, w, w) += val*mbars[d%mbars.size()];
                    }
                }
            }
        }

        const double thresh = mean(himg) + 4*stddev(himg);
        return matrix_cast<unsigned char>(upperbound(round(himg*255/thresh),255));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename mm
        >
    matrix<unsigned char> draw_fhog(
        const array2d<matrix<T,31,1>,mm>& hog,
        const long w = 15
    )
    {
        dlib::array<matrix<float> > mbars;
        impl_fhog::create_fhog_bar_images(mbars,w);

        // now draw the bars onto the HOG cells
        matrix<float> himg(hog.nr()*w, hog.nc()*w);
        himg = 0;
        for (unsigned long d = 0; d < mbars.size(); ++d)
        {
            for (long r = 0; r < himg.nr(); r+=w)
            {
                for (long c = 0; c < himg.nc(); c+=w)
                {
                    const float val = hog[r/w][c/w](d) + hog[r/w][c/w](d+mbars.size()) + hog[r/w][c/w](d+mbars.size()*2);
                    if (val > 0)
                    {
                        set_subm(himg, r, c, w, w) += val*mbars[d%mbars.size()];
                    }
                }
            }
        }

        const double thresh = mean(himg) + 4*stddev(himg);
        return matrix_cast<unsigned char>(upperbound(round(himg*255/thresh),255));
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_fHOG_H__

