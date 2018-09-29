// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_THRESHOLDINg_
#define DLIB_THRESHOLDINg_ 

#include "../pixel.h"
#include "thresholding_abstract.h"
#include "equalize_histogram.h"
#include "../enable_if.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    const unsigned char on_pixel = 255;
    const unsigned char off_pixel = 0;

// ----------------------------------------------------------------------------------------
    
    namespace impl
    {
        template <
            typename U,
            typename V,
            typename basic_pixel_type
            >
        void partition_pixels_float_work (
            unsigned long begin,
            unsigned long end,
            U&& cumsum,
            V&& sorted,
            basic_pixel_type& pix_thresh,
            unsigned long& int_thresh
        ) 
        {

            auto histsum = [&](long begin, long end) 
            { 
                return end-begin;
            };
            auto histsumi = [&](long begin, long end) 
            { 
                return cumsum[end]-cumsum[begin]; 
            };

            // If we split the pixels into two groups, those < thresh (the left group) and
            // those >= thresh (the right group), what would the sum of absolute deviations of
            // each pixel from the mean of its group be?  total_abs(thresh) computes that
            // value.
            unsigned long left_idx = 0;
            unsigned long right_idx = 0;
            auto total_abs = [&](unsigned long thresh)
            {
                auto left_avg = histsumi(begin,thresh);
                auto tmp = histsum(begin,thresh);
                if (tmp != 0)
                    left_avg /= tmp;
                auto right_avg = histsumi(thresh,end);
                tmp = histsum(thresh,end);
                if (tmp != 0)
                    right_avg /= tmp;


                while(left_idx+1 < sorted.size() && sorted[left_idx] <= left_avg)
                    ++left_idx;
                while(right_idx+1 < sorted.size() && sorted[right_idx] <= right_avg)
                    ++right_idx;

                double score = 0;
                score += left_avg*histsum(begin,left_idx) - histsumi(begin,left_idx); 
                score -= left_avg*histsum(left_idx,thresh) - histsumi(left_idx,thresh); 
                score += right_avg*histsum(thresh,right_idx) - histsumi(thresh,right_idx); 
                score -= right_avg*histsum(right_idx,end) - histsumi(right_idx,end); 
                return score;
            };



            int_thresh = begin;
            double min_sad = std::numeric_limits<double>::infinity();
            for (unsigned long i = begin; i < end; ++i)
            {
                // You can't drop a threshold in-between pixels with identical values.  So
                // skip thresholds corresponding to this degenerate case.
                if (i > 0 && sorted[i-1]==sorted[i])
                    continue;

                double sad = total_abs(i);
                if (sad <= min_sad)
                {
                    min_sad = sad;
                    int_thresh = i;
                }
            }

            pix_thresh = sorted[int_thresh];
        }

        template <
            typename U,
            typename V,
            typename basic_pixel_type
            >
        void recursive_partition_pixels_float (
            unsigned long begin,
            unsigned long end,
            U&& cumsum,
            V&& sorted,
            basic_pixel_type& pix_thresh
        ) 
        {
            unsigned long int_thresh;
            partition_pixels_float_work(begin, end, cumsum, sorted, pix_thresh, int_thresh);
        }

        template <
            typename U,
            typename V,
            typename basic_pixel_type,
            typename ...T
            >
        void recursive_partition_pixels_float (
            unsigned long begin,
            unsigned long end,
            U&& cumsum,
            V&& sorted,
            basic_pixel_type& pix_thresh,
            T&& ...more_thresholds
        ) 
        {
            unsigned long int_thresh;
            partition_pixels_float_work(begin, end, cumsum, sorted, pix_thresh, int_thresh);
            recursive_partition_pixels_float(int_thresh, end, cumsum, sorted, more_thresholds...);
        }

        template <
            typename image_type,
            typename ...T
            >
        void partition_pixels_float (
            const image_type& img_,
            typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type& pix_thresh,
            T&& ...more_thresholds
        ) 
        {
            /*
              This is a version of partition_pixels() that doesn't use the histogram to
              perform a radix sort but rather uses std::sort() as the first processing
              step.  It is therefor useful in cases where the range of possible pixels is
              too large for the faster histogram version.
            */

            COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<image_type>::pixel_type>::has_alpha == false );

            typedef typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type basic_pixel_type;

            const_image_view<image_type> img(img_);

            std::vector<basic_pixel_type> sorted;
            sorted.reserve(img.size());

            for (long r = 0; r < img.nr(); ++r)
            {
                for (long c = 0; c < img.nc(); ++c)
                    sorted.emplace_back(get_pixel_intensity(img[r][c]));
            }
            std::sort(sorted.begin(), sorted.end());

            std::vector<double> cumsum;
            cumsum.reserve(sorted.size()+1);

            // create integral array 
            cumsum.emplace_back(0);
            for (auto& v : sorted)
                cumsum.emplace_back(cumsum.back()+v);



            recursive_partition_pixels_float(0, img.size(), cumsum, sorted, pix_thresh, more_thresholds...);

        }


        template <typename image_type>
        struct is_u16img_or_less
        {
            typedef typename image_traits<image_type>::pixel_type pixel_type;
            typedef typename pixel_traits<pixel_type>::basic_pixel_type basic_pixel_type;

            const static bool value = sizeof(basic_pixel_type) <= 2 && pixel_traits<pixel_type>::is_unsigned;
        };
    }

    template <
        typename image_type,
        typename ...T
        >
    typename disable_if<impl::is_u16img_or_less<image_type>>::type
    partition_pixels (
        const image_type& img,
        typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type& pix_thresh,
        T&& ...more_thresholds
    ) 
    {
        impl::partition_pixels_float(img, pix_thresh, more_thresholds...);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace impl 
    {
        template <
            typename U,
            typename basic_pixel_type
            >
        void partition_pixels_work (
            unsigned long begin,
            unsigned long end,
            U&& total_abs,
            basic_pixel_type& pix_thresh,
            unsigned long& int_thresh
        ) 
        {
            int_thresh = begin;
            double min_sad = std::numeric_limits<double>::infinity();
            for (unsigned long i = begin; i < end; ++i)
            {
                double sad = total_abs(begin, i);
                if (sad <= min_sad)
                {
                    min_sad = sad;
                    int_thresh = i;
                }
            }

            pix_thresh = int_thresh;
        }

        template <
            typename U,
            typename basic_pixel_type
            >
        void recursive_partition_pixels (
            unsigned long begin,
            unsigned long end,
            U&& total_abs,
            basic_pixel_type& pix_thresh
        ) 
        {
            unsigned long int_thresh;
            partition_pixels_work(begin, end, total_abs, pix_thresh, int_thresh);
        }

        template <
            typename U,
            typename basic_pixel_type,
            typename ...T
            >
        void recursive_partition_pixels (
            unsigned long begin,
            unsigned long end,
            U&& total_abs,
            basic_pixel_type& pix_thresh,
            T&& ...more_thresholds
        ) 
        {
            unsigned long int_thresh;
            partition_pixels_work(begin, end, total_abs, pix_thresh, int_thresh);
            recursive_partition_pixels(int_thresh, end, total_abs, more_thresholds...);
        }

    }

    template <
        typename image_type,
        typename ...T
        >
    typename enable_if<impl::is_u16img_or_less<image_type>>::type
    partition_pixels (
        const image_type& img,
        typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type& pix_thresh,
        T&& ...more_thresholds
    ) 
    {
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<image_type>::pixel_type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<image_type>::pixel_type>::is_unsigned == true );


        matrix<unsigned long,1> hist;
        get_histogram(img,hist);

        // create integral histograms
        matrix<double,1> cum_hist(hist.size()+1), cum_histi(hist.size()+1);
        cum_hist(0) = 0;
        cum_histi(0) = 0;
        for (long i = 0; i < hist.size(); ++i)
        {
            cum_hist(i+1) = cum_hist(i) + hist(i);
            cum_histi(i+1) = cum_histi(i) + hist(i)*(double)i;
        }

        auto histsum = [&](long begin, long end) 
        { 
            return cum_hist(end)-cum_hist(begin); 
        };
        auto histsumi = [&](long begin, long end) 
        { 
            return cum_histi(end)-cum_histi(begin); 
        };

        // If we split the pixels into two groups, those < thresh (the left group) and
        // those >= thresh (the right group), what would the sum of absolute deviations of
        // each pixel from the mean of its group be?  total_abs(thresh) computes that
        // value.
        auto total_abs = [&](unsigned long begin, unsigned long thresh)
        {
            auto left_avg = histsumi(begin,thresh);
            auto tmp = histsum(begin,thresh);
            if (tmp != 0)
                left_avg /= tmp;
            auto right_avg = histsumi(thresh,hist.size());
            tmp = histsum(thresh,hist.size());
            if (tmp != 0)
                right_avg /= tmp;


            const long left_idx = (long)std::ceil(left_avg);
            const long right_idx = (long)std::ceil(right_avg);

            double score = 0;
            score += left_avg*histsum(begin,left_idx) - histsumi(begin,left_idx); 
            score -= left_avg*histsum(left_idx,thresh) - histsumi(left_idx,thresh); 
            score += right_avg*histsum(thresh,right_idx) - histsumi(thresh,right_idx); 
            score -= right_avg*histsum(right_idx,hist.size()) - histsumi(right_idx,hist.size()); 
            return score;
        };


        impl::recursive_partition_pixels(0, hist.size(), total_abs, pix_thresh, more_thresholds...);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type
    partition_pixels (
        const image_type& img
    ) 
    {
        typedef typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type basic_pixel_type;
        basic_pixel_type thresh;
        partition_pixels(img, thresh);
        return thresh;
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void threshold_image (
        const in_image_type& in_img_,
        out_image_type& out_img_,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type thresh
    )
    {
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<in_image_type>::pixel_type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<out_image_type>::pixel_type>::has_alpha == false );

        COMPILE_TIME_ASSERT(pixel_traits<typename image_traits<out_image_type>::pixel_type>::grayscale);

        const_image_view<in_image_type> in_img(in_img_);
        image_view<out_image_type> out_img(out_img_);

        // if there isn't any input image then don't do anything
        if (in_img.size() == 0)
        {
            out_img.clear();
            return;
        }

        out_img.set_size(in_img.nr(),in_img.nc());

        for (long r = 0; r < in_img.nr(); ++r)
        {
            for (long c = 0; c < in_img.nc(); ++c)
            {
                if (get_pixel_intensity(in_img[r][c]) >= thresh)
                    assign_pixel(out_img[r][c], on_pixel);
                else
                    assign_pixel(out_img[r][c], off_pixel);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void threshold_image (
        const in_image_type& in_img,
        out_image_type& out_img
    )
    {
        threshold_image(in_img,out_img,partition_pixels(in_img));
    }

    template <
        typename image_type
        >
    void threshold_image (
        image_type& img,
        typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type thresh
    )
    {
        threshold_image(img,img,thresh);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void threshold_image (
        image_type& img
    )
    {
        threshold_image(img,img,partition_pixels(img));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void auto_threshold_image (
        const in_image_type& in_img_,
        out_image_type& out_img_
    )
    {
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<in_image_type>::pixel_type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<out_image_type>::pixel_type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<in_image_type>::pixel_type>::is_unsigned == true );
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<out_image_type>::pixel_type>::is_unsigned == true );

        COMPILE_TIME_ASSERT(pixel_traits<typename image_traits<out_image_type>::pixel_type>::grayscale);

        image_view<out_image_type> out_img(out_img_);

        // if there isn't any input image then don't do anything
        if (image_size(in_img_) == 0)
        {
            out_img.clear();
            return;
        }

        unsigned long thresh;
        // find the threshold we should use
        matrix<unsigned long,1> hist;
        get_histogram(in_img_,hist);

        const_image_view<in_image_type> in_img(in_img_);

        // Start our two means (a and b) out at the ends of the histogram
        long a = 0;
        long b = hist.size()-1;
        bool moved_a = true;
        bool moved_b = true;
        while (moved_a || moved_b)
        {
            moved_a = false;
            moved_b = false;

            // catch the degenerate case where the histogram is empty
            if (a >= b)
                break;

            if (hist(a) == 0)
            {
                ++a;
                moved_a = true;
            }

            if (hist(b) == 0)
            {
                --b;
                moved_b = true;
            }
        }
        
        // now do k-means clustering with k = 2 on the histogram. 
        moved_a = true;
        moved_b = true;
        while (moved_a || moved_b)
        {
            moved_a = false;
            moved_b = false;

            int64 a_hits = 0;
            int64 b_hits = 0;
            int64 a_mass = 0;
            int64 b_mass = 0;

            for (long i = 0; i < hist.size(); ++i)
            {
                // if i is closer to a
                if (std::abs(i-a) < std::abs(i-b))
                {
                    a_mass += hist(i)*i;
                    a_hits += hist(i);
                }
                else // if i is closer to b
                {
                    b_mass += hist(i)*i;
                    b_hits += hist(i);
                }
            }

            long new_a = (a_mass + a_hits/2)/a_hits;
            long new_b = (b_mass + b_hits/2)/b_hits;

            if (new_a != a)
            {
                moved_a = true;
                a = new_a;
            }

            if (new_b != b)
            {
                moved_b = true;
                b = new_b;
            }
        }
        
        // put the threshold between the two means we found
        thresh = (a + b)/2;

        // now actually apply the threshold
        threshold_image(in_img_,out_img_,thresh);
    }

    template <
        typename image_type
        >
    void auto_threshold_image (
        image_type& img
    )
    {
        auto_threshold_image(img,img);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void hysteresis_threshold (
        const in_image_type& in_img_,
        out_image_type& out_img_,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type lower_thresh,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type upper_thresh
    )
    {
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<in_image_type>::pixel_type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename image_traits<out_image_type>::pixel_type>::has_alpha == false );

        COMPILE_TIME_ASSERT(pixel_traits<typename image_traits<out_image_type>::pixel_type>::grayscale);

        DLIB_ASSERT(is_same_object(in_img_, out_img_) == false,
            "\tvoid hysteresis_threshold(in_img_, out_img_, lower_thresh, upper_thresh)"
            << "\n\tis_same_object(in_img_,out_img_): " << is_same_object(in_img_,out_img_) 
            );

        const_image_view<in_image_type> in_img(in_img_);
        image_view<out_image_type> out_img(out_img_);

        // if there isn't any input image then don't do anything
        if (in_img.size() == 0)
        {
            out_img.clear();
            return;
        }

        out_img.set_size(in_img.nr(),in_img.nc());
        assign_all_pixels(out_img, off_pixel);

        std::vector<std::pair<long,long>> stack;
        using std::make_pair;

        // now do the thresholding
        for (long r = 0; r < in_img.nr(); ++r)
        {
            for (long c = 0; c < in_img.nc(); ++c)
            {
                typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type p;
                assign_pixel(p,in_img[r][c]);
                if (p >= upper_thresh)
                {
                    // now do line following for pixels >= lower_thresh.
                    // set the stack position to 0.
                    stack.push_back(make_pair(r,c));

                    while (stack.size() > 0)
                    {
                        const long r = stack.back().first;
                        const long c = stack.back().second;
                        stack.pop_back();

                        // This is the base case of our recursion.  We want to stop if we hit a
                        // pixel we have already visited.
                        if (out_img[r][c] == on_pixel)
                            continue;

                        out_img[r][c] = on_pixel;

                        // put the neighbors of this pixel on the stack if they are bright enough
                        if (r-1 >= 0)
                        {
                            if (get_pixel_intensity(in_img[r-1][c]) >= lower_thresh)
                                stack.push_back(make_pair(r-1, c));
                            if (c-1 >= 0 && get_pixel_intensity(in_img[r-1][c-1]) >= lower_thresh)
                                stack.push_back(make_pair(r-1, c-1));
                            if (c+1 < in_img.nc() && get_pixel_intensity(in_img[r-1][c+1]) >= lower_thresh)
                                stack.push_back(make_pair(r-1, c+1));
                        }

                        if (c-1 >= 0 && get_pixel_intensity(in_img[r][c-1]) >= lower_thresh)
                            stack.push_back(make_pair(r,c-1));
                        if (c+1 < in_img.nc() && get_pixel_intensity(in_img[r][c+1]) >= lower_thresh)
                            stack.push_back(make_pair(r,c+1));

                        if (r+1 < in_img.nr())
                        {
                            if (get_pixel_intensity(in_img[r+1][c]) >= lower_thresh)
                                stack.push_back(make_pair(r+1,c));
                            if (c-1 >= 0 && get_pixel_intensity(in_img[r+1][c-1]) >= lower_thresh)
                                stack.push_back(make_pair(r+1,c-1));
                            if (c+1 < in_img.nc() && get_pixel_intensity(in_img[r+1][c+1]) >= lower_thresh)
                                stack.push_back(make_pair(r+1,c+1));
                        }

                    } // end while (stack.size() > 0)
                }
            }
        }
    }

    template <
        typename in_image_type,
        typename out_image_type
        >
    void hysteresis_threshold (
        const in_image_type& in_img,
        out_image_type& out_img
    )
    {
        using basic_pixel_type = typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type;

        basic_pixel_type t1, t2;
        partition_pixels(in_img, t1, t2);
        hysteresis_threshold(in_img, out_img, t1, t2);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_THRESHOLDINg_ 

