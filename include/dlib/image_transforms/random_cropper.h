// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RaNDOM_CROPPER_H_
#define DLIB_RaNDOM_CROPPER_H_

#include "random_cropper_abstract.h"
#include "../threads.h"
#include <mutex>
#include <vector>
#include "interpolation.h"
#include "../image_processing/full_object_detection.h"
#include "../rand.h"

namespace dlib
{
    class random_cropper
    {
        chip_dims dims = chip_dims(300,300);
        bool randomly_flip = true;
        double max_rotation_degrees = 30;
        long min_object_length_long_dim = 75; // cropped object will be at least this many pixels along its longest edge.
        long min_object_length_short_dim = 30; // cropped object will be at least this many pixels along its shortest edge.
        double max_object_size = 0.7; // cropped object will be at most this fraction of the size of the image.
        double background_crops_fraction = 0.5;
        double translate_amount = 0.10;

        std::mutex rnd_mutex;
        dlib::rand rnd;
    public:

        void set_seed (
            time_t seed
        ) { rnd = dlib::rand(seed); }

        double get_translate_amount (
        ) const { return translate_amount; }

        void set_translate_amount (
            double value
        )  
        { 
            DLIB_CASSERT(0 <= value);
            translate_amount = value;
        }

        double get_background_crops_fraction (
        ) const { return background_crops_fraction; }

        void set_background_crops_fraction (
            double value
        )
        {
            DLIB_CASSERT(0 <= value && value <= 1);
            background_crops_fraction = value;
        }

        const chip_dims& get_chip_dims(
        ) const { return dims; }

        void set_chip_dims (
            const chip_dims& dims_
        ) { dims = dims_; }

        void set_chip_dims (
            unsigned long rows,
            unsigned long cols
        ) { set_chip_dims(chip_dims(rows,cols)); }

        bool get_randomly_flip (
        ) const { return randomly_flip; }

        void set_randomly_flip (
            bool value
        ) { randomly_flip = value; }

        double get_max_rotation_degrees (
        ) const { return max_rotation_degrees; }
        void set_max_rotation_degrees (
            double value
        ) { max_rotation_degrees = std::abs(value); }

        long get_min_object_length_long_dim (
        ) const { return min_object_length_long_dim; }
        long get_min_object_length_short_dim (
        ) const { return min_object_length_short_dim; }

        void set_min_object_size (
            long long_dim,
            long short_dim
        ) 
        { 
            DLIB_CASSERT(0 < short_dim && short_dim <= long_dim);
            min_object_length_long_dim = long_dim; 
            min_object_length_short_dim = short_dim; 
        }

        double get_max_object_size (
        ) const { return max_object_size; }
        void set_max_object_size (
            double value
        ) 
        { 
            DLIB_CASSERT(0 < value);
            max_object_size = value; 
        }

        template <
            typename array_type
            >
        void operator() (
            size_t num_crops,
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            array_type& crops,
            std::vector<std::vector<mmod_rect>>& crop_rects
        )
        {
            DLIB_CASSERT(images.size() == rects.size());
            crops.clear();
            crop_rects.clear();
            append(num_crops, images, rects, crops, crop_rects);
        }

        template <
            typename array_type
            >
        void append (
            size_t num_crops,
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            array_type& crops,
            std::vector<std::vector<mmod_rect>>& crop_rects
        )
        {
            DLIB_CASSERT(images.size() == rects.size());
            DLIB_CASSERT(crops.size() == crop_rects.size());
            auto original_size = crops.size();
            crops.resize(crops.size()+num_crops);
            crop_rects.resize(crop_rects.size()+num_crops);
            parallel_for(original_size, original_size+num_crops, [&](long i) {
                (*this)(images, rects, crops[i], crop_rects[i]);
            });
        }


        template <
            typename array_type,
            typename image_type
            >
        void operator() (
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            image_type& crop,
            std::vector<mmod_rect>& crop_rects
        )
        {
            DLIB_CASSERT(images.size() == rects.size());
            size_t idx;
            { std::lock_guard<std::mutex> lock(rnd_mutex);
                idx = rnd.get_integer(images.size());
            }
            (*this)(images[idx], rects[idx], crop, crop_rects);
        }

        template <
            typename image_type1
            >
        image_type1 operator() (
            const image_type1& img
        )
        {
            image_type1 crop;
            std::vector<mmod_rect> junk1, junk2;
            (*this)(img, junk1, crop, junk2);
            return crop;
        }

        template <
            typename image_type1,
            typename image_type2
            >
        void operator() (
            const image_type1& img,
            const std::vector<mmod_rect>& rects,
            image_type2& crop,
            std::vector<mmod_rect>& crop_rects
        )
        {
            DLIB_CASSERT(num_rows(img)*num_columns(img) != 0);
            chip_details crop_plan;
            bool should_flip_crop;
            make_crop_plan(img, rects, crop_plan, should_flip_crop);

            extract_image_chip(img, crop_plan, crop);
            const rectangle_transform tform = get_mapping_to_chip(crop_plan);

            // copy rects into crop_rects and set ones that are outside the crop to ignore or
            // drop entirely as appropriate.
            crop_rects.clear();
            for (auto rect : rects)
            {
                // map to crop
                rect.rect = tform(rect.rect);

                // if the rect is at least partly in the crop
                if (get_rect(crop).intersect(rect.rect).area() != 0)
                {
                    // set to ignore if not totally in the crop or if too small.
                    if (!get_rect(crop).contains(rect.rect) || 
                        ((long)rect.rect.height() < min_object_length_long_dim  && (long)rect.rect.width() < min_object_length_long_dim) || 
                        ((long)rect.rect.height() < min_object_length_short_dim || (long)rect.rect.width() < min_object_length_short_dim))
                    {
                        rect.ignore = true;
                    }

                    crop_rects.push_back(rect);
                }
            }

            // Also randomly flip the image
            if (should_flip_crop)
            {
                image_type2 temp;
                flip_image_left_right(crop, temp); 
                swap(crop,temp);
                for (auto&& rect : crop_rects)
                    rect.rect = impl::flip_rect_left_right(rect.rect, get_rect(crop));
            }
        }

    private:

        template <typename image_type1>
        void make_crop_plan (
            const image_type1& img,
            const std::vector<mmod_rect>& rects,
            chip_details& crop_plan,
            bool& should_flip_crop
        )
        {
            std::lock_guard<std::mutex> lock(rnd_mutex);
            rectangle crop_rect;
            if (has_non_ignored_box(rects) && rnd.get_random_double() >= background_crops_fraction)
            {
                auto rect = rects[randomly_pick_rect(rects)].rect;

                // perturb the location of the crop by a small fraction of the object's size.
                const point rand_translate = dpoint(rnd.get_double_in_range(-translate_amount,translate_amount)*std::max(rect.height(),rect.width()), 
                                                    rnd.get_double_in_range(-translate_amount,translate_amount)*std::max(rect.height(),rect.width()));

                // We are going to grow rect into the cropping rect.  First, we grow it a
                // little so that it has the desired minimum border around it.  
                drectangle drect = centered_drect(center(rect)+rand_translate, rect.width()/max_object_size, rect.height()/max_object_size);

                // Now make rect have the same aspect ratio as dims so that there won't be
                // any funny stretching when we crop it.  We do this by growing it along
                // whichever dimension is too short.
                const double target_aspect = dims.cols/(double)dims.rows;
                if (drect.width()/drect.height() < target_aspect)
                    drect = centered_drect(drect, target_aspect*drect.height(), drect.height());
                else 
                    drect = centered_drect(drect, drect.width(), drect.width()/target_aspect);

                // Now perturb the scale of the crop.  We do this by shrinking it, but not
                // so much that it gets smaller than the min object sizes require. 
                double current_width = dims.cols*rect.width()/drect.width();
                double current_height = dims.rows*rect.height()/drect.height();

                // never make any dimension smaller than the short dim.
                double min_scale1 = std::max(min_object_length_short_dim/current_width, min_object_length_short_dim/current_height);
                // at least one dimension needs to be longer than the long dim.
                double min_scale2 = std::min(min_object_length_long_dim/current_width, min_object_length_long_dim/current_height);
                double min_scale = std::max(min_scale1, min_scale2); 

                const double rand_scale_perturb = 1.0/rnd.get_double_in_range(min_scale, 1); 
                crop_rect = centered_drect(drect, drect.width()*rand_scale_perturb, drect.height()*rand_scale_perturb);

            }
            else
            {
                crop_rect = make_random_cropping_rect(img);
            }
            should_flip_crop = randomly_flip && rnd.get_random_double() > 0.5;
            const double angle = rnd.get_double_in_range(-max_rotation_degrees, max_rotation_degrees)*pi/180;
            crop_plan = chip_details(crop_rect, dims, angle);
        }

        bool has_non_ignored_box (
            const std::vector<mmod_rect>& rects
        ) const
        {
            for (auto&& b : rects)
            {
                if (!b.ignore)
                    return true;
            }
            return false;
        }

        size_t randomly_pick_rect (
            const std::vector<mmod_rect>& rects
        ) 
        {
            DLIB_CASSERT(has_non_ignored_box(rects));
            size_t idx = rnd.get_integer(rects.size());
            while(rects[idx].ignore)
                idx = rnd.get_integer(rects.size());
            return idx;
        }

        template <typename image_type>
        rectangle make_random_cropping_rect(
            const image_type& img_
        )
        {
            const_image_view<image_type> img(img_);
            // Figure out what rectangle we want to crop from the image.  We are going to
            // crop out an image of size this->dims, so we pick a random scale factor that
            // lets this random box be either as big as it can be while still fitting in
            // the image or as small as a 3x zoomed in box randomly somewhere in the image. 
            double mins = 1.0/3.0, maxs = std::min(img.nr()/(double)dims.rows, img.nc()/(double)dims.cols);
            mins = std::min(mins, maxs);
            auto scale = rnd.get_double_in_range(mins, maxs);
            rectangle rect(scale*dims.cols, scale*dims.rows);
            // randomly shift the box around
            point offset(rnd.get_integer(1+img.nc()-rect.width()),
                         rnd.get_integer(1+img.nr()-rect.height()));
            return move_rect(rect, offset);
        }



    };

// ----------------------------------------------------------------------------------------

    inline std::ostream& operator<< (
        std::ostream& out,
        const random_cropper& item
    )
    {
        using std::endl;
        out << "random_cropper details: " << endl;
        out << "  chip_dims.rows:              " << item.get_chip_dims().rows << endl;
        out << "  chip_dims.cols:              " << item.get_chip_dims().cols << endl;
        out << "  randomly_flip:               " << std::boolalpha << item.get_randomly_flip() << endl;
        out << "  max_rotation_degrees:        " << item.get_max_rotation_degrees() << endl;
        out << "  min_object_length_long_dim:  " << item.get_min_object_length_long_dim() << endl;
        out << "  min_object_length_short_dim: " << item.get_min_object_length_short_dim() << endl;
        out << "  max_object_size:             " << item.get_max_object_size() << endl;
        out << "  background_crops_fraction:   " << item.get_background_crops_fraction() << endl;
        out << "  translate_amount:            " << item.get_translate_amount() << endl;
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RaNDOM_CROPPER_H_

