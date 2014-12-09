// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMPLE_ObJECT_DETECTOR_H__
#define DLIB_SIMPLE_ObJECT_DETECTOR_H__

#include "simple_object_detector_abstract.h"
#include "dlib/image_processing/object_detector.h"
#include "dlib/string.h"
#include "dlib/image_processing/scan_fhog_pyramid.h"
#include "dlib/svm/structural_object_detection_trainer.h"
#include "dlib/geometry.h"
#include "dlib/data_io/load_image_dataset.h"
#include "dlib/image_processing/remove_unobtainable_rectangles.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > simple_object_detector;

// ----------------------------------------------------------------------------------------

    struct simple_object_detector_training_options
    {
        simple_object_detector_training_options()
        {
            be_verbose = false;
            add_left_right_image_flips = false;
            num_threads = 4;
            detection_window_size = 80*80;
            C = 1;
            epsilon = 0.01;
        }

        bool be_verbose;
        bool add_left_right_image_flips;
        unsigned long num_threads;
        unsigned long detection_window_size;
        double C;
        double epsilon;
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline void pick_best_window_size (
            const std::vector<std::vector<rectangle> >& boxes,
            unsigned long& width,
            unsigned long& height,
            const unsigned long target_size
        )
        {
            // find the average width and height
            running_stats<double> avg_width, avg_height;
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {
                for (unsigned long j = 0; j < boxes[i].size(); ++j)
                {
                    avg_width.add(boxes[i][j].width());
                    avg_height.add(boxes[i][j].height());
                }
            }

            // now adjust the box size so that it is about target_pixels pixels in size
            double size = avg_width.mean()*avg_height.mean();
            double scale = std::sqrt(target_size/size);

            width = (unsigned long)(avg_width.mean()*scale+0.5);
            height = (unsigned long)(avg_height.mean()*scale+0.5);
            // make sure the width and height never round to zero.
            if (width == 0)
                width = 1;
            if (height == 0)
                height = 1;
        }

        inline bool contains_any_boxes (
            const std::vector<std::vector<rectangle> >& boxes
        )
        {
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {
                if (boxes[i].size() != 0)
                    return true;
            }
            return false;
        }

        inline void throw_invalid_box_error_message (
            const std::string& dataset_filename,
            const std::vector<std::vector<rectangle> >& removed,
            const simple_object_detector_training_options& options
        )
        {

            std::ostringstream sout;
            // Note that the 1/16 factor is here because we will try to upsample the image
            // 2 times to accommodate small boxes.  We also take the max because we want to
            // lower bound the size of the smallest recommended box.  This is because the
            // 8x8 HOG cells can't really deal with really small object boxes.
            sout << "Error!  An impossible set of object boxes was given for training. ";
            sout << "All the boxes need to have a similar aspect ratio and also not be ";
            sout << "smaller than about " << std::max<long>(20*20,options.detection_window_size/16) << " pixels in area. ";

            std::ostringstream sout2;
            if (dataset_filename.size() != 0)
            {
                sout << "The following images contain invalid boxes:\n";
                image_dataset_metadata::dataset data;
                load_image_dataset_metadata(data, dataset_filename);
                for (unsigned long i = 0; i < removed.size(); ++i)
                {
                    if (removed[i].size() != 0)
                    {
                        const std::string imgname = data.images[i].filename;
                        sout2 << "  " << imgname << "\n";
                    }
                }
            }
            throw error("\n"+wrap_string(sout.str()) + "\n" + sout2.str());
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename image_array>
    inline void train_simple_object_detector_on_images (
        const std::string& dataset_filename, // can be "" if it's not applicable
        image_array& images,
        std::vector<std::vector<rectangle> >& boxes,
        std::vector<std::vector<rectangle> >& ignore,
        const std::string& detector_output_filename,
        const simple_object_detector_training_options& options 
    )
    {
        if (options.C <= 0)
            throw error("Invalid C value given to train_simple_object_detector(), C must be > 0.");
        if (options.epsilon <= 0)
            throw error("Invalid epsilon value given to train_simple_object_detector(), epsilon must be > 0.");

        if (images.size() != boxes.size())
            throw error("The list of images must have the same length as the list of boxes.");
        if (images.size() != ignore.size())
            throw error("The list of images must have the same length as the list of ignore boxes.");

        if (impl::contains_any_boxes(boxes) == false)
            throw error("Error, the training dataset does not have any labeled object boxes in it.");

        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        image_scanner_type scanner;
        unsigned long width, height;
        impl::pick_best_window_size(boxes, width, height, options.detection_window_size);
        scanner.set_detection_window_size(width, height); 
        structural_object_detection_trainer<image_scanner_type> trainer(scanner);
        trainer.set_num_threads(options.num_threads);  
        trainer.set_c(options.C);
        trainer.set_epsilon(options.epsilon);
        if (options.be_verbose)
        {
            std::cout << "Training with C: " << options.C << std::endl;
            std::cout << "Training with epsilon: " << options.epsilon << std::endl;
            std::cout << "Training using " << options.num_threads << " threads."<< std::endl;
            std::cout << "Training with sliding window " << width << " pixels wide by " << height << " pixels tall." << std::endl;
            if (options.add_left_right_image_flips)
                std::cout << "Training on both left and right flipped versions of images." << std::endl;
            trainer.be_verbose();
        }


        unsigned long upsample_amount = 0;

        // now make sure all the boxes are obtainable by the scanner.  We will try and
        // upsample the images at most two times to help make the boxes obtainable.
        std::vector<std::vector<rectangle> > temp(boxes), removed;
        removed = remove_unobtainable_rectangles(trainer, images, temp);
        if (impl::contains_any_boxes(removed))
        {
            ++upsample_amount;
            if (options.be_verbose)
                std::cout << "upsample images..." << std::endl;
            upsample_image_dataset<pyramid_down<2> >(images, boxes, ignore);
            temp = boxes;
            removed = remove_unobtainable_rectangles(trainer, images, temp);
            if (impl::contains_any_boxes(removed))
            {
                ++upsample_amount;
                if (options.be_verbose)
                    std::cout << "upsample images..." << std::endl;
                upsample_image_dataset<pyramid_down<2> >(images, boxes, ignore);
                temp = boxes;
                removed = remove_unobtainable_rectangles(trainer, images, temp);
            }
        }
        // if we weren't able to get all the boxes to match then throw an error 
        if (impl::contains_any_boxes(removed))
            impl::throw_invalid_box_error_message(dataset_filename, removed, options);

        if (options.add_left_right_image_flips)
            add_image_left_right_flips(images, boxes, ignore);

        simple_object_detector detector = trainer.train(images, boxes, ignore);

        std::ofstream fout(detector_output_filename.c_str(), std::ios::binary);
        int version = 1;
        serialize(detector, fout);
        serialize(version, fout);
        serialize(upsample_amount, fout);

        if (options.be_verbose)
        {
            std::cout << "Training complete, saved detector to file " << detector_output_filename << std::endl;
            std::cout << "Trained with C: " << options.C << std::endl;
            std::cout << "Training with epsilon: " << options.epsilon << std::endl;
            std::cout << "Trained using " << options.num_threads << " threads."<< std::endl;
            std::cout << "Trained with sliding window " << width << " pixels wide by " << height << " pixels tall." << std::endl;
            if (upsample_amount != 0)
            {
                if (upsample_amount == 1)
                    std::cout << "Upsampled images " << upsample_amount << " time to allow detection of small boxes." << std::endl;
                else
                    std::cout << "Upsampled images " << upsample_amount << " times to allow detection of small boxes." << std::endl;
            }
            if (options.add_left_right_image_flips)
                std::cout << "Trained on both left and right flipped versions of images." << std::endl;
        }
    }

// ----------------------------------------------------------------------------------------

    inline void train_simple_object_detector (
        const std::string& dataset_filename,
        const std::string& detector_output_filename,
        const simple_object_detector_training_options& options 
    )
    {
        dlib::array<array2d<rgb_pixel> > images;
        std::vector<std::vector<rectangle> > boxes, ignore;
        ignore = load_image_dataset(images, boxes, dataset_filename);

        train_simple_object_detector_on_images(dataset_filename, images, boxes, ignore, detector_output_filename, options);
    }

// ----------------------------------------------------------------------------------------

    struct simple_test_results
    {
        double precision;
        double recall;
        double average_precision;
    };

    template <typename image_array>
    inline const simple_test_results test_simple_object_detector_with_images (
            image_array& images,
            std::vector<std::vector<rectangle> >& boxes,
            std::vector<std::vector<rectangle> >& ignore,
            const std::string& detector_filename
    )
    {
        simple_object_detector detector;
        int version = 0;
        unsigned int upsample_amount = 0;
        std::ifstream fin(detector_filename.c_str(), std::ios::binary);
        if (!fin)
            throw error("Unable to open file " + detector_filename);
        deserialize(detector, fin);
        deserialize(version, fin);
        if (version != 1)
            throw error("Unknown simple_object_detector format.");
        deserialize(upsample_amount, fin);

        for (unsigned int i = 0; i < upsample_amount; ++i)
            upsample_image_dataset<pyramid_down<2> >(images, boxes);

        matrix<double,1,3> res = test_object_detection_function(detector, images, boxes, ignore);
        simple_test_results ret;
        ret.precision = res(0);
        ret.recall = res(1);
        ret.average_precision = res(2);
        return ret;
    }

    inline const simple_test_results test_simple_object_detector (
        const std::string& dataset_filename,
        const std::string& detector_filename
    )
    {
        dlib::array<array2d<rgb_pixel> > images;
        std::vector<std::vector<rectangle> > boxes, ignore;
        ignore = load_image_dataset(images, boxes, dataset_filename);

        return test_simple_object_detector_with_images(images, boxes, ignore, detector_filename);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SIMPLE_ObJECT_DETECTOR_H__

