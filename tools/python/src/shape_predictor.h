// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICTOR_H__
#define DLIB_SHAPE_PREDICTOR_H__

#include "dlib/string.h"
#include "dlib/geometry.h"
#include "dlib/data_io/load_image_dataset.h"
#include "dlib/image_processing.h"

using namespace std;

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct shape_predictor_training_options
    {
        shape_predictor_training_options()
        {
            be_verbose = false;
            cascade_depth = 10;
            tree_depth = 4;
            num_trees_per_cascade_level = 500;
            nu = 0.1;
            oversampling_amount = 20;
            oversampling_translation_jitter = 0;
            feature_pool_size = 400;
            lambda_param = 0.1;
            num_test_splits = 20;
            feature_pool_region_padding = 0;
            random_seed = "";
            num_threads = 0;
            landmark_relative_padding_mode = true;
        }

        bool be_verbose;
        unsigned long cascade_depth;
        unsigned long tree_depth;
        unsigned long num_trees_per_cascade_level;
        double nu;
        unsigned long oversampling_amount;
        double oversampling_translation_jitter;
        unsigned long feature_pool_size;
        double lambda_param;
        unsigned long num_test_splits;
        double feature_pool_region_padding;
        std::string random_seed;
        bool landmark_relative_padding_mode;

        // not serialized
        unsigned long num_threads;
    };

    inline void serialize (
        const shape_predictor_training_options& item,
        std::ostream& out
    )
    {
        try
        {
            serialize("shape_predictor_training_options_v2", out);
            serialize(item.be_verbose,out);
            serialize(item.cascade_depth,out);
            serialize(item.tree_depth,out);
            serialize(item.num_trees_per_cascade_level,out);
            serialize(item.nu,out);
            serialize(item.oversampling_amount,out);
            serialize(item.oversampling_translation_jitter,out);
            serialize(item.feature_pool_size,out);
            serialize(item.lambda_param,out);
            serialize(item.num_test_splits,out);
            serialize(item.feature_pool_region_padding,out);
            serialize(item.random_seed,out);
            serialize(item.landmark_relative_padding_mode,out);
        }
        catch (serialization_error& e)
        {
            throw serialization_error(e.info + "\n   while serializing an object of type shape_predictor_training_options");
        }
    }

    inline void deserialize (
        shape_predictor_training_options& item,
        std::istream& in
    )
    {
        try
        {
            check_serialized_version("shape_predictor_training_options_v2", in);
            deserialize(item.be_verbose,in);
            deserialize(item.cascade_depth,in);
            deserialize(item.tree_depth,in);
            deserialize(item.num_trees_per_cascade_level,in);
            deserialize(item.nu,in);
            deserialize(item.oversampling_amount,in);
            deserialize(item.oversampling_translation_jitter,in);
            deserialize(item.feature_pool_size,in);
            deserialize(item.lambda_param,in);
            deserialize(item.num_test_splits,in);
            deserialize(item.feature_pool_region_padding,in);
            deserialize(item.random_seed,in);
            deserialize(item.landmark_relative_padding_mode,in);
        }
        catch (serialization_error& e)
        {
            throw serialization_error(e.info + "\n   while deserializing an object of type shape_predictor_training_options");
        }
    }

    inline string print_shape_predictor_training_options(const shape_predictor_training_options& o)
    {
        std::ostringstream sout;
        sout << "shape_predictor_training_options("
            << "be_verbose=" << o.be_verbose << ", "
            << "cascade_depth=" << o.cascade_depth << ", "
            << "tree_depth=" << o.tree_depth << ", "
            << "num_trees_per_cascade_level=" << o.num_trees_per_cascade_level << ", "
            << "nu=" << o.nu << ", "
            << "oversampling_amount=" << o.oversampling_amount << ", "
            << "oversampling_translation_jitter=" << o.oversampling_translation_jitter << ", "
            << "feature_pool_size=" << o.feature_pool_size << ", "
            << "lambda_param=" << o.lambda_param << ", "
            << "num_test_splits=" << o.num_test_splits << ", "
            << "feature_pool_region_padding=" << o.feature_pool_region_padding << ", "
            << "random_seed=" << o.random_seed << ", "
            << "num_threads=" << o.num_threads << ", "
            << "landmark_relative_padding_mode=" << o.landmark_relative_padding_mode
        << ")";
        return sout.str();
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline bool contains_any_detections (
            const std::vector<std::vector<full_object_detection> >& detections
        )
        {
            for (unsigned long i = 0; i < detections.size(); ++i)
            {
                if (detections[i].size() != 0)
                    return true;
            }
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename image_array>
    inline shape_predictor train_shape_predictor_on_images (
        image_array& images,
        std::vector<std::vector<full_object_detection> >& detections,
        const shape_predictor_training_options& options
    )
    {
        if (options.lambda_param <= 0)
            throw error("Invalid lambda_param value given to train_shape_predictor(), lambda_param must be > 0.");
        if (!(0 < options.nu && options.nu <= 1))
            throw error("Invalid nu value given to train_shape_predictor(). It is required that 0 < nu <= 1.");
        if (options.feature_pool_region_padding <= -0.5)
            throw error("Invalid feature_pool_region_padding value given to train_shape_predictor(), feature_pool_region_padding must be > -0.5.");

        if (images.size() != detections.size())
            throw error("The list of images must have the same length as the list of detections.");

        if (!impl::contains_any_detections(detections))
            throw error("Error, the training dataset does not have any labeled object detections in it.");

        shape_predictor_trainer trainer;

        trainer.set_cascade_depth(options.cascade_depth);
        trainer.set_tree_depth(options.tree_depth);
        trainer.set_num_trees_per_cascade_level(options.num_trees_per_cascade_level);
        trainer.set_nu(options.nu);
        trainer.set_random_seed(options.random_seed);
        trainer.set_oversampling_amount(options.oversampling_amount);
        trainer.set_oversampling_translation_jitter(options.oversampling_translation_jitter);
        trainer.set_feature_pool_size(options.feature_pool_size);
        trainer.set_feature_pool_region_padding(options.feature_pool_region_padding);
        trainer.set_lambda(options.lambda_param);
        trainer.set_num_test_splits(options.num_test_splits);
        trainer.set_num_threads(options.num_threads);
        if (options.landmark_relative_padding_mode)
            trainer.set_padding_mode(shape_predictor_trainer::landmark_relative);
        else
            trainer.set_padding_mode(shape_predictor_trainer::bounding_box_relative);

        if (options.be_verbose)
        {
            std::cout << "Training with cascade depth: " << options.cascade_depth << std::endl;
            std::cout << "Training with tree depth: " << options.tree_depth << std::endl;
            std::cout << "Training with " << options.num_trees_per_cascade_level << " trees per cascade level."<< std::endl;
            std::cout << "Training with nu: " << options.nu << std::endl;
            std::cout << "Training with random seed: " << options.random_seed << std::endl;
            std::cout << "Training with oversampling amount: " << options.oversampling_amount << std::endl;
            std::cout << "Training with oversampling translation jitter: " << options.oversampling_translation_jitter << std::endl;
            std::cout << "Training with landmark_relative_padding_mode: " << options.landmark_relative_padding_mode << std::endl;
            std::cout << "Training with feature pool size: " << options.feature_pool_size << std::endl;
            std::cout << "Training with feature pool region padding: " << options.feature_pool_region_padding << std::endl;
            std::cout << "Training with " << options.num_threads << " threads." << std::endl;
            std::cout << "Training with lambda_param: " << options.lambda_param << std::endl;
            std::cout << "Training with " << options.num_test_splits << " split tests."<< std::endl;
            trainer.be_verbose();
        }

        shape_predictor predictor = trainer.train(images, detections);

        return predictor;
    }

    inline void train_shape_predictor (
        const std::string& dataset_filename,
        const std::string& predictor_output_filename,
        const shape_predictor_training_options& options
    )
    {
        dlib::array<array2d<unsigned char> > images;
        std::vector<std::vector<full_object_detection> > objects;
        load_image_dataset(images, objects, dataset_filename);

        shape_predictor predictor = train_shape_predictor_on_images(images, objects, options);

        serialize(predictor_output_filename) << predictor;

        if (options.be_verbose)
            std::cout << "Training complete, saved predictor to file " << predictor_output_filename << std::endl;
    }

// ----------------------------------------------------------------------------------------

    template <typename image_array>
    inline double test_shape_predictor_with_images (
            image_array& images,
            std::vector<std::vector<full_object_detection> >& detections,
            std::vector<std::vector<double> >& scales,
            const shape_predictor& predictor
    )
    {
        if (images.size() != detections.size())
            throw error("The list of images must have the same length as the list of detections.");
        if (scales.size() > 0  && scales.size() != images.size())
            throw error("The list of scales must have the same length as the list of detections.");

        if (scales.size() > 0)
            return test_shape_predictor(predictor, images, detections, scales);
        else
            return test_shape_predictor(predictor, images, detections);
    }

    inline double test_shape_predictor_py (
        const std::string& dataset_filename,
        const std::string& predictor_filename
    )
    {
        // Load the images, no scales can be provided
        dlib::array<array2d<unsigned char> > images;
        // This interface cannot take the scales parameter.
        std::vector<std::vector<double> > scales;
        std::vector<std::vector<full_object_detection> > objects;
        load_image_dataset(images, objects, dataset_filename);

        // Load the shape predictor
        shape_predictor predictor;
        deserialize(predictor_filename) >> predictor;

        return test_shape_predictor_with_images(images, objects, scales, predictor);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICTOR_H__

