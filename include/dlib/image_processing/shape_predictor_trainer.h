// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_TRAINER_H_
#define DLIB_SHAPE_PREDICToR_TRAINER_H_

#include "shape_predictor_trainer_abstract.h"
#include "shape_predictor.h"
#include "../console_progress_indicator.h"
#include "../threads.h"
#include "../data_io/image_dataset_metadata.h"
#include "box_overlap_testing.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class shape_predictor_trainer
    {
        /*!
            This thing really only works with unsigned char or rgb_pixel images (since we assume the threshold 
            should be in the range [-128,128]).
        !*/
    public:

        enum padding_mode_t
        {
            bounding_box_relative,
            landmark_relative 
        };

        shape_predictor_trainer (
        )
        {
            _cascade_depth = 10;
            _tree_depth = 4;
            _num_trees_per_cascade_level = 500;
            _nu = 0.1;
            _oversampling_amount = 20;
            _oversampling_translation_jitter = 0;
            _feature_pool_size = 400;
            _lambda = 0.1;
            _num_test_splits = 20;
            _feature_pool_region_padding = 0;
            _verbose = false;
            _num_threads = 0;
            _padding_mode = landmark_relative;
        }

        unsigned long get_cascade_depth (
        ) const { return _cascade_depth; }

        void set_cascade_depth (
            unsigned long depth
        )
        {
            DLIB_CASSERT(depth > 0, 
                "\t void shape_predictor_trainer::set_cascade_depth()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t depth:  " << depth
            );

            _cascade_depth = depth;
        }

        unsigned long get_tree_depth (
        ) const { return _tree_depth; }

        void set_tree_depth (
            unsigned long depth
        )
        {
            DLIB_CASSERT(depth > 0, 
                "\t void shape_predictor_trainer::set_tree_depth()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t depth:  " << depth
            );

            _tree_depth = depth;
        }

        unsigned long get_num_trees_per_cascade_level (
        ) const { return _num_trees_per_cascade_level; }

        void set_num_trees_per_cascade_level (
            unsigned long num
        )
        {
            DLIB_CASSERT( num > 0,
                "\t void shape_predictor_trainer::set_num_trees_per_cascade_level()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t num:  " << num
            );
            _num_trees_per_cascade_level = num;
        }

        double get_nu (
        ) const { return _nu; } 
        void set_nu (
            double nu
        )
        {
            DLIB_CASSERT(0 < nu && nu <= 1,
                "\t void shape_predictor_trainer::set_nu()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t nu:  " << nu 
            );

            _nu = nu;
        }

        std::string get_random_seed (
        ) const { return rnd.get_seed(); }
        void set_random_seed (
            const std::string& seed
        ) { rnd.set_seed(seed); }

        unsigned long get_oversampling_amount (
        ) const { return _oversampling_amount; }

        void set_oversampling_amount (
            unsigned long amount
        )
        {
            DLIB_CASSERT(amount > 0, 
                "\t void shape_predictor_trainer::set_oversampling_amount()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t amount: " << amount 
            );

            _oversampling_amount = amount;
        }

        double get_oversampling_translation_jitter (
        ) const { return _oversampling_translation_jitter; }

        void set_oversampling_translation_jitter (
            double amount
        )
        {
            DLIB_CASSERT(amount >= 0, 
                "\t void shape_predictor_trainer::set_oversampling_translation_jitter()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t amount: " << amount 
            );

            _oversampling_translation_jitter = amount;
        }

        unsigned long get_feature_pool_size (
        ) const { return _feature_pool_size; }
        void set_feature_pool_size (
            unsigned long size
        ) 
        {
            DLIB_CASSERT(size > 1, 
                "\t void shape_predictor_trainer::set_feature_pool_size()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t size: " << size 
            );

            _feature_pool_size = size;
        }

        double get_lambda (
        ) const { return _lambda; }
        void set_lambda (
            double lambda
        )
        {
            DLIB_CASSERT(lambda > 0,
                "\t void shape_predictor_trainer::set_lambda()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t lambda: " << lambda 
            );

            _lambda = lambda;
        }

        unsigned long get_num_test_splits (
        ) const { return _num_test_splits; }
        void set_num_test_splits (
            unsigned long num
        )
        {
            DLIB_CASSERT(num > 0, 
                "\t void shape_predictor_trainer::set_num_test_splits()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t num: " << num 
            );

            _num_test_splits = num;
        }

        void set_padding_mode (
            padding_mode_t mode
        )
        {
            _padding_mode = mode;
        }

        padding_mode_t get_padding_mode (
        ) const { return _padding_mode; }

        double get_feature_pool_region_padding (
        ) const { return _feature_pool_region_padding; }
        void set_feature_pool_region_padding (
            double padding 
        )
        {
            DLIB_CASSERT(padding > -0.5,
                "\t void shape_predictor_trainer::set_feature_pool_region_padding()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t padding: " << padding 
            );

            _feature_pool_region_padding = padding;
        }

        void be_verbose (
        )
        {
            _verbose = true;
        }

        void be_quiet (
        )
        {
            _verbose = false;
        }

        unsigned long get_num_threads (
        ) const { return _num_threads; }
        void set_num_threads (
                unsigned long num
        )
        {
            _num_threads = num;
        }

        template <typename image_array>
        shape_predictor train (
            const image_array& images,
            const std::vector<std::vector<full_object_detection> >& objects
        ) const
        {
            using namespace impl;
            DLIB_CASSERT(images.size() == objects.size() && images.size() > 0,
                "\t shape_predictor shape_predictor_trainer::train()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t images.size():  " << images.size() 
                << "\n\t objects.size(): " << objects.size() 
            );
            // make sure the objects agree on the number of parts and that there is at
            // least one full_object_detection. 
            unsigned long num_parts = 0;
            std::vector<int> part_present;
            for (unsigned long i = 0; i < objects.size(); ++i)
            {
                for (unsigned long j = 0; j < objects[i].size(); ++j)
                {
                    if (num_parts == 0)
                    {
                        num_parts = objects[i][j].num_parts();
                        DLIB_CASSERT(objects[i][j].num_parts() != 0,
                            "\t shape_predictor shape_predictor_trainer::train()"
                            << "\n\t You can't give objects that don't have any parts to the trainer."
                        );
                        part_present.resize(num_parts);
                    }
                    else
                    {
                        DLIB_CASSERT(objects[i][j].num_parts() == num_parts,
                            "\t shape_predictor shape_predictor_trainer::train()"
                            << "\n\t All the objects must agree on the number of parts. "
                            << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                            << "\n\t num_parts:  " << num_parts 
                        );
                    }
                    for (unsigned long p = 0; p < objects[i][j].num_parts(); ++p)
                    {
                        if (objects[i][j].part(p) != OBJECT_PART_NOT_PRESENT)
                            part_present[p] = 1;
                    }
                }
            }
            DLIB_CASSERT(num_parts != 0,
                "\t shape_predictor shape_predictor_trainer::train()"
                << "\n\t You must give at least one full_object_detection if you want to train a shape model and it must have parts."
            );
            DLIB_CASSERT(sum(mat(part_present)) == (long)num_parts,
                "\t shape_predictor shape_predictor_trainer::train()"
                << "\n\t Each part must appear at least once in this training data.  That is, "
                << "\n\t you can't have a part that is always set to OBJECT_PART_NOT_PRESENT."
            );

            // creating thread pool. if num_threads <= 1, trainer should work in caller thread
            thread_pool tp(_num_threads > 1 ? _num_threads : 0);

            // determining the type of features used for this type of images
            typedef typename std::remove_const<typename std::remove_reference<decltype(images[0])>::type>::type image_type;
            typedef typename image_traits<image_type>::pixel_type pixel_type;
            typedef typename pixel_traits<pixel_type>::basic_pixel_type feature_type;

            rnd.set_seed(get_random_seed());

            std::vector<training_sample<feature_type>> samples;
            const matrix<float,0,1> initial_shape = populate_training_sample_shapes(objects, samples);
            const std::vector<std::vector<dlib::vector<float,2> > > pixel_coordinates = randomly_sample_pixel_coordinates(initial_shape);

            unsigned long trees_fit_so_far = 0;
            console_progress_indicator pbar(get_cascade_depth()*get_num_trees_per_cascade_level());
            if (_verbose)
                std::cout << "Fitting trees..." << std::endl;

            std::vector<std::vector<impl::regression_tree> > forests(get_cascade_depth());
            // Now start doing the actual training by filling in the forests
            for (unsigned long cascade = 0; cascade < get_cascade_depth(); ++cascade)
            {
                // Each cascade uses a different set of pixels for its features.  We compute
                // their representations relative to the initial shape first.
                std::vector<unsigned long> anchor_idx; 
                std::vector<dlib::vector<float,2> > deltas;
                create_shape_relative_encoding(initial_shape, pixel_coordinates[cascade], anchor_idx, deltas);

                // First compute the feature_pixel_values for each training sample at this
                // level of the cascade.
                parallel_for(tp, 0, samples.size(), [&](unsigned long i)
                {
                    impl::extract_feature_pixel_values(images[samples[i].image_idx], samples[i].rect,
                                                 samples[i].current_shape, initial_shape, anchor_idx,
                                                 deltas, samples[i].feature_pixel_values);
                }, 1);

                // Now start building the trees at this cascade level.
                for (unsigned long i = 0; i < get_num_trees_per_cascade_level(); ++i)
                {
                    forests[cascade].push_back(make_regression_tree(tp, samples, pixel_coordinates[cascade]));

                    if (_verbose)
                    {
                        ++trees_fit_so_far;
                        pbar.print_status(trees_fit_so_far);
                    }
                }
            }

            if (_verbose)
                std::cout << "Training complete                          " << std::endl;

            return shape_predictor(initial_shape, forests, pixel_coordinates);
        }

    private:

        static void object_to_shape (
            const full_object_detection& obj,
            matrix<float,0,1>& shape,
            matrix<float,0,1>& present // a mask telling which elements of #shape are present.
        )
        {
            shape.set_size(obj.num_parts()*2);
            present.set_size(obj.num_parts()*2);
            const point_transform_affine tform_from_img = impl::normalizing_tform(obj.get_rect());
            for (unsigned long i = 0; i < obj.num_parts(); ++i)
            {
                if (obj.part(i) != OBJECT_PART_NOT_PRESENT)
                {
                    vector<float,2> p = tform_from_img(obj.part(i));
                    shape(2*i)   = p.x();
                    shape(2*i+1) = p.y();
                    present(2*i)   = 1;
                    present(2*i+1) = 1;

                    if (length(p) > 100)
                    {
                        std::cout << "Warning, one of your objects has parts that are way outside its bounding box!  This is probably an error in your annotation." << std::endl;
                    }
                }
                else
                {
                    shape(2*i)   = 0;
                    shape(2*i+1) = 0;
                    present(2*i)   = 0;
                    present(2*i+1) = 0;
                }
            }
        }

        template<typename feature_type>
        struct training_sample
        {
            /*!

            CONVENTION
                - feature_pixel_values.size() == get_feature_pool_size()
                - feature_pixel_values[j] == the value of the j-th feature pool
                  pixel when you look it up relative to the shape in current_shape.

                - target_shape == The truth shape.  Stays constant during the whole
                  training process (except for the parts that are not present, those are
                  always equal to the current_shape values).
                - present == 0/1 mask saying which parts of target_shape are present.
                - rect == the position of the object in the image_idx-th image.  All shape
                  coordinates are coded relative to this rectangle.
                - diff_shape == temporary value for holding difference between current
                  shape and target shape
            !*/

            unsigned long image_idx;
            rectangle rect;
            matrix<float,0,1> target_shape;
            matrix<float,0,1> present;

            matrix<float,0,1> current_shape;
            matrix<float,0,1> diff_shape;
            std::vector<feature_type> feature_pixel_values;

            void swap(training_sample& item)
            {
                std::swap(image_idx, item.image_idx);
                std::swap(rect, item.rect);
                target_shape.swap(item.target_shape);
                present.swap(item.present);
                current_shape.swap(item.current_shape);
                diff_shape.swap(item.diff_shape);
                feature_pixel_values.swap(item.feature_pixel_values);
            }
        };

        template<typename feature_type>
        impl::regression_tree make_regression_tree (
            thread_pool& tp,
            std::vector<training_sample<feature_type>>& samples,
            const std::vector<dlib::vector<float,2> >& pixel_coordinates
        ) const
        {
            using namespace impl;
            std::deque<std::pair<unsigned long, unsigned long> > parts;
            parts.push_back(std::make_pair(0, (unsigned long)samples.size()));

            impl::regression_tree tree;

            // walk the tree in breadth first order
            const unsigned long num_split_nodes = static_cast<unsigned long>(std::pow(2.0, (double)get_tree_depth())-1);
            std::vector<matrix<float,0,1> > sums(num_split_nodes*2+1);
            if (tp.num_threads_in_pool() > 1)
            {
                // Here we need to calculate shape differences and store sum of differences into sums[0]
                // to make it. I am splitting samples into blocks, each block will be processed by
                // separate thread, and the sum of differences of each block is stored into separate
                // place in block_sums

                const unsigned long num_workers = std::max(1UL, tp.num_threads_in_pool());
                const unsigned long num =  samples.size();
                const unsigned long block_size = std::max(1UL, (num + num_workers - 1) / num_workers);
                std::vector<matrix<float,0,1> > block_sums(num_workers);

                parallel_for(tp, 0, num_workers, [&](unsigned long block)
                {
                    const unsigned long block_begin = block * block_size;
                    const unsigned long block_end =  std::min(num, block_begin + block_size);
                    for (unsigned long i = block_begin; i < block_end; ++i)
                    {
                        samples[i].diff_shape = samples[i].target_shape - samples[i].current_shape;
                        block_sums[block] += samples[i].diff_shape;
                    }
                }, 1);

                // now calculate the total result from separate blocks
                for (unsigned long i = 0; i < block_sums.size(); ++i)
                    sums[0] += block_sums[i];
            }
            else
            {
                // synchronous implementation
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    samples[i].diff_shape = samples[i].target_shape - samples[i].current_shape;
                    sums[0] += samples[i].diff_shape;
                }
            }

            for (unsigned long i = 0; i < num_split_nodes; ++i)
            {
                std::pair<unsigned long,unsigned long> range = parts.front();
                parts.pop_front();

                const impl::split_feature split = generate_split(tp, samples, range.first,
                    range.second, pixel_coordinates, sums[i], sums[left_child(i)],
                    sums[right_child(i)]);
                tree.splits.push_back(split);
                const unsigned long mid = partition_samples(split, samples, range.first, range.second);

                parts.push_back(std::make_pair(range.first, mid));
                parts.push_back(std::make_pair(mid, range.second));
            }

            // Now all the parts contain the ranges for the leaves so we can use them to
            // compute the average leaf values.
            matrix<float,0,1> present_counts(samples[0].target_shape.size());
            tree.leaf_values.resize(parts.size());
            for (unsigned long i = 0; i < parts.size(); ++i)
            {
                // Get the present counts for each dimension so we can divide each
                // dimension by the number of observations we have on it to find the mean
                // displacement in each leaf.
                present_counts = 0;
                for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
                    present_counts += samples[j].present;
                present_counts = dlib::reciprocal(present_counts);

                if (parts[i].second != parts[i].first)
                    tree.leaf_values[i] = pointwise_multiply(present_counts,sums[num_split_nodes+i]*get_nu());
                else
                    tree.leaf_values[i] = zeros_matrix(samples[0].target_shape);

                // now adjust the current shape based on these predictions
                parallel_for(tp, parts[i].first, parts[i].second, [&](unsigned long j)
                {
                    samples[j].current_shape += tree.leaf_values[i];
                    // For parts that aren't present in the training data, we just make
                    // sure that the target shape always matches and therefore gives zero
                    // error.  So this makes the algorithm simply ignore non-present
                    // landmarks.
                    for (long k = 0; k < samples[j].present.size(); ++k)
                    {
                        // if this part is not present
                        if (samples[j].present(k) == 0)
                            samples[j].target_shape(k) = samples[j].current_shape(k);
                    }
                }, 1);
            }

            return tree;
        }

        impl::split_feature randomly_generate_split_feature (
            const std::vector<dlib::vector<float,2> >& pixel_coordinates
        ) const
        {
            const double lambda = get_lambda(); 
            impl::split_feature feat;
            const size_t max_iters = get_feature_pool_size()*get_feature_pool_size();
            for (size_t i = 0; i < max_iters; ++i)
            {
                feat.idx1   = rnd.get_integer(get_feature_pool_size());
                feat.idx2   = rnd.get_integer(get_feature_pool_size());
                while (feat.idx1 == feat.idx2)
                    feat.idx2   = rnd.get_integer(get_feature_pool_size());
                const double dist = length(pixel_coordinates[feat.idx1]-pixel_coordinates[feat.idx2]);
                const double accept_prob = std::exp(-dist/lambda);
                if (accept_prob > rnd.get_random_double())
                    break;
            }

            feat.thresh = (rnd.get_random_double()*256 - 128)/2.0;

            return feat;
        }

        template<typename feature_type>
        impl::split_feature generate_split (
            thread_pool& tp,
            const std::vector<training_sample<feature_type>>& samples,
            unsigned long begin,
            unsigned long end,
            const std::vector<dlib::vector<float,2> >& pixel_coordinates,
            const matrix<float,0,1>& sum,
            matrix<float,0,1>& left_sum,
            matrix<float,0,1>& right_sum 
        ) const
        {
            // generate a bunch of random splits and test them and return the best one.

            const unsigned long num_test_splits = get_num_test_splits();  

            // sample the random features we test in this function
            std::vector<impl::split_feature> feats;
            feats.reserve(num_test_splits);
            for (unsigned long i = 0; i < num_test_splits; ++i)
                feats.push_back(randomly_generate_split_feature(pixel_coordinates));

            std::vector<matrix<float,0,1> > left_sums(num_test_splits);
            std::vector<unsigned long> left_cnt(num_test_splits);

            const unsigned long num_workers = std::max(1UL, tp.num_threads_in_pool());
            const unsigned long block_size = std::max(1UL, (num_test_splits + num_workers - 1) / num_workers);

            // now compute the sums of vectors that go left for each feature
            parallel_for(tp, 0, num_workers, [&](unsigned long block)
            {
                const unsigned long block_begin = block * block_size;
                const unsigned long block_end   = std::min(block_begin + block_size, num_test_splits);

                for (unsigned long j = begin; j < end; ++j)
                {
                    for (unsigned long i = block_begin; i < block_end; ++i)
                    {
                        if ((float)samples[j].feature_pixel_values[feats[i].idx1] - (float)samples[j].feature_pixel_values[feats[i].idx2] > feats[i].thresh)
                        {
                            left_sums[i] += samples[j].diff_shape;
                            ++left_cnt[i];
                        }
                    }
                }

            }, 1);

            // now figure out which feature is the best
            double best_score = -1;
            unsigned long best_feat = 0;
            matrix<float,0,1> temp;
            for (unsigned long i = 0; i < num_test_splits; ++i)
            {
                // check how well the feature splits the space.
                double score = 0;
                unsigned long right_cnt = end-begin-left_cnt[i];
                if (left_cnt[i] != 0 && right_cnt != 0)
                {
                    temp = sum - left_sums[i];
                    score = dot(left_sums[i],left_sums[i])/left_cnt[i] + dot(temp,temp)/right_cnt;
                    if (score > best_score)
                    {
                        best_score = score;
                        best_feat = i;
                    }
                }
            }

            left_sums[best_feat].swap(left_sum);
            if (left_sum.size() != 0)
            {
                right_sum = sum - left_sum;
            }
            else
            {
                right_sum = sum;
                left_sum = zeros_matrix(sum);
            }
            return feats[best_feat];
        }

        template<typename feature_type>
        unsigned long partition_samples (
            const impl::split_feature& split,
            std::vector<training_sample<feature_type>>& samples,
            unsigned long begin,
            unsigned long end
        ) const
        {
            // splits samples based on split (sorta like in quick sort) and returns the mid
            // point.  make sure you return the mid in a way compatible with how we walk
            // through the tree.

            unsigned long i = begin;
            for (unsigned long j = begin; j < end; ++j)
            {
                if ((float)samples[j].feature_pixel_values[split.idx1] - (float)samples[j].feature_pixel_values[split.idx2] > split.thresh)
                {
                    samples[i].swap(samples[j]);
                    ++i;
                }
            }
            return i;
        }



        template<typename feature_type>
        matrix<float,0,1> populate_training_sample_shapes(
            const std::vector<std::vector<full_object_detection> >& objects,
            std::vector<training_sample<feature_type>>& samples
        ) const
        {
            samples.clear();
            matrix<float,0,1> mean_shape;
            matrix<float,0,1> count;
            // first fill out the target shapes
            for (unsigned long i = 0; i < objects.size(); ++i)
            {
                for (unsigned long j = 0; j < objects[i].size(); ++j)
                {
                    training_sample<feature_type> sample;
                    sample.image_idx = i;
                    sample.rect = objects[i][j].get_rect();
                    object_to_shape(objects[i][j], sample.target_shape, sample.present);
                    for (unsigned long itr = 0; itr < get_oversampling_amount(); ++itr)
                        samples.push_back(sample);
                    mean_shape += sample.target_shape;
                    count += sample.present;
                }
            }

            mean_shape = pointwise_multiply(mean_shape,reciprocal(count));

            // now go pick random initial shapes
            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                if ((i%get_oversampling_amount()) == 0)
                {
                    // The mean shape is what we really use as an initial shape so always
                    // include it in the training set as an example starting shape.
                    samples[i].current_shape = mean_shape;
                }
                else
                {
                    samples[i].current_shape.set_size(0);

                    matrix<float,0,1> hits(mean_shape.size());
                    hits = 0;

                    int iter = 0;
                    // Pick a few samples at random and randomly average them together to
                    // make the initial shape.  Note that we make sure we get at least one
                    // observation (i.e. non-OBJECT_PART_NOT_PRESENT) on each part
                    // location.
                    while(min(hits) == 0 || iter < 2)
                    {
                        ++iter;
                        const unsigned long rand_idx = rnd.get_random_32bit_number()%samples.size();
                        const double alpha = rnd.get_random_double()+0.1;
                        samples[i].current_shape += alpha*samples[rand_idx].target_shape;
                        hits += alpha*samples[rand_idx].present;
                    }
                    samples[i].current_shape = pointwise_multiply(samples[i].current_shape, reciprocal(hits));

                    if (_oversampling_translation_jitter != 0)
                    {
                        dpoint off;
                        off.x() = rnd.get_double_in_range(-_oversampling_translation_jitter,_oversampling_translation_jitter);
                        off.y() = rnd.get_double_in_range(-_oversampling_translation_jitter,_oversampling_translation_jitter);
                        for (long j = 0; j < samples[i].current_shape.size()/2; ++j)
                        {
                            samples[i].current_shape(2*j) += off.x();
                            samples[i].current_shape(2*j+1) += off.y();
                        }
                    }
                }

            }
            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                for (long k = 0; k < samples[i].present.size(); ++k)
                {
                    // if this part is not present
                    if (samples[i].present(k) == 0)
                        samples[i].target_shape(k) = samples[i].current_shape(k);
                }
            }


            return mean_shape;
        }


        void randomly_sample_pixel_coordinates (
            std::vector<dlib::vector<float,2> >& pixel_coordinates,
            const double min_x,
            const double min_y,
            const double max_x,
            const double max_y
        ) const
        /*!
            ensures
                - #pixel_coordinates.size() == get_feature_pool_size() 
                - for all valid i:
                    - pixel_coordinates[i] == a point in the box defined by the min/max x/y arguments.
        !*/
        {
            pixel_coordinates.resize(get_feature_pool_size());
            for (unsigned long i = 0; i < get_feature_pool_size(); ++i)
            {
                pixel_coordinates[i].x() = rnd.get_random_double()*(max_x-min_x) + min_x;
                pixel_coordinates[i].y() = rnd.get_random_double()*(max_y-min_y) + min_y;
            }
        }

        std::vector<std::vector<dlib::vector<float,2> > > randomly_sample_pixel_coordinates (
            const matrix<float,0,1>& initial_shape
        ) const
        {
            const double padding = get_feature_pool_region_padding();
            // Figure out the bounds on the object shapes.  We will sample uniformly
            // from this box.
            matrix<float> temp = reshape(initial_shape, initial_shape.size()/2, 2);
            double min_x = min(colm(temp,0));
            double min_y = min(colm(temp,1));
            double max_x = max(colm(temp,0));
            double max_y = max(colm(temp,1));

            if (get_padding_mode() == bounding_box_relative)
            {
                min_x = std::min(0.0, min_x);
                min_y = std::min(0.0, min_y);
                max_x = std::max(1.0, max_x);
                max_y = std::max(1.0, max_y);
            }

            min_x -= padding;
            min_y -= padding;
            max_x += padding;
            max_y += padding;

            std::vector<std::vector<dlib::vector<float,2> > > pixel_coordinates;
            pixel_coordinates.resize(get_cascade_depth());
            for (unsigned long i = 0; i < get_cascade_depth(); ++i)
                randomly_sample_pixel_coordinates(pixel_coordinates[i], min_x, min_y, max_x, max_y);
            return pixel_coordinates;
        }



        mutable dlib::rand rnd;

        unsigned long _cascade_depth;
        unsigned long _tree_depth;
        unsigned long _num_trees_per_cascade_level;
        double _nu;
        unsigned long _oversampling_amount;
        unsigned long _feature_pool_size;
        double _lambda;
        unsigned long _num_test_splits;
        double _feature_pool_region_padding;
        bool _verbose;
        unsigned long _num_threads;
        padding_mode_t _padding_mode;
        double _oversampling_translation_jitter;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename some_type_of_rectangle
        >
    image_dataset_metadata::dataset make_bounding_box_regression_training_data (
        const image_dataset_metadata::dataset& truth,
        const std::vector<std::vector<some_type_of_rectangle>>& detections
    )
    {
        DLIB_CASSERT(truth.images.size() == detections.size(), 
            "truth.images.size(): "<< truth.images.size() <<
            "\tdetections.size(): "<< detections.size()
        );
        image_dataset_metadata::dataset result = truth;

        for (size_t i = 0; i < truth.images.size(); ++i)
        {
            result.images[i].boxes.clear();
            for (auto truth_box : truth.images[i].boxes)
            {
                if (truth_box.ignore)
                    continue;

                // Find the detection that best matches the current truth_box.
                auto det = max_scoring_element(detections[i], [&truth_box](const rectangle& r) { return box_intersection_over_union(r, truth_box.rect); });
                if (det.second > 0.5)
                {
                    // Remove any existing parts and replace them with the truth_box corners.
                    truth_box.parts.clear();
                    auto b = truth_box.rect;
                    truth_box.parts["left"]     = (b.tl_corner()+b.bl_corner())/2;
                    truth_box.parts["right"]    = (b.tr_corner()+b.br_corner())/2;
                    truth_box.parts["top"]      = (b.tl_corner()+b.tr_corner())/2;
                    truth_box.parts["bottom"]   = (b.bl_corner()+b.br_corner())/2;
                    truth_box.parts["middle"]   = center(b);

                    // Now replace the bounding truth_box with the detector's bounding truth_box.
                    truth_box.rect = det.first;

                    result.images[i].boxes.push_back(truth_box);
                }
            }
        }
        return result;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_TRAINER_H_

