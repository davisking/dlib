// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to train a instance segmentation net using the PASCAL VOC2012
    dataset.  For an introduction to what segmentation is, see the accompanying header file
    dnn_instance_segmentation_ex.h.

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_instance_segmentation_train_ex example program.
    3. Run:
       ./dnn_instance_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_instance_segmentation_ex example program.
    6. Run:
       ./dnn_instance_segmentation_ex /path/to/VOC2012-or-other-images

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp, dnn_introduction2_ex.cpp,
    and dnn_semantic_segmentation_train_ex.cpp before reading this example program.
*/

#include "dnn_instance_segmentation_ex.h"
#include "pascal_voc_2012.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <execution>
#endif // __cplusplus >= 201703L

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// A single training sample for detection. A mini-batch comprises many of these.
struct det_training_sample
{
    matrix<rgb_pixel> input_image;
    std::vector<dlib::mmod_rect> mmod_rects;
};

// A single training sample for segmentation. A mini-batch comprises many of these.
struct seg_training_sample
{
    matrix<rgb_pixel> input_image;
    matrix<float> label_image; // The ground-truth label of each pixel. (+1 or -1)
};

// ----------------------------------------------------------------------------------------

bool is_instance_pixel(const dlib::rgb_pixel& rgb_label)
{
    if (rgb_label == dlib::rgb_pixel(0, 0, 0))
        return false; // Background
    if (rgb_label == dlib::rgb_pixel(224, 224, 192))
        return false; // The cream-colored `void' label is used in border regions and to mask difficult objects

    return true;
}

// Provide hash function for dlib::rgb_pixel
namespace std {
    template <>
    struct hash<dlib::rgb_pixel>
    {
        std::size_t operator()(const dlib::rgb_pixel& p) const
        {
            return (static_cast<uint32_t>(p.red) << 16)
                 | (static_cast<uint32_t>(p.green) << 8)
                 | (static_cast<uint32_t>(p.blue));
        }
    };
}

struct truth_instance
{
    dlib::rgb_pixel rgb_label;
    dlib::mmod_rect mmod_rect;
};

std::vector<truth_instance> rgb_label_images_to_truth_instances(
    const dlib::matrix<dlib::rgb_pixel>& instance_label_image,
    const dlib::matrix<dlib::rgb_pixel>& class_label_image
)
{
    std::unordered_map<dlib::rgb_pixel, mmod_rect> result_map;

    DLIB_CASSERT(instance_label_image.nr() == class_label_image.nr());
    DLIB_CASSERT(instance_label_image.nc() == class_label_image.nc());

    const auto nr = instance_label_image.nr();
    const auto nc = instance_label_image.nc();

    for (int r = 0; r < nr; ++r)
    {
        for (int c = 0; c < nc; ++c)
        {
            const auto rgb_instance_label = instance_label_image(r, c);

            if (!is_instance_pixel(rgb_instance_label))
                continue;

            const auto rgb_class_label = class_label_image(r, c);
            const Voc2012class& voc2012_class = find_voc2012_class(rgb_class_label);

            const auto i = result_map.find(rgb_instance_label);
            if (i == result_map.end())
            {
                // Encountered a new instance
                result_map[rgb_instance_label] = rectangle(c, r, c, r);
                result_map[rgb_instance_label].label = voc2012_class.classlabel;
            }
            else
            {
                // Not the first occurrence - update the rect
                auto& rect = i->second.rect;

                if (c < rect.left())
                    rect.set_left(c);
                else if (c > rect.right())
                    rect.set_right(c);

                if (r > rect.bottom())
                    rect.set_bottom(r);

                DLIB_CASSERT(i->second.label == voc2012_class.classlabel);
            }
        }
    }

    std::vector<truth_instance> flat_result;
    flat_result.reserve(result_map.size());

    for (const auto& i : result_map) {
        flat_result.push_back(truth_instance{
            i.first, i.second
        });
    }

    return flat_result;
}

// ----------------------------------------------------------------------------------------

struct truth_image
{
    image_info info;
    std::vector<truth_instance> truth_instances;
};

std::vector<mmod_rect> extract_mmod_rects(
    const std::vector<truth_instance>& truth_instances
)
{
    std::vector<mmod_rect> mmod_rects(truth_instances.size());

    std::transform(
        truth_instances.begin(),
        truth_instances.end(),
        mmod_rects.begin(),
        [](const truth_instance& truth) { return truth.mmod_rect; }
    );

    return mmod_rects;
};

std::vector<std::vector<mmod_rect>> extract_mmod_rect_vectors(
    const std::vector<truth_image>& truth_images
)
{
    std::vector<std::vector<mmod_rect>> mmod_rects(truth_images.size());

    const auto extract_mmod_rects_from_truth_image = [](const truth_image& truth_image)
    {
        return extract_mmod_rects(truth_image.truth_instances);
    };

    std::transform(
        truth_images.begin(),
        truth_images.end(),
        mmod_rects.begin(),
        extract_mmod_rects_from_truth_image
    );

    return mmod_rects;
}

det_bnet_type train_detection_network(
    const std::vector<truth_image>& truth_images,
    unsigned int det_minibatch_size
)
{
    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;
    const double min_detector_window_overlap_iou = 0.65;

    const int target_size = 70;
    const int min_target_size = 30;

    mmod_options options(
        extract_mmod_rect_vectors(truth_images),
        target_size, min_target_size,
        min_detector_window_overlap_iou
    );

    options.overlaps_ignore = test_box_overlap(0.5, 0.9);

    det_bnet_type det_net(options);

    det_net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    dlib::pipe<det_training_sample> data(200);
    auto f = [&data, &truth_images, target_size, min_target_size](time_t seed)
    {
        dlib::rand rnd(time(0) + seed);
        matrix<rgb_pixel> input_image;

        random_cropper cropper;
        cropper.set_seed(time(0));
        cropper.set_chip_dims(350, 350);

        // Usually you want to give the cropper whatever min sizes you passed to the
        // mmod_options constructor, or very slightly smaller sizes, which is what we do here.
        cropper.set_min_object_size(target_size - 2, min_target_size - 2);
        cropper.set_max_rotation_degrees(2);

        det_training_sample temp;

        while (data.is_enabled())
        {
            // Pick a random input image.
            const auto random_index = rnd.get_random_32bit_number() % truth_images.size();
            const auto& truth_image = truth_images[random_index];

            // Load the input image.
            load_image(input_image, truth_image.info.image_filename);

            // Get a random crop of the input.
            const auto mmod_rects = extract_mmod_rects(truth_image.truth_instances);
            cropper(input_image, mmod_rects, temp.input_image, temp.mmod_rects);

            disturb_colors(temp.input_image, rnd);

            // Push the result to be used by the trainer.
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f]() { f(1); });
    std::thread data_loader2([f]() { f(2); });
    std::thread data_loader3([f]() { f(3); });
    std::thread data_loader4([f]() { f(4); });

    const auto stop_data_loaders = [&]()
    {
        data.disable();
        data_loader1.join();
        data_loader2.join();
        data_loader3.join();
        data_loader4.join();
    };

    dnn_trainer<det_bnet_type> det_trainer(det_net, sgd(weight_decay, momentum));

    try
    {
        det_trainer.be_verbose();
        det_trainer.set_learning_rate(initial_learning_rate);
        det_trainer.set_synchronization_file("pascal_voc2012_det_trainer_state_file.dat", std::chrono::minutes(10));
        det_trainer.set_iterations_without_progress_threshold(5000);

        // Output training parameters.
        cout << det_trainer << endl;

        std::vector<matrix<rgb_pixel>> samples;
        std::vector<std::vector<mmod_rect>> labels;

        // The main training loop.  Keep making mini-batches and giving them to the trainer.
        // We will run until the learning rate becomes small enough.
        while (det_trainer.get_learning_rate() >= 1e-4)
        {
            samples.clear();
            labels.clear();

            // make a mini-batch
            det_training_sample temp;
            while (samples.size() < det_minibatch_size)
            {
                data.dequeue(temp);

                samples.push_back(std::move(temp.input_image));
                labels.push_back(std::move(temp.mmod_rects));
            }

            det_trainer.train_one_step(samples, labels);
        }
    }
    catch (std::exception&)
    {
        stop_data_loaders();
        throw;
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    stop_data_loaders();

    // also wait for threaded processing to stop in the trainer.
    det_trainer.get_net();

    det_net.clean();

    return det_net;
}

// ----------------------------------------------------------------------------------------

matrix<float> keep_only_current_instance(const matrix<rgb_pixel>& rgb_label_image, const rgb_pixel rgb_label)
{
    const auto nr = rgb_label_image.nr();
    const auto nc = rgb_label_image.nc();

    matrix<float> result(nr, nc);

    for (long r = 0; r < nr; ++r)
    {
        for (long c = 0; c < nc; ++c)
        {
            const auto& index = rgb_label_image(r, c);
            if (index == rgb_label)
                result(r, c) = +1;
            else if (index == dlib::rgb_pixel(224, 224, 192))
                result(r, c) = 0;
            else
                result(r, c) = -1;
        }
    }

    return result;
}

seg_bnet_type train_segmentation_network(
    const std::vector<truth_image>& truth_images,
    unsigned int seg_minibatch_size,
    const std::string& classlabel
)
{
    seg_bnet_type seg_net;

    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    const std::string synchronization_file_name
        = "pascal_voc2012_seg_trainer_state_file"
        + (classlabel.empty() ? "" : ("_" + classlabel))
        + ".dat";

    dnn_trainer<seg_bnet_type> seg_trainer(seg_net, sgd(weight_decay, momentum));
    seg_trainer.be_verbose();
    seg_trainer.set_learning_rate(initial_learning_rate);
    seg_trainer.set_synchronization_file(synchronization_file_name, std::chrono::minutes(10));
    seg_trainer.set_iterations_without_progress_threshold(2000);
    set_all_bn_running_stats_window_sizes(seg_net, 1000);

    // Output training parameters.
    cout << seg_trainer << endl;

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<matrix<float>> labels;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<seg_training_sample> data(200);
    auto f = [&data, &truth_images](time_t seed)
    {
        dlib::rand rnd(time(0) + seed);
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<rgb_pixel> rgb_label_chip;
        seg_training_sample temp;
        while (data.is_enabled())
        {
            // Pick a random input image.
            const auto random_index = rnd.get_random_32bit_number() % truth_images.size();
            const auto& truth_image = truth_images[random_index];
            const auto image_truths = truth_image.truth_instances;

            if (!image_truths.empty())
            {
                const image_info& info = truth_image.info;

                // Load the input image.
                load_image(input_image, info.image_filename);

                // Load the ground-truth (RGB) instance labels.
                load_image(rgb_label_image, info.instance_label_filename);

                // Pick a random training instance.
                const auto& truth_instance = image_truths[rnd.get_random_32bit_number() % image_truths.size()];
                const auto& truth_rect = truth_instance.mmod_rect.rect;
                const auto cropping_rect = get_cropping_rect(truth_rect);

                // Pick a random crop around the instance.
                const auto max_x_translate_amount = static_cast<long>(truth_rect.width() / 10.0);
                const auto max_y_translate_amount = static_cast<long>(truth_rect.height() / 10.0);

                const auto random_translate = point(
                    rnd.get_integer_in_range(-max_x_translate_amount, max_x_translate_amount + 1),
                    rnd.get_integer_in_range(-max_y_translate_amount, max_y_translate_amount + 1)
                );

                const rectangle random_rect(
                    cropping_rect.left()   + random_translate.x(),
                    cropping_rect.top()    + random_translate.y(),
                    cropping_rect.right()  + random_translate.x(),
                    cropping_rect.bottom() + random_translate.y()
                );

                const chip_details chip_details(random_rect, chip_dims(seg_dim, seg_dim));

                // Crop the input image.
                extract_image_chip(input_image, chip_details, temp.input_image, interpolate_bilinear());

                disturb_colors(temp.input_image, rnd);

                // Crop the labels correspondingly. However, note that here bilinear
                // interpolation would make absolutely no sense - you wouldn't say that
                // a bicycle is half-way between an aeroplane and a bird, would you?
                extract_image_chip(rgb_label_image, chip_details, rgb_label_chip, interpolate_nearest_neighbor());

                // Clear pixels not related to the current instance.
                temp.label_image = keep_only_current_instance(rgb_label_chip, truth_instance.rgb_label);

                // Push the result to be used by the trainer.
                data.enqueue(temp);
            }
            else
            {
                // TODO: use background samples as well
            }
        }
    };
    std::thread data_loader1([f]() { f(1); });
    std::thread data_loader2([f]() { f(2); });
    std::thread data_loader3([f]() { f(3); });
    std::thread data_loader4([f]() { f(4); });

    const auto stop_data_loaders = [&]()
    {
        data.disable();
        data_loader1.join();
        data_loader2.join();
        data_loader3.join();
        data_loader4.join();
    };

    try
    {
        // The main training loop.  Keep making mini-batches and giving them to the trainer.
        // We will run until the learning rate has dropped by a factor of 1e-4.
        while (seg_trainer.get_learning_rate() >= 1e-4)
        {
            samples.clear();
            labels.clear();

            // make a mini-batch
            seg_training_sample temp;
            while (samples.size() < seg_minibatch_size)
            {
                data.dequeue(temp);

                samples.push_back(std::move(temp.input_image));
                labels.push_back(std::move(temp.label_image));
            }

            seg_trainer.train_one_step(samples, labels);
        }
    }
    catch (std::exception&)
    {
        stop_data_loaders();
        throw;
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    stop_data_loaders();

    // also wait for threaded processing to stop in the trainer.
    seg_trainer.get_net();

    seg_net.clean();

    return seg_net;
}

// ----------------------------------------------------------------------------------------

int ignore_overlapped_boxes(
    std::vector<truth_instance>& truth_instances,
    const test_box_overlap& overlaps
)
/*!
    ensures
        - Whenever two rectangles in boxes overlap, according to overlaps(), we set the
          smallest box to ignore.
        - returns the number of newly ignored boxes.
!*/
{
    int num_ignored = 0;
    for (size_t i = 0, end = truth_instances.size(); i < end; ++i)
    {
        auto& box_i = truth_instances[i].mmod_rect;
        if (box_i.ignore)
            continue;
        for (size_t j = i+1; j < end; ++j)
        {
            auto& box_j = truth_instances[j].mmod_rect;
            if (box_j.ignore)
                continue;
            if (overlaps(box_i, box_j))
            {
                ++num_ignored;
                if(box_i.rect.area() < box_j.rect.area())
                    box_i.ignore = true;
                else
                    box_j.ignore = true;
            }
        }
    }
    return num_ignored;
}

std::vector<truth_instance> load_truth_instances(const image_info& info)
{
    matrix<rgb_pixel> instance_label_image;
    matrix<rgb_pixel> class_label_image;

    load_image(instance_label_image, info.instance_label_filename);
    load_image(class_label_image, info.class_label_filename);

    return rgb_label_images_to_truth_instances(instance_label_image, class_label_image);
};

std::vector<std::vector<truth_instance>> load_all_truth_instances(const std::vector<image_info>& listing)
{
    std::vector<std::vector<truth_instance>> truth_instances(listing.size());

    std::transform(
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
        std::execution::par,
#endif // __cplusplus >= 201703L
        listing.begin(),
        listing.end(),
        truth_instances.begin(),
        load_truth_instances
    );

    return truth_instances;
}

// ----------------------------------------------------------------------------------------

std::vector<truth_image> filter_based_on_classlabel(
    const std::vector<truth_image>& truth_images,
    const std::vector<std::string>& desired_classlabels
)
{
    std::vector<truth_image> result;

    const auto represents_desired_class = [&desired_classlabels](const truth_instance& truth_instance) {
        return std::find(
            desired_classlabels.begin(),
            desired_classlabels.end(),
            truth_instance.mmod_rect.label
        ) != desired_classlabels.end();
    };

    for (const auto& input : truth_images)
    {
        const auto has_desired_class = std::any_of(
            input.truth_instances.begin(),
            input.truth_instances.end(),
            represents_desired_class
        );

        if (has_desired_class) {

            // NB: This keeps only MMOD rects belonging to any of the desired classes.
            //     A reasonable alternative could be to keep all rects, but mark those
            //     belonging in other classes to be ignored during training.
            std::vector<truth_instance> temp;
            std::copy_if(
                input.truth_instances.begin(),
                input.truth_instances.end(),
                std::back_inserter(temp),
                represents_desired_class
            );

            result.push_back(truth_image{ input.info, temp });
        }
    }

    return result;
}

// Ignore truth boxes that overlap too much, are too small, or have a large aspect ratio.
void ignore_some_truth_boxes(std::vector<truth_image>& truth_images)
{
    for (auto& i : truth_images)
    {
        auto& truth_instances = i.truth_instances;

        ignore_overlapped_boxes(truth_instances, test_box_overlap(0.90, 0.95));

        for (auto& truth : truth_instances)
        {
            if (truth.mmod_rect.ignore)
                continue;

            const auto& rect = truth.mmod_rect.rect;

            constexpr unsigned long min_width  = 35;
            constexpr unsigned long min_height = 35;
            if (rect.width() < min_width && rect.height() < min_height)
            {
                truth.mmod_rect.ignore = true;
                continue;
            }

            constexpr double max_aspect_ratio_width_to_height = 3.0;
            constexpr double max_aspect_ratio_height_to_width = 1.5;
            const double aspect_ratio_width_to_height = rect.width() / static_cast<double>(rect.height());
            const double aspect_ratio_height_to_width = 1.0 / aspect_ratio_width_to_height;
            const bool is_aspect_ratio_too_large
                =  aspect_ratio_width_to_height > max_aspect_ratio_width_to_height
                || aspect_ratio_height_to_width > max_aspect_ratio_height_to_width;

            if (is_aspect_ratio_too_large)
                truth.mmod_rect.ignore = true;
        }
    }
}

// Filter images that have no (non-ignored) truth
std::vector<truth_image> filter_images_with_no_truth(const std::vector<truth_image>& truth_images)
{
    std::vector<truth_image> result;

    for (const auto& truth_image : truth_images)
    {
        const auto ignored = [](const truth_instance& truth) { return truth.mmod_rect.ignore; };
        const auto& truth_instances = truth_image.truth_instances;
        if (!std::all_of(truth_instances.begin(), truth_instances.end(), ignored))
            result.push_back(truth_image);
    }

    return result;
}

int main(int argc, char** argv) try
{
    if (argc < 2)
    {
        cout << "To run this program you need a copy of the PASCAL VOC2012 dataset." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./dnn_instance_segmentation_train_ex /path/to/VOC2012 [det-minibatch-size] [seg-minibatch-size] [class-1] [class-2] [class-3] ..." << endl;
        return 1;
    }

    cout << "\nSCANNING PASCAL VOC2012 DATASET\n" << endl;

    const auto listing = get_pascal_voc2012_train_listing(argv[1]);
    cout << "images in entire dataset: " << listing.size() << endl;
    if (listing.size() == 0)
    {
        cout << "Didn't find the VOC2012 dataset. " << endl;
        return 1;
    }

    // mini-batches smaller than the default can be used with GPUs having less memory
    const unsigned int det_minibatch_size = argc >= 3 ? std::stoi(argv[2]) : 35;
    const unsigned int seg_minibatch_size = argc >= 4 ? std::stoi(argv[3]) : 100;
    cout << "det mini-batch size: " << det_minibatch_size << endl;
    cout << "seg mini-batch size: " << seg_minibatch_size << endl;

    std::vector<std::string> desired_classlabels;

    for (int arg = 4; arg < argc; ++arg)
        desired_classlabels.push_back(argv[arg]);

    if (desired_classlabels.empty())
    {
        desired_classlabels.push_back("bicycle");
        desired_classlabels.push_back("car");
        desired_classlabels.push_back("cat");
    }

    cout << "desired classlabels:";
    for (const auto& desired_classlabel : desired_classlabels)
        cout << " " << desired_classlabel;
    cout << endl;

    // extract the MMOD rects
    cout << endl << "Extracting all truth instances...";
    const auto truth_instances = load_all_truth_instances(listing);
    cout << " Done!" << endl << endl;

    DLIB_CASSERT(listing.size() == truth_instances.size());

    std::vector<truth_image> original_truth_images;
    for (size_t i = 0, end = listing.size(); i < end; ++i)
    {
        original_truth_images.push_back(truth_image{
            listing[i], truth_instances[i]
        });
    }

    auto truth_images_filtered_by_class = filter_based_on_classlabel(original_truth_images, desired_classlabels);

    cout << "images in dataset filtered by class: " << truth_images_filtered_by_class.size() << endl;

    ignore_some_truth_boxes(truth_images_filtered_by_class);
    const auto truth_images = filter_images_with_no_truth(truth_images_filtered_by_class);

    cout << "images in dataset after ignoring some truth boxes: " << truth_images.size() << endl;

    // First train an object detector network (loss_mmod).
    cout << endl << "Training detector network:" << endl;
    const auto det_net = train_detection_network(truth_images, det_minibatch_size);

    // Then train mask predictors (segmentation).
    std::map<std::string, seg_bnet_type> seg_nets_by_class;

    // This flag controls if a separate mask predictor is trained for each class.
    // Note that it would also be possible to train a separate mask predictor for
    // class groups, each containing somehow similar classes -- for example, one
    // mask predictor for cars and buses, another for cats and dogs, and so on.
    constexpr bool separate_seg_net_for_each_class = true;

    if (separate_seg_net_for_each_class)
    {
        for (const auto& classlabel : desired_classlabels)
        {
            // Consider only the truth images belonging to this class.
            const auto class_images = filter_based_on_classlabel(truth_images, { classlabel });

            cout << endl << "Training segmentation network for class " << classlabel << ":" << endl;
            seg_nets_by_class[classlabel] = train_segmentation_network(class_images, seg_minibatch_size, classlabel);
        }
    }
    else
    {
        cout << "Training a single segmentation network:" << endl;
        seg_nets_by_class[""] = train_segmentation_network(truth_images, seg_minibatch_size, "");
    }

    cout << "Saving networks" << endl;
    serialize(instance_segmentation_net_filename) << det_net << seg_nets_by_class;
}

catch(std::exception& e)
{
    cout << e.what() << endl;
}

