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
    matrix<uint16_t> label_image; // The ground-truth label of each pixel.
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
            const auto class_label_index = rgb_label_to_index_label(rgb_class_label);
            const Voc2012class& voc2012_class = classes[class_label_index];

            const auto i = result_map.find(rgb_instance_label);
            if (i == result_map.end())
            {
                // Encountered a new instance
                result_map[rgb_instance_label] = rectangle(c, r, c, r);
                result_map[rgb_instance_label].label = voc2012_class.classlabel;
                // TODO: read the instance's class from the other png!
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
    const std::vector<std::vector<truth_instance>>& truth_instances
)
{
    std::vector<std::vector<mmod_rect>> mmod_rects(truth_instances.size());

    std::transform(
        truth_instances.begin(),
        truth_instances.end(),
        mmod_rects.begin(),
        extract_mmod_rects
    );

    return mmod_rects;
}

det_bnet_type train_detection_network(
    const std::vector<image_info>& listing,
    const std::vector<std::vector<truth_instance>>& truth_instances,
    unsigned int det_minibatch_size
)
{
    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    mmod_options options(extract_mmod_rect_vectors(truth_instances), 70, 30);
    det_bnet_type det_net(options);

    det_net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    dlib::pipe<det_training_sample> data(200);
    auto f = [&data, &listing, &truth_instances](time_t seed)
    {
        dlib::rand rnd(time(0) + seed);
        matrix<rgb_pixel> input_image;

        random_cropper cropper;
        cropper.set_seed(time(0));
        cropper.set_chip_dims(350, 350);

        // Usually you want to give the cropper whatever min sizes you passed to the
        // mmod_options constructor, or very slightly smaller sizes, which is what we do here.
        cropper.set_min_object_size(69, 28);
        cropper.set_max_rotation_degrees(2);

        det_training_sample temp;

        while (data.is_enabled())
        {
            // Pick a random input image.
            const auto random_index = rnd.get_random_32bit_number() % listing.size();
            const image_info& image_info = listing[random_index];

            // Load the input image.
            load_image(input_image, image_info.image_filename);

            // Get a random crop of the input.
            cropper(input_image, extract_mmod_rects(truth_instances[random_index]), temp.input_image, temp.mmod_rects);

            disturb_colors(temp.input_image, rnd);

            // Push the result to be used by the trainer.
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f]() { f(1); });
    std::thread data_loader2([f]() { f(2); });
    std::thread data_loader3([f]() { f(3); });
    std::thread data_loader4([f]() { f(4); });

    dnn_trainer<det_bnet_type> det_trainer(det_net, sgd(weight_decay, momentum));
    det_trainer.be_verbose();
    det_trainer.set_learning_rate(initial_learning_rate);
    det_trainer.set_synchronization_file("pascal_voc2012_det_trainer_state_file.dat", std::chrono::minutes(10));
    det_trainer.set_iterations_without_progress_threshold(5000);

    // Output training parameters.
    cout << "Training detector network:" << endl << det_trainer << endl;

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<std::vector<mmod_rect>> labels;

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-4.
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

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // also wait for threaded processing to stop in the trainer.
    det_trainer.get_net();

    det_net.clean();

    return det_net;
}

// ----------------------------------------------------------------------------------------

matrix<uint16_t> keep_only_current_instance(const matrix<rgb_pixel>& rgb_label_image, const rgb_pixel rgb_label)
{
    const auto nr = rgb_label_image.nr();
    const auto nc = rgb_label_image.nc();

    matrix<uint16_t> result(nr, nc);

    for (long r = 0; r < nr; ++r)
    {
        for (long c = 0; c < nc; ++c)
        {
            const auto& index = rgb_label_image(r, c);
            if (index == rgb_label)
                result(r, c) = 1;
            else if (index == dlib::rgb_pixel(224, 224, 192))
                result(r, c) = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
            else
                result(r, c) = 0;
        }
    }

    return result;
}

seg_bnet_type train_segmentation_network(
    const std::vector<image_info>& listing,
    const std::vector<std::vector<truth_instance>>& truth_instances,
    unsigned int seg_minibatch_size
)
{
    seg_bnet_type seg_net;

    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    dnn_trainer<seg_bnet_type> seg_trainer(seg_net, sgd(weight_decay, momentum));
    seg_trainer.be_verbose();
    seg_trainer.set_learning_rate(initial_learning_rate);
    seg_trainer.set_synchronization_file("pascal_voc2012_seg_trainer_state_file.dat", std::chrono::minutes(10));
    seg_trainer.set_iterations_without_progress_threshold(5000);
    set_all_bn_running_stats_window_sizes(seg_net, 1000);

    // Output training parameters.
    cout << "Training segmentation network:" << endl << seg_trainer << endl;

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<matrix<uint16_t>> labels;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<seg_training_sample> data(200);
    auto f = [&data, &listing, &truth_instances](time_t seed)
    {
        dlib::rand rnd(time(0) + seed);
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<rgb_pixel> rgb_label_chip;
        seg_training_sample temp;
        while (data.is_enabled())
        {
            // Pick a random input image.
            const auto random_index = rnd.get_random_32bit_number() % listing.size();
            const auto image_truths = truth_instances[random_index];
            
            if (!image_truths.empty())
            {
                const image_info& image_info = listing[random_index];

                // Load the input image.
                load_image(input_image, image_info.image_filename);

                // Load the ground-truth (RGB) instance labels.
                load_image(rgb_label_image, image_info.instance_label_filename);

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

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

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

std::vector<truth_instance> load_truth_instances(const image_info& image_info)
{
    matrix<rgb_pixel> instance_label_image;
    matrix<rgb_pixel> class_label_image;

    load_image(instance_label_image, image_info.instance_label_filename);
    load_image(class_label_image, image_info.class_label_filename);

    auto truth_instances = rgb_label_images_to_truth_instances(instance_label_image, class_label_image);

    ignore_overlapped_boxes(truth_instances, test_box_overlap(0.50, 0.95));

    for (auto& truth : truth_instances)
    {
        const auto& rect = truth.mmod_rect.rect;
        if (rect.width() < 35 && rect.height() < 35)
            truth.mmod_rect.ignore = true;
    }

    return truth_instances;
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

int main(int argc, char** argv) try
{
    if (argc < 2 || argc > 4)
    {
        cout << "To run this program you need a copy of the PASCAL VOC2012 dataset." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./dnn_instance_segmentation_train_ex /path/to/VOC2012 [det-minibatch-size] [seg-minibatch-size]" << endl;
        return 1;
    }

    cout << "\nSCANNING PASCAL VOC2012 DATASET\n" << endl;

    const auto listing = get_pascal_voc2012_train_listing(argv[1]);
    cout << "images in dataset: " << listing.size() << endl;
    if (listing.size() == 0)
    {
        cout << "Didn't find the VOC2012 dataset. " << endl;
        return 1;
    }

    // mini-batches smaller than the default can be used with GPUs having less memory
    const unsigned int det_minibatch_size = argc >= 3 ? std::stoi(argv[2]) : 40;
    const unsigned int seg_minibatch_size = argc >= 4 ? std::stoi(argv[3]) : 25;
    cout << "det mini-batch size: " << det_minibatch_size << endl;
    cout << "seg mini-batch size: " << seg_minibatch_size << endl;

    // extract the MMOD rects
    cout << "\nExtracting all truth instances...";
    const auto truth_instances = load_all_truth_instances(listing);
    cout << " Done!" << endl << endl;

    // First train a detection network (loss_mmod), and then a mask segmentation network (loss_log_per_pixel)
    const auto det_net = train_detection_network    (listing, truth_instances, det_minibatch_size);
    const auto seg_net = train_segmentation_network (listing, truth_instances, seg_minibatch_size);

    cout << "saving network" << endl;
    serialize(instance_segmentation_net_filename) << det_net << seg_net;
}

catch(std::exception& e)
{
    cout << e.what() << endl;
}

