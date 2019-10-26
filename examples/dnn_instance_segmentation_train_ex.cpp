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

// The names of the input image and the associated RGB label image in the PASCAL VOC 2012
// data set.
struct image_info
{
    string image_filename;
    string label_filename;
};

// Read the list of image files belonging to either the "train", "trainval", or "val" set
// of the PASCAL VOC2012 data.
std::vector<image_info> get_pascal_voc2012_listing(
    const std::string& voc2012_folder,
    const std::string& file = "train" // "train", "trainval", or "val"
)
{
    std::ifstream in(voc2012_folder + "/ImageSets/Segmentation/" + file + ".txt");

    std::vector<image_info> results;

    while (in)
    {
        std::string basename;
        in >> basename;

        if (!basename.empty())
        {
            image_info image_info;
            image_info.image_filename = voc2012_folder + "/JPEGImages/" + basename + ".jpg";
            image_info.label_filename = voc2012_folder + "/SegmentationObject/" + basename + ".png";
            results.push_back(image_info);
        }
    }

    return results;
}

// Read the list of image files belong to the "train" set of the PASCAL VOC2012 data.
std::vector<image_info> get_pascal_voc2012_train_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "train");
}

// Read the list of image files belong to the "val" set of the PASCAL VOC2012 data.
std::vector<image_info> get_pascal_voc2012_val_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "val");
}

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

std::vector<dlib::mmod_rect> rgb_label_image_to_mmod_rects(
    const dlib::matrix<dlib::rgb_pixel>& rgb_label_image
)
{
    std::deque<dlib::mmod_rect> mmod_rects;

    std::unordered_map<dlib::rgb_pixel, int> instance_indexes;

    const auto nr = rgb_label_image.nr();
    const auto nc = rgb_label_image.nc();

    for (int r = 0; r < nr; ++r)
    {
        for (int c = 0; c < nc; ++c)
        {
            const auto rgb_label = rgb_label_image(r, c);

            if (!is_instance_pixel(rgb_label))
                continue;

            const auto i = instance_indexes.find(rgb_label);
            if (i == instance_indexes.end())
            {
                // Encountered a new instance
                instance_indexes[rgb_label] = mmod_rects.size();
                mmod_rects.emplace_back(dlib::rectangle(c, r, c, r));

                // TODO: read the instance's class from the other png!
            }
            else
            {
                // Not the first occurrence - update the rect
                auto& rect = mmod_rects[i->second].rect;

                if (c < rect.left())
                    rect.set_left(c);
                else if (c > rect.right())
                    rect.set_right(c);

                if (r > rect.bottom())
                    rect.set_bottom(r);
            }
        }
    }

    return std::vector<dlib::mmod_rect>(mmod_rects.begin(), mmod_rects.end());
}

// ----------------------------------------------------------------------------------------

det_bnet_type train_detection_network(
    const std::vector<image_info>& listing,
    const std::vector<std::vector<mmod_rect>>& mmod_rects,
    unsigned int det_minibatch_size
)
{
    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    mmod_options options(mmod_rects, 70, 30);
    det_bnet_type det_net(options);

    det_net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    dlib::pipe<det_training_sample> data(200);
    auto f = [&data, &listing, &mmod_rects](time_t seed)
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
            cropper(input_image, mmod_rects[random_index], temp.input_image, temp.mmod_rects);

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

rgb_pixel decide_main_instance(const matrix<rgb_pixel>& rgb_label_image)
{
    // TODO: should perhaps consider other aspects as well - not just the center pixel?
    const auto nr = rgb_label_image.nr();
    const auto nc = rgb_label_image.nc();
    return rgb_label_image(nr / 2, nc / 2);
}

matrix<uint16_t> keep_only_main_instance(const matrix<rgb_pixel>& rgb_label_image)
{
    matrix<uint16_t> result;

    const auto main_instance = decide_main_instance(rgb_label_image);

    const auto nr = rgb_label_image.nr();
    const auto nc = rgb_label_image.nc();
    for (long r = 0; r < nr; ++r)
    {
        for (long c = 0; c < nc; ++c)
        {
            const auto& index = rgb_label_image(r, c);
            result(r, c) = (index == main_instance) ? 1 : 0;
        }
    }

    return result;
}

seg_bnet_type train_segmentation_network(
    const std::vector<image_info>& listing,
    const std::vector<std::vector<mmod_rect>>& mmod_rects,
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
    auto f = [&data, &listing, &mmod_rects](time_t seed)
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

            // Get the MMOD rects from the label image.
            const auto image_rects = mmod_rects[random_index];
            
            if (!image_rects.empty())
            {
                const image_info& image_info = listing[random_index];

                // Load the input image.
                load_image(input_image, image_info.image_filename);

                // Load the ground-truth (RGB) labels.
                load_image(rgb_label_image, image_info.label_filename);

                // Pick a random training instance.
                const auto& mmod_rect = image_rects[rnd.get_random_32bit_number() % image_rects.size()];
                const chip_details chip_details(mmod_rect.rect, chip_dims(seg_dim, seg_dim));

                // TODO: expand the rect by 20% or so (by a hard-coded value that will be available for inference as well)

                // Crop the input image.
                extract_image_chip(input_image, chip_details, temp.input_image, interpolate_bilinear());

                // Crop the labels correspondingly. However, note that here bilinear
                // interpolation would make absolutely no sense - you wouldn't say that
                // a bicycle is half-way between an aeroplane and a bird, would you?
                extract_image_chip(rgb_label_image, chip_details, rgb_label_chip, interpolate_nearest_neighbor());

                // Clear pixels not related to the main instance.
                temp.label_image = keep_only_main_instance(rgb_label_chip);

                // TODO: Add some perturbation to the inputs.

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
    std::vector<mmod_rect>& boxes,
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
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i].ignore)
            continue;
        for (size_t j = i+1; j < boxes.size(); ++j)
        {
            if (boxes[j].ignore)
                continue;
            if (overlaps(boxes[i], boxes[j]))
            {
                ++num_ignored;
                if(boxes[i].rect.area() < boxes[j].rect.area())
                    boxes[i].ignore = true;
                else
                    boxes[j].ignore = true;
            }
        }
    }
    return num_ignored;
}

std::vector<mmod_rect> load_mmod_rects(const image_info& image_info)
{
    matrix<rgb_pixel> rgb_label_image;

    load_image(rgb_label_image, image_info.label_filename);

    auto mmod_rects = rgb_label_image_to_mmod_rects(rgb_label_image);

    ignore_overlapped_boxes(mmod_rects, test_box_overlap(0.50, 0.95));

    for (auto& rect : mmod_rects)
        if (rect.rect.width() < 35 && rect.rect.height() < 35)
            rect.ignore = true;

    return mmod_rects;
};

std::vector<std::vector<dlib::mmod_rect>> load_all_mmod_rects(const std::vector<image_info>& listing)
{
    std::vector<std::vector<mmod_rect>> mmod_rects(listing.size());

    std::transform(
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
        std::execution::par,
#endif // __cplusplus >= 201703L
        listing.begin(),
        listing.end(),
        mmod_rects.begin(),
        load_mmod_rects
    );

    return mmod_rects;
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
    const unsigned int det_minibatch_size = argc >= 3 ? std::stoi(argv[2]) : 75;
    const unsigned int seg_minibatch_size = argc >= 4 ? std::stoi(argv[3]) : 25;
    cout << "det mini-batch size: " << det_minibatch_size << endl;
    cout << "seg mini-batch size: " << seg_minibatch_size << endl;

    // extract the MMOD rects
    cout << "\nExtracting all MMOD rects...";
    const auto mmod_rects = load_all_mmod_rects(listing);
    cout << " Done!" << endl << endl;

    // First train a detection network (loss_mmod), and then a mask segmentation network (loss_log_per_pixel)
    const auto det_net = train_detection_network    (listing, mmod_rects, det_minibatch_size);
    const auto seg_net = train_segmentation_network (listing, mmod_rects, seg_minibatch_size);

    cout << "saving network" << endl;
    serialize(instance_segmentation_net_filename) << det_net << seg_net;
}

catch(std::exception& e)
{
    cout << e.what() << endl;
}

