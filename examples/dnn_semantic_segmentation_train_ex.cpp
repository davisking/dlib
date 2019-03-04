// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to train a semantic segmentation net using the PASCAL VOC2012
    dataset.  For an introduction to what segmentation is, see the accompanying header file
    dnn_semantic_segmentation_ex.h.

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_semantic_segmentation_train_ex example program.
    3. Run:
       ./dnn_semantic_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_semantic_segmentation_ex example program.
    6. Run:
       ./dnn_semantic_segmentation_ex /path/to/VOC2012-or-other-images

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#include "dnn_semantic_segmentation_ex.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>

using namespace std;
using namespace dlib;

// A single training sample. A mini-batch comprises many of these.
struct training_sample
{
    matrix<rgb_pixel> input_image;
    matrix<uint16_t> label_image; // The ground-truth label of each pixel.
};

// ----------------------------------------------------------------------------------------

rectangle make_random_cropping_rect(
    const matrix<rgb_pixel>& img,
    dlib::rand& rnd
)
{
    // figure out what rectangle we want to crop from the image
    double mins = 0.466666666, maxs = 0.875;
    auto scale = mins + rnd.get_random_double()*(maxs-mins);
    auto size = scale*std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------

void randomly_crop_image (
    const matrix<rgb_pixel>& input_image,
    const matrix<uint16_t>& label_image,
    training_sample& crop,
    dlib::rand& rnd
)
{
    const auto rect = make_random_cropping_rect(input_image, rnd);

    const chip_details chip_details(rect, chip_dims(227, 227));

    // Crop the input image.
    extract_image_chip(input_image, chip_details, crop.input_image, interpolate_bilinear());

    // Crop the labels correspondingly. However, note that here bilinear
    // interpolation would make absolutely no sense - you wouldn't say that
    // a bicycle is half-way between an aeroplane and a bird, would you?
    extract_image_chip(label_image, chip_details, crop.label_image, interpolate_nearest_neighbor());

    // Also randomly flip the input image and the labels.
    if (rnd.get_random_double() > 0.5)
    {
        crop.input_image = fliplr(crop.input_image);
        crop.label_image = fliplr(crop.label_image);
    }

    // And then randomly adjust the colors.
    apply_random_color_offset(crop.input_image, rnd);
}

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
            image_info.label_filename = voc2012_folder + "/SegmentationClass/" + basename + ".png";
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

// The PASCAL VOC2012 dataset contains 20 ground-truth classes + background.  Each class
// is represented using an RGB color value.  We associate each class also to an index in the
// range [0, 20], used internally by the network.  To convert the ground-truth data to
// something that the network can efficiently digest, we need to be able to map the RGB
// values to the corresponding indexes.

// Given an RGB representation, find the corresponding PASCAL VOC2012 class
// (e.g., 'dog').
const Voc2012class& find_voc2012_class(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(
        [&rgb_label](const Voc2012class& voc2012class)
        {
            return rgb_label == voc2012class.rgb_label;
        }
    );
}

// Convert an RGB class label to an index in the range [0, 20].
inline uint16_t rgb_label_to_index_label(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(rgb_label).index;
}

// Convert an image containing RGB class labels to a corresponding
// image containing indexes in the range [0, 20].
void rgb_label_image_to_index_label_image(
    const dlib::matrix<dlib::rgb_pixel>& rgb_label_image,
    dlib::matrix<uint16_t>& index_label_image
)
{
    const long nr = rgb_label_image.nr();
    const long nc = rgb_label_image.nc();

    index_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r)
    {
        for (long c = 0; c < nc; ++c)
        {
            index_label_image(r, c) = rgb_label_to_index_label(rgb_label_image(r, c));
        }
    }
}

// ----------------------------------------------------------------------------------------

// Calculate the per-pixel accuracy on a dataset whose file names are supplied as a parameter.
double calculate_accuracy(anet_type& anet, const std::vector<image_info>& dataset)
{
    int num_right = 0;
    int num_wrong = 0;

    matrix<rgb_pixel> input_image;
    matrix<rgb_pixel> rgb_label_image;
    matrix<uint16_t> index_label_image;
    matrix<uint16_t> net_output;

    for (const auto& image_info : dataset)
    {
        // Load the input image.
        load_image(input_image, image_info.image_filename);

        // Load the ground-truth (RGB) labels.
        load_image(rgb_label_image, image_info.label_filename);

        // Create predictions for each pixel. At this point, the type of each prediction
        // is an index (a value between 0 and 20). Note that the net may return an image
        // that is not exactly the same size as the input.
        const matrix<uint16_t> temp = anet(input_image);

        // Convert the indexes to RGB values.
        rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);

        // Crop the net output to be exactly the same size as the input.
        const chip_details chip_details(
            centered_rect(temp.nc() / 2, temp.nr() / 2, input_image.nc(), input_image.nr()),
            chip_dims(input_image.nr(), input_image.nc())
        );
        extract_image_chip(temp, chip_details, net_output, interpolate_nearest_neighbor());

        const long nr = index_label_image.nr();
        const long nc = index_label_image.nc();

        // Compare the predicted values to the ground-truth values.
        for (long r = 0; r < nr; ++r)
        {
            for (long c = 0; c < nc; ++c)
            {
                const uint16_t truth = index_label_image(r, c);
                if (truth != dlib::loss_multiclass_log_per_pixel_::label_to_ignore)
                {
                    const uint16_t prediction = net_output(r, c);
                    if (prediction == truth)
                    {
                        ++num_right;
                    }
                    else
                    {
                        ++num_wrong;
                    }
                }
            }
        }
    }

    // Return the accuracy estimate.
    return num_right / static_cast<double>(num_right + num_wrong);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc < 2 || argc > 3)
    {
        cout << "To run this program you need a copy of the PASCAL VOC2012 dataset." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./dnn_semantic_segmentation_train_ex /path/to/VOC2012 [minibatch-size]" << endl;
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

    // a mini-batch smaller than the default can be used with GPUs having less memory
    const unsigned int minibatch_size = argc == 3 ? std::stoi(argv[2]) : 23;
    cout << "mini-batch size: " << minibatch_size << endl;

    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    bnet_type bnet;
    dnn_trainer<bnet_type> trainer(bnet,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("pascal_voc2012_trainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.
    trainer.set_iterations_without_progress_threshold(5000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(bnet, 1000);

    // Output training parameters.
    cout << endl << trainer << endl;

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<matrix<uint16_t>> labels;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<training_sample> data(200);
    auto f = [&data, &listing](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<uint16_t> index_label_image;
        training_sample temp;
        while(data.is_enabled())
        {
            // Pick a random input image.
            const image_info& image_info = listing[rnd.get_random_32bit_number()%listing.size()];

            // Load the input image.
            load_image(input_image, image_info.image_filename);

            // Load the ground-truth (RGB) labels.
            load_image(rgb_label_image, image_info.label_filename);

            // Convert the indexes to RGB values.
            rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);

            // Randomly pick a part of the image.
            randomly_crop_image(input_image, index_label_image, temp, rnd);

            // Push the result to be used by the trainer.
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-4.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        samples.clear();
        labels.clear();

        // make a mini-batch
        training_sample temp;
        while(samples.size() < minibatch_size)
        {
            data.dequeue(temp);

            samples.push_back(std::move(temp.input_image));
            labels.push_back(std::move(temp.label_image));
        }

        trainer.train_one_step(samples, labels);
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // also wait for threaded processing to stop in the trainer.
    trainer.get_net();

    bnet.clean();
    cout << "saving network" << endl;
    serialize(semantic_segmentation_net_filename) << bnet;


    // Make a copy of the network to use it for inference.
    anet_type anet = bnet;

    cout << "Testing the network..." << endl;

    // Find the accuracy of the newly trained network on both the training and the validation sets.
    cout << "train accuracy  :  " << calculate_accuracy(anet, get_pascal_voc2012_train_listing(argv[1])) << endl;
    cout << "val accuracy    :  " << calculate_accuracy(anet, get_pascal_voc2012_val_listing(argv[1])) << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

