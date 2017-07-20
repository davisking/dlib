// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to train a semantic segmentation net using the PASCAL VOC2012
    dataset.

    Instructions:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_semantic_segmentation_train_ex example program.
    3. Run:
       ./dnn_semantic_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_semantic_segmentation_ex example program.
    6. Run:
       ./dnn_semantic_segmentation_ex /path/to/VOC2012/JPEGImages/2007_000033.jpg
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

typedef std::pair<matrix<rgb_pixel>, matrix<uint16_t>> training_sample;

// ----------------------------------------------------------------------------------------

rectangle make_random_cropping_rect_resnet(
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
    const auto rect = make_random_cropping_rect_resnet(input_image, rnd);

    const chip_details chip_details(rect, chip_dims(227, 227));

    // Crop the input image.
    extract_image_chip(input_image, chip_details, crop.first, interpolate_bilinear());

    // Crop the labels correspondingly. However, note that here bilinear
    // interpolation would make absolutely no sense.
    extract_image_chip(label_image, chip_details, crop.second, interpolate_nearest_neighbor());

    // Also randomly flip the input image and the labels.
    if (rnd.get_random_double() > 0.5) {
        crop.first = fliplr(crop.first);
        crop.second = fliplr(crop.second);
    }

    // And then randomly adjust the colors.
    apply_random_color_offset(crop.first, rnd);
}

// ----------------------------------------------------------------------------------------

struct image_info
{
    string image_filename;
    string label_filename;
};

std::vector<image_info> get_pascal_voc2012_listing(
    const std::string& voc2012_folder,
    const std::string& file = "train" // "train", "trainval", or "val"
)
{
    std::ifstream in(voc2012_folder + "/ImageSets/Segmentation/" + file + ".txt");

    std::vector<image_info> results;

    while (in) {
        std::string basename;
        in >> basename;

        if (!basename.empty()) {
            image_info image_info;
            image_info.image_filename = voc2012_folder + "/JPEGImages/" + basename + ".jpg";
            image_info.label_filename = voc2012_folder + "/SegmentationClass/" + basename + ".png";
            results.push_back(image_info);
        }
    }

    return results;
}

std::vector<image_info> get_pascal_voc2012_train_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "train");
}

std::vector<image_info> get_pascal_voc2012_val_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "val");
}

// ----------------------------------------------------------------------------------------

const Voc2012class& find_voc2012_class(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(
        [&rgb_label](const Voc2012class& voc2012class) {
            return rgb_label == voc2012class.rgb_label;
        }
    );
}

inline uint16_t rgb_label_to_index_label(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(rgb_label).index;
}

void rgb_label_image_to_index_label_image(const dlib::matrix<dlib::rgb_pixel>& rgb_label_image, dlib::matrix<uint16_t>& index_label_image)
{
    const long nr = rgb_label_image.nr();
    const long nc = rgb_label_image.nc();

    index_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            index_label_image(r, c) = rgb_label_to_index_label(rgb_label_image(r, c));
        }
    }
}

// ----------------------------------------------------------------------------------------

double calculate_accuracy(anet_type& anet, const std::vector<image_info>& dataset)
{
    int num_right = 0;
    int num_wrong = 0;

    matrix<rgb_pixel> input_image;
    matrix<rgb_pixel> rgb_label_image;
    matrix<uint16_t> index_label_image;

    for (const auto& image_info : dataset) {
        load_image(input_image, image_info.image_filename);
        load_image(rgb_label_image, image_info.label_filename);

        matrix<uint16_t> net_output = anet(input_image);

        rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);

        const long nr = index_label_image.nr();
        const long nc = index_label_image.nc();

        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                const uint16_t truth = index_label_image(r, c);
                if (truth != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                    const uint16_t prediction = net_output(r, c);
                    if (prediction == truth) {
                        ++num_right;
                    }
                    else {
                        ++num_wrong;
                    }
                }
            }
        }
    }

    return num_right / static_cast<double>(num_right + num_wrong);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "To run this program you need a copy of the PASCAL VOC2012 dataset." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./dnn_semantic_segmentation_train_ex /path/to/VOC2012" << endl;
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
        
    set_dnn_prefer_smallest_algorithms();


    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    net_type net;
    dnn_trainer<net_type> trainer(net,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("pascal_voc2012_trainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.
    trainer.set_iterations_without_progress_threshold(20000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(net, 1000);

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
            const image_info& image_info = listing[rnd.get_random_32bit_number()%listing.size()];
            load_image(input_image, image_info.image_filename);
            load_image(rgb_label_image, image_info.label_filename);
            rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);
            randomly_crop_image(input_image, index_label_image, temp, rnd);
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-3.
    while(trainer.get_learning_rate() >= initial_learning_rate*1e-3)
    {
        samples.clear();
        labels.clear();

        // make a 50 image mini-batch
        training_sample temp;
        while(samples.size() < 50)
        {
            data.dequeue(temp);

            samples.push_back(std::move(temp.first));
            labels.push_back(std::move(temp.second));
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

    net.clean();
    cout << "saving network" << endl;
    serialize("voc2012net.dnn") << net;




    anet_type anet = net;

    cout << "Testing the network..." << endl;

    cout << "train accuracy  :  " << calculate_accuracy(anet, get_pascal_voc2012_train_listing(argv[1])) << endl;
    cout << "val accuracy    :  " << calculate_accuracy(anet, get_pascal_voc2012_val_listing(argv[1])) << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

