// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to retrain an existing model.

    Specifically we're going to first run a number of images through the lower
    layers of the net, and store the output from them to disk. Then we will
    train only some of the top layers, without having to run everything through
    the entire net.

    The ResNet34 architecture is from the paper Deep Residual Learning for Image
    Recognition by He, Zhang, Ren, and Sun.  The model file that comes with dlib
    was trained using the dnn_imagenet_train_ex.cpp program on a Titan X for
    about 2 weeks.  This pretrained model has a top5 error of 7.572% on the 2012
    imagenet validation dataset.

    For an introduction to dlib's DNN module read the dnn_introduction_ex.cpp and
    dnn_introduction2_ex.cpp example programs.

    Finally, these tools will use CUDA and cuDNN to drastically accelerate
    network training and testing.  CMake should automatically find them if they
    are installed and configure things appropriately.  If not, the program will
    still run but will be much slower to execute.
*/



#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>

using namespace std;
using namespace dlib;

// An example of a custom input layer
// This input layer is to feed an already calculated tensor directly into the
// "middle" of the net
class raw_input
{
public:
    typedef matrix<float> input_type;

    template <typename forward_iterator>
    void to_tensor (
        forward_iterator ibegin,
        forward_iterator iend,
        resizable_tensor& data
    ) const {
        data.set_size(std::distance(ibegin, iend), 512, 1, 1);

        long long pos = 0;
        for (auto i = ibegin; i != iend; ++i) {
            data.set_sample(pos++, *i);
        }
    }
};

void serialize(const raw_input &item, ostream &out)
{
    serialize("raw_input", out);
}

void deserialize(raw_input &item, istream &in)
{
    string ver;
    deserialize(ver, in);
    if (ver != "raw_input") {
        throw serialization_error("Unexpected version found while deserializing: " + ver);
    }
}


// ----------------------------------------------------------------------------------------

// This block of statements defines the normal resnet-34 network

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using level1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = ares<64,ares<64,ares<64,SUBNET>>>;

using anet_type = loss_multiclass_log<fc<1000,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;

// The number of classes we're going to retrain to
// This needs to match the number of folders with images
#define CLASSES 20

// This is the "head" of the full resnet network that we're going to retrain.
// Basically it's just a few of the top layers, and we use the custom class
// above to feed tensors straight into it
using retrain_net = loss_multiclass_log<fc<CLASSES, raw_input>>;

// This is the definition for the final new net that we're going to save
using new_net_type = loss_multiclass_log<fc<CLASSES,
      anet_type::subnet_type::subnet_type>>;

// ----------------------------------------------------------------------------------------

struct image_info
{
    string filename;
    string label;
    long numeric_label = 0l;
};

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

// We do some random crops as in the normal full training
void randomly_crop_images (
    const matrix<rgb_pixel>& img,
    dlib::array<matrix<rgb_pixel>>& crops,
    dlib::rand& rnd,
    long num_crops
)
{
    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i)
    {
        auto rect = make_random_cropping_rect_resnet(img, rnd);
        dets.push_back(chip_details(rect, chip_dims(227,227)));
    }

    extract_image_chips(img, dets, crops);

    for (auto&& img : crops)
    {
        // Also randomly flip the image
        if (rnd.get_random_double() > 0.5)
            img = fliplr(img);

        // And then randomly adjust the colors.
        apply_random_color_offset(img, rnd);
    }
}


void generate_tensors(const anet_type &orig_net, const std::vector<image_info> &files)
{
    anet_type::subnet_type::subnet_type decapped = orig_net.subnet().subnet();

    dlib::rand rnd;

    matrix<rgb_pixel> in_image;

    for (size_t i=0; i<files.size(); i++) {

        // Print some progress status
        cout << "\033[2K\r" << (i+1) << "/" << files.size() << " " << flush;

        string tensorfile = files[i].filename + ".tensors";
        if (file_exists(tensorfile) && file(tensorfile).size() > 0) {
            continue;
        }

        load_image(in_image, files[i].filename);
        dlib::array<matrix<rgb_pixel>> crops;
        randomly_crop_images(in_image, crops, rnd, 16);

        resizable_tensor t = decapped(crops.begin(), crops.end());
        matrix<float> m = mat(t);
        serialize(tensorfile) << m;
    }

    cout << "images" << endl;
}

int main(int argc, char** argv) try
{
    if (argc < 2)
    {
        cout << "Give this program a folder with images as command line argument.\n" << endl;
        cout << "You will also need a copy of the file resnet34_1000_imagenet_classifier.dnn " << endl;
        cout << "available at http://dlib.net/files/resnet34_1000_imagenet_classifier.dnn.bz2" << endl;
        cout << endl;
        return 1;
    }

    anet_type orig_net;
    std::vector<string> labels_;
    deserialize("resnet34_1000_imagenet_classifier.dnn") >> orig_net >> labels_;
    labels_.clear();

    std::vector<image_info> listing;
    image_info info;
    for (const directory &dir : directory(argv[1]).get_dirs()) {
        info.label = dir.name();
        labels_.push_back(dir.name());
        for (const file &f : dir.get_files()) {
            const string file_ending = right_substr(f.name(), ".");
            if (file_ending == "tensors") {
                continue;
            }
            info.filename = f;
            listing.push_back(info);
        }
        info.numeric_label++;
    }

    if (labels_.size() != CLASSES) {
        cerr << "Wrong number of classes (" << labels_.size() << "), expected " << CLASSES << endl;
    }

    generate_tensors(orig_net, listing);

    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    retrain_net rnet;
    dnn_trainer<retrain_net> trainer(rnet,sgd(weight_decay, momentum));

    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("retrainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.  You could likely get good results
    // with a smaller value but if you aren't in a hurry this value will surely work well.
    trainer.set_iterations_without_progress_threshold(100000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(rnet, 1000);

    dlib::pipe<std::pair<image_info,matrix<float>>> data(1000);
    auto f = [&data, &listing](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        while(data.is_enabled())
        {
            std::pair<image_info, matrix<float>> temp;
            temp.first = listing[rnd.get_random_32bit_number()%listing.size()];
            deserialize(temp.first.filename + ".tensors") >> temp.second;

            data.enqueue(std::move(temp));
        }
    };

    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-3.
    std::pair<image_info, matrix<float>> img;
    while(trainer.get_learning_rate() >= initial_learning_rate*1e-5)
    {
        std::vector<matrix<float>> samples;
        std::vector<unsigned long> labels;

        while(samples.size() < 50)
        {
            data.dequeue(img);

            samples.push_back(std::move(img.second));
            labels.push_back(img.first.numeric_label);
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

    new_net_type new_net;

    // Copy in the weights for the lower layers from the old network
    new_net.subnet().subnet() = orig_net.subnet().subnet();

    // Copy over the weights for the newly trained layers
    layer<0>(new_net).loss_details() = layer<0>(rnet).loss_details();
    layer<1>(new_net).layer_details() = layer<1>(rnet).layer_details();

    new_net.clean();

    cout << "saving network" << endl;

    serialize("retrained.dnn") << new_net << labels_;

    return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

