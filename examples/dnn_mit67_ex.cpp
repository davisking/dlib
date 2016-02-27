


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/svm.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <iterator>

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

template <typename T> using ares = relu<affine<add_prev1<con<relu<affine<con<tag1<T>>>>>>>>;

template <typename T> using res = relu<bn<add_prev1<con<relu<bn<con<tag1<T>>>>>>>>;
std::tuple<relu_,bn_,add_prev1_,con_,relu_,bn_,con_> res_ (
    unsigned long outputs,
    unsigned long stride = 1
) 
{
    return std::make_tuple(relu_(),
                           bn_(CONV_MODE),
                           add_prev1_(),
                           con_(outputs,3,3,stride,stride),
                           relu_(),
                           bn_(CONV_MODE),
                           con_(outputs,3,3,stride,stride));
}

// ----------------------------------------------------------------------------------------

void randomly_crop_image (
    const matrix<rgb_pixel>& img,
    matrix<rgb_pixel>& crop,
    dlib::rand& rnd
)
{
    // figure out what rectangle we want to crop from the image
    //auto scale = 1-rnd.get_random_double()*0.2;
    double mins = 0.466666666, maxs = 0.875;
    auto scale = mins + rnd.get_random_double()*(maxs-mins);
    auto size = scale*std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    rect = move_rect(rect, offset);

    // now crop it out as a 224x224 image.
    extract_image_chip(img, chip_details(rect, chip_dims(224,224)), crop);

    // Also randomly flip the image
    if (rnd.get_random_double() > 0.5)
        crop = fliplr(crop);

    // And then randomly adjust the color balance and gamma.
    disturb_colors(crop, rnd);
}

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
        // figure out what rectangle we want to crop from the image
        //auto scale = 1-rnd.get_random_double()*0.2;
        double mins = 0.466666666, maxs = 0.875;
        auto scale = mins + rnd.get_random_double()*(maxs-mins);
        auto size = scale*std::min(img.nr(), img.nc());
        rectangle rect(size, size);
        // randomly shift the box around
        point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
            rnd.get_random_32bit_number()%(img.nr()-rect.height()));
        rect = move_rect(rect, offset);

        dets.push_back(chip_details(rect, chip_dims(224,224)));
    }

    extract_image_chips(img, dets, crops);

    for (auto&& img : crops)
    {
        // Also randomly flip the image
        if (rnd.get_random_double() > 0.5)
            img = fliplr(img);

        // And then randomly adjust the color balance and gamma.
        disturb_colors(img, rnd);
    }
}

// ----------------------------------------------------------------------------------------

struct image_info
{
    string filename;
    string label;
    unsigned long numeric_label;
};

std::vector<image_info> get_imagenet_listing(
    const std::string& images_folder
)
{
    std::vector<image_info> results;
    image_info temp;
    temp.numeric_label = 0;
    // loop over all the scene types in the dataset, each is contained in a subfolder.
    auto subdirs = directory(images_folder).get_dirs();
    // sort the sub directories so the numeric labels will be assigned in sorted order.
    std::sort(subdirs.begin(), subdirs.end());
    for (auto subdir : subdirs)
    {
        // Now get all the images in this scene type
        temp.label = subdir.name();
        for (auto image_file : subdir.get_files())
        {
            temp.filename = image_file;
            results.push_back(temp);
        }
        ++temp.numeric_label;
    }
    return results;
}

unsigned long vote (
    const std::vector<unsigned long>& votes
)
{
    std::vector<unsigned long> counts(max(mat(votes))+1);
    for (auto i : votes)
        counts[i]++;
    return index_of_max(mat(counts));
}

int main(int argc, char** argv) try
{
    if (argc != 3)
    {
        cout << "give MIT 67 scene folder as input and a weight decay value!" << endl;
        return 1;
    }

    auto listing = get_imagenet_listing(argv[1]);
    cout << "images in dataset: " << listing.size() << endl;
    const auto number_of_classes = listing.back().numeric_label+1;
    if (listing.size() == 0 || number_of_classes != 1000)
    {
        cout << "Didn't find the MIT 67 scene dataset.  Are you sure you gave the correct folder?" << endl;
        cout << "Give the Images folder as an argument to this program." << endl;
        return 1;
    }
        

    const double initial_step_size = 0.1;
    const double weight_decay = sa = argv[2];

    typedef loss_multiclass_log<fc<avg_pool<
                                res<res<res<
                                res<res<res<res<res<res<
                                res<res<res<res<
                                res<res<res<
                                max_pool<relu<bn<con<
                                input<matrix<rgb_pixel>
                                >>>>>>>>>>>>>>>>>>>>>>>> net_type;


    net_type net(fc_(number_of_classes),
                 avg_pool_(1000,1000,1000,1000),
                 res_(512),res_(512),res_(512,2),
                 res_(256),res_(256),res_(256),res_(256),res_(256),res_(256,2),
                 res_(128),res_(128),res_(128),res_(128,2),
                 res_(64), res_(64), res_(64),
                 max_pool_(3,3,2,2), relu_(), bn_(CONV_MODE), con_(64,7,7,2,2)
                );


    cout << "initial step size: "<< initial_step_size << endl;
    cout << "weight decay: " << weight_decay << endl;

    dnn_trainer<net_type> trainer(net,sgd(initial_step_size, weight_decay));
    trainer.be_verbose();
    trainer.set_synchronization_file("sync_imagenet_full_training_set_40000_minstep_"+cast_to_string(weight_decay), std::chrono::minutes(5));
    trainer.set_iterations_between_step_size_adjust(40000);
    std::vector<matrix<rgb_pixel>> samples;
    std::vector<unsigned long> labels;

    randomize_samples(listing);
    const size_t training_part = listing.size()*1.0;

    dlib::rand rnd;


    const bool do_training = true;
    if (do_training)
    {
        while(trainer.get_step_size() >= 1e-3)
        {
            samples.clear();
            labels.clear();

            // make a 128 image mini-batch
            matrix<rgb_pixel> img, crop;
            while(samples.size() < 128)
            {
                auto l = listing[rnd.get_random_32bit_number()%training_part];
                load_image(img, l.filename);
                randomly_crop_image(img, crop, rnd);
                samples.push_back(crop);
                labels.push_back(l.numeric_label);
            }

            trainer.train_one_step(samples, labels);
        }

        // wait for threaded processing to stop.
        trainer.get_net();

        net.clean();
        cout << "saving network" << endl;
        serialize("imagenet_full_training_set_40000_minstep_"+cast_to_string(weight_decay)+".dat") << net;
    }


    const bool test_network = false;
    if (test_network)
    {

        typedef loss_multiclass_log<fc<avg_pool<
            ares<ares<ares<
            ares<ares<ares<ares<ares<ares<
            ares<ares<ares<ares<
            ares<ares<ares<
            max_pool<relu<affine<con<
            input<matrix<rgb_pixel>
            >>>>>>>>>>>>>>>>>>>>>>>> anet_type;
    
        anet_type net;
        deserialize("imagenet_network3_"+cast_to_string(weight_decay)+".dat") >> net;

        dlib::array<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        matrix<rgb_pixel> img, crop;
        cout << "loading images..." << endl;
        int num_right = 0;
        int num_wrong = 0;
        console_progress_indicator pbar(training_part);
        /*
        for (size_t i = 0; i < training_part; ++i)
        {
            pbar.print_status(i);
            load_image(img, listing[i].filename);

            randomly_crop_images(img, images, rnd, 16);
            unsigned long predicted_label = vote(net(images, 32));
            if (predicted_label == listing[i].numeric_label)
                ++num_right;
            else
                ++num_wrong;
        }
        */

        cout << "\ntraining num_right: " << num_right << endl;
        cout << "training num_wrong: " << num_wrong << endl;
        cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

        pbar.reset(listing.size()-training_part);
        num_right = 0;
        num_wrong = 0;
        for (size_t i = training_part; i < listing.size(); ++i)
        {
            pbar.print_status(i-training_part);
            load_image(img, listing[i].filename);

            randomly_crop_images(img, images, rnd, 16);
            unsigned long predicted_label = vote(net(images, 32));
            if (predicted_label == listing[i].numeric_label)
                ++num_right;
            else
                ++num_wrong;
        }
        cout << "\ntesting num_right: " << num_right << endl;
        cout << "testing num_wrong: " << num_wrong << endl;
        cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
        return 0;
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

