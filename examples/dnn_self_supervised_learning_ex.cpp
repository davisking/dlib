// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  I'm assuming you have already read the dnn_introduction_ex.cpp, the
    dnn_introduction2_ex.cpp and the dnn_introduction3_ex.cpp examples.  In this example
    program we are going to show how one can train a neural network using an unsupervised
    loss function.  In particular, we will train a DenseNet with 100 layers and growth
    rate k = 12 on the CIFAR-10 dataset.  This network was introduced in this paper:
    "Densely Connected Convolutional Networks" by Gao Huang, Zhuang Liu,
    Laurens van der Maaten, Kilian Q. Weinberger.
    According to the paper, this model is able to achive around 95% accuracy on CIFAR-10.

    To train the unsupervised loss, we will use the self-supervised learning (SSL) method
    called Barlow Twins, introduced in this paper:
    "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" by Jure Zbontar,
    Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny.

    The paper contains a good explanation on how and why this works, but the main idea
    behind the Barlow Twins method is:
        - generate two distorted view of a batch of images: YA, YB
        - feed them to a deep neural network and obtain their representations and
          and batch normalize them: ZA, ZB
        - compute the empirical cross-correlation matrix between both feature
          representations as: C = trans(ZA) * ZB.
        - make C as close as possible to the identity matrix.

    This removes the redundancy of the feature representations, by maximizing the
    encoded information about the images themselves, while minimizing the information
    about the transforms and data augmentations used.

    The original Barlow Twins paper uses the ImageNet dataset, but in this example we
    are using CIFAR-10, so we will follow the recommendations of this paper, instead:
    "A Note on Connecting Barlow Twins with Negative-Sample-Free Contrastive Learning"
    by Yao-Hung Hubert Tsai, Shaojie Bai, Louis-Philippe Morency, Ruslan Salakhutdinov,
    in which they experiment with Barlow Twins on CIFAR-10 and Tiny ImageNet.
*/
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/gui_widgets.h>

#include "resnet.h"

using namespace std;
using namespace dlib;

rectangle make_random_cropping_rect(const matrix<rgb_pixel>& image, dlib::rand& rnd)
{
    const double mins = 7. / 15.;
    const double maxs = 7. / 8.;
    const auto scale = rnd.get_double_in_range(mins, maxs);
    const auto size = scale * std::min(image.nr(), image.nc());
    const rectangle rect(size, size);
    const point offset(rnd.get_random_32bit_number() % (image.nc() - rect.width()),
                       rnd.get_random_32bit_number() % (image.nr() - rect.height()));
    return move_rect(rect, offset);
}

void augment(
    const matrix<rgb_pixel>& image,
    matrix<rgb_pixel>& crop,
    const unsigned long size,
    dlib::rand& rnd
)
{
    // randomly crop
    const auto rect = make_random_cropping_rect(image, rnd);
    extract_image_chip(image, chip_details(rect, chip_dims(size, size)), crop);

    // image left-right flip
    if (rnd.get_random_double() < 0.5)
        flip_image_left_right(crop);

    // color augmentation
    if (rnd.get_random_double() < 0.8)
        disturb_colors(crop, rnd);

    // grayscale
    if (rnd.get_random_double() < 0.2)
    {
        matrix<unsigned char> gray;
        assign_image(gray, crop);
        assign_image(crop, gray);
    }
}

// This model definition contains the definitions for:
// - SSL model using the Barlow Twins loss, a projector head and an input_rgb_image_pair.
// - Classifier model using the loss_multiclass_log, a fc layer and an input_rgb_image.
namespace model
{
    template <typename SUBNET>
    using projector = fc<128, relu<bn_fc<fc<512, SUBNET>>>>;

    template <typename SUBNET>
    using classifier = fc<10, SUBNET>;

    template <
        template <typename> class LOSS,
        template <typename> class MLP,
        template <typename> class BN,
        typename INPUT>
    using net_type = LOSS<MLP<avg_pool_everything<
        typename resnet::def<BN>::template backbone_50<
        INPUT>>>>;

    using train = net_type<loss_barlow_twins, projector, bn_con, input_rgb_image_pair>;
    using infer = net_type<loss_multiclass_log, classifier, affine, input_rgb_image>;
}


int main(const int argc, const char** argv)
try
{
    // The default settings are fine for the example already.
    command_line_parser parser;
    parser.add_option("batch", "set the mini batch size (default: 64)", 1);
    parser.add_option("dims", "set the projector dimensions (default: 128)", 1);
    parser.add_option("lambda", "penalize off-diagonal terms (default: 1/dims)", 1);
    parser.add_option("learning-rate", "set the initial learning rate (default: 1e-3)", 1);
    parser.add_option("min-learning-rate", "set the min learning rate (default: 1e-6)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() < 1 || parser.option("h") || parser.option("help"))
    {
        cout << "This example needs the CIFAR-10 dataset to run." << endl;
        cout << "You can get CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html" << endl;
        cout << "Download the binary version the dataset, decompress it, and put the 6" << endl;
        cout << "bin files in a folder.  Then give that folder as input to this program." << endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const size_t batch_size = get_option(parser, "batch", 64);
    const long dims = get_option(parser, "dims", 128);
    const double lambda = get_option(parser, "lambda", 1.0 / dims);
    const double learning_rate = get_option(parser, "learning-rate", 1e-3);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-6);

    // Load the CIFAR-10 dataset into memory.
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;
    load_cifar_10_dataset(parser[0], training_images, training_labels, testing_images, testing_labels);
    dlib::rand rnd;

    // Initialize the model with the specified projector dimensions and lambda.  According to the
    // third paper, lambda = 1/dims works well on CIFAR-10.
    model::train net((loss_barlow_twins_(lambda)));
    layer<1>(net).layer_details().set_num_outputs(dims);
    disable_duplicative_biases(net);

    // Train the feature extractor using the Barlow Twins method
    {
        dnn_trainer<model::train, adam> trainer(net, adam(1e-6, 0.9, 0.999));
        trainer.set_mini_batch_size(batch_size);
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.set_iterations_without_progress_threshold(10000);
        trainer.set_synchronization_file("barlow_twins_sync");
        trainer.be_verbose();
        cout << trainer << endl;

        image_window win;
        // During the training, we will compute the empirical cross-correlation matrix
        // between the features of both versions of the augmented images.  This matrix
        // should be getting close to the identity matrix as the training progresses.
        // Note that this step is already done in the loss layer, and it's not necessary
        // to do it here for the example to work.  However, it provides a nice
        // visualization of the training progress: the closer to the identity matrix,
        // the better.
        resizable_tensor eccm;
        eccm.set_size(dims, dims);
        // Some tensors needed to perform batch normalization
        resizable_tensor za_norm, zb_norm, means, invstds, rms, rvs, gamma, beta;
        alias_tensor split(batch_size, dims);
        const double eps = DEFAULT_BATCH_NORM_EPS;
        gamma.set_size(1, dims);
        beta.set_size(1, dims);

        std::vector<std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch;
        while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        {
            batch.clear();
            while (batch.size() < trainer.get_mini_batch_size())
            {
                const auto idx = rnd.get_random_64bit_number() % training_images.size();
                auto image = training_images[idx];
                matrix<rgb_pixel> aug_a, aug_b;
                augment(image, aug_a, 64, rnd);
                augment(image, aug_b, 64, rnd);
                batch.emplace_back(aug_a, aug_b);
                // win.set_image(join_rows(crop_a, crop_b));
                // cin.get();
            }
            trainer.train_one_step(batch);

            // Compute the empirical cross-correlation matrix every 100 steps. Again,
            // this is not needed for the training to work, but it's nice to visualize.
            if (trainer.get_train_one_step_calls() % 100 == 0)
            {
                // Wait for threaded processing to stop in the trainer.
                trainer.get_net(force_flush_to_disk::no);
                // Get the output from the last fc layer
                const auto& out = net.subnet().get_output();
                // The trainer might have syncronized its state to the disk and cleaned
                // the network state. If that happens, the output will be empty, in
                // which case, we just skip the empirical cross-correlation matrix
                // computation.
                if (out.size() == 0)
                    continue;
                // Separate both augmented versions of the images
                auto za = split(out);
                auto zb = split(out, split.size());
                gamma = 1;
                beta = 0;
                // Perform batch normalization on each feature representation, independently.
                tt::batch_normalize(eps, za_norm, means, invstds, 1, rms, rvs, za, gamma, beta);
                tt::batch_normalize(eps, zb_norm, means, invstds, 1, rms, rvs, za, gamma, beta);
                // Compute the empirical cross-correlation matrix between the features and
                // visualize it.
                tt::gemm(0, eccm, 1, za_norm, true, zb_norm, false);
                eccm /= batch_size;
                matrix<unsigned char> c_img;
                assign_image(c_img, mat(eccm) * 255);
                win.set_image(c_img);
                win.set_title("Barlow Twins step#: " + to_string(trainer.get_train_one_step_calls()));
            }
        }
        trainer.get_net();
        net.clean();
        serialize("barlow_twins.net") << net;
    }

    // Hopefully, the model has now learned some useful representations, which we can
    // use to learn a classifier on top of them.  We will now build the classifier model
    // and freeze the weights, so that we only train the top-most fc layer.
    model::infer inet;
    // Assign the network, without the projector, which is only used for the self-supervised
    // training.
    layer<2>(inet).subnet() = layer<5>(net).subnet();
    set_all_learning_rate_multipliers(layer<2>(inet), 0);
    cout << inet << endl;
    layer<1>(inet).layer_details().set_num_outputs(10);
    // Train the network
    {
        dnn_trainer<model::infer> trainer(inet);
        trainer.set_learning_rate(0.1);
        trainer.set_min_learning_rate(1e-4);
        // Since this model doesn't train with pairs, just single images, we can increase the
        // batch-size by a factor of 2.
        trainer.set_mini_batch_size(2 * batch_size);
        trainer.set_iterations_without_progress_threshold(5000);
        trainer.set_synchronization_file("cifar_10_sync");
        trainer.be_verbose();
        cout << trainer << endl;

        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        {
            images.clear();
            labels.clear();
            while (images.size() < trainer.get_mini_batch_size())
            {
                const auto idx = rnd.get_random_64bit_number() % training_images.size();
                auto image = training_images[idx];
                matrix<rgb_pixel> crop;
                augment(image, crop, 64, rnd);
                images.push_back(std::move(crop));
                labels.push_back(training_labels[idx]);
            }
            trainer.train_one_step(images, labels);
        }
        trainer.get_net();
        inet.clean();
        serialize("resnet50.dnn") << inet;
    }

    // Finally, we can compute the accuracy of the model on the CIFAR-10 train and test images.
    // The model used in this example (DenseNet-100, k=12) can achieve an accuracy of over 90%
    // on CIFAR-10, so we should expect something similar.
    auto compute_accuracy = [&inet](const std::vector<matrix<rgb_pixel>>& images, const std::vector<unsigned long>& labels)
    {
        size_t num_right = 0;
        size_t num_wrong = 0;
        for (size_t i = 0; i < images.size(); ++i)
        {
            matrix<rgb_pixel> image(64, 64);
            resize_image(images[i], image);
            const auto pred = inet(image);
            if (labels[i] == pred)
                ++num_right;
            else
                ++num_wrong;
        }
        cout << "num right:  " << num_right << endl;
        cout << "num wrong:  " << num_wrong << endl;
        cout << "accuracy:   " << num_right / static_cast<double>(num_right + num_wrong) << endl;
        cout << "error rate: " << num_wrong / static_cast<double>(num_right + num_wrong) << endl;
    };

    cout << "training accuracy" << endl;
    compute_accuracy(training_images, training_labels);
    cout << "\ntesting accuracy" << endl;
    compute_accuracy(testing_images, testing_labels);
    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
