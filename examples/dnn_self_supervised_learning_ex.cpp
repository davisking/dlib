// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  I'm assuming you have already read the dnn_introduction_ex.cpp, the
    dnn_introduction2_ex.cpp and the dnn_introduction3_ex.cpp examples.  In this
    example program we are going to show how one can train a neural network using an
    unsupervised loss function.  In particular, we will train the ResNet50 model from
    the paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu
    Zhang, Shaoqing Ren, Jian Sun.

    To train the unsupervised loss, we will use the self-supervised learning (SSL)
    method called Barlow Twins, introduced in this paper:
    "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" by Jure Zbontar,
    Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny.

    The paper contains a good explanation on how and why this works, but the main idea
    behind the Barlow Twins method is:
        - generate two distorted views of a batch of images: YA, YB
        - feed them to a deep neural network and obtain their representations and
          and batch normalize them: ZA, ZB
        - compute the empirical cross-correlation matrix between both feature
          representations as: C = trans(ZA) * ZB.
        - make C as close as possible to the identity matrix.

    This removes the redundancy of the feature representations by maximizing the
    encoded information about the images themselves, while minimizing the information
    about the transforms and data augmentations used to obtain the representations.

    The original Barlow Twins paper uses the ImageNet dataset, but in this example we
    are using CIFAR-10, so we will follow the recommendations of this paper, instead:
    "A Note on Connecting Barlow Twins with Negative-Sample-Free Contrastive Learning"
    by Yao-Hung Hubert Tsai, Shaojie Bai, Louis-Philippe Morency, Ruslan Salakhutdinov,
    in which they experiment with Barlow Twins on CIFAR-10 and Tiny ImageNet.  Since the
    CIFAR-10 contains relatively small images, we will define a ResNet50 architecture
    that doesn't downsample the input in the first convolutional layer, and doesn't
    have a max pooling layer afterwards, like the paper does.

    This example shows how to use the Barlow Twins loss for this common scenario:
    Let's imagine that we have collected an image data set but we don't have enough
    resources to label it all, just a small fraction of it.  We can use the Barlow
    Twins loss on all the available training data (both labeled and unlabeled images)
    to train a feature extractor and learn meaningful representations for the data set.
    Once the feature extractor is trained, we proceed to train a linear multiclass
    SVM classifier on top of it using only the fraction of labeled data.
*/

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;

// A custom definition of ResNet50 with a downsampling factor of 8 instead of 32.
// It is essentially the original ResNet50, but without the max pooling and a
// convolutional layer with a stride of 1 instead of 2 at the input.
namespace resnet50
{
    using namespace dlib;
    template <template <typename> class BN>
    struct def
    {
        template <long N, int K, int S, typename SUBNET>
        using conv = add_layer<con_<N, K, K, S, S, K / 2, K / 2>, SUBNET>;

        template<long N, int S, typename SUBNET>
        using bottleneck = BN<conv<4 * N, 1, 1, relu<BN<conv<N, 3, S, relu<BN<conv<N, 1, 1, SUBNET>>>>>>>>;

        template <long N,  typename SUBNET>
        using residual = add_prev1<bottleneck<N, 1, tag1<SUBNET>>>;

        template <typename SUBNET> using res_512 = relu<residual<512, SUBNET>>;
        template <typename SUBNET> using res_256 = relu<residual<256, SUBNET>>;
        template <typename SUBNET> using res_128 = relu<residual<128, SUBNET>>;
        template <typename SUBNET> using res_64  = relu<residual<64, SUBNET>>;

        template <long N, int S, typename SUBNET>
        using transition = add_prev2<BN<conv<4 * N, 1, S, skip1<tag2<bottleneck<N, S, tag1<SUBNET>>>>>>>;

        template <typename INPUT>
        using backbone = avg_pool_everything<
            repeat<2, res_512, transition<512, 2,
            repeat<5, res_256, transition<256, 2,
            repeat<3, res_128, transition<128, 2,
            repeat<2, res_64,  transition<64, 1,
            relu<BN<conv<64, 3, 1,INPUT>>>>>>>>>>>>;
    };
}

// This model namespace contains the definitions for:
// - SSL model with the Barlow Twins loss, a projector head and an input_rgb_image_pair.
// - Feature extractor with the loss_metric (to get the outputs) and an input_rgb_image.
namespace model
{
    template <typename SUBNET> using projector = fc<128, relu<bn_fc<fc<512, SUBNET>>>>;
    using train = loss_barlow_twins<projector<resnet50::def<bn_con>::backbone<input_rgb_image_pair>>>;
    using feats = loss_metric<resnet50::def<affine>::backbone<input_rgb_image>>;
}

rectangle make_random_cropping_rect(
    const matrix<rgb_pixel>& image,
    dlib::rand& rnd
)
{
    const auto scale = rnd.get_double_in_range(0.5, 1.0);
    const auto ratio = exp(rnd.get_double_in_range(log(3.0 / 4.0), log(4.0 / 3.0)));
    const auto size = scale * min(image.nr(), image.nc());
    const auto rect = move_rect(set_aspect_ratio(rectangle(size, size), ratio), 0, 0);
    const point offset(rnd.get_integer(max(0, static_cast<int>(image.nc() - rect.width()))),
                       rnd.get_integer(max(0, static_cast<int>(image.nr() - rect.height()))));
    return move_rect(rect, offset);
}

// A helper function to generate different kinds of augmentations.
matrix<rgb_pixel> augment(
    const matrix<rgb_pixel>& image,
    dlib::rand& rnd
)
{
    // randomly crop
    matrix<rgb_pixel> crop;
    const auto rect = make_random_cropping_rect(image, rnd);
    extract_image_chip(image, chip_details(rect, chip_dims(32, 32)), crop);

    // image left-right flip
    if (rnd.get_random_double() < 0.5)
        flip_image_left_right(crop);

    // color augmentation
    if (rnd.get_random_double() < 0.8)
        disturb_colors(crop, rnd, 1.0, 0.5);

    // grayscale
    if (rnd.get_random_double() < 0.2)
    {
        matrix<unsigned char> gray;
        assign_image(gray, crop);
        assign_image(crop, gray);
    }
    return crop;
}

int main(const int argc, const char** argv)
try
{
    // The default settings are fine for the example already.
    command_line_parser parser;
    parser.add_option("batch", "set the mini batch size per GPU (default: 64)", 1);
    parser.add_option("dims", "set the projector dimensions (default: 128)", 1);
    parser.add_option("lambda", "off-diagonal terms penalty (default: 1/dims)", 1);
    parser.add_option("learning-rate", "set the initial learning rate (default: 1e-3)", 1);
    parser.add_option("min-learning-rate", "set the min learning rate (default: 1e-5)", 1);
    parser.add_option("num-gpus", "number of GPUs (default: 1)", 1);
    parser.add_option("fraction", "fraction of labels to use (default: 0.1)", 1);
    parser.add_option("patience", "steps without progress threshold (default: 10000)", 1);
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

    parser.check_option_arg_range("fraction", 0.0, 1.0);
    const double labels_fraction = get_option(parser, "fraction", 0.1);
    const size_t num_gpus = get_option(parser, "num-gpus", 1);
    const size_t batch_size = get_option(parser, "batch", 64) * num_gpus;
    const long dims = get_option(parser, "dims", 128);
    const double lambda = get_option(parser, "lambda", 1.0 / dims);
    const double learning_rate = get_option(parser, "learning-rate", 1e-3);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-5);
    const size_t patience = get_option(parser, "patience", 10000);

    // Load the CIFAR-10 dataset into memory.
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;
    load_cifar_10_dataset(parser[0], training_images, training_labels, testing_images, testing_labels);

    // Initialize the model with the specified projector dimensions and lambda.
    // According to the second paper, lambda = 1/dims works well on CIFAR-10.
    model::train net((loss_barlow_twins_(lambda)));
    layer<1>(net).layer_details().set_num_outputs(dims);
    disable_duplicative_biases(net);
    dlib::rand rnd;
    std::vector<int> gpus(num_gpus);
    iota(gpus.begin(), gpus.end(), 0);

    // Train the feature extractor using the Barlow Twins method on all the training
    // data.
    {
        dnn_trainer<model::train, adam> trainer(net, adam(1e-6, 0.9, 0.999), gpus);
        trainer.set_mini_batch_size(batch_size);
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.set_iterations_without_progress_threshold(patience);
        trainer.set_synchronization_file("barlow_twins_sync");
        trainer.be_verbose();
        cout << trainer << endl;

        // During the training, we will visualize the empirical cross-correlation
        // matrix between the features of both versions of the augmented images.
        // This matrix should be getting close to the identity matrix as the training
        // progresses.  Note that this is done here for visualization purposes only.
        image_window win;

        std::vector<pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch;
        while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        {
            batch.clear();
            while (batch.size() < trainer.get_mini_batch_size())
            {
                const auto idx = rnd.get_random_32bit_number() % training_images.size();
                const auto& image = training_images[idx];
                batch.emplace_back(augment(image, rnd), augment(image, rnd));
            }
            trainer.train_one_step(batch);

            // Get the empirical cross-correlation matrix every 100 steps. Again,
            // this is not needed for the training to work, but it's nice to visualize.
            if (trainer.get_train_one_step_calls() % 100 == 0)
            {
                // Wait for threaded processing to stop in the trainer.
                trainer.get_net(force_flush_to_disk::no);
                const matrix<float> eccm = mat(net.loss_details().get_eccm());
                win.set_image(round(abs(mat(eccm)) * 255));
                win.set_title("Barlow Twins step#: " + to_string(trainer.get_train_one_step_calls()));
            }
        }
        trainer.get_net();
        net.clean();
        // After training, we can discard the projector head and just keep the backone
        // to train it or finetune it on other downstream tasks.
        serialize("resnet50_self_supervised_cifar_10.net") << layer<5>(net);
    }

    // Now, we initialize the feature extractor with the backbone we have just learned.
    model::feats fnet(layer<5>(net));

    // Use only the specified fraction of training labels
    if (labels_fraction < 1.0)
    {
        randomize_samples(training_images, training_labels);
        std::vector<matrix<rgb_pixel>> sub_images(
            training_images.begin(),
            training_images.begin() + lround(training_images.size() * labels_fraction));

        std::vector<unsigned long> sub_labels(
            training_labels.begin(),
            training_labels.begin() + lround(training_labels.size() * labels_fraction));

        swap(sub_images, training_images);
        swap(sub_labels, training_labels);
    }

    // Let's generate the features for those samples that have labels to train a
    // multiclass SVM classifier.
    std::vector<matrix<float, 0, 1>> features;
    cout << "Extracting features for linear classifier from " << training_images.size() << " samples..." << endl;
    features = fnet(training_images, 4 * batch_size);

    const auto df = auto_train_multiclass_svm_linear_classifier(features, training_labels, chrono::minutes(1));
    serialize("multiclass_svm_cifar_10.dat") << df;

    // Finally, we can compute the accuracy of the model on the CIFAR-10 train and
    // test images.
    const auto compute_accuracy = [&df](
        const std::vector<matrix<float, 0, 1>>& samples,
        const std::vector<unsigned long>& labels
    )
    {
        size_t num_right = 0;
        size_t num_wrong = 0;
        for (size_t i = 0; i < labels.size(); ++i)
        {
            if (labels[i] == df(samples[i]))
                ++num_right;
            else
                ++num_wrong;
        }
        cout << "  num right:  " << num_right << endl;
        cout << "  num wrong:  " << num_wrong << endl;
        cout << "  accuracy:   " << num_right / static_cast<double>(num_right + num_wrong) << endl;
        cout << "  error rate: " << num_wrong / static_cast<double>(num_right + num_wrong) << endl;
    };

    // Using 10% of the training labels should result in a testing accuracy of
    // around 88%.  Had we used all labels to train the multiclass SVM classifier,
    // we would have got a testing accuracy of around 90%, instead.
    cout << "\ntraining accuracy" << endl;
    compute_accuracy(features, training_labels);
    cout << "\ntesting accuracy" << endl;
    features = fnet(testing_images, 4 * batch_size);
    compute_accuracy(features, testing_labels);
    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
