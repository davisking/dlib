// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  I'm assuming you have already read the dnn_introduction_ex.cpp, the
    dnn_introduction2_ex.cpp and the dnn_introduction3_ex.cpp examples.  In this example
    program we are going to show how one can train a neural network using an unsupervised
    loss function.  In particular, we will train the ResNet50 model from the paper
    "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing
    Ren, Jian Sun.

    To train the unsupervised loss, we will use the self-supervised learning (SSL) method
    called Barlow Twins, introduced in this paper:
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

    This removes the redundancy of the feature representations, by maximizing the
    encoded information about the images themselves, while minimizing the information
    about the transforms and data augmentations used to obtain the representations.

    The original Barlow Twins paper uses the ImageNet dataset, but in this example we
    are using CIFAR-10, so we will follow the recommendations of this paper, instead:
    "A Note on Connecting Barlow Twins with Negative-Sample-Free Contrastive Learning"
    by Yao-Hung Hubert Tsai, Shaojie Bai, Louis-Philippe Morency, Ruslan Salakhutdinov,
    in which they experiment with Barlow Twins on CIFAR-10 and Tiny ImageNet.  Since
    the CIFAR-10 contains relatively small images, we will define a ResNet50 architecture
    that doesn't downsample the input in the first convolutional layer, and doesn't have
    a max pooling layer afterwards, like the paper does.
*/

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/global_optimization.h>
#include <dlib/gui_widgets.h>
#include <dlib/svm_threaded.h>

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
};

// This model namespace contains the definitions for:
// - SSL model using the Barlow Twins loss, a projector head and an input_rgb_image_pair.
// - A feature extractor model using the loss_metric (to get the outputs) and an input_rgb_image.
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
    const double mins = 7. / 15.;
    const double maxs = 7. / 8.;
    const auto scale = rnd.get_double_in_range(mins, maxs);
    const auto size = scale * std::min(image.nr(), image.nc());
    const rectangle rect(size, size);
    const point offset(rnd.get_random_32bit_number() % (image.nc() - rect.width()),
                       rnd.get_random_32bit_number() % (image.nr() - rect.height()));
    return move_rect(rect, offset);
}

matrix<rgb_pixel> augment(
    const matrix<rgb_pixel>& image,
    const bool prime,
    dlib::rand& rnd
)
{
    matrix<rgb_pixel> crop;
    // blur
    matrix<rgb_pixel> blurred;
    const double sigma = rnd.get_double_in_range(0.1, 1.1);
    if (!prime || (prime && rnd.get_random_double() < 0.1))
    {
        const auto rect = gaussian_blur(image, blurred, sigma);
        extract_image_chip(blurred, rect, crop);
        blurred = crop;
    }
    else
    {
        blurred = image;
    }

    // randomly crop
    const auto rect = make_random_cropping_rect(image, rnd);
    extract_image_chip(blurred, chip_details(rect, chip_dims(32, 32)), crop);

    // image left-right flip
    if (rnd.get_random_double() < 0.5)
        flip_image_left_right(crop);

    // color augmentation
    if (rnd.get_random_double() < 0.8)
        disturb_colors(crop, rnd, 0.5, 0.5);

    // grayscale
    if (rnd.get_random_double() < 0.2)
    {
        matrix<unsigned char> gray;
        assign_image(gray, crop);
        assign_image(crop, gray);
    }

    // solarize
    if (prime && rnd.get_random_double() < 0.2)
    {
        for (auto& p : crop)
        {
            if (p.red > 128)
                p.red = 255 - p.red;
            if (p.green > 128)
                p.green = 255 - p.green;
            if (p.blue > 128)
                p.blue = 255 - p.blue;
        }
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
    parser.add_option("lambda", "penalize off-diagonal terms (default: 1/dims)", 1);
    parser.add_option("learning-rate", "set the initial learning rate (default: 1e-3)", 1);
    parser.add_option("min-learning-rate", "set the min learning rate (default: 1e-5)", 1);
    parser.add_option("num-gpus", "number of GPUs (default: 1)", 1);
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

    const size_t num_gpus = get_option(parser, "num-gpus", 1);
    const size_t batch_size = get_option(parser, "batch", 64) * num_gpus;
    const long dims = get_option(parser, "dims", 128);
    const double lambda = get_option(parser, "lambda", 1.0 / dims);
    const double learning_rate = get_option(parser, "learning-rate", 1e-3);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-5);

    // Load the CIFAR-10 dataset into memory.
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;
    load_cifar_10_dataset(parser[0], training_images, training_labels, testing_images, testing_labels);

    // Initialize the model with the specified projector dimensions and lambda.  According to the
    // second paper, lambda = 1/dims works well on CIFAR-10.
    model::train net((loss_barlow_twins_(lambda)));
    layer<1>(net).layer_details().set_num_outputs(dims);
    disable_duplicative_biases(net);
    dlib::rand rnd;
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);

    // Train the feature extractor using the Barlow Twins method
    {
        dnn_trainer<model::train, adam> trainer(net, adam(1e-6, 0.9, 0.999), gpus);
        trainer.set_mini_batch_size(batch_size);
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.set_iterations_without_progress_threshold(10000);
        trainer.set_synchronization_file("barlow_twins_sync");
        trainer.be_verbose();
        cout << trainer << endl;

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
        const double eps = DEFAULT_BATCH_NORM_EPS;
        gamma.set_size(1, dims);
        beta.set_size(1, dims);
        image_window win;

        std::vector<std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch;
        while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        {
            batch.clear();
            while (batch.size() < trainer.get_mini_batch_size())
            {
                const auto idx = rnd.get_random_32bit_number() % training_images.size();
                auto image = training_images[idx];
                batch.emplace_back(augment(image, false, rnd), augment(image, true, rnd));
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
                // The trainer might have synchronized its state to the disk and cleaned
                // the network state. If that happens, the output will be empty, in
                // which case, we just skip the empirical cross-correlation matrix
                // computation.
                if (out.size() == 0)
                    continue;
                // Separate both augmented versions of the images
                alias_tensor split(out.num_samples() / 2, dims);
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

    // Now, we initialize the feature extractor model with the backbone we have just learned.
    model::feats fnet(layer<5>(net));
    // And we will generate all the features for the training set to train a multiclass SVM
    // classifier.
    std::vector<matrix<float, 0, 1>> features;
    cout << "Extracting features for linear classifier..." << endl;
    features = fnet(training_images, 4 * batch_size);
    vector_normalizer<matrix<float, 0, 1>> normalizer;
    normalizer.train(features);
    for (auto& feature : features)
        feature = normalizer(feature);

    // Find the most appropriate C setting using find_max_global.
    auto cross_validation_score = [&](const double c)
    {
        svm_multiclass_linear_trainer<linear_kernel<matrix<float, 0, 1>>, unsigned long> trainer;
        trainer.set_c(c);
        trainer.set_epsilon(0.01);
        trainer.set_max_iterations(100);
        trainer.set_num_threads(std::thread::hardware_concurrency());
        cout << "C: " << c << endl;
        const auto cm = cross_validate_multiclass_trainer(trainer, features, training_labels, 3);
        const double accuracy = sum(diag(cm)) / sum(cm);
        cout << "cross validation accuracy: " << accuracy << endl;
        cout << "confusion matrix:\n " << cm << endl;
        return accuracy;
    };
    const auto result = find_max_global(cross_validation_score, 1e-3, 1000, max_function_calls(50));
    cout << "Best C: " << result.x(0) << endl;

    // Proceed to train the SVM classifier with the best C.
    svm_multiclass_linear_trainer<linear_kernel<matrix<float, 0, 1>>, unsigned long> trainer;
    trainer.set_num_threads(std::thread::hardware_concurrency());
    trainer.set_c(result.x(0));
    cout << "Training Multiclass SVM..." << endl;
    const auto df = trainer.train(features, training_labels);
    serialize("multiclass_svm_cifar_10.dat") << df;

    // Finally, we can compute the accuracy of the model on the CIFAR-10 train and test images.
    auto compute_accuracy = [&fnet, &df, batch_size](
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

    // We should get a training accuracy of around 93% and a testing accuracy of around 89%.
    cout << "\ntraining accuracy" << endl;
    compute_accuracy(features, training_labels);
    cout << "\ntesting accuracy" << endl;
    features = fnet(testing_images, 4 * batch_size);
    for (auto& feature : features)
        feature = normalizer(feature);
    compute_accuracy(features, testing_labels);
    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
