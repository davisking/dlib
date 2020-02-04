#include <algorithm>
#include <iostream>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

// some helper definitions for the noise generation
constexpr size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;

noise_t make_noise(dlib::rand& rnd)
{
    noise_t noise;
    std::for_each(begin(noise), end(noise),
        [&rnd] (matrix<float, 1, 1> &m)
        {
            m = rnd.get_random_gaussian();
        });
    return noise;
}

// a custom convolution definition to allow for padding size specification
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// Let's define a transposed convolution to with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// The generator is made of a bunch of deconvolutional layers. It's input is a
// 1 x 1 x k noise tensor, and the output is the score of the generated image
// (decided by the discriminator, which we'll define afterwards)
using generator_type =
    loss_binary_log<
    htan<contp<1, 4, 2, 1,
    relu<bn_con<contp<64, 4, 2, 1,
    relu<bn_con<contp<128, 3, 2, 1,
    relu<bn_con<contp<256, 4, 1, 0,
    input<noise_t>
    >>>>>>>>>>>>;

// Now, let's proceed to define the discriminator, whose role will be to decide
// whether an image is fake or not.
using discriminator_type =
    loss_binary_log<
    bn_con<conp<1, 3, 1, 0,
    prelu<bn_con<conp<256, 4, 2, 1,
    prelu<bn_con<conp<128, 4, 2, 1,
    prelu<conp<64, 4, 2, 1,
    input<matrix<unsigned char>>
    >>>>>>>>>>>;

// Now, let's define a way to easily get the generated image
matrix<unsigned char> generated_image(generator_type& net)
{
    matrix<float> output = image_plane(layer<1>(net).get_output());
    matrix<unsigned char> image;
    assign_image_scaled(image, output);
    return image;
}

void generated_images(generator_type& net, std::vector<matrix<unsigned char>>& images)
{
    images.clear();
    const resizable_tensor& out = layer<1>(net).get_output();
    // std::cout << out.num_samples() << 'x' << out.k() << 'x' << out.nr() << 'x' << out.nc() << '\n';
    for (size_t n = 0; n < out.num_samples(); ++n)
    {
        matrix<float> output = image_plane(out, n);
        matrix<unsigned char> image;
        assign_image_scaled(image, output);
        // dlib::save_png(image, "generated_image_" + std::to_string(n) + ".png");
        images.push_back(std::move(image));
    }
}

int main(int argc, char** argv) try
{
    // This example is going to run on the MNIST dataset.
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return EXIT_FAILURE;
    }

    // MNIST is broken into two parts, a training set of 60000 images and a test set of
    // 10000 images.  Each image is labeled so that we know what hand written digit is
    // depicted.  These next statements load the dataset into memory.
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);

    generator_type generator;
    discriminator_type discriminator(
            prelu_(0.2), prelu_(0.2), prelu_(0.2));
    cout << "generator" << endl;
    cout << generator << endl;
    cout << "discriminator" << endl;
    cout << discriminator << endl;

    dnn_trainer<generator_type, adam> g_trainer(generator, adam(0, 0.5, 0.999));
    g_trainer.set_synchronization_file("dcgan_generator_sync", std::chrono::minutes(15));
    g_trainer.be_verbose();
    g_trainer.set_learning_rate(2e-4);
    g_trainer.set_learning_rate_shrink_factor(1);
    cout << g_trainer << endl;
    dnn_trainer<discriminator_type, adam> d_trainer(discriminator, adam(0, 0.5, 0.999));
    d_trainer.set_synchronization_file("dcgan_discriminator_sync", std::chrono::minutes(15));
    d_trainer.be_verbose();
    d_trainer.set_learning_rate(2e-4);
    d_trainer.set_learning_rate_shrink_factor(1);
    cout << d_trainer << endl;

    const long minibatch_size = 64;
    const auto mini_batch_real_labels = std::vector<float>(minibatch_size, 1.0f);
    const auto mini_batch_fake_labels = std::vector<float>(minibatch_size, -1.0f);
    dlib::rand rnd(time(nullptr));
    while (g_trainer.get_train_one_step_calls() < 1000)
    {
        // train the discriminator with real images
        std::vector<matrix<unsigned char>> mini_batch_real_samples;
        while (mini_batch_real_samples.size() < minibatch_size)
        {
            auto idx = rnd.get_random_32bit_number() % training_images.size();
            mini_batch_real_samples.push_back(training_images[idx]);
        }
        d_trainer.train_one_step(mini_batch_real_samples, mini_batch_real_labels);

        // train the discriminator with fake images
        // 1. generate some random noise
        std::vector<noise_t> noises;
        while (noises.size() < minibatch_size)
        {
            noises.push_back(std::move(make_noise(rnd)));
        }

        // 2. forward the noise through the generator
        resizable_tensor noises_tensor;
        generator.to_tensor(noises.begin(), noises.end(), noises_tensor);
        generator.subnet().forward(noises_tensor);
        // 3. get the generated images from the generator
        std::vector<matrix<unsigned char>> mini_batch_fake_samples;
        generated_images(generator, mini_batch_fake_samples);
        // 4. finally train the discriminator
        d_trainer.train_one_step(mini_batch_fake_samples, mini_batch_fake_labels);

        // train the generator
        // get the gradient with regards to the input of the discriminator for the fake images
        const resizable_tensor& dis_grad_out = discriminator.subnet().get_final_data_gradient();
        // check that the gradient hasn't been zeroed because we synced the network to the disk
        if (dis_grad_out.nr() > 0 && dis_grad_out.nc() > 0)
        {
            auto solvers = g_trainer.get_solvers();
            generator.subnet().back_propagate_error(noises_tensor, dis_grad_out);
            generator.subnet().update_parameters(
                make_sstack<adam>(solvers), g_trainer.get_learning_rate());
        }
        else // it means the discriminator has been synced to disk, so we sync the generator, too
        {
            g_trainer.get_net();
        }

        auto iteration = d_trainer.get_train_one_step_calls();
        if (iteration % 1000 == 0)
        {
            for (size_t i = 0; i < mini_batch_fake_samples.size(); ++i)
            {
                dlib::save_png(mini_batch_fake_samples[i], "fake_" + std::to_string(i) + ".png");
                dlib::save_png(mini_batch_real_samples[i], "real_" + std::to_string(i) + ".png");
            }
        }
    }

    return EXIT_SUCCESS;
}
catch(exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
