#include <algorithm>
#include <iostream>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

// A simple visitor to disable bias learning in a network.
// By default, biases are initialized to 0, so setting the multipliers to 0,
// disables bias learning.
class visitor_no_bias
{
public:
    template <typename input_layer_type>
    void operator()(size_t , input_layer_type& ) const
    {
        // ignore other layers
    }

    template <typename T, typename U, typename E>
    void operator()(size_t , add_layer<T, U, E>& l) const
    {
        set_bias_learning_rate_multiplier(l.layer_details(), 0);
        set_bias_weight_decay_multiplier(l.layer_details(), 0);
    }
};

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

// A convolution with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// A transposed convolution to with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// The generator is made of a bunch of deconvolutional layers. Its input is a
// 1 x 1 x k noise tensor, and the output is the generated image.
// The loss layer does not matter, we just put one to be able to call it.
using generator_type =
    loss_binary_log_per_pixel<
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
    htan<conp<1, 3, 1, 0,
    leaky_relu<bn_con<conp<256, 4, 2, 1,
    leaky_relu<bn_con<conp<128, 4, 2, 1,
    leaky_relu<conp<64, 4, 2, 1,
    input<matrix<unsigned char>>
    >>>>>>>>>>>;

// Some helper functions to get the images from the generator
matrix<unsigned char> generated_image(generator_type& net)
{
    matrix<float> output = image_plane(layer<1>(net).get_output());
    matrix<unsigned char> image;
    assign_image_scaled(image, output);
    return image;
}

std::vector<matrix<unsigned char>> get_generated_images(generator_type &net)
{
    std::vector<matrix<unsigned char>> images;
    const resizable_tensor &out = layer<1>(net).get_output();
    for (size_t n = 0; n < out.num_samples(); ++n)
    {
        matrix<float> output = image_plane(out, n);
        matrix<unsigned char> image;
        assign_image_scaled(image, output);
        images.push_back(std::move(image));
    }
    return images;
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

    dlib::rand rnd;
    // Instantiate both generator and discriminator
    generator_type generator;
    discriminator_type discriminator(
        leaky_relu_(0.2), leaky_relu_(0.2), leaky_relu_(0.2));
    // Remove the bias learning from the networks
    visit_layers(generator, visitor_no_bias());
    visit_layers(discriminator, visitor_no_bias());
    // Forward random noise so that we see the tensor size at each layer
    cout << "generator" << endl;
    generator(make_noise(rnd));
    cout << generator << endl;
    // Forward output of generator so that we see the tensor size at each layer
    cout << "discriminator" << endl;
    discriminator(generated_image(generator));
    cout << discriminator << endl;

    // The solvers for the generator network
    std::vector<adam> g_solvers(generator.num_computational_layers, adam(0, 0.5, 0.999));
    auto g_sstack = make_sstack(g_solvers);
    double g_learning_rate = 2e-4;
    // The discriminator trainer
    dnn_trainer<discriminator_type, adam> d_trainer(discriminator, adam(0, 0.5, 0.999));
    // d_trainer.set_synchronization_file("dcgan_discriminator_sync", std::chrono::minutes(5));
    d_trainer.be_quiet();
    d_trainer.set_learning_rate(2e-4);
    d_trainer.set_learning_rate_shrink_factor(1);
    cout << d_trainer << endl;

    const size_t minibatch_size = 64;
    std::vector<float> real_labels(minibatch_size, 1.f);
    std::vector<float> fake_labels(minibatch_size, -1.f);
    while (true)
    {
        // Train the discriminator with real images
        std::vector<matrix<unsigned char>> real_samples;
        while (real_samples.size() < minibatch_size)
        {
            auto idx = rnd.get_random_32bit_number() % training_images.size();
            real_samples.push_back(training_images[idx]);
        }
        d_trainer.train_one_step(real_samples, real_labels);

        // Train the discriminator with fake images
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
        const auto fake_samples = get_generated_images(generator);
        // 4. finally train the discriminator and wait for the threads to stop
        d_trainer.train_one_step(fake_samples, fake_labels);
        d_trainer.get_net(force_flush_to_disk::no);
        auto d_loss = d_trainer.get_average_loss();

        // Train the generator
        // This part is the essence of the Generative Adversarial Networks.
        // Until now, we have just trained a binary classifier that the generator is not
        // aware of. But now the discriminator is going to give feedback to the generator
        // on how it should update itself to generate more realistic images.
        // The following lines perform the same actions as train_one_step() except for the
        // network update part. They can also be seen as test_one_step() plus the error
        // back propagation.
        resizable_tensor samples_tensor;
        // Convert the fake samples to a tensor
        discriminator.subnet().to_tensor(fake_samples.begin(), fake_samples.end(), samples_tensor);
        // Forward the samples and compute the loss with real labels
        auto g_loss = discriminator.compute_loss(samples_tensor, real_labels.begin());
        // Back propagate the error to fill the final data gradient
        discriminator.subnet().back_propagate_error(samples_tensor);
        // Get the gradient that will tell the generator how to update itself
        const resizable_tensor& out_fake= discriminator.subnet().get_final_data_gradient();
        generator.subnet().back_propagate_error(noises_tensor, out_fake);
        generator.subnet().update_parameters(g_sstack, g_learning_rate);

        // The number of training iterations is half of the step calls
        auto iteration = d_trainer.get_train_one_step_calls() / 2;
        if (iteration % 1000 == 0)
        {
            std::cout <<
                "step#: " << iteration <<
                "\tdiscriminator loss: " << d_loss <<
                "\tgenerator loss: " << g_loss << '\n';
            save_png(tile_images(fake_samples), "fake_" + std::to_string(iteration) + ".png");
        }
    }

    return EXIT_SUCCESS;
}
catch(exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
