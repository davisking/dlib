// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  I'm assuming you have already read the dnn_introduction_ex.cpp, the
    dnn_introduction2_ex.cpp and the dnn_introduction3_ex.cpp examples.  In this example
    program we are going to show how one can train Generative Adversarial Networks (GANs).  In
    particular, we will train a Deep Convolutional Generative Adversarial Network (DCGAN) like
    the one introduced in this paper:
    "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
    by Alec Radford, Luke Metz, Soumith Chintala.

    The main idea is that there are two neural networks training at the same time:
    - the generator is in charge of generating images that look as close as possible as the
      ones from the dataset.
    - the discriminator will decide whether an image is fake (created by the generator) or real
      (selected from the dataset).

    Each training iteration alternates between training the discriminator and the generator.
    We first train the discriminator with real and fake images and then use the gradient from
    the discriminator to update the generator.

    In this example, we are going to learn how to generate digits from the MNIST dataset, but
    the same code can be run using the Fashion MNIST datset:
    https://github.com/zalandoresearch/fashion-mnist
*/

#include <algorithm>
#include <iostream>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

// We start by defining a simple visitor to disable bias learning in a network.  By default,
// biases are initialized to 0, so setting the multipliers to 0 disables bias learning.
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

// Some helper definitions for the noise generation
const size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;

noise_t make_noise(dlib::rand& rnd)
{
    noise_t noise;
    for (auto& n : noise)
    {
        n = rnd.get_random_gaussian();
    }
    return noise;
}

// A convolution with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// A transposed convolution to with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// The generator is made of a bunch of deconvolutional layers.  Its input is a 1 x 1 x k noise
// tensor, and the output is the generated image.  The loss layer does not matter for the
// training, we just stack a compatible one on top to be able to have a () operator on the
// generator.
using generator_type =
    loss_binary_log_per_pixel<
    htan<contp<1, 4, 2, 1,
    relu<bn_con<contp<64, 4, 2, 1,
    relu<bn_con<contp<128, 3, 2, 1,
    relu<bn_con<contp<256, 4, 1, 0,
    input<noise_t>
    >>>>>>>>>>>>;

// Now, let's proceed to define the discriminator, whose role will be to decide whether an
// image is fake or not.
using discriminator_type =
    loss_binary_log<
    htan<conp<1, 3, 1, 0,
    leaky_relu<bn_con<conp<256, 4, 2, 1,
    leaky_relu<bn_con<conp<128, 4, 2, 1,
    leaky_relu<conp<64, 4, 2, 1,
    input<matrix<unsigned char>>
    >>>>>>>>>>>;

// Some helper functions to generate and get the images from the generator
matrix<unsigned char> generate_image(generator_type& net, const noise_t& noise)
{
    const matrix<float> output = net(noise);
    matrix<unsigned char> image;
    assign_image(image, 127.5f * (output + 1.f));
    return image;
}

std::vector<matrix<unsigned char>> get_generated_images(generator_type& net)
{
    std::vector<matrix<unsigned char>> images;
    const resizable_tensor &out = layer<1>(net).get_output();
    for (size_t n = 0; n < out.num_samples(); ++n)
    {
        matrix<float> output = image_plane(out, n);
        matrix<unsigned char> image;
        assign_image(image, 127.5f * (output + 1.f));
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

    // MNIST is broken into two parts, a training set of 60000 images and a test set of 10000
    // images.  Each image is labeled so that we know what hand written digit is depicted.
    // These next statements load the dataset into memory.
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);

    // Fix the random generator seeds for network initialization and noise
    srand(1234);
    dlib::rand rnd(std::rand());

    // Instantiate both generator and discriminator
    generator_type generator;
    discriminator_type discriminator(
        leaky_relu_(0.2f), leaky_relu_(0.2f), leaky_relu_(0.2f));
    // Remove the bias learning from the networks
    visit_layers(generator, visitor_no_bias());
    visit_layers(discriminator, visitor_no_bias());
    // Forward random noise so that we see the tensor size at each layer
    discriminator(generate_image(generator, make_noise(rnd)));
    cout << "generator" << endl;
    cout << generator << endl;
    cout << "discriminator" << endl;
    cout << discriminator << endl;

    // The solvers for the generator network.  In this case, we don't need to use a dnn_trainer
    // for the generator, since we are going to train it manually
    adam solver(0.f, 0.5f, 0.999f);
    std::vector<adam> g_solvers(generator.num_computational_layers, solver);
    double learning_rate = 2e-4;
    // The discriminator trainer
    dnn_trainer<discriminator_type, adam> d_trainer(discriminator, solver);
    d_trainer.be_quiet();
    d_trainer.set_learning_rate(learning_rate);
    d_trainer.set_learning_rate_shrink_factor(1);
    cout << d_trainer << endl;

    // Resume training from last sync file
    if (file_exists("dcgan_sync"))
    {
        deserialize("dcgan_sync") >> generator >> d_trainer;
    }

    const size_t minibatch_size = 64;
    const std::vector<float> real_labels(minibatch_size, 1.f);
    const std::vector<float> fake_labels(minibatch_size, -1.f);
    dlib::image_window win;
    size_t iteration = 0;
    while (iteration < 50000)
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
        // 4. finally train the discriminator and wait for the threading to stop
        d_trainer.train_one_step(fake_samples, fake_labels);
        d_trainer.get_net(force_flush_to_disk::no);
        const auto d_loss = d_trainer.get_average_loss();

        // Train the generator
        // This part is the essence of the Generative Adversarial Networks.  Until now, we have
        // just trained a binary classifier that the generator is not aware of.  But now, the
        // discriminator is going to give feedback to the generator on how it should update
        // itself to generate more realistic images.  The following lines perform the same
        // actions as train_one_step() except for the network update part.  They can also be
        // seen as test_one_step() plus the error back propagation.

        // Convert the fake samples to a tensor
        resizable_tensor fake_samples_tensor;
        discriminator.subnet().to_tensor(fake_samples.begin(), fake_samples.end(), fake_samples_tensor);
        // Forward the samples and compute the loss with real labels
        const auto g_loss = discriminator.compute_loss(fake_samples_tensor, real_labels.begin());
        // Back propagate the error to fill the final data gradient
        discriminator.subnet().back_propagate_error(fake_samples_tensor);
        // Get the gradient that will tell the generator how to update itself
        const resizable_tensor& d_grad = discriminator.subnet().get_final_data_gradient();
        generator.subnet().back_propagate_error(noises_tensor, d_grad);
        generator.subnet().update_parameters(g_solvers, learning_rate);

        // At some point, we should see that the generated images start looking like samples from
        // the MNIST dataset
        iteration = d_trainer.get_train_one_step_calls() / 2;
        if (iteration % 1000 == 0)
        {
            serialize("dcgan_sync") << generator << d_trainer;
            std::cout <<
                "step#: " << iteration <<
                "\tdiscriminator loss: " << d_loss <<
                "\tgenerator loss: " << g_loss << '\n';
            win.set_image(tile_images(fake_samples));
            win.set_title("DCGAN step#: " + to_string(iteration));
        }
    }

    // Once the training has finished, we don't need the generator any more. We just keep the
    // generator.
    generator.clean();
    serialize("dcgan_mnist.dnn") << generator;

    // To test the generator, we just forward some random noise through it and visualize the
    // output.
    while (not win.is_closed())
    {
        win.set_image(generate_image(generator, make_noise(rnd)));
        cout << "Hit enter to generate a new image";
        cin.get();
    }

    return EXIT_SUCCESS;
}
catch(exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
