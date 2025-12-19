/*!
    @file slm_vision_transformer_hybrid_ex.cpp
    @brief Vision Transformer with Dlib loss hybridization demonstration

    This program demonstrates how to build a Vision Transformer using Dlib's modern
    transformer architecture (canonical_transformer) and the new patch_embeddings layer,
    showing hybridization with existing Dlib loss functions.

    Key features:
    - Modern patch embeddings with learned projection (replaces manual patch extraction)
    - Dlib's canonical_transformer with RoPE positioning
    - Hybridization examples with Dlib losses:
      * Barlow Twins (self-supervised learning, no labels needed)
      * Multiclass log (standard supervised classification)

    Vision Transformers (ViT) process images as sequences of patches, making them
    compatible with standard transformer architectures. This example shows how to
    seamlessly integrate ViT with Dlib's existing deep learning ecosystem.

    Dataset: CIFAR-10 (32x32 RGB images, 10 classes)

    Usage:
    # Self-supervised learning (Barlow Twins)
    ./slm_vision_transformer_hybrid_ex /path/to/cifar10 --ssl

    # Supervised classification
    ./slm_vision_transformer_hybrid_ex /path/to/cifar10 --supervised
!*/

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <iostream>
#include <chrono>
#include <csignal>
#include <algorithm>
#include <random>

using namespace std;
using namespace dlib;

// Signal handling for clean termination
namespace {
    std::atomic<bool> g_terminate_flag(false);

#ifdef _WIN32
    BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
        if (ctrl_type == CTRL_C_EVENT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up..." << endl;
            return TRUE;
        }
        return FALSE;
    }

    void setup_interrupt_handler() {
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    }
#else
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up..." << endl;
        }
    }

    void setup_interrupt_handler() {
        std::signal(SIGINT, signal_handler);
    }
#endif
}

// Vision Transformer Architecture
namespace dlib
{
    /*!
        Vision Transformer configuration for CIFAR-10.
        
        This demonstrates a modern, clean ViT implementation using:
        - patch_embeddings: splits image into patches + learned projection
        - canonical_transformer: Dlib's transformer with RoPE positioning
        - Standard Dlib layers: fc, dropout, ...
        
        Architecture summary:
        Input (32x32 RGB) => Patches (4x4) => Embeddings (192-dim)
            => Transformer (3 layers, 6 heads) => Pooling => Output
    !*/
    template<
        long num_layers = 3,
        long num_heads = 6,
        long embedding_dim = 192
    >
    struct vit_cifar10_config
    {
        static_assert(embedding_dim % num_heads == 0, 
            "Embedding dimension must be divisible by number of heads");
        
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long PATCH_SIZE = 4;     // 32/4 = 8x8 = 64 patches
        static constexpr long NUM_PATCHES = 64;   // (32/4)^2
        static constexpr long DONT_USE_ClASS_TOKEN = 0;
        static constexpr long DONT_USE_POSITION_EMBEDDINGS = 0;

        // Backbone: patch embeddings => transformer => pooling
        // Returns: (batch, embedding_dim) feature vectors
        template <template <typename> class DO, typename INPUT>
        using backbone_training = rms_norm<
            canonical_transformer::transformer_stack<NUM_LAYERS, gelu, DO, EMBEDDING_DIM, NUM_HEADS,
            patch_embeddings<PATCH_SIZE, EMBEDDING_DIM, DONT_USE_ClASS_TOKEN, DONT_USE_POSITION_EMBEDDINGS,
            INPUT>>>;

        template <typename INPUT>
        using backbone_inference = rms_norm<
            canonical_transformer::transformer_stack<NUM_LAYERS, gelu, multiply, EMBEDDING_DIM, NUM_HEADS,
            patch_embeddings<PATCH_SIZE, EMBEDDING_DIM, DONT_USE_ClASS_TOKEN, DONT_USE_POSITION_EMBEDDINGS,
            INPUT>>>;

        static std::string describe() {
            std::stringstream ss;
            ss << "Vision Transformer (ViT) - CIFAR-10 configuration:\n"
               << "  Input: 32x32 RGB images\n"
               << "  Patch size: " << PATCH_SIZE << "x" << PATCH_SIZE << "\n"
               << "  Number of patches: " << NUM_PATCHES << " (8x8 grid)\n"
               << "  Embedding dimension: " << EMBEDDING_DIM << "\n"
               << "  Transformer layers: " << NUM_LAYERS << "\n"
               << "  Attention heads: " << NUM_HEADS << "\n"
               << "  Head dimension: " << (EMBEDDING_DIM / NUM_HEADS) << "\n";
            return ss.str();
        }
    };
}

// Model definitions - Hybridization with Dlib losses
namespace model
{
    using my_vit = vit_cifar10_config<>;

    // Configuration 1: Self-Supervised Learning (Barlow Twins)
    // Barlow Twins learns representations without labels by maximizing agreement
    // between augmented views while decorrelating feature dimensions.
    // 
    // Architecture: ViT backbone => projector head => Barlow Twins loss
    // Input: pairs of augmented views of the same image
    // Output: self-supervised feature representations
    
    template <typename SUBNET> 
    using projector = fc<128, relu<bn_fc<fc<256, SUBNET>>>>;

    using ssl_train = loss_barlow_twins<
        projector<my_vit::backbone_training<dropout, input_rgb_image_pair>>>;
    
    using ssl_inference = loss_metric<
        my_vit::backbone_inference<input_rgb_image>>;

    // Configuration 2: Supervised classification
    // Standard supervised learning with labeled data.
    //
    // Architecture: ViT backbone => classification head => multiclass log loss
    // Input: single images with class labels
    // Output: class predictions (10 classes for CIFAR-10)

    using supervised_train = loss_multiclass_log<
        fc<10, my_vit::backbone_training<dropout, input<matrix<rgb_pixel>>>>>;
    
    using supervised_inference = loss_multiclass_log<
        fc<10, my_vit::backbone_inference<input<matrix<rgb_pixel>>>>>;
}

// Data augmentation
rectangle make_random_cropping_rect(
    const matrix<rgb_pixel>& image,
    dlib::rand& rnd
)
{
    const double min_scale = 0.7;
    const double max_scale = 1.0;
    const auto scale = rnd.get_double_in_range(min_scale, max_scale);
    const auto size = scale * std::min(image.nr(), image.nc());
    const rectangle rect(size, size);
    const point offset(
        rnd.get_random_32bit_number() % std::max<long>(1, image.nc() - rect.width() + 1),
        rnd.get_random_32bit_number() % std::max<long>(1, image.nr() - rect.height() + 1)
    );
    return move_rect(rect, offset);
}

matrix<rgb_pixel> augment_image(
    const matrix<rgb_pixel>& image,
    dlib::rand& rnd,
    bool strong_augmentation = false
)
{
    matrix<rgb_pixel> crop;
    
    // Random cropping
    const auto rect = make_random_cropping_rect(image, rnd);
    extract_image_chip(image, chip_details(rect, chip_dims(32, 32)), crop);

    // Random horizontal flip
    if (rnd.get_random_double() < 0.5)
        flip_image_left_right(crop);

    // Color augmentation
    if (rnd.get_random_double() < 0.8)
        disturb_colors(crop, rnd, 0.4, 0.4);

    // Stronger augmentations for SSL
    if (strong_augmentation)
    {
        // Grayscale conversion
        if (rnd.get_random_double() < 0.2)
        {
            matrix<unsigned char> gray;
            assign_image(gray, crop);
            assign_image(crop, gray);
        }

        // Gaussian blur
        if (rnd.get_random_double() < 0.5)
        {
            matrix<rgb_pixel> blurred;
            const double sigma = rnd.get_double_in_range(0.1, 2.0);
            gaussian_blur(crop, blurred, sigma);
            crop = blurred;
        }
    }

    return crop;
}

// Training functions
void train_ssl(
    const std::vector<matrix<rgb_pixel>>& training_images,
    const string& model_file,
    size_t batch_size,
    double learning_rate,
    double min_learning_rate,
    double lambda
)
{
    cout << "\n=== SELF-SUPERVISED LEARNING MODE (Barlow Twins) ===" << endl;
    cout << "Training without labels - Learning representations from augmentations\n" << endl;

    model::ssl_train net((loss_barlow_twins_(lambda)));
    dnn_trainer<model::ssl_train, adamw> trainer(net, adamw(0.01, 0.9, 0.999));
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(min_learning_rate);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_iterations_without_progress_threshold(15000);
    trainer.be_verbose();
    set_all_bn_running_stats_window_sizes(net, 100);    
    
    cout << "\nBarlow Twins lambda: " << lambda << endl;
    if (file_exists(model_file)) {
        deserialize(model_file) >> net;
        cout << "Number of trainable parameters: " << count_parameters(net) << "\n" << endl;
    }
    cout << "Network architecture:\n" << net << endl;

    dlib::rand rnd(time(0));    

    cout << "Starting self-supervised training...\n";
    cout << "Press Ctrl+C to stop and save the model\n" << endl;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
    {
        if (g_terminate_flag.load()) {
            cout << "\nInterrupted by user. Saving model..." << endl;
            break;
        }

        // Create pairs of augmented views
        std::vector<std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch_pairs;
        while (batch_pairs.size() < batch_size) {
            const auto idx = rnd.get_random_32bit_number() % training_images.size();
            const auto& img = training_images[idx];
            batch_pairs.emplace_back(augment_image(img, rnd, false), augment_image(img, rnd, true));
        }

        trainer.train_one_step(batch_pairs);
    }

    // Save SSL model
    trainer.get_net();
    net.clean();
    serialize(model_file) << net;
    cout << "\nSelf-supervised model saved to: " << model_file << endl;
}

void train_supervised(
    const std::vector<matrix<rgb_pixel>>& training_images,
    const std::vector<unsigned long>& training_labels,
    const std::vector<matrix<rgb_pixel>>& testing_images,
    const std::vector<unsigned long>& testing_labels,
    const string& model_file,
    size_t batch_size,
    double learning_rate,
    double min_learning_rate
)
{
    cout << "\n=== SUPERVISED LEARNING MODE (classification) ===" << endl;
    cout << "Training with labeled data for 10-class classification\n" << endl;

    model::supervised_train net;
    model::supervised_inference inference_net;
    dnn_trainer<model::supervised_train, adamw> trainer(net, adamw(0.01, 0.9, 0.999));
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(min_learning_rate);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_iterations_without_progress_threshold(15000);
    trainer.be_verbose();    
    
    if (file_exists(model_file)) {
        deserialize(model_file) >> net;
        cout << "Number of trainable parameters: " << count_parameters(net) << "\n" << endl;
    }
    cout << "Network architecture:\n" << net << endl;

    dlib::rand rnd(time(0));
    std::vector<matrix<rgb_pixel>> batch_images;
    std::vector<unsigned long> batch_labels;

    cout << "Starting supervised training...\n";
    cout << "Press Ctrl+C to stop and save the model\n" << endl;

    size_t epoch = 0;
    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
    {
        if (g_terminate_flag.load()) {
            cout << "\nInterrupted by user. Saving model..." << endl;
            break;
        }
        ++epoch;

        // Shuffle training data
        std::vector<size_t> indices(training_images.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

        // Train for one epoch
        for (size_t i = 0; i < training_images.size() && !g_terminate_flag.load(); ++i)
        {
            const auto idx = indices[i];
            batch_images.push_back(augment_image(training_images[idx], rnd, false));
            batch_labels.push_back(training_labels[idx]);

            if (batch_images.size() == batch_size)
            {
                trainer.train_one_step(batch_images, batch_labels);
                batch_images.clear();
                batch_labels.clear();
            }
        }

        // Evaluate every 10 epochs
        if (epoch % 10 == 0)
        {
            trainer.get_net();
            net.clean();
            inference_net = net;

            const size_t test_subset = std::min<size_t>(1000, testing_images.size());
            std::vector<unsigned long> predicted = inference_net(
                std::vector<matrix<rgb_pixel>>(
                    testing_images.begin(), 
                    testing_images.begin() + test_subset
                )
            );

            int num_correct = 0;
            for (size_t i = 0; i < test_subset; ++i)
                if (predicted[i] == testing_labels[i])
                    ++num_correct;

            const double accuracy = 100.0 * num_correct / test_subset;
            cout << "Epoch " << epoch << " - Validation accuracy: " << accuracy 
                 << "% (" << num_correct << "/" << test_subset << ")" << endl;
        }
    }

    // Final evaluation
    trainer.get_net();
    net.clean();    
    inference_net = net;
    cout << "\nFinal evaluation on full test set..." << endl;
    std::vector<unsigned long> predicted = inference_net(testing_images);

    int num_correct = 0;
    for (size_t i = 0; i < testing_labels.size(); ++i)
        if (predicted[i] == testing_labels[i])
            ++num_correct;

    const double final_accuracy = 100.0 * num_correct / testing_labels.size();
    cout << "Test accuracy: " << final_accuracy << "% (" 
         << num_correct << "/" << testing_labels.size() << ")" << endl;

    // Save supervised model
    serialize(model_file) << net;
    cout << "\nSupervised model saved to: " << model_file << endl;
}

int main(const int argc, const char** argv)
try
{
    setup_interrupt_handler();

    command_line_parser parser;
    parser.add_option("ssl", "Use self-supervised learning (Barlow Twins)");
    parser.add_option("supervised", "Use supervised classification");
    parser.add_option("batch-size", "Mini-batch size (default: 128 for SSL, 64 for supervised)", 1);
    parser.add_option("learning-rate", "Initial learning rate (default: 1e-3)", 1);
    parser.add_option("min-learning-rate", "Minimum learning rate (default: 1e-5)", 1);
    parser.add_option("lambda", "Barlow Twins lambda parameter (default: 0.0078)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() < 1 || parser.option("h") || parser.option("help") ||
        (!parser.option("ssl") && !parser.option("supervised")))
    {
        cout << "Vision Transformer with Dlib loss hybridization\n\n";
        cout << "This example demonstrates using modern ViT architecture\n";
        cout << "(patch_embeddings + canonical_transformer) with different\n";
        cout << "Dlib loss functions:\n\n";
        cout << "  --ssl         : Barlow Twins (self-supervised, no labels)\n";
        cout << "  --supervised  : Standard classification (with labels)\n\n";
        cout << "Dataset: CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html\n";
        cout << "Download the binary version and provide the folder path.\n\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const string cifar_dir = parser[0];
    const bool use_ssl = parser.option("ssl");
    const bool use_supervised = parser.option("supervised");

    // Load CIFAR-10
    cout << "Loading CIFAR-10 dataset from: " << cifar_dir << endl;
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;

    load_cifar_10_dataset(cifar_dir, training_images, training_labels, 
                          testing_images, testing_labels);

    cout << "Training images: " << training_images.size() << endl;
    cout << "Testing images: " << testing_images.size() << endl;

    // Display ViT configuration
    cout << "\n" << model::my_vit::describe() << "\n" << endl;

    // Training parameters
    const double learning_rate = get_option(parser, "learning-rate", 1e-3);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-5);

    if (use_ssl)
    {
        const size_t batch_size = get_option(parser, "batch-size", 128);
        const double lambda = get_option(parser, "lambda", 0.0078);
        
        train_ssl(
            training_images,
            "vit_ssl_model.dat",
            batch_size,
            learning_rate,
            min_learning_rate,
            lambda
        );
    }

    if (use_supervised)
    {
        const size_t batch_size = get_option(parser, "batch-size", 64);
        
        train_supervised(
            training_images, training_labels,
            testing_images, testing_labels,
            "vit_supervised_model.dat",
            batch_size,
            learning_rate,
            min_learning_rate
        );
    }

    return EXIT_SUCCESS;
}
catch (exception& e)
{
    cerr << "Exception: " << e.what() << endl;
    return EXIT_FAILURE;
}
