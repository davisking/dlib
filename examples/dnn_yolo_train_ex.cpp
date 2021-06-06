#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>

using namespace std;
using namespace dlib;

namespace darknet
{
    // backbone tags
    template <typename SUBNET> using btag8 = add_tag_layer<8008, SUBNET>;
    template <typename SUBNET> using btag16 = add_tag_layer<8016, SUBNET>;
    template <typename SUBNET> using bskip8 = add_skip_layer<btag8, SUBNET>;
    template <typename SUBNET> using bskip16 = add_skip_layer<btag16, SUBNET>;
    // neck tags
    template <typename SUBNET> using ntag8 = add_tag_layer<6008, SUBNET>;
    template <typename SUBNET> using ntag16 = add_tag_layer<6016, SUBNET>;
    template <typename SUBNET> using ntag32 = add_tag_layer<6032, SUBNET>;
    template <typename SUBNET> using nskip8 = add_skip_layer<ntag8, SUBNET>;
    template <typename SUBNET> using nskip16 = add_skip_layer<ntag16, SUBNET>;
    template <typename SUBNET> using nskip32 = add_skip_layer<ntag32, SUBNET>;
    // head tags
    template <typename SUBNET> using htag8 = add_tag_layer<7008, SUBNET>;
    template <typename SUBNET> using htag16 = add_tag_layer<7016, SUBNET>;
    template <typename SUBNET> using htag32 = add_tag_layer<7032, SUBNET>;
    template <typename SUBNET> using hskip8 = add_skip_layer<htag8, SUBNET>;
    template <typename SUBNET> using hskip16 = add_skip_layer<htag16, SUBNET>;
    template <typename SUBNET> using hskip32 = add_skip_layer<htag32, SUBNET>;
    // yolo tags
    template <typename SUBNET> using ytag8 = add_tag_layer<4008, SUBNET>;
    template <typename SUBNET> using ytag16 = add_tag_layer<4016, SUBNET>;
    template <typename SUBNET> using ytag32 = add_tag_layer<4032, SUBNET>;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long nf, long ks, int s, typename SUBNET>
        using conblock = ACT<BN<add_layer<con_<nf, ks, ks, s, s, ks/2, ks/2>, SUBNET>>>;

        template <long nf1, long nf2, typename SUBNET>
        using residual = add_prev1<
                         conblock<nf1, 3, 1,
                         conblock<nf2, 1, 1,
                    tag1<SUBNET>>>>;

        template <long nf, long factor, typename SUBNET>
        using conblock5 = conblock<nf, 1, 1,
                          conblock<nf * factor, 3, 1,
                          conblock<nf, 1, 1,
                          conblock<nf * factor, 3, 1,
                          conblock<nf, 1, 1, SUBNET>>>>>;

        // the residual block introduced in YOLOv3 (with bottleneck)
        template <long nf, typename SUBNET> using resv3 = residual<nf, nf / 2, SUBNET>;
        // the residual block introduced in YOLOv4 (without bottleneck)
        template <long nf, typename SUBNET> using resv4 = residual<nf, nf, SUBNET>;

        template <typename SUBNET> using resv3_64= resv3<64, SUBNET>;
        template <typename SUBNET> using resv3_128 = resv3<128, SUBNET>;
        template <typename SUBNET> using resv3_256 = resv3<256, SUBNET>;
        template <typename SUBNET> using resv3_512 = resv3<512, SUBNET>;
        template <typename SUBNET> using resv3_1024 = resv3<1024, SUBNET>;

        template <typename INPUT>
        using backbone53 = repeat<4, resv3_1024, conblock<1024, 3, 2,
                    btag16<repeat<8, resv3_512,  conblock<512, 3, 2,
                     btag8<repeat<8, resv3_256,  conblock<256, 3, 2,
                           repeat<2, resv3_128,  conblock<128, 3, 2,
                                     resv3_64<   conblock<64, 3, 2,
                                                 conblock<32, 3, 1,
                                                 INPUT>>>>>>>>>>>>>;


        template <long nf, int classes, template <typename> class YTAG, template <typename> class NTAG, typename SUBNET>
        using yolo = YTAG<sig<con<3 * (classes + 5), 1, 1, 1, 1,
                     conblock<nf, 3, 1,
                NTAG<conblock5<nf / 2, 2,
                     SUBNET>>>>>>;

        template <long num_classes>
        using yolov3 = yolo<256, num_classes, ytag8, ntag8,
                       concat2<htag8, btag8,
                 htag8<upsample<2, conblock<128, 1, 1,
                       nskip16<
                       yolo<512, num_classes, ytag16, ntag16,
                       concat2<htag16, btag16,
                htag16<upsample<2, conblock<256, 1, 1,
                       nskip32<
                       yolo<1024, num_classes, ytag32, ntag32,
                       backbone53<tag1<input_rgb_image>>
                       >>>>>>>>>>>>>;

    };

    using yolov3_train = def<leaky_relu, bn_con>::yolov3<80>;
    using yolov3_infer = def<leaky_relu, affine>::yolov3<80>;

    template <typename net_type>
    void setup_detector(net_type& net, int num_classes = 80, size_t img_size = 416)
    {
        // remove bias
        disable_duplicative_biases(net);
        // remove mean from input image
        layer<net_type::num_layers - 1>(net) = input_rgb_image(0, 0, 0);
        // setup leaky relus
        visit_computational_layers(net, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
        // set the number of filters
        layer<ytag8, 2>(net).layer_details().set_num_filters(3 * (num_classes + 5));
        layer<ytag16, 2>(net).layer_details().set_num_filters(3 * (num_classes + 5));
        layer<ytag32, 2>(net).layer_details().set_num_filters(3 * (num_classes + 5));
        // allocate the network
        matrix<rgb_pixel> image(img_size, img_size);
        net(image);
    }

}

using net_type = dlib::loss_yolo<darknet::ytag8, darknet::ytag16, darknet::ytag32, darknet::yolov3_train>;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("size", "image size for training (default: 416)", 1);
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("batch-size", "mini batch size (default: 8)", 1);
    parser.add_option("warmup", "number of warmup steps (default: 5000)", 1);
    parser.add_option("steps", "number of training steps (defaule: 500000)", 1);
    parser.add_option("workers", "number of worker threads to load data (default: 4)", 1);
    parser.add_option("gpus", "number of GPUs to run the training on (default: 1)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 || parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        cout << "Give the path to a folder containing the training.xml file." << endl;
        return 0;
    }
    const double learning_rate = get_option(parser, "learning-rate", 0.001);
    const size_t batch_size = get_option(parser, "batch-size", 8);
    const size_t warmup = get_option(parser, "warmup", 5000);
    const size_t num_steps = get_option(parser, "steps", 100000);
    const size_t image_size = get_option(parser, "size", 416);
    const size_t num_workers = get_option(parser, "workers", 4);
    const size_t num_gpus = get_option(parser, "gpus", 1);
    const std::string data_directory = parser[0];
    image_dataset_metadata::dataset dataset;
    image_dataset_metadata::load_image_dataset_metadata(dataset, data_directory + "/training.xml");
    std::cout << "# images: " << dataset.images.size() << std::endl;
    std::map<std::string, size_t> labels;
    size_t num_objects = 0;
    for (const auto& im : dataset.images)
    {
        for (const auto& b : im.boxes)
        {
            labels[b.label]++;
            ++num_objects;
        }
    }
    std::cout << "# labels: " << labels.size() << std::endl;

    yolo_options options;
    for (const auto& label : labels)
    {
        std::cout <<  " - " << label.first << ": " << label.second;
        std::cout << " (" << (100.0*label.second)/num_objects << "%)\n";
        options.labels.push_back(label.first);
    }
    options.confidence_threshold = 0.25;
    options.add_anchors<darknet::ytag8>({{10, 13}, {16, 30}, {33, 23}});
    options.add_anchors<darknet::ytag16>({{30, 61}, {62, 45}, {59, 119}});
    options.add_anchors<darknet::ytag32>({{116, 90}, {156, 198}, {373, 326}});
    options.overlaps_nms = dlib::test_box_overlap(0.45);
    net_type net(options);
    darknet::setup_detector(net, options.labels.size(), image_size);

    // Cosine scheduler with warm-up:
    // - learning_rate is the highest learning rate value, e.g. 0.01
    // - warmup: number of steps to linearly increase the learning rate
    // - steps: maximum number of steps of the training session
    const matrix<double> learning_rate_schedule = learning_rate * join_rows(
        linspace(0, 1, warmup),
        ((1 + cos(pi / (num_steps - warmup) * linspace(0, num_steps - warmup, num_steps - warmup))) / 2)
    ) + std::numeric_limits<double>::epsilon();  // this prevents the learning rates from being 0

    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    dnn_trainer<net_type> trainer(net, sgd(0.0005, 0.9), gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_learning_rate_schedule(learning_rate_schedule);
    trainer.set_synchronization_file("yolov3_sync", std::chrono::minutes(15));
    std::cout << trainer;
    std::cout << "  burnin: " << warmup << std::endl;
    std::cout << "  #steps: " << num_steps << std::endl;

    dlib::pipe<std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>>> train_data(1000);
    auto loader = [&dataset, &data_directory, &train_data, &image_size](time_t seed) {
        dlib::rand rnd(time(nullptr) + seed);
        matrix<rgb_pixel> image, letterbox;
        std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
        while (train_data.is_enabled())
        {
            std::vector<yolo_rect> boxes;
            auto idx = rnd.get_random_32bit_number() % dataset.images.size();
            load_image(image, data_directory + "/" + dataset.images[idx].filename);

            const auto tform = rectangle_transform(letterbox_image(image, letterbox, image_size));
            for (const auto& box : dataset.images[idx].boxes)
            {
                boxes.push_back(yolo_rect(tform(box.rect), 1, box.label));
            }
            // here you should do more data augmentation
            disturb_colors(image, rnd);
            temp.first = letterbox;
            temp.second = boxes;
            train_data.enqueue(temp);
        }
    };

    std::vector<std::thread> data_loaders;
    for (size_t i = 0; i < num_workers; ++i)
        data_loaders.emplace_back([loader, i]() { loader(i + 1); });

    std::vector<matrix<rgb_pixel>> images;
    std::vector<std::vector<yolo_rect>> bboxes;
    while (trainer.get_train_one_step_calls() < num_steps)
    {
        images.clear();
        bboxes.clear();
        std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
        while (images.size() < trainer.get_mini_batch_size())
        {
            train_data.dequeue(temp);
            images.push_back(std::move(temp.first));
            bboxes.push_back(std::move(temp.second));
        }
        trainer.train_one_step(images, bboxes);
    }

    trainer.get_net();
    train_data.disable();
    for (auto& worker : data_loaders)
        worker.join();

    serialize("yolov3.dnn") << net;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
