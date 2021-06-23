#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <tools/imglab/src/metadata_editor.h>

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
    // yolo tags
    template <typename SUBNET> using ytag8 = add_tag_layer<4008, SUBNET>;
    template <typename SUBNET> using ytag16 = add_tag_layer<4016, SUBNET>;
    template <typename SUBNET> using ytag32 = add_tag_layer<4032, SUBNET>;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long nf, long ks, int s, typename SUBNET>
        using conblock = ACT<BN<add_layer<con_<nf, ks, ks, s, s, ks / 2, ks / 2>, SUBNET>>>;

        template <long nf, typename SUBNET>
        using residual = add_prev1<conblock<nf, 3, 1, conblock<nf / 2, 1, 1, tag1<SUBNET>>>>;

        template <long nf, long factor, typename SUBNET>
        using conblock5 = conblock<nf, 1, 1,
                          conblock<nf * factor, 3, 1,
                          conblock<nf, 1, 1,
                          conblock<nf * factor, 3, 1,
                          conblock<nf, 1, 1, SUBNET>>>>>;

        template <typename SUBNET> using res_64 = residual<64, SUBNET>;
        template <typename SUBNET> using res_128 = residual<128, SUBNET>;
        template <typename SUBNET> using res_256 = residual<256, SUBNET>;
        template <typename SUBNET> using res_512 = residual<512, SUBNET>;
        template <typename SUBNET> using res_1024 = residual<1024, SUBNET>;

        template <typename INPUT>
        using backbone53 = repeat<4, res_1024, conblock<1024, 3, 2,
                    btag16<repeat<8, res_512,  conblock<512, 3, 2,
                     btag8<repeat<8, res_256,  conblock<256, 3, 2,
                           repeat<2, res_128,  conblock<128, 3, 2,
                                     res_64<   conblock<64, 3, 2,
                                               conblock<32, 3, 1,
                                               INPUT>>>>>>>>>>>>>;

        template <long num_classes, long nf, template <typename> class YTAG, template <typename> class NTAG, typename SUBNET>
        using yolo = YTAG<sig<con<3 * (num_classes + 5), 1, 1, 1, 1,
                     conblock<nf, 3, 1,
                NTAG<conblock5<nf / 2, 2,
                     SUBNET>>>>>>;

        template <long num_classes>
        using yolov3 = yolo<num_classes, 256, ytag8, ntag8,
                       concat2<htag8, btag8,
                 htag8<upsample<2, conblock<128, 1, 1,
                       nskip16<
                       yolo<num_classes, 512, ytag16, ntag16,
                       concat2<htag16, btag16,
                htag16<upsample<2, conblock<256, 1, 1,
                       nskip32<
                       yolo<num_classes, 1024, ytag32, ntag32,
                       backbone53<input_rgb_image>>>>>>>>>>>>>>;

    };

    using yolov3_train_type = loss_yolo<ytag8, ytag16, ytag32, def<leaky_relu, bn_con>::yolov3<80>>;
    using yolov3_infer_type = loss_yolo<ytag8, ytag16, ytag32, def<leaky_relu, affine>::yolov3<80>>;

    void setup_detector(yolov3_train_type& net, const yolo_options& options)
    {
        // remove bias from bn inputs
        disable_duplicative_biases(net);
        // setup leaky relus
        visit_computational_layers(net, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
        // enlarge the batch normalization stats window
        set_all_bn_running_stats_window_sizes(net, 1000);
        // set the number of filters for detection layers (they are located after the tag and sig layers)
        const long nfo1 = options.anchors.at(tag_id<ytag8>::id).size() * (options.labels.size() + 5);
        const long nfo2 = options.anchors.at(tag_id<ytag16>::id).size() * (options.labels.size() + 5);
        const long nfo3 = options.anchors.at(tag_id<ytag32>::id).size() * (options.labels.size() + 5);
        layer<ytag8, 2>(net).layer_details().set_num_filters(nfo1);
        layer<ytag16, 2>(net).layer_details().set_num_filters(nfo2);
        layer<ytag32, 2>(net).layer_details().set_num_filters(nfo3);
    }
}

rectangle_transform preprocess_image(const matrix<rgb_pixel>& image, matrix<rgb_pixel>& output, const long image_size)
{
    return rectangle_transform(inv(letterbox_image(image, output, image_size)));
}

void postprocess_detections(const rectangle_transform& tform, std::vector<yolo_rect>& detections)
{
    for (auto& d : detections)
        d.rect = tform(d.rect);
}

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("size", "image size for training (default: 416)", 1);
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("batch-size", "mini batch size (default: 8)", 1);
    parser.add_option("burnin", "learning rate burnin steps (default: 1000)", 1);
    parser.add_option("patience", "number of steps without progress (default: 10000)", 1);
    parser.add_option("workers", "number of worker threads to load data (default: 4)", 1);
    parser.add_option("gpus", "number of GPUs to run the training on (default: 1)", 1);
    parser.add_option("test", "test the detector with a threshold (default: 0.01)", 1);
    parser.add_option("visualize", "visualize data augmentation instead of training");
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
    const size_t patience = get_option(parser, "patience", 10000);
    const size_t batch_size = get_option(parser, "batch-size", 8);
    const size_t burnin = get_option(parser, "burnin", 1000);
    const size_t image_size = get_option(parser, "size", 416);
    const size_t num_workers = get_option(parser, "workers", 4);
    const size_t num_gpus = get_option(parser, "gpus", 1);

    const string data_directory = parser[0];
    const string sync_file_name = "yolov3_sync";

    image_dataset_metadata::dataset dataset;
    image_dataset_metadata::load_image_dataset_metadata(dataset, data_directory + "/training.xml");
    std::cout << "# images: " << dataset.images.size() << std::endl;
    std::map<string, size_t> labels;
    size_t num_objects = 0;
    for (const auto& im : dataset.images)
    {
        for (const auto& b : im.boxes)
        {
            labels[b.label]++;
            ++num_objects;
        }
    }
    cout << "# labels: " << labels.size() << endl;

    yolo_options options;
    for (const auto& label : labels)
    {
        cout <<  " - " << label.first << ": " << label.second;
        cout << " (" << (100.0*label.second)/num_objects << "%)\n";
        options.labels.push_back(label.first);
    }
    options.truth_match_iou_threshold = 0.7;
    options.add_anchors<darknet::ytag8>({{10, 13}, {16, 30}, {33, 23}});
    options.add_anchors<darknet::ytag16>({{30, 61}, {62, 45}, {59, 119}});
    options.add_anchors<darknet::ytag32>({{116, 90}, {156, 198}, {373, 326}});
    darknet::yolov3_train_type net(options);
    darknet::setup_detector(net, options);
    cerr << net << endl;

    const matrix<double> learning_rate_schedule = learning_rate * pow(linspace(1e-12, 1, burnin), 4);

    std::vector<int> gpus(num_gpus);
    iota(gpus.begin(), gpus.end(), 0);
    dnn_trainer<darknet::yolov3_train_type> trainer(net, sgd(0.0005, 0.9), gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_learning_rate_schedule(learning_rate_schedule);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_synchronization_file(sync_file_name, chrono::minutes(15));
    cout << trainer;

    if (parser.option("test"))
    {
        if (!file_exists(sync_file_name))
        {
            cout << "Could not find file " << sync_file_name << endl;
            return EXIT_FAILURE;
        }
        const double threshold = get_option(parser, "test", 0.01);
        net.loss_details().adjust_threshold(threshold);
        image_window win;
        matrix<rgb_pixel> image, resized;
        color_mapper string_to_color;
        for (const auto& im : dataset.images)
        {
            win.clear_overlay();
            load_image(image, data_directory + "/" + im.filename);
            win.set_title(im.filename);
            win.set_image(image);
            const auto tform = preprocess_image(image, resized, image_size);
            auto detections = net(resized);
            postprocess_detections(tform, detections);
            cout << "# detections: " << detections.size() << endl;
            for (const auto& det : detections)
            {
                win.add_overlay(det.rect, string_to_color(det.label), det.label);
                cout << det.label << ": " << det.rect << " " << det.detection_confidence << endl;
            }
            cin.get();
        }
        return EXIT_SUCCESS;
    }

    dlib::pipe<std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>>> train_data(1000);
    auto loader = [&dataset, &data_directory, &train_data, &image_size](time_t seed) {
        dlib::rand rnd(time(nullptr) + seed);
        matrix<rgb_pixel> image, rotated;
        std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
        random_cropper cropper;
        cropper.set_seed(time_t(nullptr) + seed);
        cropper.set_chip_dims(image_size, image_size);
        cropper.set_max_object_size(0.9);
        cropper.set_min_object_size(10, 10);
        cropper.set_max_rotation_degrees(10);
        cropper.set_translate_amount(0.1);
        cropper.set_randomly_flip(true);
        cropper.set_background_crops_fraction(0);
        while (train_data.is_enabled())
        {
            const auto idx = rnd.get_random_32bit_number() % dataset.images.size();
            load_image(image, data_directory + "/" + dataset.images[idx].filename);
            for (const auto& box : dataset.images[idx].boxes)
                temp.second.emplace_back(box.rect, 1, box.label);

            if (rnd.get_random_double() > 0.5)
            {
                // Some data augmentation
                rectangle_transform tform = rotate_image(
                    image,
                    rotated,
                    rnd.get_double_in_range(-5 * pi / 180, 5 * pi / 180),
                    interpolate_bilinear());
                for (auto& box : temp.second)
                    box.rect = tform(box.rect);

                tform = letterbox_image(rotated, temp.first, image_size);
                for (auto& box : temp.second)
                    box.rect = tform(box.rect);

                if (rnd.get_random_double() > 0.5)
                {
                    tform = flip_image_left_right(temp.first);
                    for (auto& box : temp.second)
                        box.rect = tform(box.rect);
                }
            }
            else
            {
                std::vector<yolo_rect> boxes = temp.second;
                cropper(image, boxes, temp.first, temp.second);
            }
            disturb_colors(temp.first, rnd);
            train_data.enqueue(temp);
        }
    };

    std::vector<thread> data_loaders;
    for (size_t i = 0; i < num_workers; ++i)
        data_loaders.emplace_back([loader, i]() { loader(i + 1); });

    if (parser.option("visualize"))
    {
        image_window win;
        color_mapper string_to_color;
        while (true)
        {
            std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
            train_data.dequeue(temp);
            win.clear_overlay();
            win.set_image(temp.first);
            for (const auto& r : temp.second)
            {
                auto color = string_to_color(r.label);
                if (r.ignore)
                {
                    color.alpha = 128;
                    win.add_overlay(r.rect.tl_corner(), r.rect.br_corner(), color);
                    win.add_overlay(r.rect.tr_corner(), r.rect.bl_corner(), color);
                }
                win.add_overlay(r.rect, color, r.label);
            }
            cout << "Press enter to visualize the next training sample.";
            cin.get();
        }
    }

    std::vector<matrix<rgb_pixel>> images;
    std::vector<std::vector<yolo_rect>> bboxes;

    const auto train = [&images, &bboxes, &train_data, &trainer]()
    {
        images.clear();
        bboxes.clear();
        pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
        while (images.size() < trainer.get_mini_batch_size())
        {
            train_data.dequeue(temp);
            images.push_back(move(temp.first));
            bboxes.push_back(move(temp.second));
        }
        trainer.train_one_step(images, bboxes);
    };

    std::cerr << "training started with " << burnin << " burn-in steps" << std::endl;
    while (trainer.get_train_one_step_calls() < burnin)
        train();

    std::cerr << "burn-in finished" << std::endl;
    trainer.get_net();
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(learning_rate * 1e-3);
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_iterations_without_progress_threshold(patience);
    std::cerr << trainer << std::endl;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        train();

    std::cerr << "training done" << std::endl;

    trainer.get_net();
    train_data.disable();
    for (auto& worker : data_loaders)
        worker.join();

    serialize("yolov3.dnn") << net;
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
