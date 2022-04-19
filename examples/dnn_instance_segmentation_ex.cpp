// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to do instance segmentation on an image using net pretrained
    on the PASCAL VOC2012 dataset.  For an introduction to what instance segmentation is,
    see the accompanying header file dnn_instance_segmentation_ex.h.

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_instance_segmentation_train_ex example program.
    3. Run:
       ./dnn_instance_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_instance_segmentation_ex example program.
    6. Run:
       ./dnn_instance_segmentation_ex /path/to/VOC2012-or-other-images

    An alternative to steps 2-4 above is to download a pre-trained network
    from here: http://dlib.net/files/instance_segmentation_voc2012net_v2.dnn

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#include "dnn_instance_segmentation_ex.h"
#include "pascal_voc_2012.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "You call this program like this: " << endl;
        cout << "./dnn_instance_segmentation_train_ex /path/to/images" << endl;
        cout << endl;
        cout << "You will also need a trained '" << instance_segmentation_net_filename << "' file." << endl;
        cout << "You can either train it yourself (see example program" << endl;
        cout << "dnn_instance_segmentation_train_ex), or download a" << endl;
        cout << "copy from here: http://dlib.net/files/" << instance_segmentation_net_filename << endl;
        return 1;
    }

    // Read the file containing the trained networks from the working directory.
    det_anet_type det_net;
    std::map<std::string, seg_bnet_type> seg_nets_by_class;
    deserialize(instance_segmentation_net_filename) >> det_net >> seg_nets_by_class;

    // Show inference results in a window.
    image_window win;

    matrix<rgb_pixel> input_image;

    // Find supported image files.
    const std::vector<file> files = dlib::get_files_in_directory_tree(argv[1],
        dlib::match_endings(".jpeg .jpg .png .webp"));

    dlib::rand rnd;

    cout << "Found " << files.size() << " images, processing..." << endl;

    for (const file& file : files)
    {
        // Load the input image.
        load_image(input_image, file.full_name());
        
        // Find instances in the input image
        const auto instances = det_net(input_image);

        matrix<rgb_pixel> rgb_label_image;
        matrix<float> label_image_confidence;

        matrix<rgb_pixel> input_chip;

        rgb_label_image.set_size(input_image.nr(), input_image.nc());
        rgb_label_image = rgb_pixel(0, 0, 0);

        label_image_confidence.set_size(input_image.nr(), input_image.nc());
        label_image_confidence = 0.0;

        bool found_something = false;

        for (const auto& instance : instances)
        {
            if (!found_something)
            {
                cout << "Found ";
                found_something = true;
            }
            else
            {
                cout << ", ";
            }
            cout << instance.label;

            const auto cropping_rect = get_cropping_rect(instance.rect);
            const chip_details chip_details(cropping_rect, chip_dims(seg_dim, seg_dim));
            extract_image_chip(input_image, chip_details, input_chip, interpolate_bilinear());

            const auto i = seg_nets_by_class.find(instance.label);
            if (i == seg_nets_by_class.end())
            {
                // per-class segmentation net not found, so we must be using the same net for all classes
                // (see bool separate_seg_net_for_each_class in dnn_instance_segmentation_train_ex.cpp)
                DLIB_CASSERT(seg_nets_by_class.size() == 1);
                DLIB_CASSERT(seg_nets_by_class.begin()->first == "");
            }

            auto& seg_net = i != seg_nets_by_class.end()
                ? i->second // use the segmentation net trained for this class
                : seg_nets_by_class.begin()->second; // use the same segmentation net for all classes

            const auto mask = seg_net(input_chip);

            const rgb_pixel random_color(
                rnd.get_random_8bit_number(),
                rnd.get_random_8bit_number(),
                rnd.get_random_8bit_number()
            );

            dlib::matrix<float> resized_mask(
                static_cast<int>(chip_details.rect.height()),
                static_cast<int>(chip_details.rect.width())
            );

            dlib::resize_image(mask, resized_mask);

            for (int r = 0; r < resized_mask.nr(); ++r)
            {
                for (int c = 0; c < resized_mask.nc(); ++c)
                {
                    const auto new_confidence = resized_mask(r, c);
                    if (new_confidence > 0)
                    {
                        const auto y = chip_details.rect.top() + r;
                        const auto x = chip_details.rect.left() + c;
                        if (y >= 0 && y < rgb_label_image.nr() && x >= 0 && x < rgb_label_image.nc())
                        {
                            auto& current_confidence = label_image_confidence(y, x);
                            if (new_confidence > current_confidence)
                            {
                                auto rgb_label = random_color;
                                const auto baseline_confidence = 5;
                                if (new_confidence < baseline_confidence)
                                {
                                    // Scale label intensity if confidence isn't high
                                    rgb_label.red   *= new_confidence / baseline_confidence;
                                    rgb_label.green *= new_confidence / baseline_confidence;
                                    rgb_label.blue  *= new_confidence / baseline_confidence;
                                }
                                rgb_label_image(y, x) = rgb_label;
                                current_confidence = new_confidence;
                            }
                        }
                    }
                }
            }

            const Voc2012class& voc2012_class = find_voc2012_class(
                [&instance](const Voc2012class& candidate) {
                    return candidate.classlabel == instance.label;
                }
            );

            dlib::draw_rectangle(rgb_label_image, instance.rect, voc2012_class.rgb_label, 1);
        }

        // Show the input image on the left, and the predicted RGB labels on the right.
        win.set_image(join_rows(input_image, rgb_label_image));

        if (!instances.empty())
        {
            cout << " in " << file.name() << " - hit enter to process the next image";
            cin.get();
        }
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

