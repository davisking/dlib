// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to do semantic segmentation on an image using net pretrained
    on the PASCAL VOC2012 dataset.

    Instructions:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_semantic_segmentation_train_ex example program.
    3. Run:
       ./dnn_semantic_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_semantic_segmentation_ex example program.
    6. Run:
       ./dnn_semantic_segmentation_ex /path/to/VOC2012-or-other-images
*/

#include "dnn_semantic_segmentation_ex.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

const Voc2012class& find_voc2012_class(const uint16_t& index_label)
{
    return find_voc2012_class(
        [&index_label](const Voc2012class& voc2012class) {
            return index_label == voc2012class.index;
        }
    );
}

inline rgb_pixel index_label_to_rgb_label(uint16_t index_label)
{
    return find_voc2012_class(index_label).rgb_label;
}

void index_label_image_to_rgb_label_image(const matrix<uint16_t>& index_label_image, matrix<rgb_pixel>& rgb_label_image)
{
    const long nr = index_label_image.nr();
    const long nc = index_label_image.nc();

    rgb_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            rgb_label_image(r, c) = index_label_to_rgb_label(index_label_image(r, c));
        }
    }
}

std::string get_most_prominent_non_background_classlabel(const matrix<uint16_t>& index_label_image)
{
    const long nr = index_label_image.nr();
    const long nc = index_label_image.nc();

    std::vector<unsigned int> counters(class_count);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            const uint16_t label = index_label_image(r, c);
            ++counters[label];
        }
    }

    const auto max_element = std::max_element(counters.begin() + 1, counters.end());
    const uint16_t most_prominent_index_label = max_element - counters.begin();

    return find_voc2012_class(most_prominent_index_label).classlabel;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "You call this program like this: " << endl;
        cout << "./dnn_semantic_segmentation_train_ex /path/to/images" << endl;
        cout << endl;
        cout << "You will also need a trained 'voc2012net.dnn' file." << endl;
        cout << "You can either train it yourself (see example program" << endl;
        cout << "dnn_semantic_segmentation_train_ex), or download a" << endl;
        cout << "copy from here: http://dlib.net/files/voc2012net.dnn" << endl;
        return 1;
    }

    anet_type net;
    deserialize("voc2012net.dnn") >> net;

    image_window win;

    matrix<rgb_pixel> input_image;
    matrix<rgb_pixel> rgb_label_image;

    const std::vector<file> files = dlib::get_files_in_directory_tree(argv[1],
        dlib::match_endings(".jpeg *.jpg *.png"));

    cout << "Found " << files.size() << " images, processing..." << endl;

    for (const file& file : files)
    {
        // Load the input image.
        load_image(input_image, file.full_name());

        // Create predictions for each pixel. At this point, the type of each prediction
        // is an index (a value between 0 and 20).
        const matrix<uint16_t> index_label_image = net(input_image);

        // Convert the indexes to RGB values.
        index_label_image_to_rgb_label_image(index_label_image, rgb_label_image);

        // Show the input image on the left, and the predicted RGB labels on the right.
        win.set_image(join_rows(input_image, rgb_label_image));

        // Find the most prominent class label from amongst the per-pixel predictions.
        const std::string classlabel = get_most_prominent_non_background_classlabel(index_label_image);

        cout << file.name() << " : " << classlabel << " - hit enter to process the next image";
        cin.get();
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

