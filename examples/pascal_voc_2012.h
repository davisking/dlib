// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Helper definitions for working with the PASCAL VOC2012 dataset.
*/

#ifndef PASCAL_VOC_2012_H_
#define PASCAL_VOC_2012_H_

#include <dlib/pixel.h>
#include <dlib/dnn.h>

// ----------------------------------------------------------------------------------------

// The PASCAL VOC2012 dataset contains 20 ground-truth classes + background.  Each class
// is represented using an RGB color value.  We associate each class also to an index in the
// range [0, 20], used internally by the network. To convert the ground-truth data to
// something that the network can efficiently digest, we need to be able to map the RGB
// values to the corresponding indexes.

struct Voc2012class {
    Voc2012class(uint16_t index, const dlib::rgb_pixel& rgb_label, const std::string& classlabel)
        : index(index), rgb_label(rgb_label), classlabel(classlabel)
    {}

    // The index of the class. In the PASCAL VOC 2012 dataset, indexes from 0 to 20 are valid.
    const uint16_t index = 0;

    // The corresponding RGB representation of the class.
    const dlib::rgb_pixel rgb_label;

    // The label of the class in plain text.
    const std::string classlabel;
};

namespace {
    constexpr int class_count = 21; // background + 20 classes

    const std::vector<Voc2012class> classes = {
        Voc2012class(0, dlib::rgb_pixel(0, 0, 0), ""), // background

        // The cream-colored `void' label is used in border regions and to mask difficult objects
        // (see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)
        Voc2012class(dlib::loss_multiclass_log_per_pixel_::label_to_ignore,
            dlib::rgb_pixel(224, 224, 192), "border"),

        Voc2012class(1,  dlib::rgb_pixel(128,   0,   0), "aeroplane"),
        Voc2012class(2,  dlib::rgb_pixel(  0, 128,   0), "bicycle"),
        Voc2012class(3,  dlib::rgb_pixel(128, 128,   0), "bird"),
        Voc2012class(4,  dlib::rgb_pixel(  0,   0, 128), "boat"),
        Voc2012class(5,  dlib::rgb_pixel(128,   0, 128), "bottle"),
        Voc2012class(6,  dlib::rgb_pixel(  0, 128, 128), "bus"),
        Voc2012class(7,  dlib::rgb_pixel(128, 128, 128), "car"),
        Voc2012class(8,  dlib::rgb_pixel( 64,   0,   0), "cat"),
        Voc2012class(9,  dlib::rgb_pixel(192,   0,   0), "chair"),
        Voc2012class(10, dlib::rgb_pixel( 64, 128,   0), "cow"),
        Voc2012class(11, dlib::rgb_pixel(192, 128,   0), "diningtable"),
        Voc2012class(12, dlib::rgb_pixel( 64,   0, 128), "dog"),
        Voc2012class(13, dlib::rgb_pixel(192,   0, 128), "horse"),
        Voc2012class(14, dlib::rgb_pixel( 64, 128, 128), "motorbike"),
        Voc2012class(15, dlib::rgb_pixel(192, 128, 128), "person"),
        Voc2012class(16, dlib::rgb_pixel(  0,  64,   0), "pottedplant"),
        Voc2012class(17, dlib::rgb_pixel(128,  64,   0), "sheep"),
        Voc2012class(18, dlib::rgb_pixel(  0, 192,   0), "sofa"),
        Voc2012class(19, dlib::rgb_pixel(128, 192,   0), "train"),
        Voc2012class(20, dlib::rgb_pixel(  0,  64, 128), "tvmonitor"),
    };
}

template <typename Predicate>
const Voc2012class& find_voc2012_class(Predicate predicate)
{
    const auto i = std::find_if(classes.begin(), classes.end(), predicate);

    if (i != classes.end())
    {
        return *i;
    }
    else
    {
        throw std::runtime_error("Unable to find a matching VOC2012 class");
    }
}

// ----------------------------------------------------------------------------------------

// The names of the input image and the associated RGB label image in the PASCAL VOC 2012
// data set.
struct image_info
{
    std::string image_filename;
    std::string class_label_filename;
    std::string instance_label_filename;
};

// Read the list of image files belonging to either the "train", "trainval", or "val" set
// of the PASCAL VOC2012 data.
std::vector<image_info> get_pascal_voc2012_listing(
    const std::string& voc2012_folder,
    const std::string& file = "train" // "train", "trainval", or "val"
)
{
    std::ifstream in(voc2012_folder + "/ImageSets/Segmentation/" + file + ".txt");

    std::vector<image_info> results;

    while (in)
    {
        std::string basename;
        in >> basename;

        if (!basename.empty())
        {
            image_info info;
            info.image_filename          = voc2012_folder + "/JPEGImages/"         + basename + ".jpg";
            info.class_label_filename    = voc2012_folder + "/SegmentationClass/"  + basename + ".png";
            info.instance_label_filename = voc2012_folder + "/SegmentationObject/" + basename + ".png";
            results.push_back(info);
        }
    }

    return results;
}

// Read the list of image files belong to the "train" set of the PASCAL VOC2012 data.
std::vector<image_info> get_pascal_voc2012_train_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "train");
}

// Read the list of image files belong to the "val" set of the PASCAL VOC2012 data.
std::vector<image_info> get_pascal_voc2012_val_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "val");
}

// Given an RGB representation, find the corresponding PASCAL VOC2012 class
// (e.g., 'dog').
const Voc2012class& find_voc2012_class(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(
        [&rgb_label](const Voc2012class& voc2012class)
        {
            return rgb_label == voc2012class.rgb_label;
        }
    );
}

// ----------------------------------------------------------------------------------------

// Convert an RGB class label to an index in the range [0, 20].
inline uint16_t rgb_label_to_index_label(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(rgb_label).index;
}

// Convert an image containing RGB class labels to a corresponding
// image containing indexes in the range [0, 20].
void rgb_label_image_to_index_label_image(
    const dlib::matrix<dlib::rgb_pixel>& rgb_label_image,
    dlib::matrix<uint16_t>& index_label_image
)
{
    const long nr = rgb_label_image.nr();
    const long nc = rgb_label_image.nc();

    index_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r)
    {
        for (long c = 0; c < nc; ++c)
        {
            index_label_image(r, c) = rgb_label_to_index_label(rgb_label_image(r, c));
        }
    }
}

#endif // PASCAL_VOC_2012_H_
