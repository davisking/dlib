
#include "dlib/data_io.h"
#include "dlib/string.h"
#include "metadata_editor.h"
#include "convert_pascal_xml.h"
#include "convert_pascal_v1.h"
#include "convert_idl.h"
#include "cluster.h"
#include "flip_dataset.h"
#include <dlib/cmd_line_parser.h>
#include <dlib/image_transforms.h>
#include <dlib/svm.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/md5.h>

#include <iostream>
#include <fstream>
#include <string>
#include <set>

#include <dlib/dir_nav.h>


const char* VERSION = "1.16";



using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

void create_new_dataset (
    const command_line_parser& parser
)
{
    using namespace dlib::image_dataset_metadata;

    const std::string filename = parser.option("c").argument();
    // make sure the file exists so we can use the get_parent_directory() command to
    // figure out it's parent directory.
    make_empty_file(filename);
    const std::string parent_dir = get_parent_directory(file(filename));

    unsigned long depth = 0;
    if (parser.option("r"))
        depth = 30;

    dataset meta;
    meta.name = "imglab dataset";
    meta.comment = "Created by imglab tool.";
    for (unsigned long i = 0; i < parser.number_of_arguments(); ++i)
    {
        try
        {
            const string temp = strip_path(file(parser[i]), parent_dir);
            meta.images.push_back(image(temp));
        }
        catch (dlib::file::file_not_found&)
        {
            // then parser[i] should be a directory

            std::vector<file> files = get_files_in_directory_tree(parser[i], 
                                                                  match_endings(".png .PNG .jpeg .JPEG .jpg .JPG .bmp .BMP .dng .DNG .gif .GIF"),
                                                                  depth);
            sort(files.begin(), files.end());

            for (unsigned long j = 0; j < files.size(); ++j)
            {
                meta.images.push_back(image(strip_path(files[j], parent_dir)));
            }
        }
    }

    save_image_dataset_metadata(meta, filename);
}

// ----------------------------------------------------------------------------------------

int split_dataset (
    const command_line_parser& parser
)
{
    if (parser.number_of_arguments() != 1)
    {
        cerr << "The --split option requires you to give one XML file on the command line." << endl;
        return EXIT_FAILURE;
    }

    const std::string label = parser.option("split").argument();

    dlib::image_dataset_metadata::dataset data, data_with, data_without;
    load_image_dataset_metadata(data, parser[0]);

    data_with.name = data.name;
    data_with.comment = data.comment;
    data_without.name = data.name;
    data_without.comment = data.comment;

    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        auto&& temp = data.images[i];

        bool has_the_label = false;
        // check for the label we are looking for
        for (unsigned long j = 0; j < temp.boxes.size(); ++j)
        {
            if (temp.boxes[j].label == label)
            {
                has_the_label = true;
                break;
            }
        }

        if (has_the_label)
            data_with.images.push_back(temp);
        else
            data_without.images.push_back(temp);
    }


    save_image_dataset_metadata(data_with, left_substr(parser[0],".") + "_with_"+label + ".xml");
    save_image_dataset_metadata(data_without, left_substr(parser[0],".") + "_without_"+label + ".xml");

    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

int make_train_test_splits (
    const command_line_parser& parser
)
{
    if (parser.number_of_arguments() != 1)
    {
        cerr << "The --split-train-test option requires you to give one XML file on the command line." << endl;
        return EXIT_FAILURE;
    }

    const double train_frac = get_option(parser, "split-train-test", 0.5);

    dlib::image_dataset_metadata::dataset data, data_train, data_test;
    load_image_dataset_metadata(data, parser[0]);

    data_train.name = data.name;
    data_train.comment = data.comment;
    data_test.name = data.name;
    data_test.comment = data.comment;

    const unsigned long num_train_images = static_cast<unsigned long>(std::round(train_frac*data.images.size()));

    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        if (i < num_train_images)
            data_train.images.push_back(data.images[i]);
        else
            data_test.images.push_back(data.images[i]);
    }

    save_image_dataset_metadata(data_train, left_substr(parser[0],".") + "_train.xml");
    save_image_dataset_metadata(data_test, left_substr(parser[0],".") + "_test.xml");

    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

void print_all_labels (
    const dlib::image_dataset_metadata::dataset& data
)
{
    std::set<std::string> labels;
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            labels.insert(data.images[i].boxes[j].label);
        }
    }

    for (std::set<std::string>::iterator i = labels.begin(); i != labels.end(); ++i)
    {
        if (i->size() != 0)
        {
            cout << *i << endl;
        }
    }
}

// ----------------------------------------------------------------------------------------

void print_all_label_stats (
    const dlib::image_dataset_metadata::dataset& data
)
{
    std::map<std::string, running_stats<double> > area_stats, aspect_ratio;
    std::map<std::string, int> image_hits;
    std::set<std::string> labels;
    unsigned long num_unignored_boxes = 0;
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        std::set<std::string> temp;
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            labels.insert(data.images[i].boxes[j].label);
            temp.insert(data.images[i].boxes[j].label);

            area_stats[data.images[i].boxes[j].label].add(data.images[i].boxes[j].rect.area());
            aspect_ratio[data.images[i].boxes[j].label].add(data.images[i].boxes[j].rect.width()/
                                                    (double)data.images[i].boxes[j].rect.height());

            if (!data.images[i].boxes[j].ignore)
                ++num_unignored_boxes;
        }

        // count the number of images for each label
        for (std::set<std::string>::iterator i = temp.begin(); i != temp.end(); ++i)
            image_hits[*i] += 1;
    }

    cout << "Number of images: "<< data.images.size() << endl;
    cout << "Number of different labels: "<< labels.size() << endl;
    cout << "Number of non-ignored boxes: " << num_unignored_boxes << endl << endl;

    for (std::set<std::string>::iterator i = labels.begin(); i != labels.end(); ++i)
    {
        if (i->size() == 0)
            cout << "Unlabeled Boxes:" << endl;
        else
            cout << "Label: "<< *i << endl;
        cout << "   number of images:      " << image_hits[*i] << endl;
        cout << "   number of occurrences: " << area_stats[*i].current_n() << endl;
        cout << "   min box area:    " << area_stats[*i].min() << endl;
        cout << "   max box area:    " << area_stats[*i].max() << endl;
        cout << "   mean box area:   " << area_stats[*i].mean() << endl;
        cout << "   stddev box area: " << area_stats[*i].stddev() << endl;
        cout << "   mean width/height ratio:   " << aspect_ratio[*i].mean() << endl;
        cout << "   stddev width/height ratio: " << aspect_ratio[*i].stddev() << endl;
        cout << endl;
    }
}

// ----------------------------------------------------------------------------------------

void rename_labels (
    dlib::image_dataset_metadata::dataset& data,
    const std::string& from,
    const std::string& to
)
{
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            if (data.images[i].boxes[j].label == from)
                data.images[i].boxes[j].label = to;
        }
    }

}

// ----------------------------------------------------------------------------------------

void ignore_labels (
    dlib::image_dataset_metadata::dataset& data,
    const std::string& label
)
{
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            if (data.images[i].boxes[j].label == label)
                data.images[i].boxes[j].ignore = true;
        }
    }
}

// ----------------------------------------------------------------------------------------

void merge_metadata_files (
    const command_line_parser& parser
)
{
    image_dataset_metadata::dataset src, dest;
    load_image_dataset_metadata(src, parser.option("add").argument(0));
    load_image_dataset_metadata(dest, parser.option("add").argument(1));

    std::map<string,image_dataset_metadata::image> merged_data;
    for (unsigned long i = 0; i < dest.images.size(); ++i)
        merged_data[dest.images[i].filename] = dest.images[i];
    // now add in the src data and overwrite anything if there are duplicate entries.
    for (unsigned long i = 0; i < src.images.size(); ++i)
        merged_data[src.images[i].filename] = src.images[i];

    // copy merged data into dest
    dest.images.clear();
    for (std::map<string,image_dataset_metadata::image>::const_iterator i = merged_data.begin(); 
        i != merged_data.end(); ++i)
    {
        dest.images.push_back(i->second);
    }

    save_image_dataset_metadata(dest, "merged.xml");
}

// ----------------------------------------------------------------------------------------

void rotate_dataset(const command_line_parser& parser)
{
    image_dataset_metadata::dataset metadata;
    const string datasource = parser[0];
    load_image_dataset_metadata(metadata,datasource);

    double angle = get_option(parser, "rotate", 0);

    // Set the current directory to be the one that contains the
    // metadata file. We do this because the file might contain
    // file paths which are relative to this folder.
    set_current_dir(get_parent_directory(file(datasource)));

    const string file_prefix = "rotated_"+ cast_to_string(angle) + "_";
    const string metadata_filename = get_parent_directory(file(datasource)).full_name() +
        directory::get_separator() + file_prefix + file(datasource).name();


    array2d<rgb_pixel> img, temp;
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        file f(metadata.images[i].filename);
        string filename = get_parent_directory(f).full_name() + directory::get_separator() + file_prefix + to_png_name(f.name());

        load_image(img, metadata.images[i].filename);
        const point_transform_affine tran = rotate_image(img, temp, angle*pi/180);
        if (parser.option("jpg"))
        {
            filename = to_jpg_name(filename);
            save_jpeg(temp, filename,JPEG_QUALITY);
        }
        else
        {
            save_png(temp, filename);
        }

        rectangle_transform rtran = tran;
        for (unsigned long j = 0; j < metadata.images[i].boxes.size(); ++j)
        {
            metadata.images[i].boxes[j].rect = rtran(metadata.images[i].boxes[j].rect);

            for (auto& p : metadata.images[i].boxes[j].parts)
                p.second = tran(p.second);
        }

        metadata.images[i].filename = filename;
    }

    save_image_dataset_metadata(metadata, metadata_filename);
}

// ----------------------------------------------------------------------------------------

int resample_dataset(const command_line_parser& parser)
{
    if (parser.number_of_arguments() != 1)
    {
        cerr << "The --resample option requires you to give one XML file on the command line." << endl;
        return EXIT_FAILURE;
    }

    const size_t obj_size = get_option(parser,"cropped-object-size",100*100); 
    const double margin_scale =  get_option(parser,"crop-size",2.5); // cropped image will be this times wider than the object.
    const unsigned long min_object_size = get_option(parser,"min-object-size",1);
    const bool one_object_per_image = parser.option("one-object-per-image");

    dlib::image_dataset_metadata::dataset data, resampled_data;
    std::ostringstream sout;
    sout << "\nThe --resample parameters which generated this dataset were:" << endl;
    sout << "   cropped-object-size: "<< obj_size << endl;
    sout << "   crop-size: "<< margin_scale << endl;
    sout << "   min-object-size: "<< min_object_size << endl;
    if (one_object_per_image)
        sout << "   one_object_per_image: true" << endl;
    resampled_data.comment = data.comment + sout.str();
    resampled_data.name = data.name + " RESAMPLED";

    load_image_dataset_metadata(data, parser[0]);
    locally_change_current_dir chdir(get_parent_directory(file(parser[0])));
    dlib::rand rnd;

    const size_t image_size = std::round(std::sqrt(obj_size*margin_scale*margin_scale));
    const chip_dims cdims(image_size, image_size);

    console_progress_indicator pbar(data.images.size());
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        // don't even bother loading images that don't have objects.
        if (data.images[i].boxes.size() == 0)
            continue;

        pbar.print_status(i);
        array2d<rgb_pixel> img, chip;
        load_image(img, data.images[i].filename);


        // figure out what chips we want to take from this image
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            const rectangle rect = data.images[i].boxes[j].rect;
            if (data.images[i].boxes[j].ignore || rect.area() < min_object_size)
                continue;

            const auto max_dim = std::max(rect.width(), rect.height());

            const double rand_scale_perturb = 1 - 0.3*(rnd.get_random_double()-0.5);
            const rectangle crop_rect = centered_rect(rect, max_dim*margin_scale*rand_scale_perturb, max_dim*margin_scale*rand_scale_perturb);

            const rectangle_transform tform = get_mapping_to_chip(chip_details(crop_rect, cdims));
            extract_image_chip(img, chip_details(crop_rect, cdims), chip);

            image_dataset_metadata::image dimg;
            // Now transform the boxes to the crop and also mark them as ignored if they
            // have already been cropped out or are outside the crop.
            for (size_t k = 0; k < data.images[i].boxes.size(); ++k)
            {
                image_dataset_metadata::box box = data.images[i].boxes[k];
                // ignore boxes outside the cropped image
                if (crop_rect.intersect(box.rect).area() == 0)
                    continue;

                // mark boxes we include in the crop as ignored.  Also mark boxes that
                // aren't totally within the crop as ignored.
                if (crop_rect.contains(grow_rect(box.rect,10)) && (!one_object_per_image || k==j))
                    data.images[i].boxes[k].ignore = true;
                else
                    box.ignore = true;

                if (box.rect.area() < min_object_size)
                    box.ignore = true;

                box.rect = tform(box.rect);
                for (auto&& p : box.parts)
                    p.second = tform.get_tform()(p.second);
                dimg.boxes.push_back(box);
            }
            // Put a 64bit hash of the image data into the name to make sure there are no
            // file name conflicts.
            std::ostringstream sout;
            sout << hex << murmur_hash3_128bit(&chip[0][0], chip.size()*sizeof(chip[0][0])).second;
            dimg.filename = data.images[i].filename + "_RESAMPLED_"+sout.str()+".png";

            if (parser.option("jpg"))
            {
                dimg.filename = to_jpg_name(dimg.filename);
                save_jpeg(chip,dimg.filename, JPEG_QUALITY);
            }
            else
            {
                save_png(chip,dimg.filename);
            }
            resampled_data.images.push_back(dimg);
        }
    }

    save_image_dataset_metadata(resampled_data, parser[0] + ".RESAMPLED.xml");

    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

int tile_dataset(const command_line_parser& parser)
{
    if (parser.number_of_arguments() != 1)
    {
        cerr << "The --tile option requires you to give one XML file on the command line." << endl;
        return EXIT_FAILURE;
    }

    string out_image = parser.option("tile").argument();
    string ext = right_substr(out_image,".");
    if (ext != "png" && ext != "jpg")
    {
        cerr << "The output image file must have either .png or .jpg extension." << endl;
        return EXIT_FAILURE;
    }

    const unsigned long chip_size = get_option(parser, "size", 8000);

    dlib::image_dataset_metadata::dataset data;
    load_image_dataset_metadata(data, parser[0]);
    locally_change_current_dir chdir(get_parent_directory(file(parser[0])));
    dlib::array<array2d<rgb_pixel> > images;
    console_progress_indicator pbar(data.images.size());
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        // don't even bother loading images that don't have objects.
        if (data.images[i].boxes.size() == 0)
            continue;

        pbar.print_status(i);
        array2d<rgb_pixel> img;
        load_image(img, data.images[i].filename);

        // figure out what chips we want to take from this image
        std::vector<chip_details> dets;
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            if (data.images[i].boxes[j].ignore)
                continue;

            rectangle rect = data.images[i].boxes[j].rect;
            dets.push_back(chip_details(rect, chip_size));
        }
        // Now grab all those chips at once.
        dlib::array<array2d<rgb_pixel> > chips;
        extract_image_chips(img, dets, chips);
        // and put the chips into the output.
        for (unsigned long j = 0; j < chips.size(); ++j)
            images.push_back(chips[j]);
    }

    chdir.revert();

    if (ext == "png")
        save_png(tile_images(images), out_image);
    else
        save_jpeg(tile_images(images), out_image);

    return EXIT_SUCCESS;
}


// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {

        command_line_parser parser;

        parser.add_option("h","Displays this information.");
        parser.add_option("v","Display version.");

        parser.set_group_name("Creating XML files");
        parser.add_option("c","Create an XML file named <arg> listing a set of images.",1);
        parser.add_option("r","Search directories recursively for images.");
        parser.add_option("convert","Convert foreign image Annotations from <arg> format to the imglab format. "
                          "Supported formats: pascal-xml, pascal-v1, idl.",1);

        parser.set_group_name("Viewing XML files");
        parser.add_option("tile","Chip out all the objects and save them as one big image called <arg>.",1);
        parser.add_option("size","When using --tile or --cluster, make each extracted object contain "
                                 "about <arg> pixels (default 8000).",1);
        parser.add_option("l","List all the labels in the given XML file.");
        parser.add_option("stats","List detailed statistics on the object labels in the given XML file.");
        parser.add_option("files","List all the files in the given XML file.");

        parser.set_group_name("Editing/Transforming XML datasets");
        parser.add_option("rename", "Rename all labels of <arg1> to <arg2>.",2);
        parser.add_option("parts","The display will allow image parts to be labeled.  The set of allowable parts "
                          "is defined by <arg> which should be a space separated list of parts.",1);
        parser.add_option("rmempty","Remove all images that don't contain non-ignored annotations and save the results to a new XML file.");
        parser.add_option("rmdupes","Remove duplicate images from the dataset.  This is done by comparing "
                                    "the md5 hash of each image file and removing duplicate images. " );
        parser.add_option("rmdiff","Set the ignored flag to true for boxes marked as difficult.");
        parser.add_option("rmtrunc","Set the ignored flag to true for boxes that are partially outside the image.");
        parser.add_option("box-images","Add a box to each image that contains the entire image.");
        parser.add_option("sort-num-objects","Sort the images listed an XML file so images with many objects are listed first.");
        parser.add_option("sort","Alphabetically sort the images in an XML file.");
        parser.add_option("shuffle","Randomly shuffle the order of the images listed in an XML file.");
        parser.add_option("seed", "When using --shuffle, set the random seed to the string <arg>.",1);
        parser.add_option("split", "Split the contents of an XML file into two separate files.  One containing the "
            "images with objects labeled <arg> and another file with all the other images. ",1);
        parser.add_option("split-train-test", "Split the contents of an XML file into two separate files.  A training "
            "file containing <arg> fraction of the images and a testing file containing the remaining (1-<arg>) images. "
            "The partitioning is done deterministically by putting the first images in the input xml file into the training split "
            "and the later images into the test split.",1);
        parser.add_option("add", "Add the image metadata from <arg1> into <arg2>.  If any of the image "
                                 "tags are in both files then the ones in <arg2> are deleted and replaced with the "
                                 "image tags from <arg1>.  The results are saved into merged.xml and neither <arg1> or "
                                 "<arg2> files are modified.",2);
        parser.add_option("flip", "Read an XML image dataset from the <arg> XML file and output a left-right flipped "
                                  "version of the dataset and an accompanying flipped XML file named flipped_<arg>. " 
                                  "We also adjust object part labels after flipping so that the new flipped dataset "
                                  "has the same average part layout as the source dataset." ,1);
        parser.add_option("flip-basic", "This option is just like --flip, except we don't adjust any object part labels after flipping. "
                                        "The parts are instead simply mirrored to the flipped dataset.", 1);
        parser.add_option("rotate", "Read an XML image dataset and output a copy that is rotated counter clockwise by <arg> degrees. "
                                  "The output is saved to an XML file prefixed with rotated_<arg>.",1);
        parser.add_option("cluster", "Cluster all the objects in an XML file into <arg> different clusters and save "
                                     "the results as cluster_###.xml and cluster_###.jpg files.",1);
        parser.add_option("ignore", "Mark boxes labeled as <arg> as ignored.  The resulting XML file is output as a separate file and the original is not modified.",1);
        parser.add_option("rmlabel","Remove all boxes labeled <arg> and save the results to a new XML file.",1);
        parser.add_option("rm-other-labels","Remove all boxes not labeled <arg> and save the results to a new XML file.",1);
        parser.add_option("rmignore","Remove all boxes marked ignore and save the results to a new XML file.");
        parser.add_option("rm-if-overlaps","Remove all boxes labeled <arg> if they overlap any box not labeled <arg> and save the results to a new XML file.",1);
        parser.add_option("jpg", "When saving images to disk, write them as jpg files instead of png.");

        parser.set_group_name("Cropping sub images");
        parser.add_option("resample", "Crop out images that are centered on each object in the dataset. "
                                      "The output is a new XML dataset."); 
        parser.add_option("cropped-object-size", "When doing --resample, make the cropped objects contain about <arg> pixels (default 10000).",1);
        parser.add_option("min-object-size", "When doing --resample, skip objects that have fewer than <arg> pixels in them (default 1).",1);
        parser.add_option("crop-size", "When doing --resample, the entire cropped image will be <arg> times wider than the object (default 2.5).",1); 
        parser.add_option("one-object-per-image", "When doing --resample, only include one non-ignored object per image (i.e. the central object).");



        parser.parse(argc, argv);

        const char* singles[] = {"h","c","r","l","files","convert","parts","rmdiff", "rmtrunc", "rmdupes", "seed", "shuffle", "split", "add", 
                                 "flip-basic", "flip", "rotate", "tile", "size", "cluster", "resample", "min-object-size", "rmempty",
                                 "crop-size", "cropped-object-size", "rmlabel", "rm-other-labels", "rm-if-overlaps", "sort-num-objects", 
                                 "one-object-per-image", "jpg", "rmignore", "sort", "split-train-test", "box-images"};
        parser.check_one_time_options(singles);
        const char* c_sub_ops[] = {"r", "convert"};
        parser.check_sub_options("c", c_sub_ops);
        parser.check_sub_option("shuffle", "seed");
        const char* resample_sub_ops[] = {"min-object-size", "crop-size", "cropped-object-size", "one-object-per-image"};
        parser.check_sub_options("resample", resample_sub_ops);
        const char* size_parent_ops[] = {"tile", "cluster"};
        parser.check_sub_options(size_parent_ops, "size");
        parser.check_incompatible_options("c", "l");
        parser.check_incompatible_options("c", "files");
        parser.check_incompatible_options("c", "rmdiff");
        parser.check_incompatible_options("c", "rmempty");
        parser.check_incompatible_options("c", "rmlabel");
        parser.check_incompatible_options("c", "rm-other-labels");
        parser.check_incompatible_options("c", "rmignore");
        parser.check_incompatible_options("c", "rm-if-overlaps");
        parser.check_incompatible_options("c", "rmdupes");
        parser.check_incompatible_options("c", "rmtrunc");
        parser.check_incompatible_options("c", "box-images");
        parser.check_incompatible_options("c", "add");
        parser.check_incompatible_options("c", "flip");
        parser.check_incompatible_options("c", "flip-basic");
        parser.check_incompatible_options("flip", "flip-basic");
        parser.check_incompatible_options("c", "rotate");
        parser.check_incompatible_options("c", "rename");
        parser.check_incompatible_options("c", "ignore");
        parser.check_incompatible_options("c", "parts");
        parser.check_incompatible_options("c", "tile");
        parser.check_incompatible_options("c", "cluster");
        parser.check_incompatible_options("c", "resample");
        parser.check_incompatible_options("l", "rename");
        parser.check_incompatible_options("l", "ignore");
        parser.check_incompatible_options("l", "add");
        parser.check_incompatible_options("l", "parts");
        parser.check_incompatible_options("l", "flip");
        parser.check_incompatible_options("l", "flip-basic");
        parser.check_incompatible_options("l", "rotate");
        parser.check_incompatible_options("files", "rename");
        parser.check_incompatible_options("files", "ignore");
        parser.check_incompatible_options("files", "add");
        parser.check_incompatible_options("files", "parts");
        parser.check_incompatible_options("files", "flip");
        parser.check_incompatible_options("files", "flip-basic");
        parser.check_incompatible_options("files", "rotate");
        parser.check_incompatible_options("add", "flip");
        parser.check_incompatible_options("add", "flip-basic");
        parser.check_incompatible_options("add", "rotate");
        parser.check_incompatible_options("add", "tile");
        parser.check_incompatible_options("flip", "tile");
        parser.check_incompatible_options("flip-basic", "tile");
        parser.check_incompatible_options("rotate", "tile");
        parser.check_incompatible_options("cluster", "tile");
        parser.check_incompatible_options("resample", "tile");
        parser.check_incompatible_options("flip", "cluster");
        parser.check_incompatible_options("flip-basic", "cluster");
        parser.check_incompatible_options("rotate", "cluster");
        parser.check_incompatible_options("add", "cluster");
        parser.check_incompatible_options("flip", "resample");
        parser.check_incompatible_options("flip-basic", "resample");
        parser.check_incompatible_options("rotate", "resample");
        parser.check_incompatible_options("add", "resample");
        parser.check_incompatible_options("shuffle", "tile");
        parser.check_incompatible_options("sort-num-objects", "tile");
        parser.check_incompatible_options("sort", "tile");
        parser.check_incompatible_options("convert", "l");
        parser.check_incompatible_options("convert", "files");
        parser.check_incompatible_options("convert", "rename");
        parser.check_incompatible_options("convert", "ignore");
        parser.check_incompatible_options("convert", "parts");
        parser.check_incompatible_options("convert", "cluster");
        parser.check_incompatible_options("convert", "resample");
        parser.check_incompatible_options("rmdiff", "rename");
        parser.check_incompatible_options("rmdiff", "ignore");
        parser.check_incompatible_options("rmempty", "ignore");
        parser.check_incompatible_options("rmempty", "rename");
        parser.check_incompatible_options("rmlabel", "ignore");
        parser.check_incompatible_options("rmlabel", "rename");
        parser.check_incompatible_options("rm-other-labels", "ignore");
        parser.check_incompatible_options("rm-other-labels", "rename");
        parser.check_incompatible_options("rmignore", "ignore");
        parser.check_incompatible_options("rmignore", "rename");
        parser.check_incompatible_options("rm-if-overlaps", "ignore");
        parser.check_incompatible_options("rm-if-overlaps", "rename");
        parser.check_incompatible_options("rmdupes", "rename");
        parser.check_incompatible_options("rmdupes", "ignore");
        parser.check_incompatible_options("rmtrunc", "rename");
        parser.check_incompatible_options("rmtrunc", "ignore");
        parser.check_incompatible_options("box-images", "rename");
        parser.check_incompatible_options("box-images", "ignore");
        const char* convert_args[] = {"pascal-xml","pascal-v1","idl"};
        parser.check_option_arg_range("convert", convert_args);
        parser.check_option_arg_range("cluster", 2, 999);
        parser.check_option_arg_range("rotate", -360, 360);
        parser.check_option_arg_range("size", 10*10, 1000*1000);
        parser.check_option_arg_range("min-object-size", 1, 10000*10000);
        parser.check_option_arg_range("cropped-object-size", 4, 10000*10000);
        parser.check_option_arg_range("crop-size", 1.0, 100.0);
        parser.check_option_arg_range("split-train-test", 0.0, 1.0);

        if (parser.option("h"))
        {
            cout << "Usage: imglab [options] <image files/directories or XML file>\n";
            parser.print_options(cout);
            cout << endl << endl;
            return EXIT_SUCCESS;
        }

        if (parser.option("add"))
        {
            merge_metadata_files(parser);
            return EXIT_SUCCESS;
        }

        if (parser.option("flip") || parser.option("flip-basic"))
        {
            flip_dataset(parser);
            return EXIT_SUCCESS;
        }

        if (parser.option("rotate"))
        {
            rotate_dataset(parser);
            return EXIT_SUCCESS;
        }

        if (parser.option("v"))
        {
            cout << "imglab v" << VERSION 
                 << "\nCompiled: " << __TIME__ << " " << __DATE__ 
                 << "\nWritten by Davis King\n";
            cout << "Check for updates at http://dlib.net\n\n";
            return EXIT_SUCCESS;
        }

        if (parser.option("tile"))
        {
            return tile_dataset(parser);
        }

        if (parser.option("cluster"))
        {
            return cluster_dataset(parser);
        }

        if (parser.option("resample"))
        {
            return resample_dataset(parser);
        }

        if (parser.option("c"))
        {
            if (parser.option("convert"))
            {
                if (parser.option("convert").argument() == "pascal-xml")
                    convert_pascal_xml(parser);
                else if (parser.option("convert").argument() == "pascal-v1")
                    convert_pascal_v1(parser);
                else if (parser.option("convert").argument() == "idl")
                    convert_idl(parser);
            }
            else
            {
                create_new_dataset(parser);
            }
            return EXIT_SUCCESS;
        }
        
        if (parser.option("rmdiff"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rmdiff option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            for (unsigned long i = 0; i < data.images.size(); ++i)
            {
                for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
                {
                    if (data.images[i].boxes[j].difficult)
                        data.images[i].boxes[j].ignore = true;
                }
            }
            save_image_dataset_metadata(data, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("rmempty"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rmempty option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data, data2;
            load_image_dataset_metadata(data, parser[0]);

            data2 = data;
            data2.images.clear();
            for (unsigned long i = 0; i < data.images.size(); ++i)
            {
                bool has_label = false;
                for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
                {
                    if (!data.images[i].boxes[j].ignore)
                        has_label = true;
                }
                if (has_label)
                    data2.images.push_back(data.images[i]);
            }
            save_image_dataset_metadata(data2, parser[0] + ".rmempty.xml");
            return EXIT_SUCCESS;
        }

        if (parser.option("rmlabel"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rmlabel option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);

            const auto label = parser.option("rmlabel").argument();

            for (auto&& img : data.images)
            {
                std::vector<dlib::image_dataset_metadata::box> boxes;
                for (auto&& b : img.boxes)
                {
                    if (b.label != label)
                        boxes.push_back(b);
                }
                img.boxes = boxes;
            }

            save_image_dataset_metadata(data, parser[0] + ".rmlabel-"+label+".xml");
            return EXIT_SUCCESS;
        }

        if (parser.option("rm-other-labels"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rm-other-labels option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);

            const auto labels = parser.option("rm-other-labels").argument();
            // replace comma by dash to form the file name
            std::string strlabels = labels;
            std::replace(strlabels.begin(), strlabels.end(), ',', '-');
            std::vector<string> all_labels = split(labels, ",");
            for (auto&& img : data.images)
            {
                std::vector<dlib::image_dataset_metadata::box> boxes;
                for (auto&& b : img.boxes)
                {
                    if (std::find(all_labels.begin(), all_labels.end(), b.label) != all_labels.end())
                        boxes.push_back(b);
                }
                img.boxes = boxes;
            }

            save_image_dataset_metadata(data, parser[0] + ".rm-other-labels-"+ strlabels +".xml");
            return EXIT_SUCCESS;
        }

        if (parser.option("rmignore"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rmignore option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);

            for (auto&& img : data.images)
            {
                std::vector<dlib::image_dataset_metadata::box> boxes;
                for (auto&& b : img.boxes)
                {
                    if (!b.ignore)
                        boxes.push_back(b);
                }
                img.boxes = boxes;
            }

            save_image_dataset_metadata(data, parser[0] + ".rmignore.xml");
            return EXIT_SUCCESS;
        }

        if (parser.option("rm-if-overlaps"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rm-if-overlaps option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);

            const auto label = parser.option("rm-if-overlaps").argument();

            test_box_overlap overlaps(0.5);

            for (auto&& img : data.images)
            {
                std::vector<dlib::image_dataset_metadata::box> boxes;
                for (auto&& b : img.boxes)
                {
                    if (b.label != label)
                    {
                        boxes.push_back(b);
                    }
                    else
                    {
                        bool has_overlap = false;
                        for (auto&& b2 : img.boxes)
                        {
                            if (b2.label != label && overlaps(b2.rect, b.rect))
                            {
                                has_overlap = true;
                                break;
                            }
                        }
                        if (!has_overlap)
                            boxes.push_back(b);
                    }
                }
                img.boxes = boxes;
            }

            save_image_dataset_metadata(data, parser[0] + ".rm-if-overlaps-"+label+".xml");
            return EXIT_SUCCESS;
        }

        if (parser.option("rmdupes"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rmdupes option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data, data_out;
            std::set<std::string> hashes;
            load_image_dataset_metadata(data, parser[0]);
            data_out = data;
            data_out.images.clear();

            for (unsigned long i = 0; i < data.images.size(); ++i)
            {
                ifstream fin(data.images[i].filename.c_str(), ios::binary);
                string hash = md5(fin);
                if (hashes.count(hash) == 0)
                {
                    hashes.insert(hash);
                    data_out.images.push_back(data.images[i]);
                }
            }
            save_image_dataset_metadata(data_out, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("box-images"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --box-images option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset   data;
            load_image_dataset_metadata(data, parser[0]);
            {
                locally_change_current_dir chdir(get_parent_directory(file(parser[0])));
                parallel_for(0, data.images.size(), [&](long i) 
                {
                    array2d<unsigned char> img;
                    load_image(img, data.images[i].filename);
                    data.images[i].boxes.emplace_back(get_rect(img));
                });
            }
            save_image_dataset_metadata(data, parser[0]+".boxed.xml");
            return EXIT_SUCCESS;
        }

        if (parser.option("rmtrunc"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rmtrunc option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            {
                locally_change_current_dir chdir(get_parent_directory(file(parser[0])));
                for (unsigned long i = 0; i < data.images.size(); ++i)
                {
                    array2d<unsigned char> img;
                    load_image(img, data.images[i].filename);
                    const rectangle area = get_rect(img);
                    for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
                    {
                        if (!area.contains(data.images[i].boxes[j].rect))
                            data.images[i].boxes[j].ignore = true;
                    }
                }
            }
            save_image_dataset_metadata(data, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("l"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The -l option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            print_all_labels(data);
            return EXIT_SUCCESS;
        }

        if (parser.option("files"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --files option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            for (size_t i = 0; i < data.images.size(); ++i)
                cout << data.images[i].filename << "\n";
            return EXIT_SUCCESS;
        }

        if (parser.option("split"))
        {
            return split_dataset(parser);
        }

        if (parser.option("split-train-test"))
        {
            return make_train_test_splits(parser);
        }

        if (parser.option("shuffle"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --shuffle option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            const string default_seed = cast_to_string(time(0));
            const string seed = get_option(parser, "seed", default_seed);
            dlib::rand rnd(seed);
            randomize_samples(data.images, rnd);
            save_image_dataset_metadata(data, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("sort-num-objects"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --sort-num-objects option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            std::sort(data.images.rbegin(),  data.images.rend(), 
                [](const image_dataset_metadata::image& a, const image_dataset_metadata::image& b) { return a.boxes.size() < b.boxes.size(); });
            save_image_dataset_metadata(data, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("sort"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --sort option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            std::sort(data.images.begin(),  data.images.end(), 
                [](const image_dataset_metadata::image& a, const image_dataset_metadata::image& b) { return a.filename < b.filename; });
            save_image_dataset_metadata(data, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("stats"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --stats option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            print_all_label_stats(data);
            return EXIT_SUCCESS;
        }

        if (parser.option("rename"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --rename option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            for (unsigned long i = 0; i < parser.option("rename").count(); ++i)
            {
                rename_labels(data, parser.option("rename").argument(0,i), parser.option("rename").argument(1,i));
            }
            save_image_dataset_metadata(data, parser[0]);
            return EXIT_SUCCESS;
        }

        if (parser.option("ignore"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The --ignore option requires you to give one XML file on the command line." << endl;
                return EXIT_FAILURE;
            }

            dlib::image_dataset_metadata::dataset data;
            load_image_dataset_metadata(data, parser[0]);
            for (unsigned long i = 0; i < parser.option("ignore").count(); ++i)
            {
                ignore_labels(data, parser.option("ignore").argument());
            }
            save_image_dataset_metadata(data, parser[0]+".ignored.xml");
            return EXIT_SUCCESS;
        }

        if (parser.number_of_arguments() == 1)
        {
            metadata_editor editor(parser[0]);
            if (parser.option("parts"))
            {
                std::vector<string> parts = split(parser.option("parts").argument());
                for (unsigned long i = 0; i < parts.size(); ++i)
                {
                    editor.add_labelable_part_name(parts[i]);
                }
            }
            editor.wait_until_closed();
            return EXIT_SUCCESS;
        }

        cout << "Invalid command, give -h to see options." << endl;
        return EXIT_FAILURE;
    }
    catch (exception& e)
    {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
}

// ----------------------------------------------------------------------------------------

