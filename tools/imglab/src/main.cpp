
#include "dlib/data_io.h"
#include "dlib/string.h"
#include "metadata_editor.h"
#include "convert_pascal_xml.h"
#include "convert_pascal_v1.h"
#include "convert_idl.h"
#include <dlib/cmd_line_parser.h>
#include <dlib/svm.h>

#include <iostream>
#include <fstream>
#include <string>
#include <set>

#include <dlib/dir_nav.h>


const char* VERSION = "0.6";



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
    const std::string parent_dir = get_parent_directory(file(filename)).full_name();

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
            const string temp = strip_path(file(parser[i]).full_name(), parent_dir);
            meta.images.push_back(image(temp));
        }
        catch (dlib::file::file_not_found&)
        {
            // then parser[i] should be a directory

            std::vector<file> files = get_files_in_directory_tree(parser[i], 
                                                                  match_endings(".png .PNG .jpeg .JPEG .jpg .JPG .bmp .BMP .dng .DNG"),
                                                                  depth);
            sort(files.begin(), files.end());

            for (unsigned long j = 0; j < files.size(); ++j)
            {
                meta.images.push_back(image(strip_path(files[j].full_name(), parent_dir)));
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
        dlib::image_dataset_metadata::image temp = data.images[i];

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
        {
            std::vector<dlib::image_dataset_metadata::box> boxes;
            // remove other labels
            for (unsigned long j = 0; j < temp.boxes.size(); ++j)
            {
                if (temp.boxes[j].label == label)
                {
                    // put only the boxes with the label we want into boxes
                    boxes.push_back(temp.boxes[j]);
                }
            }
            temp.boxes = boxes;
            data_with.images.push_back(temp);
        }
        else
        {
            data_without.images.push_back(temp);
        }
    }


    save_image_dataset_metadata(data_with, left_substr(parser[0],".") + "_with_"+label + ".xml");
    save_image_dataset_metadata(data_without, left_substr(parser[0],".") + "_without_"+label + ".xml");

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
        }

        // count the number of images for each label
        for (std::set<std::string>::iterator i = temp.begin(); i != temp.end(); ++i)
            image_hits[*i] += 1;
    }

    cout << "Number of images: "<< data.images.size() << endl;
    cout << "Number of different labels: "<< labels.size() << endl << endl;

    for (std::set<std::string>::iterator i = labels.begin(); i != labels.end(); ++i)
    {
        if (i->size() != 0)
        {
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

        parser.set_group_name("Viewing/Editing XML files");
        parser.add_option("l","List all the labels in the given XML file.");
        parser.add_option("stats","List detailed statistics on the object labels in the given XML file.");
        parser.add_option("rename", "Rename all labels of <arg1> to <arg2>.",2);
        parser.add_option("parts","The display will allow image parts to be labeled.  The set of allowable parts "
                          "defined in a space separated list contained in <arg>.",1);
        parser.add_option("rmdiff","Remove boxes marked as difficult.");
        parser.add_option("shuffle","Randomly shuffle the order of the images listed in file <arg>.");
        parser.add_option("seed", "When using --shuffle, set the random seed to the string <arg>.",1);
        parser.add_option("split", "Split the contents of an XML file into two separate files.  One containing the "
            "images with objects labeled <arg> and another file with all the other images.  Additionally, the file "
            "containing the <arg> labeled objects will not contain any other labels other than <arg>. "
            "That is, the images in the first file are stripped of all labels other than the <arg> labels.",1);

        parser.parse(argc, argv);

        const char* singles[] = {"h","c","r","l","convert","parts","rmdiff","seed", "shuffle", "split"};
        parser.check_one_time_options(singles);
        const char* c_sub_ops[] = {"r", "convert"};
        parser.check_sub_options("c", c_sub_ops);
        parser.check_sub_option("shuffle", "seed");
        parser.check_incompatible_options("c", "l");
        parser.check_incompatible_options("c", "rmdiff");
        parser.check_incompatible_options("c", "rename");
        parser.check_incompatible_options("c", "parts");
        parser.check_incompatible_options("l", "rename");
        parser.check_incompatible_options("l", "parts");
        parser.check_incompatible_options("convert", "l");
        parser.check_incompatible_options("convert", "rename");
        parser.check_incompatible_options("convert", "parts");
        parser.check_incompatible_options("rmdiff", "rename");
        const char* convert_args[] = {"pascal-xml","pascal-v1","idl"};
        parser.check_option_arg_range("convert", convert_args);

        if (parser.option("h"))
        {
            cout << "Usage: imglab [options] <image files/directories or XML file>\n";
            parser.print_options(cout);
            cout << endl << endl;
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
                std::vector<dlib::image_dataset_metadata::box> boxes;
                for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
                {
                    if (!data.images[i].boxes[j].difficult)
                        boxes.push_back(data.images[i].boxes[j]);
                }
                data.images[i].boxes = boxes;
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

        if (parser.option("split"))
        {
            return split_dataset(parser);
        }

        if (parser.option("shuffle"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cerr << "The -shuffle option requires you to give one XML file on the command line." << endl;
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
        }
    }
    catch (exception& e)
    {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
}

// ----------------------------------------------------------------------------------------

