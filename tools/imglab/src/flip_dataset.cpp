// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "flip_dataset.h"
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <string>
#include "common.h"
#include <dlib/image_transforms.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

void flip_dataset(const command_line_parser& parser)
{
    image_dataset_metadata::dataset metadata;
    const string datasource = parser.option("flip").argument();
    load_image_dataset_metadata(metadata,datasource);

    // Set the current directory to be the one that contains the
    // metadata file. We do this because the file might contain
    // file paths which are relative to this folder.
    set_current_dir(get_parent_directory(file(datasource)));

    const string metadata_filename = get_parent_directory(file(datasource)).full_name() +
        directory::get_separator() + "flipped_" + file(datasource).name();


    array2d<rgb_pixel> img, temp;
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        file f(metadata.images[i].filename);
        string filename = get_parent_directory(f).full_name() + directory::get_separator() + "flipped_" + to_png_name(f.name());

        load_image(img, metadata.images[i].filename);
        flip_image_left_right(img, temp);
        if (parser.option("jpg"))
        {
            filename = to_jpg_name(filename);
            save_jpeg(temp, filename,JPEG_QUALITY);
        }
        else
        {
            save_png(temp, filename);
        }

        for (unsigned long j = 0; j < metadata.images[i].boxes.size(); ++j)
        {
            metadata.images[i].boxes[j].rect = impl::flip_rect_left_right(metadata.images[i].boxes[j].rect, get_rect(img));

            // flip all the object parts
            std::map<std::string,point>::iterator k;
            for (k = metadata.images[i].boxes[j].parts.begin(); k != metadata.images[i].boxes[j].parts.end(); ++k)
            {
                k->second = impl::flip_rect_left_right(rectangle(k->second,k->second), get_rect(img)).tl_corner();
            }
        }

        metadata.images[i].filename = filename;
    }

    save_image_dataset_metadata(metadata, metadata_filename);
}

// ----------------------------------------------------------------------------------------

