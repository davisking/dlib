// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LOAD_IMAGE_DaTASET_H__
#define DLIB_LOAD_IMAGE_DaTASET_H__

#include "load_image_dataset_abstract.h"
#include "../misc_api.h"
#include "../dir_nav.h"
#include "../image_io.h"
#include "../array.h"
#include <vector>
#include "../geometry.h"
#include "image_dataset_metadata.h"
#include <string>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type, 
        typename MM
        >
    void load_image_dataset (
        array<image_type,MM>& images,
        std::vector<std::vector<rectangle> >& object_locations,
        const std::string& filename,
        const std::string& label 
    )
    {
        images.clear();
        object_locations.clear();
        const std::string old_working_dir = get_current_dir();

        // Set the current directory to be the one that contains the
        // metadata file. We do this because the file might contain
        // file paths which are relative to this folder.
        const std::string parent_dir = get_parent_directory(file(filename)).full_name();
        set_current_dir(parent_dir);


        using namespace dlib::image_dataset_metadata;

        dataset data;
        load_image_dataset_metadata(data, filename);

        images.resize(data.images.size());
        std::vector<rectangle> rects;
        for (unsigned long i = 0; i < data.images.size(); ++i)
        {
            load_image(images[i], data.images[i].filename);
            rects.clear();
            for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
            {
                if (label.size() == 0 || data.images[i].boxes[j].label == label)
                {
                    rects.push_back(data.images[i].boxes[j].rect);
                }
            }
            object_locations.push_back(rects);
        }

        set_current_dir(old_working_dir);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type, 
        typename MM
        >
    void load_image_dataset (
        array<image_type,MM>& images,
        std::vector<std::vector<rectangle> >& object_locations,
        const std::string& filename
    )
    {
        load_image_dataset(images, object_locations, filename, "");
    }

// ----------------------------------------------------------------------------------------
}

#endif // DLIB_LOAD_IMAGE_DaTASET_H__

