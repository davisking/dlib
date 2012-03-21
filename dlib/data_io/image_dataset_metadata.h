// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IMAGE_DAtASET_METADATA_H__
#define DLIB_IMAGE_DAtASET_METADATA_H__

#include <string>
#include <vector>
#include "../geometry.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{
    namespace image_dataset_metadata
    {
        struct box
        {
            box(
            ) : 
                head(-0xFFFF,-0xFFFF), 
                difficult(false),
                truncated(false),
                occluded(false)
            {}

            rectangle rect;

            // optional fields
            std::string label;
            point head; // a value of (-0xFFFF,-0xFFFF) indicates the field not supplied
            bool difficult;
            bool truncated;
            bool occluded;

            bool has_head() const { return head != point(-0xFFFF,-0xFFFF); }
            bool has_label() const { return label.size() != 0; }
        };

        struct image
        {
            image() {}
            image(const std::string& f) : filename(f) {}

            std::string filename;
            std::vector<box> boxes;
        };

        struct dataset
        {
            std::vector<image> images;
            std::string comment;
            std::string name;
        };

    // ------------------------------------------------------------------------------------

        void save_image_dataset_metadata (
            const dataset& meta,
            const std::string& filename
        );

    // ------------------------------------------------------------------------------------

        void load_image_dataset_metadata (
            dataset& meta,
            const std::string& filename
        );

    // ------------------------------------------------------------------------------------

    }
}

// ----------------------------------------------------------------------------------------

#ifdef NO_MAKEFILE
#include "image_dataset_metadata.cpp"
#endif

#endif // DLIB_IMAGE_DAtASET_METADATA_H__

