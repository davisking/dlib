// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "flip_dataset.h"
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <string>
#include "common.h"
#include <dlib/image_transforms.h>
#include <dlib/optimization.h>
#include <dlib/image_processing.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<long> align_points(
    const std::vector<dpoint>& from,
    const std::vector<dpoint>& to,
    double min_angle = -90*pi/180.0,
    double max_angle = 90*pi/180.0,
    long num_angles = 181 
)
/*!
    ensures
        - Figures out how to align the points in from with the points in to.  Returns an
          assignment array A that indicates that from[i] matches with to[A[i]].

          We use the Hungarian algorithm with a search over reasonable angles.  This method
          works because we just need to account for a translation and a mild rotation and
          nothing else.  If there is any other more complex mapping then you probably don't
          have landmarks that make sense to flip.
!*/
{
    DLIB_CASSERT(from.size() == to.size());

    std::vector<long> best_assignment;
    double best_assignment_cost = std::numeric_limits<double>::infinity();

    matrix<double> dists(from.size(), to.size());
    matrix<long long> idists;

    for (auto angle : linspace(min_angle, max_angle, num_angles))
    {
        auto rot = rotation_matrix(angle);
        for (long r = 0; r < dists.nr(); ++r)
        {
            for (long c = 0; c < dists.nc(); ++c)
            {
                dists(r,c) = length_squared(rot*from[r]-to[c]);
            }
        }

        idists = matrix_cast<long long>(-round(std::numeric_limits<long long>::max()*(dists/max(dists))));

        auto assignment = max_cost_assignment(idists);
        auto cost = assignment_cost(dists, assignment);
        if (cost < best_assignment_cost)
        {
            best_assignment_cost = cost;
            best_assignment = std::move(assignment);
        }
    }


    // Now compute the alignment error in terms of average distance moved by each part.  We
    // do this so we can give the user a warning if it's impossible to make a good
    // alignment.
    running_stats<double> rs;
    std::vector<dpoint> tmp(to.size());
    for (size_t i = 0; i < to.size(); ++i)
        tmp[best_assignment[i]] = to[i];
    auto tform = find_similarity_transform(from, tmp);
    for (size_t i = 0; i < from.size(); ++i)
        rs.add(length(tform(from[i])-tmp[i]));
    if (rs.mean() > 0.05)
    {
        cout << "WARNING, your dataset has object part annotations and you asked imglab to " << endl;
        cout << "flip the data.  Imglab tried to adjust the part labels so that the average" << endl;
        cout << "part layout in the flipped dataset is the same as the source dataset.  " << endl;
        cout << "However, the part annotation scheme doesn't seem to be left-right symmetric." << endl;
        cout << "You should manually review the output to make sure the part annotations are " << endl;
        cout << "labeled as you expect." << endl;
    }


    return best_assignment;
}

// ----------------------------------------------------------------------------------------

std::map<string,dpoint> normalized_parts (
    const image_dataset_metadata::box& b
)
{
    auto tform = dlib::impl::normalizing_tform(b.rect);
    std::map<string,dpoint> temp;
    for (auto& p : b.parts)
        temp[p.first] = tform(p.second);
    return temp;
}

// ----------------------------------------------------------------------------------------

std::map<string,dpoint> average_parts (
    const image_dataset_metadata::dataset& data
)
/*!
    ensures
        - returns the average part layout over all objects in data.  This is done by
          centering the parts inside their rects and then averaging all the objects.
!*/
{
    std::map<string,dpoint> psum;
    std::map<string,double> pcnt;
    for (auto& image : data.images)
    {
        for (auto& box : image.boxes)
        {
            for (auto& p : normalized_parts(box))
            {
                psum[p.first] += p.second;
                pcnt[p.first] += 1;
            }
        }
    }

    // make into an average
    for (auto& p : psum)
        p.second /= pcnt[p.first];

    return psum;
}

// ----------------------------------------------------------------------------------------

void make_part_labeling_match_target_dataset (
    const image_dataset_metadata::dataset& target,
    image_dataset_metadata::dataset& data 
)
/*!
    This function tries to adjust the part labels in data so that the average part layout
    in data is the same as target, according to the string labels.  Therefore, it doesn't
    adjust part positions, instead it changes the string labels on the parts to achieve
    this.  This really only makes sense when you flipped a dataset that contains left-right
    symmetric objects and you want to remap the part labels of the flipped data so that
    they match the unflipped data's annotation scheme.
!*/
{
    auto target_parts = average_parts(target);
    auto data_parts = average_parts(data);

    // Convert to a form align_points() understands.  We also need to keep track of the
    // labels for later.
    std::vector<dpoint> from, to;
    std::vector<string> from_labels, to_labels;
    for (auto& p : target_parts)
    {
        from_labels.emplace_back(p.first);
        from.emplace_back(p.second);
    }
    for (auto& p : data_parts)
    {
        to_labels.emplace_back(p.first);
        to.emplace_back(p.second);
    }

    auto assignment = align_points(from, to);
    // so now we know that from_labels[i] should replace to_labels[assignment[i]]
    std::map<string,string> label_mapping;
    for (size_t i = 0; i < assignment.size(); ++i)
        label_mapping[to_labels[assignment[i]]] = from_labels[i];

    // now apply the label mapping to the dataset
    for (auto& image : data.images)
    {
        for (auto& box : image.boxes)
        {
            std::map<string,point> temp;
            for (auto& p : box.parts)
                temp[label_mapping[p.first]] = p.second;
            box.parts = std::move(temp);
        }
    }
}

// ----------------------------------------------------------------------------------------

void flip_dataset(const command_line_parser& parser)
{
    image_dataset_metadata::dataset metadata, orig_metadata;
    string datasource;
    if (parser.option("flip"))
        datasource = parser.option("flip").argument();
    else
        datasource = parser.option("flip-basic").argument();
    load_image_dataset_metadata(metadata,datasource);
    orig_metadata = metadata;

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
            for (auto& part : metadata.images[i].boxes[j].parts)
            {
                part.second = impl::flip_rect_left_right(rectangle(part.second,part.second), get_rect(img)).tl_corner();
            }
        }

        metadata.images[i].filename = filename;
    }

    if (!parser.option("flip-basic"))
        make_part_labeling_match_target_dataset(orig_metadata, metadata);

    save_image_dataset_metadata(metadata, metadata_filename);
}

// ----------------------------------------------------------------------------------------

