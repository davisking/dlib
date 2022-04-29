// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cluster.h" 
#include <dlib/console_progress_indicator.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/misc_api.h>
#include <dlib/dir_nav.h>
#include <dlib/clustering.h>
#include <dlib/svm.h>
#include <dlib/statistics.h>

// ----------------------------------------------------------------------------------------

using namespace std;
using namespace dlib;

//  ----------------------------------------------------------------------------

struct assignment
{
    unsigned long c;
    double dist;
    unsigned long idx;

    bool operator<(const assignment& item) const
    { return dist < item.dist; }
};

std::vector<assignment> angular_cluster (
    std::vector<matrix<double,0,1> > feats,
    const unsigned long num_clusters
)
{
    DLIB_CASSERT(feats.size() != 0, "The dataset can't be empty");
    for (unsigned long i = 0; i < feats.size(); ++i)
    {
        DLIB_CASSERT(feats[i].size() == feats[0].size(), "All feature vectors must have the same length.");
    }

    // find the centroid of feats
    const matrix<double,0,1> m = mean(mat(feats));

    // Now center feats and then project onto the unit sphere.  The reason for projecting
    // onto the unit sphere is so pick_initial_centers() works in a sensible way.
    for (auto& f : feats) 
    {
        f = normalize(f-m);
    }

    // now do angular clustering of the points
    std::vector<matrix<double,0,1> > centers;
    pick_initial_centers(num_clusters, centers, feats, linear_kernel<matrix<double,0,1> >(), 0.05);
    find_clusters_using_angular_kmeans(feats, centers);

    // and then report the resulting assignments
    std::vector<assignment> assignments;
    for (unsigned long i = 0; i < feats.size(); ++i)
    {
        assignment temp;
        temp.c = nearest_center(centers, feats[i]);
        temp.dist = length(feats[i] - centers[temp.c]);
        temp.idx = i;
        assignments.push_back(temp);
    }
    return assignments;
}
std::vector<assignment> chinese_cluster (
    std::vector<matrix<double,0,1> > feats,
    unsigned long &num_clusters
    )
{
    DLIB_CASSERT(feats.size() != 0, "The dataset can't be empty");
    for (unsigned long i = 0; i < feats.size(); ++i)
    {
        DLIB_CASSERT(feats[i].size() == feats[0].size(), "All feature vectors must have the same length.");
    }

    // Try to find a good value to select if we should add a vertex in the graph.  First we
    // normalize the features.
    const matrix<double,0,1> m = mean(mat(feats));

    for (auto& f : feats) 
    {
        f = normalize(f-m);
    }

    // Then we find the average distance between them, that average will be a good threshold to
    // decide if pairs are connected.
    running_stats<double> rs;
    for (size_t i = 0; i < feats.size(); ++i) {
        for (size_t j = i; j < feats.size(); ++j) {
            rs.add(length(feats[i] - feats[j]));
        }
    }

    // add vertices for chinese whispers to find clusters
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < feats.size(); ++i) {
        for (size_t j = i; j < feats.size(); ++j) {
            if (length(feats[i] - feats[j]) < rs.mean()) {
                edges.push_back(sample_pair(i, j, length(feats[i] - feats[j])));
            }
        }
    }

    std::vector<unsigned long> labels;
    num_clusters = chinese_whispers(edges, labels);

    std::vector<assignment> assignments;
    for (unsigned long i = 0; i < feats.size(); ++i)
    {
        assignment temp;
        temp.c = labels[i];
        temp.dist = length(feats[i]);
        temp.idx = i;
        assignments.push_back(temp);
    }
    return assignments;
}

// ----------------------------------------------------------------------------------------

bool compare_first (
    const std::pair<double,image_dataset_metadata::image>& a,
    const std::pair<double,image_dataset_metadata::image>& b
)
{
    return a.first < b.first;
}

// ----------------------------------------------------------------------------------------

double mean_aspect_ratio (
    const image_dataset_metadata::dataset& data
)
{
    double sum = 0;
    double cnt = 0;
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            rectangle rect = data.images[i].boxes[j].rect;
            if (rect.area() == 0 || data.images[i].boxes[j].ignore)
                continue;
            sum += rect.width()/(double)rect.height();
            ++cnt;
        }
    }

    if (cnt != 0)
        return sum/cnt;
    else
        return 0;
}

// ----------------------------------------------------------------------------------------

bool has_non_ignored_boxes (const image_dataset_metadata::image& img)
{
    for (auto&& b : img.boxes)
    {
        if (!b.ignore)
            return true;
    }
    return false;
}

// ----------------------------------------------------------------------------------------

int cluster_dataset(
    const dlib::command_line_parser& parser
)
{
    // make sure the user entered an argument to this program
    if (parser.number_of_arguments() != 1)
    {
        cerr << "The --cluster option requires you to give one XML file on the command line." << endl;
        return EXIT_FAILURE;
    }

    unsigned long num_clusters = get_option(parser, "cluster", 0);
    const unsigned long chip_size = get_option(parser, "size", 8000);

    image_dataset_metadata::dataset data;

    image_dataset_metadata::load_image_dataset_metadata(data, parser[0]);
    set_current_dir(get_parent_directory(file(parser[0])));

    const double aspect_ratio = mean_aspect_ratio(data);

    dlib::array<array2d<rgb_pixel> > images;
    std::vector<matrix<double,0,1> > feats;
    console_progress_indicator pbar(data.images.size());
    // extract all the object chips and HOG features.
    cout << "Loading image data..." << endl;
    for (unsigned long i = 0; i < data.images.size(); ++i)
    {
        pbar.print_status(i);
        if (!has_non_ignored_boxes(data.images[i]))
            continue;

        array2d<rgb_pixel> img, chip;
        load_image(img, data.images[i].filename);

        for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
        {
            if (data.images[i].boxes[j].ignore || data.images[i].boxes[j].rect.area() < 10)
                continue;
            drectangle rect = data.images[i].boxes[j].rect;
            rect = set_aspect_ratio(rect, aspect_ratio);
            extract_image_chip(img, chip_details(rect, chip_size), chip);
            feats.push_back(extract_fhog_features(chip));
            images.push_back(chip);
        }
    }

    if (feats.size() == 0)
    {
        cerr << "No non-ignored object boxes found in the XML dataset.  You can't cluster an empty dataset." << endl;
        return EXIT_FAILURE;
    }

    cout << "\nClustering objects..." << endl;
    std::vector<assignment> assignments;
    if (num_clusters) {
        assignments = angular_cluster(feats, num_clusters);
    } else {
        assignments = chinese_cluster(feats, num_clusters);
    }


    // Now output each cluster to disk as an XML file.
    for (unsigned long c = 0; c < num_clusters; ++c)
    {
        // We are going to accumulate all the image metadata for cluster c.  We put it
        // into idata so we can sort the images such that images with central chips
        // come before less central chips.  The idea being to get the good chips to
        // show up first in the listing, making it easy to manually remove bad ones if
        // that is desired.
        std::vector<std::pair<double,image_dataset_metadata::image> > idata(data.images.size());
        unsigned long idx = 0;
        for (unsigned long i = 0; i < data.images.size(); ++i)
        {
            idata[i].first = std::numeric_limits<double>::infinity();
            idata[i].second.filename = data.images[i].filename;
            idata[i].second.width = data.images[i].width;
            idata[i].second.height = data.images[i].height;
            if (!has_non_ignored_boxes(data.images[i]))
                continue;

            for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
            {
                idata[i].second.boxes.push_back(data.images[i].boxes[j]);

                if (data.images[i].boxes[j].ignore || data.images[i].boxes[j].rect.area() < 10)
                    continue;

                // If this box goes into cluster c then update the score for the whole
                // image based on this boxes' score.  Otherwise, mark the box as
                // ignored.
                if (assignments[idx].c == c)
                    idata[i].first = std::min(idata[i].first, assignments[idx].dist);
                else
                    idata[i].second.boxes.back().ignore = true;

                ++idx;
            }
        }

        // now save idata to an xml file.
        std::sort(idata.begin(), idata.end(), compare_first);
        image_dataset_metadata::dataset cdata;
        cdata.comment = data.comment + "\n\n This file contains objects which were clustered into group " + 
                     cast_to_string(c+1) + " of " + cast_to_string(num_clusters) + " groups with a chip size of " + 
                     cast_to_string(chip_size) + " by imglab.";
        cdata.name = data.name;
        for (unsigned long i = 0; i < idata.size(); ++i)
        {
            // if this image has non-ignored boxes in it then include it in the output.
            if (idata[i].first != std::numeric_limits<double>::infinity())
                cdata.images.push_back(idata[i].second);
        }

        string outfile = "cluster_"+pad_int_with_zeros(c+1, 3) + ".xml";
        cout << "Saving " << outfile << endl;
        save_image_dataset_metadata(cdata, outfile);
    }

    // Now output each cluster to disk as a big tiled jpeg file.  Sort everything so, just
    // like in the xml file above, the best objects come first in the tiling.
    std::sort(assignments.begin(), assignments.end());
    for (unsigned long c = 0; c < num_clusters; ++c)
    {
        dlib::array<array2d<rgb_pixel> > temp;
        for (unsigned long i = 0; i < assignments.size(); ++i)
        {
            if (assignments[i].c == c)
                temp.push_back(images[assignments[i].idx]);
        }

#ifdef DLIB_WEBP_SUPPORT
        if (parser.option("webp"))
        {
            string outfile = "cluster_"+pad_int_with_zeros(c+1, 3) + ".webp";
            cout << "Saving " << outfile << endl;
            const float webp_quality = std::stof(parser.option("webp").argument());
            save_webp(tile_images(temp), outfile, webp_quality);
        }
        else
#endif
        {
            string outfile = "cluster_"+pad_int_with_zeros(c+1, 3) + ".jpg";
            cout << "Saving " << outfile << endl;
            save_jpeg(tile_images(temp), outfile);
        }
    }


    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

