// Copyright (C) 2020  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CIFAR_CPp_
#define DLIB_CIFAR_CPp_

#include "cifar.h"
#include <fstream>

// ----------------------------------------------------------------------------------------

namespace dlib
{
    void load_cifar_10_dataset (
        const std::string& folder_name,
        std::vector<matrix<rgb_pixel>>& training_images,
        std::vector<unsigned long>& training_labels,
        std::vector<matrix<rgb_pixel>>& testing_images,
        std::vector<unsigned long>& testing_labels
    )
    {
        using namespace std;
        std::vector<string> file_names{
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
            "test_batch.bin"
        };
        std::vector<ifstream> fins;
        for (const auto& file_name : file_names)
        {
            fins.emplace_back((folder_name + "/" + file_name).c_str(), ios::binary);
        }

        for (size_t i = 0; i < fins.size(); ++i)
        {
            if (!fins[i]) throw error("Unable to open file " + file_names[i]);
        }

        const size_t images_per_batch = 10000;
        const size_t training_batches = 5;
        const size_t testing_batches = 1;
        const long nr = 32;
        const long nc = 32;
        const long plane_size = nr * nc;
        const long image_size = 3 * plane_size;

        training_images.resize(images_per_batch * training_batches);
        training_labels.resize(images_per_batch * training_batches);
        testing_images.resize(images_per_batch * testing_batches);
        testing_labels.resize(images_per_batch * testing_batches);

        for (size_t i = 0; i < fins.size(); ++i)
        {
            bool test_batch = i == fins.size() - 1;
            for (size_t j = 0; j < images_per_batch; ++j)
            {
                auto idx = i * images_per_batch + j;

                char l;
                fins[i].read(&l, 1);
                if (test_batch)
                {
                    idx -= training_images.size();
                    testing_labels[idx] = l;
                    testing_images[idx].set_size(nr, nc);
                }
                else
                {
                    training_labels[idx] = l;
                    training_images[idx].set_size(nr, nc);
                }

                std::array<unsigned char, image_size> buffer;
                fins[i].read((char*)buffer.begin(), buffer.size());
                for (long k = 0; k < plane_size; ++k)
                {
                    char r = buffer[0 * plane_size + k];
                    char g = buffer[1 * plane_size + k];
                    char b = buffer[2 * plane_size + k];
                    const long row = k / nr;
                    const long col = k % nr;
                    if (test_batch)
                        testing_images[idx](row, col) = rgb_pixel(r, g, b);
                    else
                        training_images[idx](row, col) = rgb_pixel(r, g, b);
                }
            }
        }

        for (size_t i = 0; i < fins.size(); ++i)
        {
            if (!fins[i]) throw error("Unable to read file " + file_names[i]);
        }

        for (size_t i = 0; i < fins.size(); ++i)
        {
            if (fins[i].get() != EOF) throw error("Unexpected bytes at end of " + file_names[i]);
        }
    }
}
// ----------------------------------------------------------------------------------------

#endif // DLIB_CIFAR_CPp_



