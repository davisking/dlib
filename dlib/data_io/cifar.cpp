// Copyright (C) 2020  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CIFAR_CPp_
#define DLIB_CIFAR_CPp_

#include "cifar.h"
#include <fstream>

// ----------------------------------------------------------------------------------------

namespace dlib
{
    namespace impl
    {
        void load_cifar_10_batch (
            const std::string& folder_name,
            const std::string& batch_name,
            const size_t first_idx,
            const size_t images_per_batch,
            std::vector<matrix<rgb_pixel>>& images,
            std::vector<unsigned long>& labels
        )
        {
            std::ifstream fin(folder_name + "/" + batch_name, std::ios::binary);
            if (!fin) throw error("Unable to open file " + batch_name);
            const long nr = 32;
            const long nc = 32;
            const long plane_size = nr * nc;
            const long image_size = 3 * plane_size;

            for (size_t i = 0; i < images_per_batch; ++i)
            {
                char l;
                fin.read(&l, 1);
                labels[first_idx + i] = l;
                images[first_idx + i].set_size(nr, nc);

                std::array<unsigned char, image_size> buffer;
                fin.read((char*)(buffer.data()), buffer.size());
                for (long k = 0; k < plane_size; ++k)
                {
                    char r = buffer[0 * plane_size + k];
                    char g = buffer[1 * plane_size + k];
                    char b = buffer[2 * plane_size + k];
                    const long row = k / nr;
                    const long col = k % nr;
                    images[first_idx + i](row, col) = rgb_pixel(r, g, b);
                }
            }

            if (!fin) throw error("Unable to read file " + batch_name);

            if (fin.get() != EOF) throw error("Unexpected bytes at end of " + batch_name);
        }
    }

    void load_cifar_10_dataset (
        const std::string& folder_name,
        std::vector<matrix<rgb_pixel>>& training_images,
        std::vector<unsigned long>& training_labels,
        std::vector<matrix<rgb_pixel>>& testing_images,
        std::vector<unsigned long>& testing_labels
    )
    {
        using namespace std;

        const size_t images_per_batch = 10000;
        const size_t num_training_batches = 5;
        const size_t num_testing_batches = 1;

        training_images.resize(images_per_batch * num_training_batches);
        training_labels.resize(images_per_batch * num_training_batches);
        testing_images.resize(images_per_batch * num_testing_batches);
        testing_labels.resize(images_per_batch * num_testing_batches);

        std::vector<string> training_batches_names{
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
        };

        for (size_t i = 0; i < num_training_batches; ++i)
        {
            impl::load_cifar_10_batch(
                folder_name,
                training_batches_names[i],
                i * images_per_batch,
                images_per_batch,
                training_images,
                training_labels);
        }

        impl::load_cifar_10_batch(
            folder_name,
            "test_batch.bin",
            0,
            images_per_batch,
            testing_images,
            testing_labels);
    }
}
// ----------------------------------------------------------------------------------------

#endif // DLIB_CIFAR_CPp_
