// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MNIST_CPp_
#define DLIB_MNIST_CPp_

#include "mnist.h"
#include <fstream>
#include "../byte_orderer.h"
#include "../uintn.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{
    void load_mnist_dataset (
        const std::string& folder_name,
        std::vector<matrix<unsigned char> >& training_images,
        std::vector<unsigned long>& training_labels,
        std::vector<matrix<unsigned char> >& testing_images,
        std::vector<unsigned long>& testing_labels
    )
    {
        using namespace std;
        ifstream fin1((folder_name+"/train-images-idx3-ubyte").c_str(), ios::binary);
        if (!fin1)
        {
            fin1.open((folder_name + "/train-images.idx3-ubyte").c_str(), ios::binary);
        }

        ifstream fin2((folder_name+"/train-labels-idx1-ubyte").c_str(), ios::binary);
        if (!fin2)
        {
            fin2.open((folder_name + "/train-labels.idx1-ubyte").c_str(), ios::binary);
        }

        ifstream fin3((folder_name+"/t10k-images-idx3-ubyte").c_str(), ios::binary);
        if (!fin3)
        {
            fin3.open((folder_name + "/t10k-images.idx3-ubyte").c_str(), ios::binary);
        }

        ifstream fin4((folder_name+"/t10k-labels-idx1-ubyte").c_str(), ios::binary);
        if (!fin4)
        {
            fin4.open((folder_name + "/t10k-labels.idx1-ubyte").c_str(), ios::binary);
        }

        if (!fin1) throw error("Unable to open file train-images-idx3-ubyte or train-images.idx3-ubyte");
        if (!fin2) throw error("Unable to open file train-labels-idx1-ubyte or train-labels.idx1-ubyte");
        if (!fin3) throw error("Unable to open file t10k-images-idx3-ubyte or t10k-images.idx3-ubyte");
        if (!fin4) throw error("Unable to open file t10k-labels-idx1-ubyte or t10k-labels.idx1-ubyte");
		
        byte_orderer bo;

        // make sure the files have the contents we expect.
        uint32 magic, num, nr, nc, num2, num3, num4;
        fin1.read((char*)&magic, sizeof(magic));  bo.big_to_host(magic);
        fin1.read((char*)&num, sizeof(num));  bo.big_to_host(num);
        fin1.read((char*)&nr, sizeof(nr));  bo.big_to_host(nr);
        fin1.read((char*)&nc, sizeof(nc));  bo.big_to_host(nc);
        if (magic != 2051 || num != 60000 || nr != 28 || nc != 28)
            throw error("mnist dat files are corrupted.");

        fin2.read((char*)&magic, sizeof(magic));  bo.big_to_host(magic);
        fin2.read((char*)&num2, sizeof(num2));  bo.big_to_host(num2);
        if (magic != 2049 || num2 != 60000)
            throw error("mnist dat files are corrupted.");

        fin3.read((char*)&magic, sizeof(magic));  bo.big_to_host(magic);
        fin3.read((char*)&num3, sizeof(num3));  bo.big_to_host(num3);
        fin3.read((char*)&nr, sizeof(nr));  bo.big_to_host(nr);
        fin3.read((char*)&nc, sizeof(nc));  bo.big_to_host(nc);
        if (magic != 2051 || num3 != 10000 || nr != 28 || nc != 28)
            throw error("mnist dat files are corrupted.");

        fin4.read((char*)&magic, sizeof(magic));  bo.big_to_host(magic);
        fin4.read((char*)&num4, sizeof(num4));  bo.big_to_host(num4);
        if (magic != 2049 || num4 != 10000)
            throw error("mnist dat files are corrupted.");

        if (!fin1) throw error("Unable to read train-images-idx3-ubyte");
        if (!fin2) throw error("Unable to read train-labels-idx1-ubyte");
        if (!fin3) throw error("Unable to read t10k-images-idx3-ubyte");
        if (!fin4) throw error("Unable to read t10k-labels-idx1-ubyte");


        training_images.resize(60000);
        training_labels.resize(60000);
        testing_images.resize(10000);
        testing_labels.resize(10000);

        for (size_t i = 0; i < training_images.size(); ++i)
        {
            training_images[i].set_size(nr,nc);
            fin1.read((char*)&training_images[i](0,0), nr*nc);
        }
        for (size_t i = 0; i < training_labels.size(); ++i)
        {
            char l;
            fin2.read(&l, 1);
            training_labels[i] = l;
        }

        for (size_t i = 0; i < testing_images.size(); ++i)
        {
            testing_images[i].set_size(nr,nc);
            fin3.read((char*)&testing_images[i](0,0), nr*nc);
        }
        for (size_t i = 0; i < testing_labels.size(); ++i)
        {
            char l;
            fin4.read(&l, 1);
            testing_labels[i] = l;
        }

        if (!fin1) throw error("Unable to read train-images-idx3-ubyte");
        if (!fin2) throw error("Unable to read train-labels-idx1-ubyte");
        if (!fin3) throw error("Unable to read t10k-images-idx3-ubyte");
        if (!fin4) throw error("Unable to read t10k-labels-idx1-ubyte");

        if (fin1.get() != EOF) throw error("Unexpected bytes at end of train-images-idx3-ubyte");
        if (fin2.get() != EOF) throw error("Unexpected bytes at end of train-labels-idx1-ubyte");
        if (fin3.get() != EOF) throw error("Unexpected bytes at end of t10k-images-idx3-ubyte");
        if (fin4.get() != EOF) throw error("Unexpected bytes at end of t10k-labels-idx1-ubyte");
    }
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_MNIST_CPp_



