// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <dlib/data_io.h>
#include <dlib/sparse_vector.h>
#include <boost/python/args.hpp>
#include "pyassert.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

typedef std::vector<std::pair<unsigned long,double> > sparse_vect;


void _make_sparse_vector (
    sparse_vect& v
)
{
    make_sparse_vector_inplace(v);
}

void _make_sparse_vector2 (
    std::vector<sparse_vect>& v
)
{
    for (unsigned long i = 0; i < v.size(); ++i)
        make_sparse_vector_inplace(v[i]);
}

tuple _load_libsvm_formatted_data (
    const std::string& file_name
) 
{ 
    std::vector<sparse_vect> samples;
    std::vector<double> labels;
    load_libsvm_formatted_data(file_name, samples, labels); 
    return make_tuple(samples, labels);
}

void _save_libsvm_formatted_data (
    const std::string& file_name,
    const std::vector<sparse_vect>& samples,
    const std::vector<double>& labels
) 
{ 
    pyassert(samples.size() == labels.size(), "Invalid inputs");
    save_libsvm_formatted_data(file_name, samples, labels); 
}

void bind_other()
{
    using boost::python::arg;

    def("make_sparse_vector", _make_sparse_vector , 
"This function modifies its argument so that it is a properly sorted sparse vector.    \n\
This means that the elements of the sparse vector will be ordered so that pairs    \n\
with smaller indices come first.  Additionally, there won't be any pairs with    \n\
identical indices.  If such pairs were present in the input sparse vector then    \n\
their values will be added together and only one pair with their index will be    \n\
present in the output.   " 
        );
    def("make_sparse_vector", _make_sparse_vector2 , 
        "This function modifies a sparse_vectors object so that all elements it contains are properly sorted sparse vectors.");

    def("load_libsvm_formatted_data",_load_libsvm_formatted_data, (arg("file_name")),
"ensures    \n\
    - Attempts to read a file of the given name that should contain libsvm    \n\
      formatted data.  The data is returned as a tuple where the first tuple    \n\
      element is an array of sparse vectors and the second element is an array of    \n\
      labels.    " 
    );

    def("save_libsvm_formatted_data",_save_libsvm_formatted_data, (arg("file_name"), arg("samples"), arg("labels")),
"requires    \n\
    - len(samples) == len(labels)    \n\
ensures    \n\
    - saves the data to the given file in libsvm format   " 
    );
}

