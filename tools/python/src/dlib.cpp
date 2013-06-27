// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <boost/python.hpp>

void bind_matrix();
void bind_vector();
void bind_svm_c_trainer();
void bind_decision_functions();
void bind_basic_types();
void bind_other();
void bind_svm_rank_trainer();
void bind_cca();
void bind_sequence_segmenter();
void bind_svm_struct();


BOOST_PYTHON_MODULE(dlib)
{
    // Disable printing of the C++ function signature in the python __doc__ string
    // since it is full of huge amounts of template clutter.
    boost::python::docstring_options options(true,true,false);

    bind_matrix();
    bind_vector();
    bind_svm_c_trainer();
    bind_decision_functions();
    bind_basic_types();
    bind_other();
    bind_svm_rank_trainer();
    bind_cca();
    bind_sequence_segmenter();
    bind_svm_struct();
}

