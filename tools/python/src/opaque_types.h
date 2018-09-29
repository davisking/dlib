// Copyright (C) 2017 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PyTHON_OPAQUE_TYPES_H_
#define DLIB_PyTHON_OPAQUE_TYPES_H_

#include <dlib/python.h>
#include <dlib/geometry.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <dlib/matrix.h>
#include <dlib/image_processing/full_object_detection.h>
#include <map>
#include <dlib/svm/ranking_tools.h>
#include <dlib/data_io.h>

// All uses of PYBIND11_MAKE_OPAQUE need to be in this common header to avoid ODR
// violations.
PYBIND11_MAKE_OPAQUE(std::vector<dlib::rectangle>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<dlib::rectangle>>);

PYBIND11_MAKE_OPAQUE(std::vector<double>);


typedef std::vector<dlib::matrix<double,0,1>> column_vectors;
PYBIND11_MAKE_OPAQUE(column_vectors);
PYBIND11_MAKE_OPAQUE(std::vector<column_vectors>);

typedef std::pair<unsigned long,unsigned long> ulong_pair;
PYBIND11_MAKE_OPAQUE(ulong_pair);
PYBIND11_MAKE_OPAQUE(std::vector<ulong_pair>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<ulong_pair>>);

typedef std::pair<unsigned long,double> ulong_double_pair;
PYBIND11_MAKE_OPAQUE(ulong_double_pair);
PYBIND11_MAKE_OPAQUE(std::vector<ulong_double_pair>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<ulong_double_pair>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<ulong_double_pair> > >);

PYBIND11_MAKE_OPAQUE(std::vector<dlib::mmod_rect>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<dlib::mmod_rect> >);
PYBIND11_MAKE_OPAQUE(std::vector<dlib::full_object_detection>);

typedef std::map<std::string,dlib::point> parts_list_type;
PYBIND11_MAKE_OPAQUE(parts_list_type);

typedef std::vector<dlib::ranking_pair<dlib::matrix<double,0,1>>> ranking_pairs;
typedef std::vector<std::pair<unsigned long,double> > sparse_vect;
typedef std::vector<dlib::ranking_pair<sparse_vect> > sparse_ranking_pairs;
PYBIND11_MAKE_OPAQUE(ranking_pairs);
PYBIND11_MAKE_OPAQUE(sparse_ranking_pairs);


PYBIND11_MAKE_OPAQUE(std::vector<dlib::point>);
PYBIND11_MAKE_OPAQUE(std::vector<dlib::dpoint>);

PYBIND11_MAKE_OPAQUE(std::vector<dlib::image_dataset_metadata::box>);
PYBIND11_MAKE_OPAQUE(std::vector<dlib::image_dataset_metadata::image>);

#endif // DLIB_PyTHON_OPAQUE_TYPES_H_

