// Copyright (C) 2015 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/geometry.h>
#include <pybind11/stl_bind.h>
#include "indexing.h"
#include "opaque_types.h"
#include <dlib/filtering.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;


// ----------------------------------------------------------------------------------------

long left(const rectangle& r) { return r.left(); }
long top(const rectangle& r) { return r.top(); }
long right(const rectangle& r) { return r.right(); }
long bottom(const rectangle& r) { return r.bottom(); }
long width(const rectangle& r) { return r.width(); }
long height(const rectangle& r) { return r.height(); }
unsigned long area(const rectangle& r) { return r.area(); }

double dleft(const drectangle& r) { return r.left(); }
double dtop(const drectangle& r) { return r.top(); }
double dright(const drectangle& r) { return r.right(); }
double dbottom(const drectangle& r) { return r.bottom(); }
double dwidth(const drectangle& r) { return r.width(); }
double dheight(const drectangle& r) { return r.height(); }
double darea(const drectangle& r) { return r.area(); }

template <typename rect_type>
bool is_empty(const rect_type& r) { return r.is_empty(); }

template <typename rect_type>
point center(const rect_type& r) { return center(r); }

template <typename rect_type>
point dcenter(const rect_type& r) { return dcenter(r); }

template <typename rect_type, typename ptype>
bool contains(const rect_type& r, const ptype& p) { return r.contains(p); }

template <typename rect_type>
bool contains_xy(const rect_type& r, const long x, const long y) { return r.contains(point(x, y)); }

template <typename rect_type>
bool contains_rec(const rect_type& r, const rect_type& r2) { return r.contains(r2); }

template <typename rect_type>
rect_type intersect(const rect_type& r, const rect_type& r2) { return r.intersect(r2); }

template <typename rect_type>
string print_rectangle_str(const rect_type& r)
{
    std::ostringstream sout;
    sout << r;
    return sout.str();
}

string print_rectangle_repr(const rectangle& r)
{
    std::ostringstream sout;
    sout << "rectangle(" << r.left() << "," << r.top() << "," << r.right() << "," << r.bottom() << ")";
    return sout.str();
}

string print_drectangle_repr(const drectangle& r)
{
    std::ostringstream sout;
    sout << "drectangle(" << r.left() << "," << r.top() << "," << r.right() << "," << r.bottom() << ")";
    return sout.str();
}

string print_rect_filter(const rect_filter& r)
{
    std::ostringstream sout;
    sout << "rect_filter(";
    sout << "measurement_noise="<<r.get_left().get_measurement_noise();
    sout << ", typical_acceleration="<<r.get_left().get_typical_acceleration();
    sout << ", max_measurement_deviation="<<r.get_left().get_max_measurement_deviation();
    sout << ")";
    return sout.str();
}



// ----------------------------------------------------------------------------------------

void bind_rectangles(py::module& m)
{
    {
    typedef rectangle type;
    py::class_<type>(m, "rectangle", "This object represents a rectangular area of an image.")
        .def(py::init<long,long,long,long>(), py::arg("left"),py::arg("top"),py::arg("right"),py::arg("bottom"))
        .def(py::init<drectangle>(), py::arg("rect"))
        .def(py::init<rectangle>(), py::arg("rect"))
        .def(py::init())
        .def("area",   &::area)
        .def("left",   &::left)
        .def("top",    &::top)
        .def("right",  &::right)
        .def("bottom", &::bottom)
        .def("width",  &::width)
        .def("height", &::height)
        .def("tl_corner", &type::tl_corner, "Returns the top left corner of the rectangle.")
        .def("tr_corner", &type::tr_corner, "Returns the top right corner of the rectangle.")
        .def("bl_corner", &type::bl_corner, "Returns the bottom left corner of the rectangle.")
        .def("br_corner", &type::br_corner, "Returns the bottom right corner of the rectangle.")
        .def("is_empty", &::is_empty<type>)
        .def("center", &::center<type>)
        .def("dcenter", &::dcenter<type>)
        .def("contains", &::contains<type,point>, py::arg("point"))
        .def("contains", &::contains<type,dpoint>, py::arg("point"))
        .def("contains", &::contains_xy<type>, py::arg("x"), py::arg("y"))
        .def("contains", &::contains_rec<type>, py::arg("rectangle"))
        .def("intersect", &::intersect<type>, py::arg("rectangle"))
        .def("__str__", &::print_rectangle_str<type>)
        .def("__repr__", &::print_rectangle_repr)
        .def(py::self += point())
        .def(py::self + point())
        .def(py::self += rectangle())
        .def(py::self + rectangle())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef drectangle type;
    py::class_<type>(m, "drectangle", "This object represents a rectangular area of an image with floating point coordinates.")
        .def(py::init<double,double,double,double>(), py::arg("left"), py::arg("top"), py::arg("right"), py::arg("bottom"))
        .def(py::init<rectangle>(), py::arg("rect"))
        .def(py::init<drectangle>(), py::arg("rect"))
        .def(py::init<>())
        .def("area",   &::darea)
        .def("left",   &::dleft)
        .def("top",    &::dtop)
        .def("right",  &::dright)
        .def("bottom", &::dbottom)
        .def("width",  &::dwidth)
        .def("height", &::dheight)
        .def("is_empty", &::is_empty<type>)
        .def("center", &::center<type>)
        .def("dcenter", &::dcenter<type>)
        .def("tl_corner", &type::tl_corner, "Returns the top left corner of the rectangle.")
        .def("tr_corner", &type::tr_corner, "Returns the top right corner of the rectangle.")
        .def("bl_corner", &type::bl_corner, "Returns the bottom left corner of the rectangle.")
        .def("br_corner", &type::br_corner, "Returns the bottom right corner of the rectangle.")
        .def("contains", &::contains<type,point>, py::arg("point"))
        .def("contains", &::contains<type,dpoint>, py::arg("point"))
        .def("contains", &::contains_xy<type>, py::arg("x"), py::arg("y"))
        .def("contains", &::contains_rec<type>, py::arg("rectangle"))
        .def("intersect", &::intersect<type>, py::arg("rectangle"))
        .def("__str__", &::print_rectangle_str<type>)
        .def("__repr__", &::print_drectangle_repr)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
        typedef rect_filter type;
        py::class_<type>(m, "rect_filter",
            R"asdf( 
                This object is a simple tool for filtering a rectangle that
                measures the location of a moving object that has some non-trivial
                momentum.  Importantly, the measurements are noisy and the object can
                experience sudden unpredictable accelerations.  To accomplish this
                filtering we use a simple Kalman filter with a state transition model of:

                    position_{i+1} = position_{i} + velocity_{i} 
                    velocity_{i+1} = velocity_{i} + some_unpredictable_acceleration

                and a measurement model of:
                    
                    measured_position_{i} = position_{i} + measurement_noise

                Where some_unpredictable_acceleration and measurement_noise are 0 mean Gaussian 
                noise sources with standard deviations of typical_acceleration and
                measurement_noise respectively.

                To allow for really sudden and large but infrequent accelerations, at each
                step we check if the current measured position deviates from the predicted
                filtered position by more than max_measurement_deviation*measurement_noise 
                and if so we adjust the filter's state to keep it within these bounds.
                This allows the moving object to undergo large unmodeled accelerations, far
                in excess of what would be suggested by typical_acceleration, without
                then experiencing a long lag time where the Kalman filter has to "catches
                up" to the new position.  )asdf"
        )
        .def(py::init<double,double,double>(), py::arg("measurement_noise"), py::arg("typical_acceleration"), py::arg("max_measurement_deviation"))
        .def("measurement_noise",   [](const rect_filter& a){return a.get_left().get_measurement_noise();})
        .def("typical_acceleration",   [](const rect_filter& a){return a.get_left().get_typical_acceleration();})
        .def("max_measurement_deviation",   [](const rect_filter& a){return a.get_left().get_max_measurement_deviation();})
        .def("__call__", [](rect_filter& f, const dlib::rectangle& r){return rectangle(f(r)); }, py::arg("rect"))
        .def("__repr__", print_rect_filter)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    m.def("find_optimal_rect_filter",
        [](const std::vector<rectangle>& rects, const double smoothness ) { return find_optimal_rect_filter(rects, smoothness); },
        py::arg("rects"),
        py::arg("smoothness")=1,
"requires \n\
    - rects.size() > 4 \n\
    - smoothness >= 0 \n\
ensures \n\
    - This function finds the \"optimal\" settings of a rect_filter based on recorded \n\
      measurement data stored in rects.  Here we assume that rects is a complete \n\
      track history of some object's measured positions.  Essentially, what we do \n\
      is find the rect_filter that minimizes the following objective function: \n\
         sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1]) \n\
         Where i is a time index. \n\
      The sum runs over all the data in rects.  So what we do is find the \n\
      filter settings that produce smooth filtered trajectories but also produce \n\
      filtered outputs that are as close to the measured positions as possible. \n\
      The larger the value of smoothness the less jittery the filter outputs will \n\
      be, but they might become biased or laggy if smoothness is set really high. " 
    /*!
        requires
            - rects.size() > 4
            - smoothness >= 0
        ensures
            - This function finds the "optimal" settings of a rect_filter based on recorded
              measurement data stored in rects.  Here we assume that rects is a complete
              track history of some object's measured positions.  Essentially, what we do
              is find the rect_filter that minimizes the following objective function:
                 sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1])
                 Where i is a time index.
              The sum runs over all the data in rects.  So what we do is find the
              filter settings that produce smooth filtered trajectories but also produce
              filtered outputs that are as close to the measured positions as possible.
              The larger the value of smoothness the less jittery the filter outputs will
              be, but they might become biased or laggy if smoothness is set really high. 
    !*/
    );

    {
    typedef std::vector<rectangle> type;
    py::bind_vector<type>(m, "rectangles", "An array of rectangle objects.")
        .def(py::init<size_t>(), py::arg("initial_size"))
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<rectangle>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef std::vector<std::vector<rectangle>> type;
    py::bind_vector<type>(m, "rectangless", "An array of arrays of rectangle objects.")
        .def(py::init<size_t>(), py::arg("initial_size"))
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<rectangle>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    m.def("translate_rect", [](const rectangle& rect, const point& p){return translate_rect(rect,p);},
" returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) \n\
  (i.e. moves the location of the rectangle but doesn't change its shape)",
        py::arg("rect"), py::arg("p"));

    m.def("translate_rect", [](const drectangle& rect, const point& p){return translate_rect(rect,p);},
" returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) \n\
  (i.e. moves the location of the rectangle but doesn't change its shape)",
        py::arg("rect"), py::arg("p"));

    m.def("translate_rect", [](const rectangle& rect, const dpoint& p){return translate_rect(rect,point(p));},
" returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) \n\
  (i.e. moves the location of the rectangle but doesn't change its shape)",
        py::arg("rect"), py::arg("p"));

    m.def("translate_rect", [](const drectangle& rect, const dpoint& p){return translate_rect(rect,p);},
" returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) \n\
  (i.e. moves the location of the rectangle but doesn't change its shape)",
        py::arg("rect"), py::arg("p"));



    m.def("shrink_rect", [](const rectangle& rect, long num){return shrink_rect(rect,num);},
" returns rectangle(rect.left()+num, rect.top()+num, rect.right()-num, rect.bottom()-num) \n\
  (i.e. shrinks the given rectangle by shrinking its border by num)",
        py::arg("rect"), py::arg("num"));

    m.def("grow_rect", [](const rectangle& rect, long num){return grow_rect(rect,num);},
"- return shrink_rect(rect, -num) \n\
  (i.e. grows the given rectangle by expanding its border by num)",
        py::arg("rect"), py::arg("num"));

    m.def("scale_rect", [](const rectangle& rect, double scale){return scale_rect(rect,scale);},
"- return scale_rect(rect, scale) \n\
(i.e. resizes the given rectangle by a scale factor)",
        py::arg("rect"), py::arg("scale"));

    m.def("centered_rect", [](const point& p, unsigned long width, unsigned long height) {
        return centered_rect(p, width, height); },
        py::arg("p"), py::arg("width"), py::arg("height"));

    m.def("centered_rects", [](const std::vector<point>& p, unsigned long width, unsigned long height) {
        return centered_rects(p, width, height); },
        py::arg("pts"), py::arg("width"), py::arg("height"));

    m.def("centered_rect", [](const dpoint& p, unsigned long width, unsigned long height) {
        return centered_rect(p, width, height); },
        py::arg("p"), py::arg("width"), py::arg("height"));

    m.def("centered_rect", [](const rectangle& rect, unsigned long width, unsigned long height) {
        return centered_rect(rect, width, height); },
        py::arg("rect"), py::arg("width"), py::arg("height"));

    m.def("centered_rect", [](const drectangle& rect, unsigned long width, unsigned long height) {
        return centered_rect(rect, width, height); },
        py::arg("rect"), py::arg("width"), py::arg("height"));


    m.def("center", [](const rectangle& rect){return center(rect); }, py::arg("rect"),
        "    returns the center of the given rectangle");
    m.def("center", [](const drectangle& rect){return center(rect); }, py::arg("rect"),
        "    returns the center of the given rectangle");
}

// ----------------------------------------------------------------------------------------
