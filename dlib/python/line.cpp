// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <pybind11/stl_bind.h>
#include "indexing.h"

using namespace dlib;
using namespace std;


string line__repr__ (const line& p)
{
    std::ostringstream sout;
    sout << "line(" << p.p1() << ", " << p.p2() << ")";
    return sout.str();
}

string line__str__(const line& p)
{
    std::ostringstream sout;
    sout << "(" << p.p1() << ", " << p.p2() << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

void bind_line(py::module& m)
{

    const char* class_docs = 
"This object represents a line in the 2D plane.  The line is defined by two points \n\
running through it, p1 and p2.  This object also includes a unit normal vector that \n\
is perpendicular to the line."; 

    py::class_<line>(m, "line", class_docs)
        .def(py::init<>(), "p1, p2, and normal are all the 0 vector.")
        .def(py::init<dpoint,dpoint>(), py::arg("a"), py::arg("b"),
"ensures \n\
    - #p1 == a \n\
    - #p2 == b \n\
    - #normal == A vector normal to the line passing through points a and b. \n\
      Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees." 
        /*!
            ensures
                - #p1 == a
                - #p2 == b
                - #normal == A vector normal to the line passing through points a and b.
                  Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        !*/
            )
        .def(py::init<point,point>(), py::arg("a"), py::arg("b"),
"ensures \n\
    - #p1 == a \n\
    - #p2 == b \n\
    - #normal == A vector normal to the line passing through points a and b. \n\
      Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees." 
        /*!
            ensures
                - #p1 == a
                - #p2 == b
                - #normal == A vector normal to the line passing through points a and b.
                  Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        !*/
            )
        .def_property_readonly("normal", &line::normal, "returns a unit vector that is normal to the line passing through p1 and p2.")
        .def("__repr__", &line__repr__)
        .def("__str__", &line__str__)
        .def_property_readonly("p1", &line::p1, "returns the first endpoint of the line.")
        .def_property_readonly("p2", &line::p2, "returns the second endpoint of the line.");


    m.def("signed_distance_to_line", &signed_distance_to_line<long>, py::arg("l"), py::arg("p"));
    m.def("signed_distance_to_line", &signed_distance_to_line<double>, py::arg("l"), py::arg("p"),
"ensures \n\
    - returns how far p is from the line l.  This is a signed distance.  The sign \n\
      indicates which side of the line the point is on and the magnitude is the \n\
      distance. Moreover, the direction of positive sign is pointed to by the \n\
      vector l.normal. \n\
    - To be specific, this routine returns dot(p-l.p1, l.normal)" 
    /*!
        ensures
            - returns how far p is from the line l.  This is a signed distance.  The sign
              indicates which side of the line the point is on and the magnitude is the
              distance. Moreover, the direction of positive sign is pointed to by the
              vector l.normal.
            - To be specific, this routine returns dot(p-l.p1, l.normal)
    !*/
        );

    m.def("distance_to_line", &distance_to_line<long>, py::arg("l"), py::arg("p"));
    m.def("distance_to_line", &distance_to_line<double>, py::arg("l"), py::arg("p"),
       "returns abs(signed_distance_to_line(l,p))" );

    m.def("reverse", [](const line& a){return reverse(a);}, py::arg("l"),
"ensures \n\
    - returns line(l.p2, l.p1) \n\
      (i.e. returns a line object that represents the same line as l but with the \n\
      endpoints, and therefore, the normal vector flipped.  This means that the \n\
      signed distance of operator() is also flipped)." 
    /*!
        ensures
            - returns line(l.p2, l.p1)
              (i.e. returns a line object that represents the same line as l but with the
              endpoints, and therefore, the normal vector flipped.  This means that the
              signed distance of operator() is also flipped).
    !*/
        );

    m.def("intersect", [](const line& a, const line& b){return intersect(a,b);}, py::arg("a"), py::arg("b"),
"ensures \n\
    - returns the point of intersection between lines a and b.  If no such point \n\
      exists then this function returns a point with Inf values in it." 
    /*!
        ensures
            - returns the point of intersection between lines a and b.  If no such point
              exists then this function returns a point with Inf values in it.
    !*/
        );

    m.def("angle_between_lines", [](const line& a, const line& b){return angle_between_lines(a,b);}, py::arg("a"), py::arg("b"),
"ensures \n\
    - returns the angle, in degrees, between the given lines.  This is a number in \n\
      the range [0 90]." 
    /*!
        ensures
            - returns the angle, in degrees, between the given lines.  This is a number in
              the range [0 90].
    !*/
        );

    m.def("count_points_on_side_of_line", &count_points_on_side_of_line<long>, 
        py::arg("l"), py::arg("reference_point"), py::arg("pts"), py::arg("dist_thresh_min")=0, py::arg("dist_thresh_max")=std::numeric_limits<double>::infinity());
    m.def("count_points_on_side_of_line", &count_points_on_side_of_line<double>, 
        py::arg("l"), py::arg("reference_point"), py::arg("pts"), py::arg("dist_thresh_min")=0, py::arg("dist_thresh_max")=std::numeric_limits<double>::infinity(),
"ensures \n\
    - Returns a count of how many points in pts have a distance from the line l \n\
      that is in the range [dist_thresh_min, dist_thresh_max].  This distance is a \n\
      signed value that indicates how far a point is from the line. Moreover, if \n\
      the point is on the same side as reference_point then the distance is \n\
      positive, otherwise it is negative.  So for example, If this range is [0, \n\
      infinity] then this function counts how many points are on the same side of l \n\
      as reference_point." 
    /*!
        ensures
            - Returns a count of how many points in pts have a distance from the line l
              that is in the range [dist_thresh_min, dist_thresh_max].  This distance is a
              signed value that indicates how far a point is from the line. Moreover, if
              the point is on the same side as reference_point then the distance is
              positive, otherwise it is negative.  So for example, If this range is [0,
              infinity] then this function counts how many points are on the same side of l
              as reference_point.
    !*/
        );

    m.def("count_points_between_lines", &count_points_between_lines<long>, py::arg("l1"), py::arg("l2"), py::arg("reference_point"), py::arg("pts"));
    m.def("count_points_between_lines", &count_points_between_lines<double>, py::arg("l1"), py::arg("l2"), py::arg("reference_point"), py::arg("pts"),
"ensures \n\
    - Counts and returns the number of points in pts that are between lines l1 and \n\
      l2.  Since a pair of lines will, in the general case, divide the plane into 4 \n\
      regions, we identify the region of interest as the one that contains the \n\
      reference_point.  Therefore, this function counts the number of points in pts \n\
      that appear in the same region as reference_point." 
    /*!
        ensures
            - Counts and returns the number of points in pts that are between lines l1 and
              l2.  Since a pair of lines will, in the general case, divide the plane into 4
              regions, we identify the region of interest as the one that contains the
              reference_point.  Therefore, this function counts the number of points in pts
              that appear in the same region as reference_point.
    !*/
        );


}

