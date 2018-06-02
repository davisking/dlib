// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry.h>
#include <dlib/image_transforms.h>
#include <pybind11/stl_bind.h>
#include "indexing.h"

using namespace dlib;
using namespace std;

typedef matrix<double,0,1> cv;


void cv_set_size(cv& m, long s)
{
    m.set_size(s);
    m = 0;
}

double dotprod ( const cv& a, const cv& b)
{
    return dot(a,b);
}

string cv__str__(const cv& v)
{
    ostringstream sout;
    for (long i = 0; i < v.size(); ++i)
    {
        sout << v(i);
        if (i+1 < v.size())
            sout << "\n";
    }
    return sout.str();
}

string cv__repr__ (const cv& v)
{
    std::ostringstream sout;
    sout << "dlib.vector([";
    for (long i = 0; i < v.size(); ++i)
    {
        sout << v(i);
        if (i+1 < v.size())
            sout << ", ";
    }
    sout << "])";
    return sout.str();
}

std::shared_ptr<cv> cv_from_object(py::object obj)
{
    try {
        long nr = obj.cast<long>();
        auto temp = std::make_shared<cv>(nr);
        *temp = 0;
        return temp;
    } catch(py::cast_error &e) {
        py::list li = obj.cast<py::list>();
        const long nr = len(obj);
        auto temp = std::make_shared<cv>(nr);
        for ( long r = 0; r < nr; ++r)
        {
            (*temp)(r) = li[r].cast<double>();
        }
        return temp;
    }
}

long cv__len__(cv& c)
{
    return c.size();
}


void cv__setitem__(cv& c, long p, double val)
{
    if (p < 0) {
        p = c.size() + p; // negative index
    }
    if (p > c.size()-1) {
        PyErr_SetString( PyExc_IndexError, "index out of range"
        );
        throw py::error_already_set();
    }
    c(p) = val;
}

double cv__getitem__(cv& m, long r)
{
    if (r < 0) {
        r = m.size() + r; // negative index
    }
    if (r > m.size()-1 || r < 0) {
        PyErr_SetString( PyExc_IndexError, "index out of range"
        );
        throw py::error_already_set();
    }
    return m(r);
}


cv cv__getitem2__(cv& m, py::slice r)
{
    size_t start, stop, step, slicelength;
    if (!r.compute(m.size(), &start, &stop, &step, &slicelength))
        throw py::error_already_set();

    cv temp(slicelength);

    for (size_t i = 0; i < slicelength; ++i) {
         temp(i) = m(start); start += step;
    }
    return temp;
}

py::tuple cv_get_matrix_size(cv& m)
{
    return py::make_tuple(m.nr(), m.nc());
}

// ----------------------------------------------------------------------------------------

string point_transform_projective__repr__ (const point_transform_projective& tform)
{
    std::ostringstream sout;
    sout << "point_transform_projective(\n" << csv << tform.get_m() << ")";
    return sout.str();
}

string point_transform_projective__str__(const point_transform_projective& tform)
{
    std::ostringstream sout;
    sout << "(" << csv << tform.get_m() << ")";
    return sout.str();
}

point_transform_projective init_point_transform_projective (
    const numpy_image<double>& m_
)
{
    const_image_view<numpy_image<double>> m(m_);
    DLIB_CASSERT(m.nr() == 3 && m.nc() == 3,
        "The matrix used to construct a point_transform_projective object must be 3x3.");

    return point_transform_projective(mat(m));
}

// ----------------------------------------------------------------------------------------

string point__repr__ (const point& p)
{
    std::ostringstream sout;
    sout << "point(" << p.x() << ", " << p.y() << ")";
    return sout.str();
}

string point__str__(const point& p)
{
    std::ostringstream sout;
    sout << "(" << p.x() << ", " << p.y() << ")";
    return sout.str();
}

string dpoint__repr__ (const dpoint& p)
{
    std::ostringstream sout;
    sout << "dpoint(" << p.x() << ", " << p.y() << ")";
    return sout.str();
}

string dpoint__str__(const dpoint& p)
{
    std::ostringstream sout;
    sout << "(" << p.x() << ", " << p.y() << ")";
    return sout.str();
}

long point_x(const point& p) { return p.x(); }
long point_y(const point& p) { return p.y(); }
double dpoint_x(const dpoint& p) { return p.x(); }
double dpoint_y(const dpoint& p) { return p.y(); }

// ----------------------------------------------------------------------------------------

template <typename T>
dlib::vector<T,2> numpy_to_dlib_vect (
    const py::array_t<T>& v
)
/*!
    ensures
        - converts a numpy array with 2 elements into a dlib::vector<T,2>
!*/
{
    DLIB_CASSERT(v.size() == 2, "You can only convert a numpy array to a dlib point or dpoint if it has just 2 elements.");
    DLIB_CASSERT(v.ndim() == 1 || v.ndim() == 2, "The input needs to be interpretable as a row or column vector.");
    dpoint temp;
    if (v.ndim() == 1)
    {
        temp.x() = v.at(0);
        temp.y() = v.at(1);
    }
    else if (v.shape(0) == 2)
    {
        temp.x() = v.at(0,0);
        temp.y() = v.at(1,0);
    }
    else
    {
        temp.x() = v.at(0,0);
        temp.y() = v.at(0,1);
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

point_transform_projective py_find_projective_transform (
    const std::vector<dpoint>& from_points,
    const std::vector<dpoint>& to_points
)
{
    DLIB_CASSERT(from_points.size() == to_points.size(),
        "from_points and to_points must have the same number of points.");
    DLIB_CASSERT(from_points.size() >= 4, 
        "You need at least 4 points to find a projective transform.");
    return find_projective_transform(from_points, to_points);
}

template <typename T>
point_transform_projective py_find_projective_transform2 (
    const numpy_image<T>& from_points_,
    const numpy_image<T>& to_points_
)
{
    const_image_view<numpy_image<T>> from_points(from_points_);
    const_image_view<numpy_image<T>> to_points(to_points_);

    DLIB_CASSERT(from_points.nc() == 2 && to_points.nc() == 2, 
        "Both from_points and to_points must be arrays with 2 columns.");
    DLIB_CASSERT(from_points.nr() == to_points.nr(),
        "from_points and to_points must have the same number of rows.");
    DLIB_CASSERT(from_points.nr() >= 4, 
        "You need at least 4 rows in the input matrices to find a projective transform.");
                 
    std::vector<dpoint> from, to;
    for (long r = 0; r < from_points.nr(); ++r)
    {
        from.push_back(dpoint(from_points[r][0], from_points[r][1]));
        to.push_back(dpoint(to_points[r][0], to_points[r][1]));
    }

    return find_projective_transform(from, to);
}

// ----------------------------------------------------------------------------------------

void register_point_transform_projective(
    py::module& m
)
{
                
    py::class_<point_transform_projective>(m, "point_transform_projective", 
        "This is an object that takes 2D points and applies a projective transformation to them.")
            .def(py::init<>(),
"ensures \n\
    - This object will perform the identity transform.  That is, given a point \n\
      as input it will return the same point as output.  Therefore, self.m == a 3x3 identity matrix." 
        /*!
            ensures
                - This object will perform the identity transform.  That is, given a point
                  as input it will return the same point as output.  Therefore, self.m == a 3x3 identity matrix.
        !*/
                )
            .def(py::init<>(&init_point_transform_projective), py::arg("m"),
"ensures \n\
    - self.m == m" 
                )
            .def("__repr__", &point_transform_projective__repr__)
            .def("__str__", &point_transform_projective__str__)
            .def("__call__", [](const point_transform_projective& tform, const dpoint& p){return tform(p);}, py::arg("p"),
"ensures \n\
    - Applies the projective transformation defined by this object's constructor \n\
      to p and returns the result.  To define this precisely: \n\
        - let p_h == the point p in homogeneous coordinates.  That is: \n\
            - p_h.x == p.x \n\
            - p_h.y == p.y \n\
            - p_h.z == 1  \n\
        - let x == m*p_h  \n\
        - Then this function returns the value x/x.z" 
        /*!
            ensures
                - Applies the projective transformation defined by this object's constructor
                  to p and returns the result.  To define this precisely:
                    - let p_h == the point p in homogeneous coordinates.  That is:
                        - p_h.x == p.x
                        - p_h.y == p.y
                        - p_h.z == 1 
                    - let x == m*p_h 
                    - Then this function returns the value x/x.z
        !*/
                )
            .def_property_readonly("m", [](const point_transform_projective& tform){numpy_image<double> tmp; assign_image(tmp,tform.get_m()); return tmp;},
                "m is the 3x3 matrix that defines the projective transformation.")
            .def(py::pickle(&getstate<point_transform_projective>, &setstate<point_transform_projective>));


    m.def("inv", [](const point_transform_projective& tform){return inv(tform); }, py::arg("trans"),
"ensures \n\
    - If trans is an invertible transformation then this function returns a new \n\
      transformation that is the inverse of trans. " 
    /*!
        ensures
            - If trans is an invertible transformation then this function returns a new
              transformation that is the inverse of trans. 
    !*/
        );


    m.def("find_projective_transform", &py_find_projective_transform, py::arg("from_points"), py::arg("to_points"),
"requires \n\
    - len(from_points) == len(to_points) \n\
    - len(from_points) >= 4 \n\
ensures \n\
    - returns a point_transform_projective object, T, such that for all valid i: \n\
        length(T(from_points[i]) - to_points[i]) \n\
      is minimized as often as possible.  That is, this function finds the projective \n\
      transform that maps points in from_points to points in to_points.  If no \n\
      projective transform exists which performs this mapping exactly then the one \n\
      which minimizes the mean squared error is selected. " 
    /*!
        requires
            - len(from_points) == len(to_points)
            - len(from_points) >= 4
        ensures
            - returns a point_transform_projective object, T, such that for all valid i:
                length(T(from_points[i]) - to_points[i])
              is minimized as often as possible.  That is, this function finds the projective
              transform that maps points in from_points to points in to_points.  If no
              projective transform exists which performs this mapping exactly then the one
              which minimizes the mean squared error is selected. 
    !*/
        );

    const char* docs = 
"requires \n\
    - from_points and to_points have two columns and the same number of rows. \n\
      Moreover, they have at least 4 rows. \n\
ensures \n\
    - returns a point_transform_projective object, T, such that for all valid i: \n\
        length(T(dpoint(from_points[i])) - dpoint(to_points[i])) \n\
      is minimized as often as possible.  That is, this function finds the projective \n\
      transform that maps points in from_points to points in to_points.  If no \n\
      projective transform exists which performs this mapping exactly then the one \n\
      which minimizes the mean squared error is selected. ";
    /*!
        requires
            - from_points and to_points have two columns and the same number of rows.
              Moreover, they have at least 4 rows.
        ensures
            - returns a point_transform_projective object, T, such that for all valid i:
                length(T(dpoint(from_points[i])) - dpoint(to_points[i]))
              is minimized as often as possible.  That is, this function finds the projective
              transform that maps points in from_points to points in to_points.  If no
              projective transform exists which performs this mapping exactly then the one
              which minimizes the mean squared error is selected. 
    !*/
    m.def("find_projective_transform", &py_find_projective_transform2<float>, py::arg("from_points"), py::arg("to_points"), docs);
    m.def("find_projective_transform", &py_find_projective_transform2<double>, py::arg("from_points"), py::arg("to_points"), docs);

}

// ----------------------------------------------------------------------------------------

double py_polygon_area(
    const std::vector<dpoint>& pts
)
{
    return polygon_area(pts);
}

double py_polygon_area2(
    const py::list& pts
)
{
    std::vector<dpoint> temp(len(pts));
    for (size_t i = 0; i < temp.size(); ++i)
        temp[i] = pts[i].cast<dpoint>();

    return polygon_area(temp);
}

// ----------------------------------------------------------------------------------------

void bind_vector(py::module& m)
{
    {
    py::class_<cv, std::shared_ptr<cv>>(m, "vector", "This object represents the mathematical idea of a column vector.")
        .def(py::init())
        .def("set_size", &cv_set_size)
        .def("resize", &cv_set_size)
        .def(py::init(&cv_from_object))
        .def("__repr__", &cv__repr__)
        .def("__str__", &cv__str__)
        .def("__len__", &cv__len__)
        .def("__getitem__", &cv__getitem__)
        .def("__getitem__", &cv__getitem2__)
        .def("__setitem__", &cv__setitem__)
        .def_property_readonly("shape", &cv_get_matrix_size)
        .def(py::pickle(&getstate<cv>, &setstate<cv>));

    m.def("dot", &dotprod, "Compute the dot product between two dense column vectors.");
    }
    {
    typedef point type;
    py::class_<type>(m, "point", "This object represents a single point of integer coordinates that maps directly to a dlib::point.")
            .def(py::init<long,long>(), py::arg("x"), py::arg("y"))
            .def(py::init<dpoint>(), py::arg("p"))
            .def(py::init<>(&numpy_to_dlib_vect<long>), py::arg("v"))
            .def(py::init<>(&numpy_to_dlib_vect<float>), py::arg("v"))
            .def(py::init<>(&numpy_to_dlib_vect<double>), py::arg("v"))
            .def("__repr__", &point__repr__)
            .def("__str__", &point__str__)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self / double())
            .def(py::self * double())
            .def(double() * py::self)
            .def("normalize", &type::normalize, "Returns a unit normalized copy of this vector.")
            .def_property("x", &point_x, [](point& p, long x){p.x()=x;}, "The x-coordinate of the point.")
            .def_property("y", &point_y, [](point& p, long y){p.x()=y;}, "The y-coordinate of the point.")
            .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef std::vector<point> type;
    py::bind_vector<type>(m, "points", "An array of point objects.")
        .def(py::init<size_t>(), py::arg("initial_size"))
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<point>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef dpoint type;
    py::class_<type>(m, "dpoint", "This object represents a single point of floating point coordinates that maps directly to a dlib::dpoint.")
            .def(py::init<double,double>(), py::arg("x"), py::arg("y"))
            .def(py::init<point>(), py::arg("p"))
            .def(py::init<>(&numpy_to_dlib_vect<long>), py::arg("v"))
            .def(py::init<>(&numpy_to_dlib_vect<float>), py::arg("v"))
            .def(py::init<>(&numpy_to_dlib_vect<double>), py::arg("v"))
            .def("__repr__", &dpoint__repr__)
            .def("__str__", &dpoint__str__)
            .def("normalize", &type::normalize, "Returns a unit normalized copy of this vector.")
            .def_property("x", &dpoint_x, [](dpoint& p, double x){p.x()=x;}, "The x-coordinate of the dpoint.")
            .def_property("y", &dpoint_y, [](dpoint& p, double y){p.x()=y;}, "The y-coordinate of the dpoint.")
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self / double())
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef std::vector<dpoint> type;
    py::bind_vector<type>(m, "dpoints", "An array of dpoint objects.")
        .def(py::init<size_t>(), py::arg("initial_size"))
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<dpoint>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    m.def("length", [](const point& p){return length(p); }, 
        "returns the distance from p to the origin, i.e. the L2 norm of p.", py::arg("p"));
    m.def("length", [](const dpoint& p){return length(p); }, 
        "returns the distance from p to the origin, i.e. the L2 norm of p.", py::arg("p"));

    m.def("dot", [](const point& a, const point& b){return dot(a,b); },  "Returns the dot product of the points a and b.", py::arg("a"), py::arg("b"));
    m.def("dot", [](const dpoint& a, const dpoint& b){return dot(a,b); },  "Returns the dot product of the points a and b.", py::arg("a"), py::arg("b"));

    register_point_transform_projective(m);

    m.def("polygon_area", &py_polygon_area, py::arg("pts"));
    m.def("polygon_area", &py_polygon_area2, py::arg("pts"),
"ensures \n\
    - If you walk the points pts in order to make a closed polygon, what is its \n\
      area?  This function returns that area.  It uses the shoelace formula to \n\
      compute the result and so works for general non-self-intersecting polygons." 
    /*!
        ensures
            - If you walk the points pts in order to make a closed polygon, what is its
              area?  This function returns that area.  It uses the shoelace formula to
              compute the result and so works for general non-self-intersecting polygons.
    !*/
        );

}

