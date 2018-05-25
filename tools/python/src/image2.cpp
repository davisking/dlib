// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_resize_image (
    const numpy_image<T>& img,
    unsigned long rows,
    unsigned long cols
)
{
    numpy_image<T> out;
    set_image_size(out, rows, cols);
    resize_image(img, out);
    return out;
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_equalize_histogram (
    const numpy_image<T>& img
)
{
    numpy_image<T> out;
    equalize_histogram(img,out);
    return out;
}

// ----------------------------------------------------------------------------------------

class py_hough_transform
{
public:

    py_hough_transform(
        unsigned long size
    ) : ht(size) 
    {
        DLIB_CASSERT(size > 0);
    }

    unsigned long size(
    ) const { return ht.size(); }

    long nr(
    ) const { return ht.nr(); }

    long nc(
    ) const { return ht.nc(); }

    line get_line (
        const point& p
    ) const 
    { 
        DLIB_CASSERT(rectangle(0,0,size()-1,size()-1).contains(p));
        auto temp = ht.get_line(p); 
        return line(temp.first, temp.second);
    }

    double get_line_angle_in_degrees (
        const point& p 
    ) const 
    { 
        DLIB_CASSERT(rectangle(0,0,size()-1,size()-1).contains(p));
        return ht.get_line_angle_in_degrees(p); 
    }

    py::tuple get_line_properties (
        const point& p
    ) const 
    { 
        DLIB_CASSERT(rectangle(0,0,size()-1,size()-1).contains(p));
        double angle_in_degrees;
        double radius;
        ht.get_line_properties(p, angle_in_degrees, radius);
        return py::make_tuple(angle_in_degrees, radius);
    }

    point get_best_hough_point (
        const point& p,
        const numpy_image<float>& himg
    ) 
    { 
        DLIB_ASSERT(himg.nr() == size() && himg.nc() == size() &&
            rectangle(0,0,size()-1,size()-1).contains(p) == true,
            "\t point hough_transform::get_best_hough_point()"
            << "\n\t Invalid arguments given to this function."
            << "\n\t himg.nr(): " << himg.nr()
            << "\n\t himg.nc(): " << himg.nc()
            << "\n\t size():    " << size()
            << "\n\t p:         " << p 
        );
        return ht.get_best_hough_point(p,himg); 
    }

    template <
        typename T 
        >
    numpy_image<float> compute_ht (
        const numpy_image<T>& img,
        const rectangle& box
    ) const
    {
        numpy_image<float> out;
        ht(img, box, out);
        return out;
    }

    template <
        typename T 
        >
    numpy_image<float> compute_ht2 (
        const numpy_image<T>& img
    ) const
    {
        numpy_image<float> out;
        ht(img, out);
        return out;
    }

    template <
        typename T 
        >
    py::list find_pixels_voting_for_lines (
        const numpy_image<T>& img,
        const rectangle& box,
        const std::vector<point>& hough_points,
        const unsigned long angle_window_size = 1,
        const unsigned long radius_window_size = 1
    ) const
    {
        return vector_to_python_list(ht.find_pixels_voting_for_lines(img, box, hough_points, angle_window_size, radius_window_size));
    }

    template <
        typename T 
        >
    py::list find_pixels_voting_for_lines2 (
        const numpy_image<T>& img,
        const std::vector<point>& hough_points,
        const unsigned long angle_window_size = 1,
        const unsigned long radius_window_size = 1
    ) const
    {
        return vector_to_python_list(ht.find_pixels_voting_for_lines(img, hough_points, angle_window_size, radius_window_size));
    }

    std::vector<point> find_strong_hough_points(
        const numpy_image<float>& himg,
        const float hough_count_threshold,
        const double angle_nms_thresh,
        const double radius_nms_thresh
    )
    {
        return ht.find_strong_hough_points(himg, hough_count_threshold, angle_nms_thresh, radius_nms_thresh);
    }

private:
    hough_transform ht;
};

// ----------------------------------------------------------------------------------------

void register_hough_transform(py::module& m)
{
    const char* class_docs =
"This object is a tool for computing the line finding version of the Hough transform \n\
given some kind of edge detection image as input.  It also allows the edge pixels \n\
to be weighted such that higher weighted edge pixels contribute correspondingly \n\
more to the output of the Hough transform, allowing stronger edges to create \n\
correspondingly stronger line detections in the final Hough transform.";


    const char* doc_constr = 
"requires \n\
    - size_ > 0 \n\
ensures \n\
    - This object will compute Hough transforms that are size_ by size_ pixels.   \n\
      This is in terms of both the Hough accumulator array size as well as the \n\
      input image size. \n\
    - size() == size_";
        /*!
            requires
                - size_ > 0
            ensures
                - This object will compute Hough transforms that are size_ by size_ pixels.  
                  This is in terms of both the Hough accumulator array size as well as the
                  input image size.
                - size() == size_
        !*/

    py::class_<py_hough_transform>(m, "hough_transform", class_docs)
        .def(py::init<unsigned long>(), doc_constr, py::arg("size_"))
        .def("size", &py_hough_transform::size,
            "returns the size of the Hough transforms generated by this object.  In particular, this object creates Hough transform images that are size() by size() pixels in size.")
        .def("get_line", &py_hough_transform::get_line, py::arg("p"),
"requires \n\
    - rectangle(0,0,size()-1,size()-1).contains(p) == true \n\
      (i.e. p must be a point inside the Hough accumulator array) \n\
ensures \n\
    - returns the line segment in the original image space corresponding \n\
      to Hough transform point p.  \n\
    - The returned points are inside rectangle(0,0,size()-1,size()-1).") 
    /*!
        requires
            - rectangle(0,0,size()-1,size()-1).contains(p) == true
              (i.e. p must be a point inside the Hough accumulator array)
        ensures
            - returns the line segment in the original image space corresponding
              to Hough transform point p. 
            - The returned points are inside rectangle(0,0,size()-1,size()-1).
    !*/

        .def("get_line_angle_in_degrees", &py_hough_transform::get_line_angle_in_degrees, py::arg("p"),
"requires \n\
    - rectangle(0,0,size()-1,size()-1).contains(p) == true \n\
      (i.e. p must be a point inside the Hough accumulator array) \n\
ensures \n\
    - returns the angle, in degrees, of the line corresponding to the Hough \n\
      transform point p.")
    /*!
        requires
            - rectangle(0,0,size()-1,size()-1).contains(p) == true
              (i.e. p must be a point inside the Hough accumulator array)
        ensures
            - returns the angle, in degrees, of the line corresponding to the Hough
              transform point p.
    !*/


        .def("get_line_properties", &py_hough_transform::get_line_properties, py::arg("p"),
"requires \n\
    - rectangle(0,0,size()-1,size()-1).contains(p) == true \n\
      (i.e. p must be a point inside the Hough accumulator array) \n\
ensures \n\
    - Converts a point in the Hough transform space into an angle, in degrees, \n\
      and a radius, measured in pixels from the center of the input image. \n\
    - let ANGLE_IN_DEGREES == the angle of the line corresponding to the Hough \n\
      transform point p.  Moreover: -90 <= ANGLE_IN_DEGREES < 90. \n\
    - RADIUS == the distance from the center of the input image, measured in \n\
      pixels, and the line corresponding to the Hough transform point p. \n\
      Moreover: -sqrt(size()*size()/2) <= RADIUS <= sqrt(size()*size()/2) \n\
    - returns a tuple of (ANGLE_IN_DEGREES, RADIUS)" )
    /*!
        requires
            - rectangle(0,0,size()-1,size()-1).contains(p) == true
              (i.e. p must be a point inside the Hough accumulator array)
        ensures
            - Converts a point in the Hough transform space into an angle, in degrees,
              and a radius, measured in pixels from the center of the input image.
            - let ANGLE_IN_DEGREES == the angle of the line corresponding to the Hough
              transform point p.  Moreover: -90 <= ANGLE_IN_DEGREES < 90.
            - RADIUS == the distance from the center of the input image, measured in
              pixels, and the line corresponding to the Hough transform point p.
              Moreover: -sqrt(size()*size()/2) <= RADIUS <= sqrt(size()*size()/2)
            - returns a tuple of (ANGLE_IN_DEGREES, RADIUS)
    !*/

        .def("get_best_hough_point", &py_hough_transform::get_best_hough_point, py::arg("p"), py::arg("himg"),
"requires \n\
    - himg has size() rows and columns. \n\
    - rectangle(0,0,size()-1,size()-1).contains(p) == true \n\
ensures \n\
    - This function interprets himg as a Hough image and p as a point in the \n\
      original image space.  Given this, it finds the maximum scoring line that \n\
      passes though p.  That is, it checks all the Hough accumulator bins in \n\
      himg corresponding to lines though p and returns the location with the \n\
      largest score.   \n\
    - returns a point X such that get_rect(himg).contains(X) == true")
    /*!
        requires
            - himg has size() rows and columns.
            - rectangle(0,0,size()-1,size()-1).contains(p) == true
        ensures
            - This function interprets himg as a Hough image and p as a point in the
              original image space.  Given this, it finds the maximum scoring line that
              passes though p.  That is, it checks all the Hough accumulator bins in
              himg corresponding to lines though p and returns the location with the
              largest score.  
            - returns a point X such that get_rect(himg).contains(X) == true
    !*/

        .def("__call__", &py_hough_transform::compute_ht<uint8_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<uint16_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<uint32_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<uint64_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<int8_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<int16_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<int32_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<int64_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<float>, py::arg("img"), py::arg("box"))
        .def("__call__", &py_hough_transform::compute_ht<double>, py::arg("img"), py::arg("box"),
"requires \n\
    - box.width() == size() \n\
    - box.height() == size() \n\
ensures \n\
    - Computes the Hough transform of the part of img contained within box. \n\
      In particular, we do a grayscale version of the Hough transform where any \n\
      non-zero pixel in img is treated as a potential component of a line and \n\
      accumulated into the returned Hough accumulator image.  However, rather than \n\
      adding 1 to each relevant accumulator bin we add the value of the pixel \n\
      in img to each Hough accumulator bin.  This means that, if all the \n\
      pixels in img are 0 or 1 then this routine performs a normal Hough \n\
      transform.  However, if some pixels have larger values then they will be \n\
      weighted correspondingly more in the resulting Hough transform. \n\
    - The returned hough transform image will be size() rows by size() columns. \n\
    - The returned image is the Hough transform of the part of img contained in \n\
      box.  Each point in the Hough image corresponds to a line in the input box. \n\
      In particular, the line for hough_image[y][x] is given by get_line(point(x,y)).  \n\
      Also, when viewing the Hough image, the x-axis gives the angle of the line \n\
      and the y-axis the distance of the line from the center of the box.  The \n\
      conversion between Hough coordinates and angle and pixel distance can be \n\
      obtained by calling get_line_properties()." )
    /*!
        requires
            - box.width() == size()
            - box.height() == size()
        ensures
            - Computes the Hough transform of the part of img contained within box.
              In particular, we do a grayscale version of the Hough transform where any
              non-zero pixel in img is treated as a potential component of a line and
              accumulated into the returned Hough accumulator image.  However, rather than
              adding 1 to each relevant accumulator bin we add the value of the pixel
              in img to each Hough accumulator bin.  This means that, if all the
              pixels in img are 0 or 1 then this routine performs a normal Hough
              transform.  However, if some pixels have larger values then they will be
              weighted correspondingly more in the resulting Hough transform.
            - The returned hough transform image will be size() rows by size() columns.
            - The returned image is the Hough transform of the part of img contained in
              box.  Each point in the Hough image corresponds to a line in the input box.
              In particular, the line for hough_image[y][x] is given by get_line(point(x,y)). 
              Also, when viewing the Hough image, the x-axis gives the angle of the line
              and the y-axis the distance of the line from the center of the box.  The
              conversion between Hough coordinates and angle and pixel distance can be
              obtained by calling get_line_properties().
    !*/

        .def("__call__", &py_hough_transform::compute_ht2<uint8_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<uint16_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<uint32_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<uint64_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<int8_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<int16_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<int32_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<int64_t>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<float>, py::arg("img"))
        .def("__call__", &py_hough_transform::compute_ht2<double>, py::arg("img"),
            "    simply performs: return self(img, get_rect(img)).  That is, just runs the hough transform on the whole input image.")

        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<uint8_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<uint16_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<uint32_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<uint64_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<int8_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<int16_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<int32_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<int64_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<float>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines<double>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1,
"requires \n\
    - box.width() == size() \n\
    - box.height() == size() \n\
    - for all valid i: \n\
        - rectangle(0,0,size()-1,size()-1).contains(hough_points[i]) == true \n\
          (i.e. hough_points must contain points in the output Hough transform \n\
          space generated by this object.) \n\
    - angle_window_size >= 1 \n\
    - radius_window_size >= 1 \n\
ensures \n\
    - This function computes the Hough transform of the part of img contained \n\
      within box.  It does the same computation as __call__() defined above, \n\
      except instead of accumulating into an image we create an explicit list \n\
      of all the points in img that contributed to each line (i.e each point in \n\
      the Hough image). To do this we take a list of Hough points as input and \n\
      only record hits on these specifically identified Hough points.  A \n\
      typical use of find_pixels_voting_for_lines() is to first run the normal \n\
      Hough transform using __call__(), then find the lines you are interested \n\
      in, and then call find_pixels_voting_for_lines() to determine which \n\
      pixels in the input image belong to those lines. \n\
    - This routine returns a vector, CONSTITUENT_POINTS, with the following \n\
      properties: \n\
        - CONSTITUENT_POINTS.size() == hough_points.size() \n\
        - for all valid i: \n\
            - Let HP[i] = centered_rect(hough_points[i], angle_window_size, radius_window_size) \n\
            - Any point in img with a non-zero value that lies on a line \n\
              corresponding to one of the Hough points in HP[i] is added to \n\
              CONSTITUENT_POINTS[i].  Therefore, when this routine finishes, \n\
              #CONSTITUENT_POINTS[i] will contain all the points in img that \n\
              voted for the lines associated with the Hough accumulator bins in \n\
              HP[i]. \n\
            - #CONSTITUENT_POINTS[i].size() == the number of points in img that \n\
              voted for any of the lines HP[i] in Hough space.  Note, however, \n\
              that if angle_window_size or radius_window_size are made so large \n\
              that HP[i] overlaps HP[j] for i!=j then the overlapping regions \n\
              of Hough space are assign to HP[i] or HP[j] arbitrarily. \n\
              Therefore, all points in CONSTITUENT_POINTS are unique, that is, \n\
              there is no overlap in points between any two elements of \n\
              CONSTITUENT_POINTS." )
    /*!
        requires
            - box.width() == size()
            - box.height() == size()
            - for all valid i:
                - rectangle(0,0,size()-1,size()-1).contains(hough_points[i]) == true
                  (i.e. hough_points must contain points in the output Hough transform
                  space generated by this object.)
            - angle_window_size >= 1
            - radius_window_size >= 1
        ensures
            - This function computes the Hough transform of the part of img contained
              within box.  It does the same computation as __call__() defined above,
              except instead of accumulating into an image we create an explicit list
              of all the points in img that contributed to each line (i.e each point in
              the Hough image). To do this we take a list of Hough points as input and
              only record hits on these specifically identified Hough points.  A
              typical use of find_pixels_voting_for_lines() is to first run the normal
              Hough transform using __call__(), then find the lines you are interested
              in, and then call find_pixels_voting_for_lines() to determine which
              pixels in the input image belong to those lines.
            - This routine returns a vector, CONSTITUENT_POINTS, with the following
              properties:
                - CONSTITUENT_POINTS.size() == hough_points.size()
                - for all valid i:
                    - Let HP[i] = centered_rect(hough_points[i], angle_window_size, radius_window_size)
                    - Any point in img with a non-zero value that lies on a line
                      corresponding to one of the Hough points in HP[i] is added to
                      CONSTITUENT_POINTS[i].  Therefore, when this routine finishes,
                      #CONSTITUENT_POINTS[i] will contain all the points in img that
                      voted for the lines associated with the Hough accumulator bins in
                      HP[i].
                    - #CONSTITUENT_POINTS[i].size() == the number of points in img that
                      voted for any of the lines HP[i] in Hough space.  Note, however,
                      that if angle_window_size or radius_window_size are made so large
                      that HP[i] overlaps HP[j] for i!=j then the overlapping regions
                      of Hough space are assign to HP[i] or HP[j] arbitrarily.
                      Therefore, all points in CONSTITUENT_POINTS are unique, that is,
                      there is no overlap in points between any two elements of
                      CONSTITUENT_POINTS.
    !*/
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<uint8_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<uint16_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<uint32_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<uint64_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<int8_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<int16_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<int32_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<int64_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<float>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &py_hough_transform::find_pixels_voting_for_lines2<double>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1,
"    performs: return find_pixels_voting_for_lines(img, get_rect(img), hough_points, angle_window_size, radius_window_size); \n\
That is, just runs the routine on the whole input image." )

        .def("find_strong_hough_points", &py_hough_transform::find_strong_hough_points, py::arg("himg"), py::arg("hough_count_threshold"), py::arg("angle_nms_thresh"), py::arg("radius_nms_thresh"),
"requires \n\
    - himg has size() rows and columns. \n\
    - angle_nms_thresh >= 0 \n\
    - radius_nms_thresh >= 0 \n\
ensures \n\
    - This routine finds strong lines in a Hough transform and performs \n\
      non-maximum suppression on the detected lines.  Recall that each point in \n\
      Hough space is associated with a line. Therefore, this routine finds all \n\
      the pixels in himg (a Hough transform image) with values >= \n\
      hough_count_threshold and performs non-maximum suppression on the \n\
      identified list of pixels.  It does this by discarding lines that are \n\
      within angle_nms_thresh degrees of a stronger line or within \n\
      radius_nms_thresh distance (in terms of radius as defined by \n\
      get_line_properties()) to a stronger Hough point. \n\
    - The identified lines are returned as a list of coordinates in himg." );
    /*!
        requires
            - himg has size() rows and columns.
            - angle_nms_thresh >= 0
            - radius_nms_thresh >= 0
        ensures
            - This routine finds strong lines in a Hough transform and performs
              non-maximum suppression on the detected lines.  Recall that each point in
              Hough space is associated with a line. Therefore, this routine finds all
              the pixels in himg (a Hough transform image) with values >=
              hough_count_threshold and performs non-maximum suppression on the
              identified list of pixels.  It does this by discarding lines that are
              within angle_nms_thresh degrees of a stronger line or within
              radius_nms_thresh distance (in terms of radius as defined by
              get_line_properties()) to a stronger Hough point.
            - The identified lines are returned as a list of coordinates in himg.
    !*/

}

// ----------------------------------------------------------------------------------------

std::vector<point> py_remove_incoherent_edge_pixels (
    const std::vector<point>& line,
    const numpy_image<float>& horz_gradient,
    const numpy_image<float>& vert_gradient,
    double angle_threshold
)
{

    DLIB_CASSERT(num_rows(horz_gradient) == num_rows(vert_gradient));
    DLIB_CASSERT(num_columns(horz_gradient) == num_columns(vert_gradient));
    DLIB_CASSERT(angle_threshold >= 0);
    for (auto& p : line)
        DLIB_CASSERT(get_rect(horz_gradient).contains(p), "All line points must be inside the given images.");

    return remove_incoherent_edge_pixels(line, horz_gradient, vert_gradient, angle_threshold);
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_transform_image (
    const numpy_image<T>& img,
    const point_transform_projective& map_point,
    long rows,
    long columns
)
{
    DLIB_CASSERT(rows > 0 && columns > 0, "The requested output image dimensions are invalid.");
    numpy_image<T> out_;
    image_view<numpy_image<T>> out(out_);
    out.set_size(rows, columns);

    transform_image(img, out_, interpolate_bilinear(), map_point);

    return out_;
}
// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_extract_image_4points (
    const numpy_image<T>& img,
    const py::list& corners,
    long rows,
    long columns
)
{
    DLIB_CASSERT(rows >= 0);
    DLIB_CASSERT(columns >= 0);
    DLIB_CASSERT(len(corners) == 4);

    numpy_image<T> out;
    set_image_size(out, rows, columns);
    try
    {
        extract_image_4points(img, out, python_list_to_array<dpoint,4>(corners));
        return out;
    } 
    catch (py::cast_error&){}

    try
    {
        extract_image_4points(img, out, python_list_to_array<line,4>(corners));
        return out;
    }
    catch(py::cast_error&)
    {
        throw dlib::error("extract_image_4points() requires the corners argument to be a list of 4 dpoints or 4 lines.");
    }
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_mbd (
    const numpy_image<T>& img,
    size_t iterations,
    bool do_left_right_scans 
)
{
    numpy_image<T> out;
    min_barrier_distance(img, out, iterations, do_left_right_scans);
    return out;
}

numpy_image<unsigned char> py_mbd2 (
    const numpy_image<rgb_pixel>& img,
    size_t iterations,
    bool do_left_right_scans 
)
{
    numpy_image<unsigned char> out;
    min_barrier_distance(img, out, iterations, do_left_right_scans);
    return out;
}


// ----------------------------------------------------------------------------------------

void bind_image_classes2(py::module& m)
{

    const char* docs = "Resizes img, using bilinear interpolation, to have the indicated number of rows and columns.";


    m.def("resize_image", &py_resize_image<uint8_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<uint16_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<uint32_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<uint64_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int8_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int16_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int32_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int64_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<float>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<double>, docs, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<rgb_pixel>, docs, py::arg("img"), py::arg("rows"), py::arg("cols"));


    docs = "Returns a histogram equalized version of img.";
    m.def("equalize_histogram", &py_equalize_histogram<uint8_t>, py::arg("img"));
    m.def("equalize_histogram", &py_equalize_histogram<uint16_t>, docs, py::arg("img"));

    m.def("min_barrier_distance", &py_mbd<uint8_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<uint16_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<uint32_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<uint64_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<int8_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<int16_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<int32_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<int64_t>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<float>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd<double>, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true);
    m.def("min_barrier_distance", &py_mbd2, py::arg("img"), py::arg("iterations")=10, py::arg("do_left_right_scans")=true,
"requires \n\
    - iterations > 0 \n\
ensures \n\
    - This function implements the salient object detection method described in the paper: \n\
        \"Minimum barrier salient object detection at 80 fps\" by Zhang, Jianming, et al.  \n\
      In particular, we compute the minimum barrier distance between the borders of \n\
      the image and all the other pixels.  The resulting image is returned.  Note that \n\
      the paper talks about a bunch of other things you could do beyond computing \n\
      the minimum barrier distance, but this function doesn't do any of that. It's \n\
      just the vanilla MBD. \n\
    - We will perform iterations iterations of MBD passes over the image.  Larger \n\
      values might give better results but run slower. \n\
    - During each MBD iteration we make raster scans over the image.  These pass \n\
      from top->bottom, bottom->top, left->right, and right->left.  If \n\
      do_left_right_scans==false then the left/right passes are not executed. \n\
      Skipping them makes the algorithm about 2x faster but might reduce the \n\
      quality of the output." 
    /*!
        requires
            - iterations > 0
        ensures
            - This function implements the salient object detection method described in the paper:
                "Minimum barrier salient object detection at 80 fps" by Zhang, Jianming, et al. 
              In particular, we compute the minimum barrier distance between the borders of
              the image and all the other pixels.  The resulting image is returned.  Note that
              the paper talks about a bunch of other things you could do beyond computing
              the minimum barrier distance, but this function doesn't do any of that. It's
              just the vanilla MBD.
            - We will perform iterations iterations of MBD passes over the image.  Larger
              values might give better results but run slower.
            - During each MBD iteration we make raster scans over the image.  These pass
              from top->bottom, bottom->top, left->right, and right->left.  If
              do_left_right_scans==false then the left/right passes are not executed.
              Skipping them makes the algorithm about 2x faster but might reduce the
              quality of the output.
    !*/
    );

    register_hough_transform(m);

    m.def("normalize_image_gradients", normalize_image_gradients<numpy_image<double>>, py::arg("img1"), py::arg("img2"));
    m.def("normalize_image_gradients", normalize_image_gradients<numpy_image<float>>, py::arg("img1"), py::arg("img2"),
"requires \n\
    - img1 and img2 have the same dimensions. \n\
ensures \n\
    - This function assumes img1 and img2 are the two gradient images produced by a \n\
      function like sobel_edge_detector().  It then unit normalizes the gradient \n\
      vectors. That is, for all valid r and c, this function ensures that: \n\
        - img1[r][c]*img1[r][c] + img2[r][c]*img2[r][c] == 1  \n\
          unless both img1[r][c] and img2[r][c] were 0 initially, then they stay zero.");
    /*!
        requires
            - img1 and img2 have the same dimensions.
        ensures
            - This function assumes img1 and img2 are the two gradient images produced by a
              function like sobel_edge_detector().  It then unit normalizes the gradient
              vectors. That is, for all valid r and c, this function ensures that:
                - img1[r][c]*img1[r][c] + img2[r][c]*img2[r][c] == 1 
                  unless both img1[r][c] and img2[r][c] were 0 initially, then they stay zero.
    !*/


    m.def("remove_incoherent_edge_pixels", &py_remove_incoherent_edge_pixels, py::arg("line"), py::arg("horz_gradient"),
        py::arg("vert_gradient"), py::arg("angle_thresh"),
"requires \n\
    - horz_gradient and vert_gradient have the same dimensions. \n\
    - horz_gradient and vert_gradient represent unit normalized vectors.  That is, \n\
      you should have called normalize_image_gradients(horz_gradient,vert_gradient) \n\
      or otherwise caused all the gradients to have unit norm. \n\
    - for all valid i: \n\
        get_rect(horz_gradient).contains(line[i]) \n\
ensures \n\
    - This routine looks at all the points in the given line and discards the ones that \n\
      have outlying gradient directions.  To be specific, this routine returns a set \n\
      of points PTS such that:  \n\
        - for all valid i,j: \n\
            - The difference in angle between the gradients for PTS[i] and PTS[j] is  \n\
              less than angle_threshold degrees.   \n\
        - len(PTS) <= len(line) \n\
        - PTS is just line with some elements removed." );
    /*!
        requires
            - horz_gradient and vert_gradient have the same dimensions.
            - horz_gradient and vert_gradient represent unit normalized vectors.  That is,
              you should have called normalize_image_gradients(horz_gradient,vert_gradient)
              or otherwise caused all the gradients to have unit norm.
            - for all valid i:
                get_rect(horz_gradient).contains(line[i])
        ensures
            - This routine looks at all the points in the given line and discards the ones that
              have outlying gradient directions.  To be specific, this routine returns a set
              of points PTS such that: 
                - for all valid i,j:
                    - The difference in angle between the gradients for PTS[i] and PTS[j] is 
                      less than angle_threshold degrees.  
                - len(PTS) <= len(line)
                - PTS is just line with some elements removed.
    !*/

    py::register_exception<no_convex_quadrilateral>(m, "no_convex_quadrilateral");

    m.def("extract_image_4points", &py_extract_image_4points<uint8_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<uint16_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<uint32_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<uint64_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<int8_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<int16_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<int32_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<int64_t>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<float>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<double>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"));
    m.def("extract_image_4points", &py_extract_image_4points<rgb_pixel>, py::arg("img"), py::arg("corners"), py::arg("rows"), py::arg("columns"),
"requires \n\
    - corners is a list of dpoint or line objects. \n\
    - len(corners) == 4 \n\
    - rows >= 0 \n\
    - columns >= 0 \n\
ensures \n\
    - The returned image has the given number of rows and columns. \n\
    - if (corners contains dpoints) then \n\
        - The 4 points in corners define a convex quadrilateral and this function \n\
          extracts that part of the input image img and returns it.  Therefore, \n\
          each corner of the quadrilateral is associated to a corner of the \n\
          extracted image and bilinear interpolation and a projective mapping is \n\
          used to transform the pixels in the quadrilateral into the output image. \n\
          To determine which corners of the quadrilateral map to which corners of \n\
          the returned image we fit the tightest possible rectangle to the \n\
          quadrilateral and map its vertices to their nearest rectangle corners. \n\
          These corners are then trivially mapped to the output image (i.e.  upper \n\
          left corner to upper left corner, upper right corner to upper right \n\
          corner, etc.). \n\
    - else \n\
        - This routine finds the 4 intersecting points of the given lines which \n\
          form a convex quadrilateral and uses them as described above to extract \n\
          an image.   i.e. It just then calls: extract_image_4points(img, \n\
          intersections_between_lines, rows, columns). \n\
        - If no convex quadrilateral can be made from the given lines then this \n\
          routine throws no_convex_quadrilateral." 
    /*!
        requires
            - corners is a list of dpoint or line objects.
            - len(corners) == 4
            - rows >= 0
            - columns >= 0
        ensures
            - The returned image has the given number of rows and columns.
            - if (corners contains dpoints) then
                - The 4 points in corners define a convex quadrilateral and this function
                  extracts that part of the input image img and returns it.  Therefore,
                  each corner of the quadrilateral is associated to a corner of the
                  extracted image and bilinear interpolation and a projective mapping is
                  used to transform the pixels in the quadrilateral into the output image.
                  To determine which corners of the quadrilateral map to which corners of
                  the returned image we fit the tightest possible rectangle to the
                  quadrilateral and map its vertices to their nearest rectangle corners.
                  These corners are then trivially mapped to the output image (i.e.  upper
                  left corner to upper left corner, upper right corner to upper right
                  corner, etc.).
            - else
                - This routine finds the 4 intersecting points of the given lines which
                  form a convex quadrilateral and uses them as described above to extract
                  an image.   i.e. It just then calls: extract_image_4points(img,
                  intersections_between_lines, rows, columns).
                - If no convex quadrilateral can be made from the given lines then this
                  routine throws no_convex_quadrilateral.
    !*/
          );


    m.def("transform_image", &py_transform_image<uint8_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<uint16_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<uint32_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<uint64_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int8_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int16_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int32_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int64_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<float>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<double>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<rgb_pixel>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"),
"requires \n\
    - rows > 0 \n\
    - columns > 0 \n\
ensures \n\
    - Returns an image that is the given rows by columns in size and contains a \n\
      transformed part of img.  To do this, we interpret map_point as a mapping \n\
      from pixels in the returned image to pixels in the input img.  transform_image()  \n\
      uses this mapping and bilinear interpolation to fill the output image with an \n\
      interpolated copy of img.   \n\
    - Any locations in the output image that map to pixels outside img are set to 0." 
    /*!
        requires
            - rows > 0
            - columns > 0
        ensures
            - Returns an image that is the given rows by columns in size and contains a
              transformed part of img.  To do this, we interpret map_point as a mapping
              from pixels in the returned image to pixels in the input img.  transform_image() 
              uses this mapping and bilinear interpolation to fill the output image with an
              interpolated copy of img.  
            - Any locations in the output image that map to pixels outside img are set to 0.
    !*/
        );

}


