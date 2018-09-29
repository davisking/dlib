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
py::array convert_image_scaled (
    const numpy_image<T>& img,
    const string& dtype,
    const double thresh = 4
)
{
    if (dtype == "uint8")    {numpy_image<uint8_t>   out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "uint16")   {numpy_image<uint16_t>  out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "uint32")   {numpy_image<uint32_t>  out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "uint64")   {numpy_image<uint64_t>  out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "int8")     {numpy_image<int8_t>    out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "int16")    {numpy_image<int16_t>   out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "int32")    {numpy_image<int32_t>   out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "int64")    {numpy_image<int64_t>   out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "float32")  {numpy_image<float>     out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "float64")  {numpy_image<double>    out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "float")    {numpy_image<float>     out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "double")   {numpy_image<double>    out; assign_image_scaled(out, img, thresh); return out;}
    if (dtype == "rgb_pixel"){numpy_image<rgb_pixel> out; assign_image_scaled(out, img, thresh); return out;}


    throw dlib::error("convert_image_scaled() called with invalid dtype, must be one of these strings: \n"
        "uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel");
}

// ----------------------------------------------------------------------------------------

struct py_pyramid_down
{

    void dostuff(point) {}

    py_pyramid_down(
    ) = default;

    py_pyramid_down (
        unsigned int N_
    ) : N(N_) 
    {
        DLIB_CASSERT( 1 <= N && N <= 20, "pyramid downsampling rate must be between 1 and 20.");
    }

    unsigned int pyramid_downsampling_rate (
    ) const { return N; }

    template <typename T>
    dlib::vector<double,2> point_down (
        const dlib::vector<T,2>& pp
    ) const
    {
        dpoint p = pp;
        switch(N)
        {
            case 1: return pyr1.point_down(p);
            case 2: return pyr2.point_down(p);
            case 3: return pyr3.point_down(p);
            case 4: return pyr4.point_down(p);
            case 5: return pyr5.point_down(p);
            case 6: return pyr6.point_down(p);
            case 7: return pyr7.point_down(p);
            case 8: return pyr8.point_down(p);
            case 9: return pyr9.point_down(p);
            case 10: return pyr10.point_down(p);
            case 11: return pyr11.point_down(p);
            case 12: return pyr12.point_down(p);
            case 13: return pyr13.point_down(p);
            case 14: return pyr14.point_down(p);
            case 15: return pyr15.point_down(p);
            case 16: return pyr16.point_down(p);
            case 17: return pyr17.point_down(p);
            case 18: return pyr18.point_down(p);
            case 19: return pyr19.point_down(p);
            case 20: return pyr20.point_down(p);
        }

        DLIB_CASSERT(false, "This should never happen");
    }

    template <typename T>
    dlib::vector<double,2> point_up (
        const dlib::vector<T,2>& pp
    ) const
    {
        dpoint p = pp;
        switch(N)
        {
            case 1: return pyr1.point_up(p);
            case 2: return pyr2.point_up(p);
            case 3: return pyr3.point_up(p);
            case 4: return pyr4.point_up(p);
            case 5: return pyr5.point_up(p);
            case 6: return pyr6.point_up(p);
            case 7: return pyr7.point_up(p);
            case 8: return pyr8.point_up(p);
            case 9: return pyr9.point_up(p);
            case 10: return pyr10.point_up(p);
            case 11: return pyr11.point_up(p);
            case 12: return pyr12.point_up(p);
            case 13: return pyr13.point_up(p);
            case 14: return pyr14.point_up(p);
            case 15: return pyr15.point_up(p);
            case 16: return pyr16.point_up(p);
            case 17: return pyr17.point_up(p);
            case 18: return pyr18.point_up(p);
            case 19: return pyr19.point_up(p);
            case 20: return pyr20.point_up(p);
        }
        DLIB_CASSERT(false, "This should never happen");
    }

// -----------------------------

    template <typename T>
    dlib::vector<double,2> point_down2 (
        const dlib::vector<T,2>& p,
        unsigned int levels
    ) const
    {
        dlib::vector<double,2> temp = p;
        for (unsigned int i = 0; i < levels; ++i)
            temp = point_down(temp);
        return temp;
    }

    template <typename T>
    dlib::vector<double,2> point_up2 (
        const dlib::vector<T,2>& p,
        unsigned int levels
    ) const
    {
        dlib::vector<double,2> temp = p;
        for (unsigned int i = 0; i < levels; ++i)
            temp = point_up(temp);
        return temp;
    }

// -----------------------------

    template <typename rect_type>
    rect_type rect_up (
        const rect_type& rect
    ) const
    {
        return rect_type(point_up(rect.tl_corner()), point_up(rect.br_corner()));
    }

    template <typename rect_type>
    rect_type rect_up2 (
        const rect_type& rect,
        unsigned int levels
    ) const
    {
        return rect_type(point_up2(rect.tl_corner(),levels), point_up2(rect.br_corner(),levels));
    }

// -----------------------------

    template <typename rect_type>
    rect_type rect_down (
        const rect_type& rect
    ) const
    {
        return rect_type(point_down(rect.tl_corner()), point_down(rect.br_corner()));
    }

    template <typename rect_type>
    rect_type rect_down2 (
        const rect_type& rect,
        unsigned int levels
    ) const
    {
        return rect_type(point_down2(rect.tl_corner(),levels), point_down2(rect.br_corner(),levels));
    }

    template <
        typename T
        >
    numpy_image<T> down (
        const numpy_image<T>& img
    ) const
    {

        numpy_image<T> down;
        switch(N)
        {
            case 1: pyr1(img,down); break;
            case 2: pyr2(img,down); break;
            case 3: pyr3(img,down); break;
            case 4: pyr4(img,down); break;
            case 5: pyr5(img,down); break;
            case 6: pyr6(img,down); break;
            case 7: pyr7(img,down); break;
            case 8: pyr8(img,down); break;
            case 9: pyr9(img,down); break;
            case 10: pyr10(img,down); break;
            case 11: pyr11(img,down); break;
            case 12: pyr12(img,down); break;
            case 13: pyr13(img,down); break;
            case 14: pyr14(img,down); break;
            case 15: pyr15(img,down); break;
            case 16: pyr16(img,down); break;
            case 17: pyr17(img,down); break;
            case 18: pyr18(img,down); break;
            case 19: pyr19(img,down); break;
            case 20: pyr20(img,down); break;
        }

        return down;
    }

private:
    unsigned int N = 2;

    pyramid_down<1> pyr1;
    pyramid_down<2> pyr2;
    pyramid_down<3> pyr3;
    pyramid_down<4> pyr4;
    pyramid_down<5> pyr5;
    pyramid_down<6> pyr6;
    pyramid_down<7> pyr7;
    pyramid_down<8> pyr8;
    pyramid_down<9> pyr9;
    pyramid_down<10> pyr10;
    pyramid_down<11> pyr11;
    pyramid_down<12> pyr12;
    pyramid_down<13> pyr13;
    pyramid_down<14> pyr14;
    pyramid_down<15> pyr15;
    pyramid_down<16> pyr16;
    pyramid_down<17> pyr17;
    pyramid_down<18> pyr18;
    pyramid_down<19> pyr19;
    pyramid_down<20> pyr20;

};

// ----------------------------------------------------------------------------------------

py::tuple py_find_bright_lines (
    const numpy_image<float>& xx,
    const numpy_image<float>& xy,
    const numpy_image<float>& yy
)
{
    numpy_image<float> horz, vert;
    find_bright_lines(xx,xy,yy,horz,vert);
    return py::make_tuple(horz,vert);
}

py::tuple py_find_dark_lines (
    const numpy_image<float>& xx,
    const numpy_image<float>& xy,
    const numpy_image<float>& yy
)
{
    numpy_image<float> horz, vert;
    find_dark_lines(xx,xy,yy,horz,vert);
    return py::make_tuple(horz,vert);
}

numpy_image<float> py_find_bright_keypoints (
    const numpy_image<float>& xx,
    const numpy_image<float>& xy,
    const numpy_image<float>& yy
)
{
    numpy_image<float> sal;
    find_bright_keypoints(xx,xy,yy,sal);
    return sal;
}

numpy_image<float> py_find_dark_keypoints (
    const numpy_image<float>& xx,
    const numpy_image<float>& xy,
    const numpy_image<float>& yy
)
{
    numpy_image<float> sal;
    find_dark_keypoints(xx,xy,yy,sal);
    return sal;
}

template <typename T>
py::tuple py_sobel_edge_detector (
    const numpy_image<T>& img
)
{
    numpy_image<float> horz, vert;
    sobel_edge_detector(img, horz, vert);
    return py::make_tuple(horz,vert);
}

numpy_image<float> py_suppress_non_maximum_edges (
    const numpy_image<float>& horz,
    const numpy_image<float>& vert
)
{
    numpy_image<float> out;
    suppress_non_maximum_edges(horz,vert,out);
    return out;
}

numpy_image<float> py_suppress_non_maximum_edges2 (
    const py::tuple& horz_and_vert_gradients 
)
{
    numpy_image<float> out, horz, vert;
    horz = horz_and_vert_gradients[0];
    vert = horz_and_vert_gradients[1];
    suppress_non_maximum_edges(horz,vert,out);
    return out;
}

template <typename T> 
std::vector<point> py_find_peaks (
    const numpy_image<T>& img,
    const double non_max_suppression_radius,
    const T& thresh
)
{
    return find_peaks(img, non_max_suppression_radius, thresh);
}

template <typename T> 
std::vector<point> py_find_peaks2 (
    const numpy_image<T>& img,
    const double non_max_suppression_radius
)
{
    return find_peaks(img, non_max_suppression_radius, partition_pixels(img));
}


// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<unsigned char> py_hysteresis_threshold (
    const numpy_image<T>& img,
    T lower_thresh,
    T upper_thresh
)
{
    numpy_image<unsigned char> out;
    hysteresis_threshold(img, out, lower_thresh, upper_thresh);
    return out;
}

template <typename T>
numpy_image<unsigned char> py_hysteresis_threshold2 (
    const numpy_image<T>& img
)
{
    numpy_image<unsigned char> out;
    hysteresis_threshold(img, out);
    return out;
}

// ----------------------------------------------------------------------------------------

void bind_image_classes3(py::module& m)
{
    const char* docs;

    docs = 
"requires \n\
    - thresh > 0 \n\
ensures \n\
    - Converts an image to a target pixel type.  dtype must be a string containing one of the following: \n\
      uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel \n\
 \n\
      The contents of img will be scaled to fit the dynamic range of the target \n\
      pixel type.  The thresh parameter is used to filter source pixel values which \n\
      are outliers.  These outliers will saturate at the edge of the destination \n\
      image's dynamic range. \n\
    - Specifically, for all valid r and c: \n\
        - We scale img[r][c] into the dynamic range of the target pixel type.  This \n\
          is done using the mean and standard deviation of img. Call the mean M and \n\
          the standard deviation D.  Then the scaling from source to destination is \n\
          performed using the following mapping: \n\
            let SRC_UPPER  = min(M + thresh*D, max(img)) \n\
            let SRC_LOWER  = max(M - thresh*D, min(img)) \n\
            let DEST_UPPER = max value possible for the selected dtype.  \n\
            let DEST_LOWER = min value possible for the selected dtype. \n\
 \n\
            MAPPING: [SRC_LOWER, SRC_UPPER] -> [DEST_LOWER, DEST_UPPER] \n\
 \n\
          Where this mapping is a linear mapping of values from the left range \n\
          into the right range of values.  Source pixel values outside the left \n\
          range are modified to be at the appropriate end of the range.";
    /*!
        requires
            - thresh > 0
        ensures
            - Converts an image to a target pixel type.  dtype must be a string containing one of the following:
              uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel

              The contents of img will be scaled to fit the dynamic range of the target
              pixel type.  The thresh parameter is used to filter source pixel values which
              are outliers.  These outliers will saturate at the edge of the destination
              image's dynamic range.
            - Specifically, for all valid r and c:
                - We scale img[r][c] into the dynamic range of the target pixel type.  This
                  is done using the mean and standard deviation of img. Call the mean M and
                  the standard deviation D.  Then the scaling from source to destination is
                  performed using the following mapping:
                    let SRC_UPPER  = min(M + thresh*D, max(img))
                    let SRC_LOWER  = max(M - thresh*D, min(img))
                    let DEST_UPPER = max value possible for the selected dtype. 
                    let DEST_LOWER = min value possible for the selected dtype.

                    MAPPING: [SRC_LOWER, SRC_UPPER] -> [DEST_LOWER, DEST_UPPER]

                  Where this mapping is a linear mapping of values from the left range
                  into the right range of values.  Source pixel values outside the left
                  range are modified to be at the appropriate end of the range.
    !*/
    m.def("convert_image_scaled", convert_image_scaled<uint8_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<uint16_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<uint32_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<uint64_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<int8_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<int16_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<int32_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<int64_t>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<float>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<double>, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);
    m.def("convert_image_scaled", convert_image_scaled<rgb_pixel>, docs, py::arg("img"), py::arg("dtype"), py::arg("thresh")=4);



    const char* class_docs;


    class_docs =
"This is a simple object to help create image pyramids.  In particular, it \n\
downsamples images at a ratio of N to N-1. \n\
 \n\
Note that setting N to 1 means that this object functions like \n\
pyramid_disable (defined at the bottom of this file).   \n\
 \n\
WARNING, when mapping rectangles from one layer of a pyramid \n\
to another you might end up with rectangles which extend slightly  \n\
outside your images.  This is because points on the border of an  \n\
image at a higher pyramid layer might correspond to points outside  \n\
images at lower layers.  So just keep this in mind.  Note also \n\
that it's easy to deal with.  Just say something like this: \n\
    rect = rect.intersect(get_rect(my_image)); # keep rect inside my_image ";
        /*!
                This is a simple object to help create image pyramids.  In particular, it
                downsamples images at a ratio of N to N-1.

                Note that setting N to 1 means that this object functions like
                pyramid_disable (defined at the bottom of this file).  

                WARNING, when mapping rectangles from one layer of a pyramid
                to another you might end up with rectangles which extend slightly 
                outside your images.  This is because points on the border of an 
                image at a higher pyramid layer might correspond to points outside 
                images at lower layers.  So just keep this in mind.  Note also
                that it's easy to deal with.  Just say something like this:
                    rect = rect.intersect(get_rect(my_image)); # keep rect inside my_image 
        !*/

    docs =
"- Downsamples img to make a new image that is roughly (pyramid_downsampling_rate()-1)/pyramid_downsampling_rate()  \n\
  times the size of the original image.   \n\
- The location of a point P in original image will show up at point point_down(P) \n\
  in the downsampled image.   \n\
- Note that some points on the border of the original image might correspond to  \n\
  points outside the downsampled image.";
        /*!
          - Downsamples img to make a new image that is roughly (pyramid_downsampling_rate()-1)/pyramid_downsampling_rate() 
            times the size of the original image.  
          - The location of a point P in original image will show up at point point_down(P)
            in the downsampled image.  
          - Note that some points on the border of the original image might correspond to 
            points outside the downsampled image.  
        !*/
    py::class_<py_pyramid_down>(m, "pyramid_down", class_docs)
        .def(py::init<unsigned int>(), "Creates this class with the provided downsampling rate. i.e. pyramid_downsampling_rate()==N. \nN must be in the range 1 to 20.", py::arg("N"))
        .def(py::init<>(), "Creates this class with pyramid_downsampling_rate()==2")
        .def("pyramid_downsampling_rate", &py_pyramid_down::pyramid_downsampling_rate,
            "Returns a number N that defines the downsampling rate.  In particular, images are downsampled by a factor of N to N-1.")
        .def("point_up", &py_pyramid_down::point_up<long>,   py::arg("p"))
        .def("point_up", &py_pyramid_down::point_up<double>, 
            "Maps from pixels in a downsampled image to pixels in the original image.",  py::arg("p"))
        .def("point_up", &py_pyramid_down::point_up2<long>,   py::arg("p"), py::arg("levels"))
        .def("point_up", &py_pyramid_down::point_up2<double>, 
            "Applies point_up() to p levels times and returns the result.",  py::arg("p"), py::arg("levels"))
        .def("point_down", &py_pyramid_down::point_down<long>,   py::arg("p"))
        .def("point_down", &py_pyramid_down::point_down<double>, 
            "Maps from pixels in a source image to the corresponding pixels in the downsampled image.", py::arg("p"))
        .def("point_down", &py_pyramid_down::point_down2<long>,   py::arg("p"), py::arg("levels"))
        .def("point_down", &py_pyramid_down::point_down2<double>, "Applies point_down() to p levels times and returns the result.",   
            py::arg("p"), py::arg("levels"))
        .def("rect_down", &py_pyramid_down::rect_down<rectangle>,   py::arg("rect"))
        .def("rect_down", &py_pyramid_down::rect_down<drectangle>,
          "returns drectangle(point_down(rect.tl_corner()), point_down(rect.br_corner()));\n (i.e. maps rect into a downsampled)",
          py::arg("rect"))
        .def("rect_down", &py_pyramid_down::rect_down2<rectangle>,   py::arg("rect"), py::arg("levels"))
        .def("rect_down", &py_pyramid_down::rect_down2<drectangle>, "Applies rect_down() to rect levels times and returns the result.",
            py::arg("rect"), py::arg("levels"))
        .def("rect_up", &py_pyramid_down::rect_up<rectangle>,   py::arg("rect"))
        .def("rect_up", &py_pyramid_down::rect_up<drectangle>,   
          "returns drectangle(point_up(rect.tl_corner()), point_up(rect.br_corner()));\n (i.e. maps rect into a parent image)",
            py::arg("rect"))
        .def("rect_up", &py_pyramid_down::rect_up2<rectangle>,   py::arg("rect"), py::arg("levels"))
        .def("rect_up", &py_pyramid_down::rect_up2<drectangle>,  "Applies rect_up() to rect levels times and returns the result.",
            py::arg("p"), py::arg("levels"))
        .def("__call__", &py_pyramid_down::down<uint8_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<uint16_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<uint32_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<uint64_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<int8_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<int16_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<int32_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<int64_t>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<float>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<double>,   py::arg("img"))
        .def("__call__", &py_pyramid_down::down<rgb_pixel>, docs,  py::arg("img"));


    docs =
"requires \n\
    - xx, xy, and yy all have the same dimensions. \n\
ensures \n\
    - This routine is similar to sobel_edge_detector(), except instead of finding \n\
      an edge it finds a bright/white line.  For example, the border between a \n\
      black piece of paper and a white table is an edge, but a curve drawn with a \n\
      pencil on a piece of paper makes a line.  Therefore, the output of this \n\
      routine is a vector field encoded in the horz and vert images, which are \n\
      returned in a tuple where the first element is horz and the second is vert. \n\
 \n\
      The vector obtains a large magnitude when centered on a bright line in an image and the \n\
      direction of the vector is perpendicular to the line.  To be very precise, \n\
      each vector points in the direction of greatest change in second derivative \n\
      and the magnitude of the vector encodes the derivative magnitude in that \n\
      direction.  Moreover, if the second derivative is positive then the output \n\
      vector is zero.  This zeroing if positive gradients causes the output to be \n\
      sensitive only to bright lines surrounded by darker pixels. \n\
 \n\
    - We assume that xx, xy, and yy are the 3 second order gradients of the image \n\
      in question.  You can obtain these gradients using the image_gradients class. \n\
    - The output images will have the same dimensions as the input images. ";
    /*!
        requires
            - xx, xy, and yy all have the same dimensions.
        ensures
            - This routine is similar to sobel_edge_detector(), except instead of finding
              an edge it finds a bright/white line.  For example, the border between a
              black piece of paper and a white table is an edge, but a curve drawn with a
              pencil on a piece of paper makes a line.  Therefore, the output of this
              routine is a vector field encoded in the horz and vert images, which are
              returned in a tuple where the first element is horz and the second is vert.

              The vector obtains a large magnitude when centered on a bright line in an image and the
              direction of the vector is perpendicular to the line.  To be very precise,
              each vector points in the direction of greatest change in second derivative
              and the magnitude of the vector encodes the derivative magnitude in that
              direction.  Moreover, if the second derivative is positive then the output
              vector is zero.  This zeroing if positive gradients causes the output to be
              sensitive only to bright lines surrounded by darker pixels.

            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output images will have the same dimensions as the input images. 
    !*/
    m.def("find_bright_lines",     &py_find_bright_lines,     docs, py::arg("xx"), py::arg("xy"), py::arg("yy"));



    docs =
"requires \n\
    - xx, xy, and yy all have the same dimensions. \n\
ensures \n\
    - This routine is similar to sobel_edge_detector(), except instead of finding \n\
      an edge it finds a dark line.  For example, the border between a black piece \n\
      of paper and a white table is an edge, but a curve drawn with a pencil on a \n\
      piece of paper makes a line.  Therefore, the output of this routine is a \n\
      vector field encoded in the horz and vert images, which are returned in a \n\
      tuple where the first element is horz and the second is vert. \n\
 \n\
      The vector obtains a large magnitude when centered on a dark line in an image \n\
      and the direction of the vector is perpendicular to the line.  To be very \n\
      precise, each vector points in the direction of greatest change in second \n\
      derivative and the magnitude of the vector encodes the derivative magnitude \n\
      in that direction.  Moreover, if the second derivative is negative then the \n\
      output vector is zero.  This zeroing if negative gradients causes the output \n\
      to be sensitive only to dark lines surrounded by darker pixels. \n\
 \n\
    - We assume that xx, xy, and yy are the 3 second order gradients of the image \n\
      in question.  You can obtain these gradients using the image_gradients class. \n\
    - The output images will have the same dimensions as the input images. ";
    /*!
        requires
            - xx, xy, and yy all have the same dimensions.
        ensures
            - This routine is similar to sobel_edge_detector(), except instead of finding
              an edge it finds a dark line.  For example, the border between a black piece
              of paper and a white table is an edge, but a curve drawn with a pencil on a
              piece of paper makes a line.  Therefore, the output of this routine is a
              vector field encoded in the horz and vert images, which are returned in a
              tuple where the first element is horz and the second is vert.

              The vector obtains a large magnitude when centered on a dark line in an image
              and the direction of the vector is perpendicular to the line.  To be very
              precise, each vector points in the direction of greatest change in second
              derivative and the magnitude of the vector encodes the derivative magnitude
              in that direction.  Moreover, if the second derivative is negative then the
              output vector is zero.  This zeroing if negative gradients causes the output
              to be sensitive only to dark lines surrounded by darker pixels.

            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output images will have the same dimensions as the input images. 
    !*/
    m.def("find_dark_lines",       &py_find_dark_lines,       docs, py::arg("xx"), py::arg("xy"), py::arg("yy"));



    docs =
"requires \n\
    - xx, xy, and yy all have the same dimensions. \n\
ensures \n\
    - This routine finds bright \"keypoints\" in an image.  In general, these are \n\
      bright/white localized blobs.  It does this by computing the determinant of \n\
      the image Hessian at each location and storing this value into the returned \n\
      image if both eigenvalues of the Hessian are negative.  If either eigenvalue \n\
      is positive then the output value for that pixel is 0.  I.e. \n\
        - Let OUT denote the returned image. \n\
        - for all valid r,c: \n\
            - OUT[r][c] == a number >= 0 and larger values indicate the \n\
              presence of a keypoint at this pixel location. \n\
    - We assume that xx, xy, and yy are the 3 second order gradients of the image \n\
      in question.  You can obtain these gradients using the image_gradients class. \n\
    - The output image will have the same dimensions as the input images.";
    /*!
        requires
            - xx, xy, and yy all have the same dimensions.
        ensures
            - This routine finds bright "keypoints" in an image.  In general, these are
              bright/white localized blobs.  It does this by computing the determinant of
              the image Hessian at each location and storing this value into the returned
              image if both eigenvalues of the Hessian are negative.  If either eigenvalue
              is positive then the output value for that pixel is 0.  I.e.
                - Let OUT denote the returned image.
                - for all valid r,c:
                    - OUT[r][c] == a number >= 0 and larger values indicate the
                      presence of a keypoint at this pixel location.
            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output image will have the same dimensions as the input images.
    !*/
    m.def("find_bright_keypoints", &py_find_bright_keypoints, docs, py::arg("xx"), py::arg("xy"), py::arg("yy"));



    docs =
"requires \n\
    - xx, xy, and yy all have the same dimensions. \n\
ensures \n\
    - This routine finds dark \"keypoints\" in an image.  In general, these are \n\
      dark localized blobs.  It does this by computing the determinant of \n\
      the image Hessian at each location and storing this value into the returned \n\
      image if both eigenvalues of the Hessian are negative.  If either eigenvalue \n\
      is negative then the output value for that pixel is 0.  I.e. \n\
        - Let OUT denote the returned image. \n\
        - for all valid r,c: \n\
            - OUT[r][c] == a number >= 0 and larger values indicate the \n\
              presence of a keypoint at this pixel location. \n\
    - We assume that xx, xy, and yy are the 3 second order gradients of the image \n\
      in question.  You can obtain these gradients using the image_gradients class. \n\
    - The output image will have the same dimensions as the input images.";
    /*!
        requires
            - xx, xy, and yy all have the same dimensions.
        ensures
            - This routine finds dark "keypoints" in an image.  In general, these are
              dark localized blobs.  It does this by computing the determinant of
              the image Hessian at each location and storing this value into the returned
              image if both eigenvalues of the Hessian are negative.  If either eigenvalue
              is negative then the output value for that pixel is 0.  I.e.
                - Let OUT denote the returned image.
                - for all valid r,c:
                    - OUT[r][c] == a number >= 0 and larger values indicate the
                      presence of a keypoint at this pixel location.
            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output image will have the same dimensions as the input images.
    !*/
    m.def("find_dark_keypoints",   &py_find_dark_keypoints,   docs, py::arg("xx"), py::arg("xy"), py::arg("yy"));



    docs = 
"requires \n\
    - The two input images have the same dimensions. \n\
ensures \n\
    - Returns an image, of the same dimensions as the input.  Each element in this \n\
      image holds the edge strength at that location.  Moreover, edge pixels that are not  \n\
      local maximizers have been set to 0. \n\
    - let edge_strength(r,c) == sqrt(pow(horz[r][c],2) + pow(vert[r][c],2)) \n\
      (i.e. The Euclidean norm of the gradient) \n\
    - let OUT denote the returned image. \n\
    - for all valid r and c: \n\
        - if (edge_strength(r,c) is at a maximum with respect to its 2 neighboring \n\
          pixels along the line indicated by the image gradient vector (horz[r][c],vert[r][c])) then \n\
            - OUT[r][c] == edge_strength(r,c) \n\
        - else \n\
            - OUT[r][c] == 0";
    /*!
        requires
            - The two input images have the same dimensions.
        ensures
            - Returns an image, of the same dimensions as the input.  Each element in this
              image holds the edge strength at that location.  Moreover, edge pixels that are not 
              local maximizers have been set to 0.
            - let edge_strength(r,c) == sqrt(pow(horz[r][c],2) + pow(vert[r][c],2))
              (i.e. The Euclidean norm of the gradient)
            - let OUT denote the returned image.
            - for all valid r and c:
                - if (edge_strength(r,c) is at a maximum with respect to its 2 neighboring
                  pixels along the line indicated by the image gradient vector (horz[r][c],vert[r][c])) then
                    - OUT[r][c] == edge_strength(r,c)
                - else
                    - OUT[r][c] == 0
    !*/
    m.def("suppress_non_maximum_edges", &py_suppress_non_maximum_edges, docs, py::arg("horz"), py::arg("vert"));
    m.def("suppress_non_maximum_edges", &py_suppress_non_maximum_edges2,
        "Performs: return suppress_non_maximum_edges(horz_and_vert_gradients[0], horz_and_vert_gradients[1])",
        py::arg("horz_and_vert_gradients"));



    docs =
"requires \n\
    - non_max_suppression_radius >= 0 \n\
ensures \n\
    - Scans the given image and finds all pixels with values >= thresh that are \n\
      also local maximums within their 8-connected neighborhood of the image.  Such \n\
      pixels are collected, sorted in decreasing order of their pixel values, and \n\
      then non-maximum suppression is applied to this list of points using the \n\
      given non_max_suppression_radius.  The final list of peaks is then returned. \n\
 \n\
      Therefore, the returned list, V, will have these properties: \n\
        - len(V) == the number of peaks found in the image. \n\
        - When measured in image coordinates, no elements of V are within \n\
          non_max_suppression_radius distance of each other.  That is, for all valid i!=j \n\
          it is true that length(V[i]-V[j]) > non_max_suppression_radius. \n\
        - For each element of V, that element has the maximum pixel value of all \n\
          pixels in the ball centered on that pixel with radius \n\
          non_max_suppression_radius.";
    /*!
        requires
            - non_max_suppression_radius >= 0
        ensures
            - Scans the given image and finds all pixels with values >= thresh that are
              also local maximums within their 8-connected neighborhood of the image.  Such
              pixels are collected, sorted in decreasing order of their pixel values, and
              then non-maximum suppression is applied to this list of points using the
              given non_max_suppression_radius.  The final list of peaks is then returned.

              Therefore, the returned list, V, will have these properties:
                - len(V) == the number of peaks found in the image.
                - When measured in image coordinates, no elements of V are within
                  non_max_suppression_radius distance of each other.  That is, for all valid i!=j
                  it is true that length(V[i]-V[j]) > non_max_suppression_radius.
                - For each element of V, that element has the maximum pixel value of all
                  pixels in the ball centered on that pixel with radius
                  non_max_suppression_radius.
    !*/
    m.def("find_peaks", &py_find_peaks<float>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<double>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<uint8_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<uint16_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<uint32_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<uint64_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<int8_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<int16_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<int32_t>, py::arg("img"), py::arg("non_max_suppression_radius"), py::arg("thresh"));
    m.def("find_peaks", &py_find_peaks<int64_t>, py::arg("img"), docs, py::arg("non_max_suppression_radius"), py::arg("thresh"));

    m.def("find_peaks", &py_find_peaks2<float>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<double>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<uint8_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<uint16_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<uint32_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<uint64_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<int8_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<int16_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<int32_t>, py::arg("img"), py::arg("non_max_suppression_radius")=0);
    m.def("find_peaks", &py_find_peaks2<int64_t>, py::arg("img"),
        "performs: return find_peaks(img, non_max_suppression_radius, partition_pixels(img))",
        py::arg("non_max_suppression_radius")=0);



    docs =
"Applies the sobel edge detector to the given input image and returns two gradient \n\
images in a tuple.  The first contains the x gradients and the second contains the \n\
y gradients of the image.";
    /*!
         Applies the sobel edge detector to the given input image and returns two gradient
         images in a tuple.  The first contains the x gradients and the second contains the
         y gradients of the image.
    !*/
    m.def("sobel_edge_detector", &py_sobel_edge_detector<uint8_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<uint16_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<uint32_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<uint64_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<int8_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<int16_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<int32_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<int64_t>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<float>, py::arg("img"));
    m.def("sobel_edge_detector", &py_sobel_edge_detector<double>, docs, py::arg("img"));


    docs =
"Applies hysteresis thresholding to img and returns the results.  In particular, \n\
pixels in img with values >= upper_thresh have an output value of 255 and all \n\
others have a value of 0 unless they are >= lower_thresh and are connected to a \n\
pixel with a value >= upper_thresh, in which case they have a value of 255.  Here \n\
pixels are connected if there is a path between them composed of pixels that would \n\
receive an output of 255.";
    /*!
        Applies hysteresis thresholding to img and returns the results.  In particular,
        pixels in img with values >= upper_thresh have an output value of 255 and all
        others have a value of 0 unless they are >= lower_thresh and are connected to a
        pixel with a value >= upper_thresh, in which case they have a value of 255.  Here
        pixels are connected if there is a path between them composed of pixels that would
        receive an output of 255.
    !*/
    m.def("hysteresis_threshold", &py_hysteresis_threshold<uint8_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<uint16_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<uint32_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<uint64_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<int8_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<int16_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<int32_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<int64_t>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<float>, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold<double>, docs, py::arg("img"), py::arg("lower_thresh"), py::arg("upper_thresh"));

    docs =
"performs: return hysteresis_threshold(img, t1, t2) where the thresholds \n\
are first obtained by calling [t1, t2]=partition_pixels(img).";
    /*!
        performs: return hysteresis_threshold(img, t1, t2) where the thresholds
        are first obtained by calling [t1, t2]=partition_pixels(img).
    !*/
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<uint8_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<uint16_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<uint32_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<uint64_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<int8_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<int16_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<int32_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<int64_t>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<float>, py::arg("img"));
    m.def("hysteresis_threshold", &py_hysteresis_threshold2<double>, docs, py::arg("img"));
}

