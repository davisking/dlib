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

string print_rgb_pixel_str(const rgb_pixel& p)
{
    std::ostringstream sout;
    sout << "red: "<< (int)p.red
         << ", green: "<< (int)p.green
         << ", blue: "<< (int)p.blue;
    return sout.str();
}

string print_rgb_pixel_repr(const rgb_pixel& p)
{
    std::ostringstream sout;
    sout << "rgb_pixel(" << (int)p.red << "," << (int)p.green << "," << (int)p.blue << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<unsigned char> py_threshold_image2(
    const numpy_image<T>& in_img,
    typename pixel_traits<T>::basic_pixel_type thresh
)
{
    numpy_image<unsigned char> out_img;
    threshold_image(in_img, out_img, thresh);
    return out_img;
}

template <typename T>
numpy_image<unsigned char> py_threshold_image(
    const numpy_image<T>& in_img
)
{
    numpy_image<unsigned char> out_img;
    threshold_image(in_img, out_img);
    return out_img;
}

// ----------------------------------------------------------------------------------------

template <typename T>
typename pixel_traits<T>::basic_pixel_type py_partition_pixels (
    const numpy_image<T>& img
)
{
    return partition_pixels(img);
}

template <typename T>
py::tuple py_partition_pixels2 (
    const numpy_image<T>& img,
    int num_thresholds
)
{
    DLIB_CASSERT(1 <= num_thresholds && num_thresholds <= 6);

    typename pixel_traits<T>::basic_pixel_type t1,t2,t3,t4,t5,t6;

    switch(num_thresholds)
    {
        case 1: partition_pixels(img,t1); return py::make_tuple(t1);
        case 2: partition_pixels(img,t1,t2); return py::make_tuple(t1,t2);
        case 3: partition_pixels(img,t1,t2,t3); return py::make_tuple(t1,t2,t3);
        case 4: partition_pixels(img,t1,t2,t3,t4); return py::make_tuple(t1,t2,t3,t4);
        case 5: partition_pixels(img,t1,t2,t3,t4,t5); return py::make_tuple(t1,t2,t3,t4,t5);
        case 6: partition_pixels(img,t1,t2,t3,t4,t5,t6); return py::make_tuple(t1,t2,t3,t4,t5,t6);
    }
    DLIB_CASSERT(false, "This should never happen.");
}

// ----------------------------------------------------------------------------------------

template <typename T>
py::tuple py_gaussian_blur (
    const numpy_image<T>& img,
    double sigma = 1,
    int max_size = 1001
)
{
    numpy_image<T> out;
    auto rect = gaussian_blur(img, out, sigma, max_size);
    return py::make_tuple(out, rect);
}

template <typename T>
py::tuple py_label_connected_blobs (
    const numpy_image<T>& img,
    bool zero_pixels_are_background,
    int neighborhood_connectivity,
    bool connected_if_both_not_zero
)
{
    DLIB_CASSERT(neighborhood_connectivity == 4 ||
                 neighborhood_connectivity == 8 ||
                 neighborhood_connectivity == 24);

    unsigned long num_blobs = 0;

    numpy_image<uint32_t> labels;

    if (zero_pixels_are_background && neighborhood_connectivity == 4 && connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, ::zero_pixels_are_background(), neighbors_4(), ::connected_if_both_not_zero(), labels);
    else if (zero_pixels_are_background && neighborhood_connectivity == 4 && !connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, ::zero_pixels_are_background(), neighbors_4(), connected_if_equal(), labels);
    else if (!zero_pixels_are_background && neighborhood_connectivity == 4 && connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, nothing_is_background(), neighbors_4(), ::connected_if_both_not_zero(), labels);
    else if (!zero_pixels_are_background && neighborhood_connectivity == 4 && !connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, nothing_is_background(), neighbors_4(), connected_if_equal(), labels);

    else if (zero_pixels_are_background && neighborhood_connectivity == 8 && connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, ::zero_pixels_are_background(), neighbors_8(), ::connected_if_both_not_zero(), labels);
    else if (zero_pixels_are_background && neighborhood_connectivity == 8 && !connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, ::zero_pixels_are_background(), neighbors_8(), connected_if_equal(), labels);
    else if (!zero_pixels_are_background && neighborhood_connectivity == 8 && connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, nothing_is_background(), neighbors_8(), ::connected_if_both_not_zero(), labels);
    else if (!zero_pixels_are_background && neighborhood_connectivity == 8 && !connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, nothing_is_background(), neighbors_8(), connected_if_equal(), labels);

    else if (zero_pixels_are_background && neighborhood_connectivity == 24 && connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, ::zero_pixels_are_background(), neighbors_24(), ::connected_if_both_not_zero(), labels);
    else if (zero_pixels_are_background && neighborhood_connectivity == 24 && !connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, ::zero_pixels_are_background(), neighbors_24(), connected_if_equal(), labels);
    else if (!zero_pixels_are_background && neighborhood_connectivity == 24 && connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, nothing_is_background(), neighbors_24(), ::connected_if_both_not_zero(), labels);
    else if (!zero_pixels_are_background && neighborhood_connectivity == 24 && !connected_if_both_not_zero )
        num_blobs = label_connected_blobs(img, nothing_is_background(), neighbors_24(), connected_if_equal(), labels);
    else
        DLIB_CASSERT(false, "this should never happen");

    return py::make_tuple(labels, num_blobs);
}

// ----------------------------------------------------------------------------------------

template <typename T>
py::tuple py_label_connected_blobs_watershed (
    const numpy_image<T>& img,
    const T& background_thresh, 
    const double smoothing
)
{
    numpy_image<uint32_t> labels;
    auto num_blobs = label_connected_blobs_watershed(img, labels, background_thresh, smoothing);
    return py::make_tuple(labels, num_blobs);
}

template <typename T>
py::tuple py_label_connected_blobs_watershed2 (
    const numpy_image<T>& img
)
{
    numpy_image<uint32_t> labels;
    auto num_blobs = label_connected_blobs_watershed(img, labels);
    return py::make_tuple(labels, num_blobs);
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<rgb_pixel> py_randomly_color_image (
    const numpy_image<T>& img
)
{
    numpy_image<rgb_pixel> temp;
    matrix<T> itemp;
    assign_image(itemp, numpy_image<T>(img));
    assign_image(temp, randomly_color_image(itemp));
    return temp;
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<rgb_pixel> py_jet (
    const numpy_image<T>& img
)
{
    numpy_image<rgb_pixel> temp;
    matrix<T> itemp;
    assign_image(itemp, numpy_image<T>(img));
    assign_image(temp, jet(itemp));
    return temp;
}

// ----------------------------------------------------------------------------------------

template <typename T>
py::array convert_image (
    const numpy_image<T>& img,
    const string& dtype
)
{
    if (dtype == "uint8")    {numpy_image<uint8_t>   out; assign_image(out, img); return out;}
    if (dtype == "uint16")   {numpy_image<uint16_t>  out; assign_image(out, img); return out;}
    if (dtype == "uint32")   {numpy_image<uint32_t>  out; assign_image(out, img); return out;}
    if (dtype == "uint64")   {numpy_image<uint64_t>  out; assign_image(out, img); return out;}
    if (dtype == "int8")     {numpy_image<int8_t>    out; assign_image(out, img); return out;}
    if (dtype == "int16")    {numpy_image<int16_t>   out; assign_image(out, img); return out;}
    if (dtype == "int32")    {numpy_image<int32_t>   out; assign_image(out, img); return out;}
    if (dtype == "int64")    {numpy_image<int64_t>   out; assign_image(out, img); return out;}
    if (dtype == "float32")  {numpy_image<float>     out; assign_image(out, img); return out;}
    if (dtype == "float64")  {numpy_image<double>    out; assign_image(out, img); return out;}
    if (dtype == "float")    {numpy_image<float>     out; assign_image(out, img); return out;}
    if (dtype == "double")   {numpy_image<double>    out; assign_image(out, img); return out;}
    if (dtype == "rgb_pixel"){numpy_image<rgb_pixel> out; assign_image(out, img); return out;}


    throw dlib::error("convert_image() called with invalid dtype, must be one of these strings: \n"
        "uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel");
}

py::array as_grayscale(
    const py::array& img
)
{
    if (is_image<rgb_pixel>(img))
    {
        numpy_image<unsigned char> out;
        assign_image(out, numpy_image<rgb_pixel>(img));
        return out;
    }
    else
    {
        return img;
    }
}

// ----------------------------------------------------------------------------------------

numpy_image<unsigned char> py_skeleton(
    numpy_image<unsigned char>& img
)
{
    skeleton(img);
    return img;
}

// ----------------------------------------------------------------------------------------

void bind_image_classes(py::module& m)
{



    py::class_<rgb_pixel>(m, "rgb_pixel")
        .def(py::init<unsigned char,unsigned char,unsigned char>(), py::arg("red"), py::arg("green"), py::arg("blue"))
        .def("__str__", &print_rgb_pixel_str)
        .def("__repr__", &print_rgb_pixel_repr)
        .def_readwrite("red", &rgb_pixel::red)
        .def_readwrite("green", &rgb_pixel::green)
        .def_readwrite("blue", &rgb_pixel::blue);

    const char* docs = "Thresholds img and returns the result.  Pixels in img with grayscale values >= partition_pixels(img) \n" 
              "have an output value of 255 and all others have a value of 0.";
    m.def("threshold_image", &py_threshold_image<unsigned char>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<uint16_t>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<uint32_t>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<float>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<double>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<rgb_pixel>,docs, py::arg("img") );

    docs = "Thresholds img and returns the result.  Pixels in img with grayscale values >= thresh \n"
              "have an output value of 255 and all others have a value of 0.";
    m.def("threshold_image", &py_threshold_image2<unsigned char>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<uint16_t>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<uint32_t>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<float>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<double>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<rgb_pixel>,docs, py::arg("img"), py::arg("thresh") );


    docs = 
"Finds a threshold value that would be reasonable to use with \n\
threshold_image(img, threshold).  It does this by finding the threshold that \n\
partitions the pixels in img into two groups such that the sum of absolute \n\
deviations between each pixel and the mean of its group is minimized.";
    m.def("partition_pixels", &py_partition_pixels<rgb_pixel>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<unsigned char>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<uint16_t>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<uint32_t>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<float>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<double>,docs, py::arg("img") );

    docs = 
"This version of partition_pixels() finds multiple partitions rather than just \n\
one partition.  It does this by first partitioning the pixels just as the \n\
above partition_pixels(img) does.  Then it forms a new image with only pixels \n\
>= that first partition value and recursively partitions this new image. \n\
However, the recursion is implemented in an efficient way which is faster than \n\
explicitly forming these images and calling partition_pixels(), but the \n\
output is the same as if you did.  For example, suppose you called \n\
[t1,t2,t2] = partition_pixels(img,3).  Then we would have: \n\
   - t1 == partition_pixels(img) \n\
   - t2 == partition_pixels(an image with only pixels with values >= t1 in it) \n\
   - t3 == partition_pixels(an image with only pixels with values >= t2 in it)" ;
    m.def("partition_pixels", &py_partition_pixels2<rgb_pixel>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<unsigned char>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<uint16_t>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<uint32_t>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<float>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<double>,docs, py::arg("img"), py::arg("num_thresholds") );

    docs = 
"requires \n\
    - sigma > 0 \n\
    - max_size > 0 \n\
    - max_size is an odd number \n\
ensures \n\
    - Filters img with a Gaussian filter of sigma width.  The actual spatial filter will \n\
      be applied to pixel blocks that are at most max_size wide and max_size tall (note that \n\
      this function will automatically select a smaller block size as appropriate).  The  \n\
      results are returned.  We also return a rectangle which indicates what pixels \n\
      in the returned image are considered non-border pixels and therefore contain \n\
      output from the filter.  E.g. \n\
        - filtered_img,rect = gaussian_blur(img) \n\
      would give you the filtered image and the rectangle in question. \n\
    - The filter is applied to each color channel independently. \n\
    - Pixels close enough to the edge of img to not have the filter still fit  \n\
      inside the image are set to zero. \n\
    - The returned image has the same dimensions as the input image.";
    /*!
        requires
            - sigma > 0
            - max_size > 0
            - max_size is an odd number
        ensures
            - Filters img with a Gaussian filter of sigma width.  The actual spatial filter will
              be applied to pixel blocks that are at most max_size wide and max_size tall (note that
              this function will automatically select a smaller block size as appropriate).  The 
              results are returned.  We also return a rectangle which indicates what pixels
              in the returned image are considered non-border pixels and therefore contain
              output from the filter.  E.g.
                - filtered_img,rect = gaussian_blur(img)
              would give you the filtered image and the rectangle in question.
            - The filter is applied to each color channel independently.
            - Pixels close enough to the edge of img to not have the filter still fit 
              inside the image are set to zero.
            - The returned image has the same dimensions as the input image.
    !*/
    m.def("gaussian_blur", &py_gaussian_blur<rgb_pixel>,py::arg("img"), py::arg("sigma"), py::arg("max_size")=1000 );
    m.def("gaussian_blur", &py_gaussian_blur<unsigned char>,py::arg("img"), py::arg("sigma"), py::arg("max_size")=1000 );
    m.def("gaussian_blur", &py_gaussian_blur<uint16>,py::arg("img"), py::arg("sigma"), py::arg("max_size")=1000 );
    m.def("gaussian_blur", &py_gaussian_blur<uint32>,py::arg("img"), py::arg("sigma"), py::arg("max_size")=1000 );
    m.def("gaussian_blur", &py_gaussian_blur<float>, py::arg("img"), py::arg("sigma"), py::arg("max_size")=1000 );
    m.def("gaussian_blur", &py_gaussian_blur<double>,docs, py::arg("img"), py::arg("sigma"), py::arg("max_size")=1000 );



    docs = 
"requires \n\
    - all pixels in img are set to either 255 or 0. \n\
ensures \n\
    - This function computes the skeletonization of img and stores the result in \n\
      #img.  That is, given a binary image, we progressively thin the binary blobs \n\
      (composed of on_pixel values) until only a single pixel wide skeleton of the \n\
      original blobs remains. \n\
    - Doesn't change the shape or size of img.";
    /*!
        requires
            - all pixels in img are set to either 255 or 0.
        ensures
            - This function computes the skeletonization of img and stores the result in
              img (i.e. it works in place and therefore modifies the supplied img).  That
              is, given a binary image, we progressively thin the binary blobs (composed of
              on_pixel values) until only a single pixel wide skeleton of the original
              blobs remains.
            - Doesn't change the shape or size of img.
            - Returns img.  Note that the returned object is the same object as the input
              object.
    !*/
    m.def("skeleton", py_skeleton, docs, py::arg("img"));




    docs = 
"requires \n\
    - neighborhood_connectivity == 4, 8, or 24 \n\
ensures \n\
    - This function labels each of the connected blobs in img with a unique integer  \n\
      label.   \n\
    - An image can be thought of as a graph where pixels A and B are connected if \n\
      they are close to each other and satisfy some criterion like having the same \n\
      value or both being non-zero.  Then this function can be understood as \n\
      labeling all the connected components of this pixel graph such that all \n\
      pixels in a component get the same label while pixels in different components \n\
      get different labels.   \n\
    - If zero_pixels_are_background==true then there is a special background component \n\
      and all pixels with value 0 are assigned to it. Moreover, all such background pixels \n\
      will always get a blob id of 0 regardless of any other considerations. \n\
    - This function returns a label image and a count of the number of blobs found. \n\
      I.e., if you ran this function like: \n\
        label_img, num_blobs = label_connected_blobs(img) \n\
      You would obtain the noted label image and number of blobs. \n\
    - The output label_img has the same dimensions as the input image. \n\
    - for all valid r and c: \n\
        - label_img[r][c] == the blob label number for pixel img[r][c].   \n\
        - label_img[r][c] >= 0 \n\
        - if (img[r][c]==0) then \n\
            - label_img[r][c] == 0 \n\
        - else \n\
            - label_img[r][c] != 0 \n\
    - if (len(img) != 0) then  \n\
        - The returned num_blobs will be == label_img.max()+1 \n\
          (i.e. returns a number one greater than the maximum blob id number,  \n\
          this is the number of blobs found.) \n\
    - else \n\
        - num_blobs will be 0. \n\
    - blob labels are contiguous, therefore, the number returned by this function is \n\
      the number of blobs in the image (including the background blob).";
    /*!
        requires
            - neighborhood_connectivity == 4, 8, or 24
        ensures
            - This function labels each of the connected blobs in img with a unique integer 
              label.  
            - An image can be thought of as a graph where pixels A and B are connected if
              they are close to each other and satisfy some criterion like having the same
              value or both being non-zero.  Then this function can be understood as
              labeling all the connected components of this pixel graph such that all
              pixels in a component get the same label while pixels in different components
              get different labels.  
            - If zero_pixels_are_background==true then there is a special background component
              and all pixels with value 0 are assigned to it. Moreover, all such background pixels
              will always get a blob id of 0 regardless of any other considerations.
            - This function returns a label image and a count of the number of blobs found.
              I.e., if you ran this function like:
                label_img, num_blobs = label_connected_blobs(img)
              You would obtain the noted label image and number of blobs.
            - The output label_img has the same dimensions as the input image.
            - for all valid r and c:
                - label_img[r][c] == the blob label number for pixel img[r][c].  
                - label_img[r][c] >= 0
                - if (img[r][c]==0) then
                    - label_img[r][c] == 0
                - else
                    - label_img[r][c] != 0
            - if (len(img) != 0) then 
                - The returned num_blobs will be == label_img.max()+1
                  (i.e. returns a number one greater than the maximum blob id number, 
                  this is the number of blobs found.)
            - else
                - num_blobs will be 0.
            - blob labels are contiguous, therefore, the number returned by this function is
              the number of blobs in the image (including the background blob).
    !*/
    m.def("label_connected_blobs", py_label_connected_blobs<unsigned char>, 
        py::arg("img"),py::arg("zero_pixels_are_background")=true,py::arg("neighborhood_connectivity")=8,py::arg("connected_if_both_not_zero")=false);
    m.def("label_connected_blobs", py_label_connected_blobs<uint16_t>, 
        py::arg("img"),py::arg("zero_pixels_are_background")=true,py::arg("neighborhood_connectivity")=8,py::arg("connected_if_both_not_zero")=false);
    m.def("label_connected_blobs", py_label_connected_blobs<uint32_t>,  
        py::arg("img"),py::arg("zero_pixels_are_background")=true,py::arg("neighborhood_connectivity")=8,py::arg("connected_if_both_not_zero")=false);
    m.def("label_connected_blobs", py_label_connected_blobs<uint64_t>,  
        py::arg("img"),py::arg("zero_pixels_are_background")=true,py::arg("neighborhood_connectivity")=8,py::arg("connected_if_both_not_zero")=false);
    m.def("label_connected_blobs", py_label_connected_blobs<float>,  
        py::arg("img"),py::arg("zero_pixels_are_background")=true,py::arg("neighborhood_connectivity")=8,py::arg("connected_if_both_not_zero")=false);
    m.def("label_connected_blobs", py_label_connected_blobs<double>, docs, 
        py::arg("img"),py::arg("zero_pixels_are_background")=true,py::arg("neighborhood_connectivity")=8,py::arg("connected_if_both_not_zero")=false);


    docs =
"requires \n\
    - smoothing >= 0 \n\
ensures \n\
    - This routine performs a watershed segmentation of the given input image and \n\
      labels each resulting flooding region with a unique integer label. It does \n\
      this by marking the brightest pixels as sources of flooding and then flood \n\
      fills the image outward from those sources.  Each flooded area is labeled \n\
      with the identity of the source pixel and flooding stops when another flooded \n\
      area is reached or pixels with values < background_thresh are encountered.   \n\
    - The flooding will also overrun a source pixel if that source pixel has yet to \n\
      label any neighboring pixels.  This behavior helps to mitigate spurious \n\
      splits of objects due to noise.  You can further control this behavior by \n\
      setting the smoothing parameter.  The flooding will take place on an image \n\
      that has been Gaussian blurred with a sigma==smoothing.  So setting smoothing \n\
      to a larger number will in general cause more regions to be merged together. \n\
      Note that the smoothing parameter has no effect on the interpretation of \n\
      background_thresh since the decision of \"background or not background\" is \n\
      always made relative to the unsmoothed input image. \n\
    - This function returns a tuple of the labeled image and number of blobs found.  \n\
      i.e. you can call it like this: \n\
        label_img, num_blobs = label_connected_blobs_watershed(img,background_thresh,smoothing) \n\
    - The returned label_img will have the same dimensions as img.  \n\
    - for all valid r and c: \n\
        - if (img[r][c] < background_thresh) then \n\
            - label_img[r][c] == 0, (i.e. the pixel is labeled as background) \n\
        - else \n\
            - label_img[r][c] == an integer value indicating the identity of the segment \n\
              containing the pixel img[r][c].   \n\
    - The returned num_blobs is the number of labeled segments, including the \n\
      background segment.  Therefore, the returned number is 1+(the max value in \n\
      label_img).";
    /*!
        requires
            - smoothing >= 0
        ensures
            - This routine performs a watershed segmentation of the given input image and
              labels each resulting flooding region with a unique integer label. It does
              this by marking the brightest pixels as sources of flooding and then flood
              fills the image outward from those sources.  Each flooded area is labeled
              with the identity of the source pixel and flooding stops when another flooded
              area is reached or pixels with values < background_thresh are encountered.  
            - The flooding will also overrun a source pixel if that source pixel has yet to
              label any neighboring pixels.  This behavior helps to mitigate spurious
              splits of objects due to noise.  You can further control this behavior by
              setting the smoothing parameter.  The flooding will take place on an image
              that has been Gaussian blurred with a sigma==smoothing.  So setting smoothing
              to a larger number will in general cause more regions to be merged together.
              Note that the smoothing parameter has no effect on the interpretation of
              background_thresh since the decision of "background or not background" is
              always made relative to the unsmoothed input image.
            - This function returns a tuple of the labeled image and number of blobs found. 
              i.e. you can call it like this:
                label_img, num_blobs = label_connected_blobs_watershed(img,background_thresh,smoothing)
            - The returned label_img will have the same dimensions as img. 
            - for all valid r and c:
                - if (img[r][c] < background_thresh) then
                    - label_img[r][c] == 0, (i.e. the pixel is labeled as background)
                - else
                    - label_img[r][c] == an integer value indicating the identity of the segment
                      containing the pixel img[r][c].  
            - The returned num_blobs is the number of labeled segments, including the
              background segment.  Therefore, the returned number is 1+(the max value in
              label_img).
    !*/
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed<unsigned char>, py::arg("img"),py::arg("background_thresh"),py::arg("smoothing")=0);
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed<uint16_t>, py::arg("img"),py::arg("background_thresh"),py::arg("smoothing")=0);
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed<uint32_t>, py::arg("img"),py::arg("background_thresh"),py::arg("smoothing")=0);
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed<float>, py::arg("img"),py::arg("background_thresh"),py::arg("smoothing")=0);
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed<double>, docs, py::arg("img"),py::arg("background_thresh"),py::arg("smoothing")=0);

    docs = "This version of label_connected_blobs_watershed simple invokes: \n"
           "   return label_connected_blobs_watershed(img, partition_pixels(img))";
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed2<unsigned char>, py::arg("img"));
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed2<uint16_t>, py::arg("img"));
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed2<uint32_t>, py::arg("img"));
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed2<float>, py::arg("img"));
    m.def("label_connected_blobs_watershed", py_label_connected_blobs_watershed2<double>, docs, py::arg("img"));


    docs = 
"Converts a grayscale image into a jet colored image.  This is an image where dark \n\
pixels are dark blue and larger values become light blue, then yellow, and then \n\
finally red as they approach the maximum pixel values." ;
    m.def("jet", py_jet<unsigned char>, py::arg("img"));
    m.def("jet", py_jet<uint16_t>, py::arg("img"));
    m.def("jet", py_jet<uint32_t>, py::arg("img"));
    m.def("jet", py_jet<float>, py::arg("img"));
    m.def("jet", py_jet<double>, docs, py::arg("img"));

    docs = 
"- randomly generates a mapping from gray level pixel values \n\
  to the RGB pixel space and then uses this mapping to create \n\
  a colored version of img.  Returns an image which represents \n\
  this colored version of img. \n\
- black pixels in img will remain black in the output image.  ";
    /*!
        - randomly generates a mapping from gray level pixel values
          to the RGB pixel space and then uses this mapping to create
          a colored version of img.  Returns an image which represents
          this colored version of img.
        - black pixels in img will remain black in the output image.  
    !*/
    m.def("randomly_color_image", py_randomly_color_image<unsigned char>, py::arg("img"));
    m.def("randomly_color_image", py_randomly_color_image<uint16_t>, py::arg("img"));
    m.def("randomly_color_image", py_randomly_color_image<uint32_t>, docs, py::arg("img"));



    docs =
"requires \n\
    - all pixels in img are set to either 255 or 0. \n\
      (i.e. it must be a binary image) \n\
ensures \n\
    - This routine finds endpoints of lines in a thinned binary image.  For \n\
      example, if the image was produced by skeleton() or something like a Canny \n\
      edge detector then you can use find_line_endpoints() to find the pixels \n\
      sitting on the ends of lines.";
    /*!
        requires
            - all pixels in img are set to either 255 or 0.
              (i.e. it must be a binary image)
        ensures
            - This routine finds endpoints of lines in a thinned binary image.  For
              example, if the image was produced by skeleton() or something like a Canny
              edge detector then you can use find_line_endpoints() to find the pixels
              sitting on the ends of lines.
    !*/
    m.def("find_line_endpoints", find_line_endpoints<numpy_image<unsigned char>>, docs, py::arg("img"));


    m.def("get_rect", [](const py::array& img){ return rectangle(0,0,(long)img.shape(1)-1,(long)img.shape(0)-1); },
        "returns a rectangle(0,0,img.shape(1)-1,img.shape(0)-1).  Therefore, it is the rectangle that bounds the image.", 
        py::arg("img")  );


    const char* grad_docs =
"- Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). \n\
- This routine computes the requested gradient of img at each location in VALID_AREA. \n\
  The gradients are returned in a new image of the same dimensions as img.  All pixels \n\
  outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple \n\
  where the first element is the gradient image and the second is VALID_AREA.";

    const char* filt_docs =
"- Returns the filter used by the indicated derivative to compute the image gradient. \n\
  That is, the output gradients are found by cross correlating the returned filter with \n\
  the input image. \n\
- The returned filter has get_scale()*2+1 rows and columns." ;

    const char* class_docs = 
"This class is a tool for computing first and second derivatives of an \n\
image.  It does this by fitting a quadratic surface around each pixel and \n\
then computing the gradients of that quadratic surface.  For the details \n\
see the paper: \n\
    Quadratic models for curved line detection in SAR CCD by Davis E. King \n\
    and Rhonda D. Phillips \n\
 \n\
This technique gives very accurate gradient estimates and is also very fast \n\
since the entire gradient estimation procedure, for each type of gradient, \n\
is accomplished by cross-correlating the image with a single separable \n\
filter.  This means you can compute gradients at very large scales (e.g. by \n\
fitting the quadratic to a large window, like a 99x99 window) and it still \n\
runs very quickly.";

    py::class_<image_gradients>(m, "image_gradients", class_docs)
        .def(py::init<long>(), "Creates this class with the provided scale. i.e. get_scale()==scale. \nscale must be >= 1.", py::arg("scale"))
        .def(py::init<>(), "Creates this class with a scale of 1. i.e. get_scale()==1")
        .def("gradient_x", [](image_gradients& g, const numpy_image<unsigned char>& img){
            numpy_image<float> out;
            auto rect=g.gradient_x(img,out); 
            return py::make_tuple(out,rect);
            },  py::arg("img"))
        .def("gradient_x", [](image_gradients& g, const numpy_image<float>& img){
            numpy_image<float> out;
            auto rect=g.gradient_x(img,out); 
            return py::make_tuple(out,rect);
            }, grad_docs, py::arg("img"))
        .def("gradient_y", [](image_gradients& g, const numpy_image<unsigned char>& img){
            numpy_image<float> out;
            auto rect=g.gradient_y(img,out); 
            return py::make_tuple(out,rect);
            },  py::arg("img"))
        .def("gradient_y", [](image_gradients& g, const numpy_image<float>& img){
            numpy_image<float> out;
            auto rect=g.gradient_y(img,out); 
            return py::make_tuple(out,rect);
            }, grad_docs, py::arg("img"))
        .def("gradient_xx", [](image_gradients& g, const numpy_image<unsigned char>& img){
            numpy_image<float> out;
            auto rect=g.gradient_xx(img,out); 
            return py::make_tuple(out,rect);
            }, py::arg("img"))
        .def("gradient_xx", [](image_gradients& g, const numpy_image<float>& img){
            numpy_image<float> out;
            auto rect=g.gradient_xx(img,out); 
            return py::make_tuple(out,rect);
            }, grad_docs, py::arg("img"))
        .def("gradient_xy", [](image_gradients& g, const numpy_image<unsigned char>& img){
            numpy_image<float> out;
            auto rect=g.gradient_xy(img,out); 
            return py::make_tuple(out,rect);
            }, py::arg("img"))
        .def("gradient_xy", [](image_gradients& g, const numpy_image<float>& img){
            numpy_image<float> out;
            auto rect=g.gradient_xy(img,out); 
            return py::make_tuple(out,rect);
            }, grad_docs, py::arg("img"))
        .def("gradient_yy", [](image_gradients& g, const numpy_image<unsigned char>& img){
            numpy_image<float> out;
            auto rect=g.gradient_yy(img,out); 
            return py::make_tuple(out,rect);
            }, py::arg("img"))
        .def("gradient_yy", [](image_gradients& g, const numpy_image<float>& img){
            numpy_image<float> out;
            auto rect=g.gradient_yy(img,out); 
            return py::make_tuple(out,rect);
            }, grad_docs, py::arg("img"))
        .def("get_x_filter", [](image_gradients& g){ return numpy_image<float>(g.get_x_filter()); }, filt_docs)
        .def("get_y_filter", [](image_gradients& g){ return numpy_image<float>(g.get_y_filter()); }, filt_docs)
        .def("get_xx_filter", [](image_gradients& g){ return numpy_image<float>(g.get_xx_filter()); }, filt_docs)
        .def("get_xy_filter", [](image_gradients& g){ return numpy_image<float>(g.get_xy_filter()); }, filt_docs)
        .def("get_yy_filter", [](image_gradients& g){ return numpy_image<float>(g.get_yy_filter()); }, filt_docs)
        .def("get_scale", &image_gradients::get_scale, 
"When we estimate a gradient we do so by fitting a quadratic filter to a window of size \n\
get_scale()*2+1 centered on each pixel.  Therefore, the scale parameter controls the size \n\
of gradients we will find.  For example, a very large scale will cause the gradient_xx() \n\
to be insensitive to high frequency noise in the image while smaller scales would be more \n\
sensitive to such fluctuations in the image." 
        );


    docs = 
"Converts an image to a target pixel type.  dtype must be a string containing one of the following: \n\
    uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel \n\
 \n\
When converting from a color space with more than 255 values the pixel intensity is \n\
saturated at the minimum and maximum pixel values of the target pixel type.  For \n\
example, if you convert a float valued image to uint8 then float values will be \n\
truncated to integers and values larger than 255 are converted to 255 while values less \n\
than 0 are converted to 0.";
    /*!
    Converts an image to a target pixel type.  dtype must be a string containing one of the following:
        uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel

    When converting from a color space with more than 255 values the pixel intensity is
    saturated at the minimum and maximum pixel values of the target pixel type.  For
    example, if you convert a float valued image to uint8 then float values will be
    truncated to integers and values larger than 255 are converted to 255 while values less
    than 0 are converted to 0.
    !*/
    m.def("convert_image", convert_image<uint8_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<uint16_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<uint32_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<uint64_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<int8_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<int16_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<int32_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<int64_t>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<float>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<double>, py::arg("img"), py::arg("dtype"));
    m.def("convert_image", convert_image<rgb_pixel>, docs, py::arg("img"), py::arg("dtype"));

    m.def("as_grayscale", &as_grayscale, 
        "Convert an image to 8bit grayscale.  If it's already a grayscale image do nothing and just return img.", py::arg("img"));

}

