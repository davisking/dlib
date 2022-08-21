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
numpy_image<T> py_scale_image (
    const numpy_image<T>& img,
    double scale
)
{
    DLIB_CASSERT(scale > 0, "Scale factor must be greater than 0");
    numpy_image<T> out = img;
    resize_image(scale, out);
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
    for (const auto& p : line)
        DLIB_CASSERT(get_rect(horz_gradient).contains(p), "All line points must be inside the given images.");

    return remove_incoherent_edge_pixels(line, horz_gradient, vert_gradient, angle_threshold);
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

template <typename T>
numpy_image<T> py_extract_image_chip (
    const numpy_image<T>& img,
    const chip_details& chip_location 
)
{
    numpy_image<T> out;
    extract_image_chip(img, chip_location, out);
    return out;
}

template <typename T>
py::list py_extract_image_chips (
    const numpy_image<T>& img,
    const py::list& chip_locations
)
{
    dlib::array<numpy_image<T>> out;
    extract_image_chips(img, python_list_to_vector<chip_details>(chip_locations), out);
    py::list ret;
    for (const auto& i : out)
        ret.append(i);
    return ret;
}

// ----------------------------------------------------------------------------------------

void register_extract_image_chip (py::module& m)
{
    const char* class_docs = 
"WHAT THIS OBJECT REPRESENTS \n\
    This is a simple tool for passing in a pair of row and column values to the \n\
    chip_details constructor.";


    auto print_chip_dims_str = [](const chip_dims& d)
    {
        std::ostringstream sout;
        sout << "rows="<< d.rows << ", cols=" << d.cols; 
        return sout.str();
    };
    auto print_chip_dims_repr = [](const chip_dims& d)
    {
        std::ostringstream sout;
        sout << "chip_dims(rows="<< d.rows << ", cols=" << d.cols << ")"; 
        return sout.str();
    };

    py::class_<chip_dims>(m, "chip_dims", class_docs)
        .def(py::init<unsigned long, unsigned long>(), py::arg("rows"), py::arg("cols"))
        .def("__str__", print_chip_dims_str)
        .def("__repr__", print_chip_dims_repr)
        .def_readwrite("rows", &chip_dims::rows)
        .def_readwrite("cols", &chip_dims::cols);



    auto print_chip_details_str = [](const chip_details& d)
    {
        std::ostringstream sout;
        sout << "rect=" << d.rect << ", angle="<< d.angle << ", rows="<< d.rows << ", cols=" << d.cols; 
        return sout.str();
    };
    auto print_chip_details_repr = [](const chip_details& d)
    {
        std::ostringstream sout;
        sout << "chip_details(rect=drectangle(" 
            << d.rect.left()<<","<<d.rect.top()<<","<<d.rect.right()<<","<<d.rect.bottom()
            <<"), angle="<< d.angle << ", dims=chip_dims(rows="<< d.rows << ", cols=" << d.cols << "))"; 
        return sout.str();
    };


    class_docs =
"WHAT THIS OBJECT REPRESENTS \n\
    This object describes where an image chip is to be extracted from within \n\
    another image.  In particular, it specifies that the image chip is \n\
    contained within the rectangle self.rect and that prior to extraction the \n\
    image should be rotated counter-clockwise by self.angle radians.  Finally, \n\
    the extracted chip should have self.rows rows and self.cols columns in it \n\
    regardless of the shape of self.rect.  This means that the extracted chip \n\
    will be stretched to fit via bilinear interpolation when necessary." ;
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object describes where an image chip is to be extracted from within
                another image.  In particular, it specifies that the image chip is
                contained within the rectangle self.rect and that prior to extraction the
                image should be rotated counter-clockwise by self.angle radians.  Finally,
                the extracted chip should have self.rows rows and self.cols columns in it
                regardless of the shape of self.rect.  This means that the extracted chip
                will be stretched to fit via bilinear interpolation when necessary.
        !*/
    py::class_<chip_details>(m, "chip_details", class_docs)
        .def(py::init<drectangle>(), py::arg("rect"))
        .def(py::init<rectangle>(), py::arg("rect"),
"ensures \n\
    - self.rect == rect_ \n\
    - self.angle == 0 \n\
    - self.rows == rect.height() \n\
    - self.cols == rect.width()" 
        /*!
            ensures
                - self.rect == rect_
                - self.angle == 0
                - self.rows == rect.height()
                - self.cols == rect.width()
        !*/
            )
        .def(py::init<drectangle,unsigned long>(), py::arg("rect"), py::arg("size"))
        .def(py::init<rectangle,unsigned long>(), py::arg("rect"), py::arg("size"),
"ensures \n\
    - self.rect == rect \n\
    - self.angle == 0 \n\
    - self.rows and self.cols is set such that the total size of the chip is as close \n\
      to size as possible but still matches the aspect ratio of rect. \n\
    - As long as size and the aspect ratio of rect stays constant then \n\
      self.rows and self.cols will always have the same values.  This means \n\
      that, for example, if you want all your chips to have the same dimensions \n\
      then ensure that size is always the same and also that rect always has \n\
      the same aspect ratio.  Otherwise the calculated values of self.rows and \n\
      self.cols may be different for different chips.  Alternatively, you can \n\
      use the chip_details constructor below that lets you specify the exact \n\
      values for rows and cols." 
        /*!
            ensures
                - self.rect == rect
                - self.angle == 0
                - self.rows and self.cols is set such that the total size of the chip is as close
                  to size as possible but still matches the aspect ratio of rect.
                - As long as size and the aspect ratio of rect stays constant then
                  self.rows and self.cols will always have the same values.  This means
                  that, for example, if you want all your chips to have the same dimensions
                  then ensure that size is always the same and also that rect always has
                  the same aspect ratio.  Otherwise the calculated values of self.rows and
                  self.cols may be different for different chips.  Alternatively, you can
                  use the chip_details constructor below that lets you specify the exact
                  values for rows and cols.
        !*/
            )
        .def(py::init<drectangle,unsigned long,double>(), py::arg("rect"), py::arg("size"), py::arg("angle"))
        .def(py::init<rectangle,unsigned long,double>(), py::arg("rect"), py::arg("size"), py::arg("angle"),
"ensures \n\
    - self.rect == rect \n\
    - self.angle == angle \n\
    - self.rows and self.cols is set such that the total size of the chip is as \n\
      close to size as possible but still matches the aspect ratio of rect. \n\
    - As long as size and the aspect ratio of rect stays constant then \n\
      self.rows and self.cols will always have the same values.  This means \n\
      that, for example, if you want all your chips to have the same dimensions \n\
      then ensure that size is always the same and also that rect always has \n\
      the same aspect ratio.  Otherwise the calculated values of self.rows and \n\
      self.cols may be different for different chips.  Alternatively, you can \n\
      use the chip_details constructor below that lets you specify the exact \n\
      values for rows and cols." 
        /*!
            ensures
                - self.rect == rect
                - self.angle == angle
                - self.rows and self.cols is set such that the total size of the chip is as
                  close to size as possible but still matches the aspect ratio of rect.
                - As long as size and the aspect ratio of rect stays constant then
                  self.rows and self.cols will always have the same values.  This means
                  that, for example, if you want all your chips to have the same dimensions
                  then ensure that size is always the same and also that rect always has
                  the same aspect ratio.  Otherwise the calculated values of self.rows and
                  self.cols may be different for different chips.  Alternatively, you can
                  use the chip_details constructor below that lets you specify the exact
                  values for rows and cols.
        !*/
            )
        .def(py::init<drectangle,chip_dims>(), py::arg("rect"), py::arg("dims"))
        .def(py::init<rectangle,chip_dims>(), py::arg("rect"), py::arg("dims"),
"ensures \n\
    - self.rect == rect \n\
    - self.angle == 0 \n\
    - self.rows == dims.rows \n\
    - self.cols == dims.cols" 
        /*!
            ensures
                - self.rect == rect
                - self.angle == 0
                - self.rows == dims.rows
                - self.cols == dims.cols
        !*/
            )
        .def(py::init<drectangle,chip_dims,double>(), py::arg("rect"), py::arg("dims"), py::arg("angle"))
        .def(py::init<rectangle,chip_dims,double>(), py::arg("rect"), py::arg("dims"), py::arg("angle"),
"ensures \n\
    - self.rect == rect \n\
    - self.angle == angle \n\
    - self.rows == dims.rows \n\
    - self.cols == dims.cols" 
        /*!
            ensures
                - self.rect == rect
                - self.angle == angle
                - self.rows == dims.rows
                - self.cols == dims.cols
        !*/
            )
        .def(py::init<std::vector<dpoint>,std::vector<dpoint>,chip_dims>(), py::arg("chip_points"), py::arg("img_points"), py::arg("dims"))
        .def(py::init<std::vector<point>,std::vector<point>,chip_dims>(), py::arg("chip_points"), py::arg("img_points"), py::arg("dims"),
"requires \n\
    - len(chip_points) == len(img_points) \n\
    - len(chip_points) >= 2  \n\
ensures \n\
    - The chip will be extracted such that the pixel locations chip_points[i] \n\
      in the chip are mapped to img_points[i] in the original image by a \n\
      similarity transform.  That is, if you know the pixelwize mapping you \n\
      want between the chip and the original image then you use this function \n\
      of chip_details constructor to define the mapping. \n\
    - self.rows == dims.rows \n\
    - self.cols == dims.cols \n\
    - self.rect and self.angle are computed based on the given size of the output chip \n\
      (specified by dims) and the similarity transform between the chip and \n\
      image (specified by chip_points and img_points)." 
        /*!
            requires
                - len(chip_points) == len(img_points)
                - len(chip_points) >= 2 
            ensures
                - The chip will be extracted such that the pixel locations chip_points[i]
                  in the chip are mapped to img_points[i] in the original image by a
                  similarity transform.  That is, if you know the pixelwize mapping you
                  want between the chip and the original image then you use this function
                  of chip_details constructor to define the mapping.
                - self.rows == dims.rows
                - self.cols == dims.cols
                - self.rect and self.angle are computed based on the given size of the output chip
                  (specified by dims) and the similarity transform between the chip and
                  image (specified by chip_points and img_points).
        !*/
            )
        .def("__str__", print_chip_details_str)
        .def("__repr__", print_chip_details_repr)
        .def_readwrite("rect", &chip_details::rect)
        .def_readwrite("angle", &chip_details::angle)
        .def_readwrite("rows", &chip_details::rows)
        .def_readwrite("cols", &chip_details::cols);

    {
    typedef std::vector<chip_details> type;
    py::bind_vector<type>(m, "chip_detailss", "An array of chip_details objects.")
        .def("extend", extend_vector_with_python_list<type>);
    }

    m.def("extract_image_chip", &py_extract_image_chip<uint8_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<uint16_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<uint32_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<uint64_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<int8_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<int16_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<int32_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<int64_t>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<float>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<double>, py::arg("img"), py::arg("chip_location"));
    m.def("extract_image_chip", &py_extract_image_chip<rgb_pixel>, py::arg("img"), py::arg("chip_location"),
        "    This routine is just like extract_image_chips() except it takes a single \n"
        "    chip_details object and returns a single chip image rather than a list of images."
        );

    m.def("extract_image_chips", &py_extract_image_chips<uint8_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<uint16_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<uint32_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<uint64_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<int8_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<int16_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<int32_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<int64_t>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<float>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<double>, py::arg("img"), py::arg("chip_locations"));
    m.def("extract_image_chips", &py_extract_image_chips<rgb_pixel>, py::arg("img"), py::arg("chip_locations"),
"requires \n\
    - for all valid i:  \n\
        - chip_locations[i].rect.is_empty() == false \n\
        - chip_locations[i].rows*chip_locations[i].cols != 0 \n\
ensures \n\
    - This function extracts \"chips\" from an image.  That is, it takes a list of \n\
      rectangular sub-windows (i.e. chips) within an image and extracts those \n\
      sub-windows, storing each into its own image.  It also scales and rotates the \n\
      image chips according to the instructions inside each chip_details object. \n\
      It uses bilinear interpolation. \n\
    - The extracted image chips are returned in a python list of numpy arrays.  The \n\
      length of the returned array is len(chip_locations). \n\
    - Let CHIPS be the returned array, then we have: \n\
        - for all valid i: \n\
            - #CHIPS[i] == The image chip extracted from the position \n\
              chip_locations[i].rect in img. \n\
            - #CHIPS[i].shape(0) == chip_locations[i].rows \n\
            - #CHIPS[i].shape(1) == chip_locations[i].cols \n\
            - The image will have been rotated counter-clockwise by \n\
              chip_locations[i].angle radians, around the center of \n\
              chip_locations[i].rect, before the chip was extracted.  \n\
    - Any pixels in an image chip that go outside img are set to 0 (i.e. black)." 
    /*!
        requires
            - for all valid i: 
                - chip_locations[i].rect.is_empty() == false
                - chip_locations[i].rows*chip_locations[i].cols != 0
        ensures
            - This function extracts "chips" from an image.  That is, it takes a list of
              rectangular sub-windows (i.e. chips) within an image and extracts those
              sub-windows, storing each into its own image.  It also scales and rotates the
              image chips according to the instructions inside each chip_details object.
              It uses bilinear interpolation.
            - The extracted image chips are returned in a python list of numpy arrays.  The
              length of the returned array is len(chip_locations).
            - Let CHIPS be the returned array, then we have:
                - for all valid i:
                    - #CHIPS[i] == The image chip extracted from the position
                      chip_locations[i].rect in img.
                    - #CHIPS[i].shape(0) == chip_locations[i].rows
                    - #CHIPS[i].shape(1) == chip_locations[i].cols
                    - The image will have been rotated counter-clockwise by
                      chip_locations[i].angle radians, around the center of
                      chip_locations[i].rect, before the chip was extracted. 
            - Any pixels in an image chip that go outside img are set to 0 (i.e. black).
    !*/
        );

    m.def("get_face_chip_details",
          static_cast<chip_details (*)(const full_object_detection&, const unsigned long, const double)>(&get_face_chip_details),
          py::arg("det"), py::arg("size")=200, py::arg("padding")=0.2,
        "Given a full_object_detection det, returns a chip_details object which can be \n\
         used to extract an image of given size and padding."
        );

    m.def("get_face_chip_details",
          static_cast<std::vector<chip_details> (*)(const std::vector<full_object_detection>&, const unsigned long, const double)>(&get_face_chip_details),
          py::arg("dets"), py::arg("size")=200, py::arg("padding")=0.2,
        "Given a list of full_object_detection dets, returns a chip_details object which can be \n\
         used to extract an image of given size and padding."
        );
}

// ----------------------------------------------------------------------------------------

py::array py_tile_images (
    const py::list& images
)
{
    DLIB_CASSERT(len(images) > 0);

    if (is_image<rgb_pixel>(images[0].cast<py::array>()))
    {
        std::vector<numpy_image<rgb_pixel>> tmp(len(images));
        for (size_t i = 0; i < tmp.size(); ++i)
            assign_image(tmp[i], images[i].cast<py::array>());
        return numpy_image<rgb_pixel>(tile_images(tmp));
    }
    else
    {
        std::vector<numpy_image<unsigned char>> tmp(len(images));
        for (size_t i = 0; i < tmp.size(); ++i)
            assign_image(tmp[i], images[i].cast<py::array>());
        return numpy_image<unsigned char>(tile_images(tmp));
    }
}

// ----------------------------------------------------------------------------------------

template <typename T>
py::array_t<unsigned long> py_get_histogram (
    const numpy_image<T>& img,
    size_t hist_size
)
{
    matrix<unsigned long,1> hist;
    get_histogram(img,hist,hist_size);

    return numpy_image<unsigned long>(std::move(hist)).squeeze();
}

// ----------------------------------------------------------------------------------------

py::array py_sub_image (
    const py::array& img,
    const rectangle& win
)
{
    DLIB_CASSERT(img.ndim() >= 2);

    auto width_step = img.strides(0);

    const long nr = img.shape(0);
    const long nc = img.shape(1);
    rectangle rect(0,0,nc-1,nr-1);
    rect = rect.intersect(win);

    std::vector<size_t> shape(img.ndim()), strides(img.ndim());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape[i] = img.shape(i);
        strides[i] = img.strides(i);
    }

    shape[0] = rect.height();
    shape[1] = rect.width();

    size_t col_stride = 1;
    for (size_t i = 1; i < strides.size(); ++i)
        col_stride *= strides[i];

    const void* data = (char*)img.data() + col_stride*rect.left() + rect.top()*strides[0];

    return py::array(img.dtype(), shape, strides, data, img);
}

py::array py_sub_image2 (
    const py::tuple& image_and_rect_tuple
)
{
    DLIB_CASSERT(len(image_and_rect_tuple) == 2);
    return py_sub_image(image_and_rect_tuple[0].cast<py::array>(), image_and_rect_tuple[1].cast<rectangle>());
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
    m.def("resize_image", &py_scale_image<int8_t>, py::arg("img"), py::arg("scale"));
    m.def("resize_image", &py_scale_image<int16_t>, py::arg("img"), py::arg("scale"));
    m.def("resize_image", &py_scale_image<int32_t>, py::arg("img"), py::arg("scale"));
    m.def("resize_image", &py_scale_image<int64_t>, py::arg("img"), py::arg("scale"));
    m.def("resize_image", &py_scale_image<float>, py::arg("img"), py::arg("scale"));
    m.def("resize_image", &py_scale_image<double>, py::arg("img"), py::arg("scale"));
    m.def("resize_image", &py_scale_image<rgb_pixel>, py::arg("img"), py::arg("scale"),
        "Resizes img, using bilinear interpolation, to have the new size (img rows * scale, img cols * scale)"
        );

    register_extract_image_chip(m);

    m.def("sub_image", &py_sub_image, py::arg("img"), py::arg("rect"),
"Returns a new numpy array that references the sub window in img defined by rect. \n\
If rect is larger than img then rect is cropped so that it does not go outside img. \n\
Therefore, this routine is equivalent to performing: \n\
    win = get_rect(img).intersect(rect) \n\
    subimg = img[win.top():win.bottom()-1,win.left():win.right()-1]" 
    /*!
        Returns a new numpy array that references the sub window in img defined by rect.
        If rect is larger than img then rect is cropped so that it does not go outside img.
        Therefore, this routine is equivalent to performing:
            win = get_rect(img).intersect(rect)
            subimg = img[win.top():win.bottom()-1,win.left():win.right()-1]
    !*/
        );
    m.def("sub_image", &py_sub_image2, py::arg("image_and_rect_tuple"),
        "Performs: return sub_image(image_and_rect_tuple[0], image_and_rect_tuple[1])");


    m.def("get_histogram", &py_get_histogram<uint8_t>, py::arg("img"), py::arg("hist_size"));
    m.def("get_histogram", &py_get_histogram<uint16_t>, py::arg("img"), py::arg("hist_size"));
    m.def("get_histogram", &py_get_histogram<uint32_t>, py::arg("img"), py::arg("hist_size"));
    m.def("get_histogram", &py_get_histogram<uint64_t>, py::arg("img"), py::arg("hist_size"),
"ensures \n\
    - Returns a numpy array, HIST, that contains a histogram of the pixels in img. \n\
      In particular, we will have: \n\
        - len(HIST) == hist_size \n\
        - for all valid i:  \n\
            - HIST[i] == the number of times a pixel with intensity i appears in img." 
    /*!
        ensures
            - Returns a numpy array, HIST, that contains a histogram of the pixels in img.
              In particular, we will have:
                - len(HIST) == hist_size
                - for all valid i: 
                    - HIST[i] == the number of times a pixel with intensity i appears in img.
    !*/
        );


    m.def("tile_images", py_tile_images, py::arg("images"),
"requires \n\
    - images is a list of numpy arrays that can be interpreted as images.  They \n\
      must all be the same type of image as well. \n\
ensures \n\
    - This function takes the given images and tiles them into a single large \n\
      square image and returns this new big tiled image.  Therefore, it is a \n\
      useful method to visualize many small images at once." 
        /*!
            requires
                - images is a list of numpy arrays that can be interpreted as images.  They
                  must all be the same type of image as well.
            ensures
                - This function takes the given images and tiles them into a single large
                  square image and returns this new big tiled image.  Therefore, it is a
                  useful method to visualize many small images at once.
        !*/
        );

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


}


