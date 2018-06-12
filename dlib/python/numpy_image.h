// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYTHON_NuMPY_IMAGE_Hh_
#define DLIB_PYTHON_NuMPY_IMAGE_Hh_

#include <dlib/algs.h>
#include <dlib/error.h>
#include <dlib/matrix.h>
#include <dlib/pixel.h>
#include <string>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <dlib/image_transforms/assign_image.h>
#include <stdint.h>
#include <type_traits>

namespace py = pybind11;


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename pixel_type
        >
    bool is_image (
        const py::array& img 
    )
    /*!
        ensures
            - returns true if and only if the given python numpy array can reasonably be
              interpreted as an image containing pixel_type pixels.
    !*/
    {
        using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
        const size_t expected_channels = pixel_traits<pixel_type>::num;

        const bool has_correct_number_of_dims = (img.ndim()==2 && expected_channels==1) || 
                                                (img.ndim()==3 && img.shape(2)==expected_channels);

        return img.dtype().kind() == py::dtype::of<basic_pixel_type>().kind() && 
               img.itemsize() == sizeof(basic_pixel_type) && 
               has_correct_number_of_dims;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename pixel_type
        >
    void assert_correct_num_channels_in_image (
        const py::array& img
    )
    {
        const size_t expected_channels = pixel_traits<pixel_type>::num;
        if (expected_channels == 1)
        {
            if (!(img.ndim() == 2 || (img.ndim()==3&&img.shape(2)==1)))
                throw dlib::error("Expected a 2D numpy array, but instead got one with " + std::to_string(img.ndim()) + " dimensions.");
        }
        else
        {
            if (img.ndim() != 3)
            {
                throw dlib::error("Expected a numpy array with 3 dimensions, but instead got one with " + std::to_string(img.ndim()) + " dimensions.");
            }
            else if (img.shape(2) != expected_channels)
            {
                if (pixel_traits<pixel_type>::rgb)
                    throw dlib::error("Expected a RGB image with " + std::to_string(expected_channels) + " channels but got an image with " + std::to_string(img.shape(2)) + " channels.");
                else
                    throw dlib::error("Expected an image with " + std::to_string(expected_channels) + " channels but got an image with " + std::to_string(img.shape(2)) + " channels.");
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename pixel_type
        >
    void assert_is_image (
        const py::array& obj
    )
    {
        if (!is_image<pixel_type>(obj))
        {
            assert_correct_num_channels_in_image<pixel_type>(obj);

            using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
            const char expected_type = py::dtype::of<basic_pixel_type>().kind();
            const char got_type = obj.dtype().kind();

            const size_t expected_size = sizeof(basic_pixel_type);
            const size_t got_size = obj.itemsize();

            auto toname = [](char type, size_t size) {
                if (type == 'i' && size == 1) return "int8";
                else if (type == 'i' && size == 2) return "int16";
                else if (type == 'i' && size == 4) return "int32";
                else if (type == 'i' && size == 8) return "int64";
                else if (type == 'u' && size == 1) return "uint8";
                else if (type == 'u' && size == 2) return "uint16";
                else if (type == 'u' && size == 4) return "uint32";
                else if (type == 'u' && size == 8) return "uint64";
                else if (type == 'f' && size == 4) return "float32";
                else if (type == 'd' && size == 8) return "float64";
                else DLIB_CASSERT(false, "unknown type");
            };

            throw dlib::error("Expected numpy array with elements of type " + std::string(toname(expected_type,expected_size)) + " but got " + toname(got_type, got_size) + ".");
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename pixel_type
        >
    class numpy_image : public py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style>
    {
        /*!
            REQUIREMENTS ON pixel_type
                - is a dlib pixel type, this just means that dlib::pixel_traits<pixel_type>
                  is defined.

            WHAT THIS OBJECT REPRESENTS
                This is an image object that implements dlib's generic image interface and
                is backed by a numpy array.  It therefore is easily interchanged with
                python since there is no copying.  It is functionally just a pybind11
                array_t object with the additional routines needed to conform to dlib's
                generic image API.  It also includes appropriate runtime checks to make
                sure that the numpy array is always typed and sized appropriately relative
                to the supplied pixel_type. 
        !*/
    public:

        numpy_image() = default;

        numpy_image(
            const py::array& img
        ) : py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style>(img)
        {
            assert_is_image<pixel_type>(img);
        }

        numpy_image (
            long rows,
            long cols
        )
        {
            set_size(rows,cols);
        }

        numpy_image (
            const py::object& img
        ) : numpy_image(img.cast<py::array>()) {}

        numpy_image(
            const numpy_image& img
        ) = default;

        numpy_image& operator= (
            const py::object& rhs
        )
        {
            *this = numpy_image(rhs);
            return *this;
        }

        numpy_image& operator= (
            const py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style>& rhs
        )
        {
            *this = numpy_image(rhs);
            return *this;
        }

        numpy_image& operator= (
            const numpy_image& rhs
        ) = default;

        template <long NR, long NC>
        numpy_image (
            matrix<pixel_type,NR,NC>&& rhs
        ) : numpy_image(convert_to_numpy(std::move(rhs))) {}

        template <long NR, long NC>
        numpy_image& operator= (
            matrix<pixel_type,NR,NC>&& rhs
        )
        {
            *this = numpy_image(rhs);
            return *this;
        }

        void set_size(size_t rows, size_t cols)
        {
            using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
            constexpr size_t channels = pixel_traits<pixel_type>::num;
            if (channels != 1)
                *this = py::array_t<basic_pixel_type, py::array::c_style>({rows, cols, channels});
            else
                *this = py::array_t<basic_pixel_type, py::array::c_style>({rows, cols});
        }

    private:
        static py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style> convert_to_numpy(matrix<pixel_type>&& img)
        {
            using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
            const size_t dtype_size = sizeof(basic_pixel_type);
            const auto rows = static_cast<const size_t>(num_rows(img));
            const auto cols = static_cast<const size_t>(num_columns(img));
            const size_t channels = pixel_traits<pixel_type>::num;
            const size_t image_size = dtype_size * rows * cols * channels;

            std::unique_ptr<pixel_type[]> arr_ptr = img.steal_memory();
            basic_pixel_type* arr = (basic_pixel_type *) arr_ptr.release();

            if (channels == 1)
            {
                return pybind11::template array_t<basic_pixel_type, py::array::c_style>(
                    {rows, cols},                                                       // shape
                    {dtype_size*cols, dtype_size},                                      // strides
                    arr,                                                                // pointer
                    pybind11::capsule{ arr, [](void *arr_p) { delete[] reinterpret_cast<basic_pixel_type*>(arr_p); } }
                );
            }
            else
            {
                return pybind11::template array_t<basic_pixel_type, py::array::c_style>(
                    {rows, cols, channels},                                                     // shape
                    {dtype_size * cols * channels, dtype_size * channels, dtype_size},          // strides
                    arr,                                                                        // pointer
                    pybind11::capsule{ arr, [](void *arr_p) { delete[] reinterpret_cast<basic_pixel_type*>(arr_p); } }
                );
            }
        }

    };

// ----------------------------------------------------------------------------------------

    template <typename pixel_type>
    void assign_image (
        numpy_image<pixel_type>& dest,
        const py::array& src
    )
    {
        if (is_image<pixel_type>(src))     dest = src;
        else if (is_image<uint8_t>(src))   assign_image(dest, numpy_image<uint8_t>(src));
        else if (is_image<uint16_t>(src))  assign_image(dest, numpy_image<uint16_t>(src));
        else if (is_image<uint32_t>(src))  assign_image(dest, numpy_image<uint32_t>(src));
        else if (is_image<uint64_t>(src))  assign_image(dest, numpy_image<uint64_t>(src));
        else if (is_image<int8_t>(src))    assign_image(dest, numpy_image<int8_t>(src));
        else if (is_image<int16_t>(src))   assign_image(dest, numpy_image<int16_t>(src));
        else if (is_image<int32_t>(src))   assign_image(dest, numpy_image<int32_t>(src));
        else if (is_image<int64_t>(src))   assign_image(dest, numpy_image<int64_t>(src));
        else if (is_image<float>(src))     assign_image(dest, numpy_image<float>(src));
        else if (is_image<double>(src))    assign_image(dest, numpy_image<double>(src));
        else if (is_image<rgb_pixel>(src)) assign_image(dest, numpy_image<rgb_pixel>(src));
        else DLIB_CASSERT(false, "Unsupported pixel type used in assign_image().");
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                          BORING IMPLEMENTATION STUFF
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename pixel_type>
    long num_rows(const numpy_image<pixel_type>& img)
    {
        if (img.size()==0)
            return 0;

        assert_correct_num_channels_in_image<pixel_type>(img);
        return img.shape(0);
    }

    template <typename pixel_type>
    long num_columns(const numpy_image<pixel_type>& img)
    {
        if (img.size()==0)
            return 0;

        assert_correct_num_channels_in_image<pixel_type>(img);
        return img.shape(1);
    }

    template <typename pixel_type>
    void set_image_size(numpy_image<pixel_type>& img, size_t rows, size_t cols)
    {
        img.set_size(rows, cols);
    }

    template <typename pixel_type>
    void* image_data(numpy_image<pixel_type>& img)
    {
        if (img.size()==0)
            return 0;

        assert_is_image<pixel_type>(img);
        return img.mutable_data(0);
    }

    template <typename pixel_type>
    const void* image_data (const numpy_image<pixel_type>& img)
    {
        if (img.size()==0)
            return 0;

        assert_is_image<pixel_type>(img);
        return img.data(0);
    }

    template <typename pixel_type>
    long width_step (const numpy_image<pixel_type>& img)
    {
        if (img.size()==0)
            return 0;

        assert_correct_num_channels_in_image<pixel_type>(img);
        using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
        if (img.ndim()==3 && img.strides(2) != sizeof(basic_pixel_type))
            throw dlib::error("The stride of the 3rd dimension (the channel dimension) of the numpy array must be " + std::to_string(sizeof(basic_pixel_type)));
        if (img.strides(1) != sizeof(pixel_type))
            throw dlib::error("The stride of the 2nd dimension (the columns dimension) of the numpy array must be " + std::to_string(sizeof(pixel_type)));

        return img.strides(0);
    }

    template <typename pixel_type>
    void swap(numpy_image<pixel_type>& a, numpy_image<pixel_type>& b)
    {
        std::swap(a,b);
    }


    template <typename T> 
    struct image_traits<numpy_image<T>>
    {
        typedef T pixel_type;
    };
}

// ----------------------------------------------------------------------------------------

namespace pybind11
{
    namespace detail
    {
        template <typename pixel_type> struct handle_type_name<dlib::numpy_image<pixel_type>> 
        {
            using basic_pixel_type = typename dlib::pixel_traits<pixel_type>::basic_pixel_type;

            template <size_t channels> 
            static PYBIND11_DESCR getname(typename std::enable_if<channels==1,int>::type) {
                return _("numpy.ndarray[(rows,cols),") + npy_format_descriptor<basic_pixel_type>::name() + _("]");
            }
            template <size_t channels> 
            static PYBIND11_DESCR getname(typename std::enable_if<channels!=1,int>::type) {
                if (channels == 2)
                    return _("numpy.ndarray[(rows,cols,2),") + npy_format_descriptor<basic_pixel_type>::name() + _("]");
                else if (channels == 3)
                    return _("numpy.ndarray[(rows,cols,3),") + npy_format_descriptor<basic_pixel_type>::name() + _("]");
                else if (channels == 4)
                    return _("numpy.ndarray[(rows,cols,4),") + npy_format_descriptor<basic_pixel_type>::name() + _("]");
            }

            static PYBIND11_DESCR name() {
                constexpr size_t channels = dlib::pixel_traits<pixel_type>::num;
                // The reason we have to call getname() in this wonky way is because
                // pybind11 uses a type that records the length of the returned string in
                // the type.  So we have to do this overloading to make the return type
                // from name() consistent.  In C++17 this would be a lot cleaner with
                // constexpr if, but can't use C++17 yet because of lack of wide support  :(
                return getname<channels>(0);
            }
        };

        template <typename pixel_type>
        struct pyobject_caster<dlib::numpy_image<pixel_type>> {
            using type = dlib::numpy_image<pixel_type>;

            bool load(handle src, bool convert) {
                // If passed a tuple where the first element of the tuple is a valid
                // numpy_image then bind the numpy_image to that element of the tuple.
                // We do this because there is a pattern of returning an image and some
                // associated metadata.  This allows the returned tuple from such functions
                // to also be treated as an image without needing to unpack the first
                // argument.
                if (PyTuple_Check(src.ptr()) && PyTuple_Size(src.ptr()) >= 1)
                    src = reinterpret_borrow<py::tuple>(src)[0];

                if (!type::check_(src))
                    return false;
                // stash the output of ensure into a temp variable since assigning it to
                // value (the member variable created by the PYBIND11_TYPE_CASTER)
                // apparently causes the return bool value to be ignored? 
                auto temp = type::ensure(src);
                if (!dlib::is_image<pixel_type>(temp))
                    return false;
                value = temp;
                return static_cast<bool>(value);
            }

            static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
                return src.inc_ref();
            }
            PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
        };
    }
}


// ----------------------------------------------------------------------------------------

#endif // DLIB_PYTHON_NuMPY_IMAGE_Hh_

