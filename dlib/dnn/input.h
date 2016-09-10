// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_INPUT_H_
#define DLIB_DNn_INPUT_H_

#include "input_abstract.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../image_processing.h"
#include <sstream>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename T>
    class input
    {
        const static bool always_false = sizeof(T)!=sizeof(T); 
        static_assert(always_false, "Unsupported type given to input<>.  input<> only supports "
            "dlib::matrix and dlib::array2d objects."); 
    };

// ----------------------------------------------------------------------------------------

    template <size_t NR, size_t NC=NR>
    class input_rgb_image_sized;

    class input_rgb_image
    {
    public:
        typedef matrix<rgb_pixel> input_type;

        input_rgb_image (
        ) : 
            avg_red(122.782), 
            avg_green(117.001),
            avg_blue(104.298) 
        {
        }

        input_rgb_image (
            float avg_red_,
            float avg_green_,
            float avg_blue_
        ) : avg_red(avg_red_), avg_green(avg_green_), avg_blue(avg_blue_) 
        {}

        template <size_t NR, size_t NC>
        inline input_rgb_image (
            const input_rgb_image_sized<NR,NC>& item
        ); 

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input_rgb_image::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), 3, nr, nc);


            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        rgb_pixel temp = (*i)(r,c);
                        auto p = ptr++;
                        *p = (temp.red-avg_red)/256.0; 
                        p += offset;
                        *p = (temp.green-avg_green)/256.0; 
                        p += offset;
                        *p = (temp.blue-avg_blue)/256.0; 
                        p += offset;
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input_rgb_image& item, std::ostream& out)
        {
            serialize("input_rgb_image", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
        }

        friend void deserialize(input_rgb_image& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image.");
            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image& item)
        {
            out << "input_rgb_image("<<item.avg_red<<","<<item.avg_green<<","<<item.avg_blue<<")";
            return out;
        }

        friend void to_xml(const input_rgb_image& item, std::ostream& out)
        {
            out << "<input_rgb_image r='"<<item.avg_red<<"' g='"<<item.avg_green<<"' b='"<<item.avg_blue<<"'/>";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------

    template <size_t NR, size_t NC>
    class input_rgb_image_sized
    {
    public:
        static_assert(NR != 0 && NC != 0, "The input image can't be empty.");

        typedef matrix<rgb_pixel> input_type;

        input_rgb_image_sized (
        ) : 
            avg_red(122.782), 
            avg_green(117.001),
            avg_blue(104.298) 
        {
        }

        input_rgb_image_sized (
            const input_rgb_image& item
        ) : avg_red(item.get_avg_red()),
            avg_green(item.get_avg_green()),
            avg_blue(item.get_avg_blue())
        {}

        input_rgb_image_sized (
            float avg_red_,
            float avg_green_,
            float avg_blue_
        ) : avg_red(avg_red_), avg_green(avg_green_), avg_blue(avg_blue_) 
        {}

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            // make sure all input images have the correct size
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==NR && i->nc()==NC,
                    "\t input_rgb_image_sized::to_tensor()"
                    << "\n\t All input images must have "<<NR<<" rows and "<<NC<< " columns, but we got one with "<<i->nr()<<" rows and "<<i->nc()<<" columns."
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), 3, NR, NC);


            const size_t offset = NR*NC;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (size_t r = 0; r < NR; ++r)
                {
                    for (size_t c = 0; c < NC; ++c)
                    {
                        rgb_pixel temp = (*i)(r,c);
                        auto p = ptr++;
                        *p = (temp.red-avg_red)/256.0; 
                        p += offset;
                        *p = (temp.green-avg_green)/256.0; 
                        p += offset;
                        *p = (temp.blue-avg_blue)/256.0; 
                        p += offset;
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input_rgb_image_sized& item, std::ostream& out)
        {
            serialize("input_rgb_image_sized", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
            serialize(NR, out);
            serialize(NC, out);
        }

        friend void deserialize(input_rgb_image_sized& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_sized")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_sized.");
            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);
            size_t nr, nc;
            deserialize(nr, in);
            deserialize(nc, in);
            if (nr != NR || nc != NC)
            {
                std::ostringstream sout;
                sout << "Wrong image dimensions found while deserializing dlib::input_rgb_image_sized.\n";
                sout << "Expected "<<NR<<" rows and "<<NC<< " columns, but found "<<nr<<" rows and "<<nc<<" columns.";
                throw serialization_error(sout.str());
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_sized& item)
        {
            out << "input_rgb_image_sized("<<item.avg_red<<","<<item.avg_green<<","<<item.avg_blue<<") nr="<<NR<<" nc="<<NC;
            return out;
        }

        friend void to_xml(const input_rgb_image_sized& item, std::ostream& out)
        {
            out << "<input_rgb_image_sized r='"<<item.avg_red<<"' g='"<<item.avg_green<<"' b='"<<item.avg_blue<<"' nr='"<<NR<<"' nc='"<<NC<<"'/>";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------

    template <size_t NR, size_t NC>
    input_rgb_image::
    input_rgb_image (
        const input_rgb_image_sized<NR,NC>& item
    ) : avg_red(item.get_avg_red()),
        avg_green(item.get_avg_green()),
        avg_blue(item.get_avg_blue())
    {}

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename MM, typename L>
    class input<matrix<T,NR,NC,MM,L>> 
    {
    public:
        typedef matrix<T,NR,NC,MM,L> input_type;

        input() {}
        input(const input&) {}

        template <typename mm>
        input(const input<array2d<T,mm>>&) {}

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), pixel_traits<T>::num, nr, nc);

            typedef typename pixel_traits<T>::basic_pixel_type bptype;

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto temp = pixel_to_vector<float>((*i)(r,c));
                        auto p = ptr++;
                        for (long j = 0; j < temp.size(); ++j)
                        {
                            if (is_same_type<bptype,unsigned char>::value)
                                *p = temp(j)/256.0;
                            else
                                *p = temp(j);
                            p += offset;
                        }
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input& item, std::ostream& out)
        {
            serialize("input<matrix>", out);
        }

        friend void deserialize(input& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<matrix>")
                throw serialization_error("Unexpected version found while deserializing dlib::input.");
        }

        friend std::ostream& operator<<(std::ostream& out, const input& item)
        {
            out << "input<matrix>";
            return out;
        }

        friend void to_xml(const input& item, std::ostream& out)
        {
            out << "<input/>";
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename MM>
    class input<array2d<T,MM>> 
    {
    public:
        typedef array2d<T,MM> input_type;

        input() {}
        input(const input&) {}

        template <long NR, long NC, typename mm, typename L>
        input(const input<matrix<T,NR,NC,mm,L>>&) {}

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input::to_tensor()"
                    << "\n\t All array2d objects given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), pixel_traits<T>::num, nr, nc);
            typedef typename pixel_traits<T>::basic_pixel_type bptype;

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto temp = pixel_to_vector<float>((*i)[r][c]);
                        auto p = ptr++;
                        for (long j = 0; j < temp.size(); ++j)
                        {
                            if (is_same_type<bptype,unsigned char>::value)
                                *p = temp(j)/256.0;
                            else
                                *p = temp(j);
                            p += offset;
                        }
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input& item, std::ostream& out)
        {
            serialize("input<array2d>", out);
        }

        friend void deserialize(input& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<array2d>")
                throw serialization_error("Unexpected version found while deserializing dlib::input.");
        }
        friend std::ostream& operator<<(std::ostream& out, const input& item)
        {
            out << "input<array2d>";
            return out;
        }

        friend void to_xml(const input& item, std::ostream& out)
        {
            out << "<input/>";
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename PYRAMID_TYPE>
    class input_rgb_image_pyramid
    {
    public:
        typedef matrix<rgb_pixel> input_type;
        typedef PYRAMID_TYPE pyramid_type;

        input_rgb_image_pyramid (
        ) : 
            avg_red(122.782), 
            avg_green(117.001),
            avg_blue(104.298) 
        {
        }

        input_rgb_image_pyramid (
            float avg_red_,
            float avg_green_,
            float avg_blue_
        ) : avg_red(avg_red_), avg_green(avg_green_), avg_blue(avg_blue_) 
        {}

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        bool image_contained_point (
            const tensor& data,
            const point& p
        ) const
        {
            auto&& rects = any_cast<std::vector<rectangle>>(data.annotation());
            DLIB_CASSERT(rects.size() > 0);
            return rects[0].contains(p);
        }

        drectangle tensor_space_to_image_space (
            const tensor& data,
            drectangle r
        ) const
        {
            auto&& rects = any_cast<std::vector<rectangle>>(data.annotation());
            return tiled_pyramid_to_image<pyramid_type>(rects, r);
        }

        drectangle image_space_to_tensor_space (
            const tensor& data,
            double scale,
            drectangle r 
        ) const
        {
            DLIB_CASSERT(0 < scale && scale <= 1 , "scale: "<< scale);
            auto&& rects = any_cast<std::vector<rectangle>>(data.annotation());
            return image_to_tiled_pyramid<pyramid_type>(rects, scale, r);
        }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            auto nr = ibegin->nr();
            auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input_rgb_image_pyramid::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }


            std::vector<matrix<rgb_pixel>> imgs(std::distance(ibegin,iend));
            parallel_for(0, imgs.size(), [&](long i){
                std::vector<rectangle> rects;
                if (i == 0)
                    create_tiled_pyramid<pyramid_type>(ibegin[i], imgs[i], data.annotation().get<std::vector<rectangle>>());
                else
                    create_tiled_pyramid<pyramid_type>(ibegin[i], imgs[i], rects);
            });
            nr = imgs[0].nr();
            nc = imgs[0].nc();
            data.set_size(imgs.size(), 3, nr, nc);

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto&& img : imgs)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        rgb_pixel temp = img(r,c);
                        auto p = ptr++;
                        *p = (temp.red-avg_red)/256.0; 
                        p += offset;
                        *p = (temp.green-avg_green)/256.0; 
                        p += offset;
                        *p = (temp.blue-avg_blue)/256.0; 
                        p += offset;
                    }
                }
                ptr += offset*(data.k()-1);
            }
        }

        friend void serialize(const input_rgb_image_pyramid& item, std::ostream& out)
        {
            serialize("input_rgb_image_pyramid", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
        }

        friend void deserialize(input_rgb_image_pyramid& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_pyramid")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_pyramid.");
            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_pyramid& item)
        {
            out << "input_rgb_image_pyramid("<<item.avg_red<<","<<item.avg_green<<","<<item.avg_blue<<")";
            return out;
        }

        friend void to_xml(const input_rgb_image_pyramid& item, std::ostream& out)
        {
            out << "<input_rgb_image_pyramid r='"<<item.avg_red<<"' g='"<<item.avg_green<<"' b='"<<item.avg_blue<<"'/>";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_H_

