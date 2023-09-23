// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DRAW_IMAGe_
#define DLIB_DRAW_IMAGe_

#include "draw_abstract.h"
#include "../algs.h"
#include "../pixel.h"
#include "../matrix.h"
#include "../gui_widgets/fonts.h"
#include "../geometry.h"
#include <cmath>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_line (
        long x1,
        long y1,
        long x2,
        long y2,
        image_type& c_,
        const pixel_type& val
    ) 
    {
        image_view<image_type> c(c_);
        if (x1 == x2)
        {
            // make sure y1 comes before y2
            if (y1 > y2)
                swap(y1,y2);

            if (x1 < 0 || x1 >= c.nc())
                return;


            // this is a vertical line
            for (long y = y1; y <= y2; ++y)
            {
                if (y < 0 || y >= c.nr())
                    continue;

                assign_pixel(c[y][x1], val);
            }
        }
        else if (y1 == y2)
        {

            // make sure x1 comes before x2
            if (x1 > x2)
                swap(x1,x2);

            if (y1 < 0 || y1 >= c.nr())
                return;

            // this is a horizontal line
            for (long x = x1; x <= x2; ++x)
            {
                if (x < 0 || x >= c.nc())
                    continue;

                assign_pixel(c[y1][x] , val);
            }
        }
        else
        {
            // This part is a little more complicated because we are going to perform alpha
            // blending so the diagonal lines look nice.
            const rectangle valid_area = get_rect(c);
            rgb_alpha_pixel alpha_pixel;
            assign_pixel(alpha_pixel, val);
            const unsigned char max_alpha = alpha_pixel.alpha;

            const long rise = (((long)y2) - ((long)y1));
            const long run = (((long)x2) - ((long)x1));
            if (std::abs(rise) < std::abs(run))
            {
                const double slope = ((double)rise)/run;


                double first, last;


                if (x1 > x2)                
                {
                    first = std::max(x2,valid_area.left());
                    last = std::min(x1,valid_area.right());
                }
                else
                {
                    first = std::max(x1,valid_area.left());
                    last = std::min(x2,valid_area.right());
                }                             

                long y;
                long x;
                const double x1f = x1;
                const double y1f = y1;
                for (double i = first; i <= last; ++i)
                {   
                    const double dy = slope*(i-x1f) + y1f;
                    const double dx = i;

                    y = static_cast<long>(dy);
                    x = static_cast<long>(dx);


                    if (y >= valid_area.top() && y <= valid_area.bottom())
                    {
                        alpha_pixel.alpha = static_cast<unsigned char>((1.0-(dy-y))*max_alpha);
                        assign_pixel(c[y][x], alpha_pixel);
                    }
                    if (y+1 >= valid_area.top() && y+1 <= valid_area.bottom())
                    {
                        alpha_pixel.alpha = static_cast<unsigned char>((dy-y)*max_alpha);
                        assign_pixel(c[y+1][x], alpha_pixel);
                    }
                }         
            }
            else
            {
                const double slope = ((double)run)/rise;


                double first, last;


                if (y1 > y2)                
                {
                    first = std::max(y2,valid_area.top());
                    last = std::min(y1,valid_area.bottom());
                }
                else
                {
                    first = std::max(y1,valid_area.top());
                    last = std::min(y2,valid_area.bottom());
                }                             

                long x;
                long y;
                const double x1f = x1;
                const double y1f = y1;
                for (double i = first; i <= last; ++i)
                {   
                    const double dx = slope*(i-y1f) + x1f;
                    const double dy = i;

                    y = static_cast<long>(dy);
                    x = static_cast<long>(dx);

                    if (x >= valid_area.left() && x <= valid_area.right())
                    {
                        alpha_pixel.alpha = static_cast<unsigned char>((1.0-(dx-x))*max_alpha);
                        assign_pixel(c[y][x], alpha_pixel);
                    }
                    if (x+1 >= valid_area.left() && x+1 <= valid_area.right())
                    {
                        alpha_pixel.alpha = static_cast<unsigned char>((dx-x)*max_alpha);
                        assign_pixel(c[y][x+1], alpha_pixel);
                    }
                } 
            }
        }

    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_line (
        image_type& c,
        const point& p1,
        const point& p2,
        const pixel_type& val
    ) 
    {
        draw_line(p1.x(),p1.y(),p2.x(),p2.y(),c,val);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_rectangle (
        image_type& c,
        const rectangle& rect,
        const pixel_type& val
    ) 
    {
        draw_line(c, rect.tl_corner(), rect.tr_corner(), val);
        draw_line(c, rect.bl_corner(), rect.br_corner(), val);
        draw_line(c, rect.tl_corner(), rect.bl_corner(), val);
        draw_line(c, rect.tr_corner(), rect.br_corner(), val);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_rectangle (
        image_type& c,
        const rectangle& rect,
        const pixel_type& val,
        unsigned int thickness
    ) 
    {
        for (unsigned int i = 0; i < thickness; ++i)
        {
            if ((i%2)==0)
                draw_rectangle(c,shrink_rect(rect,(i+1)/2),val);
            else
                draw_rectangle(c,grow_rect(rect,(i+1)/2),val);
        }
    }

// ----------------------------------------------------------------------------------------

    struct string_dims
    {
        string_dims() = default;
        string_dims (
            unsigned long width,
            unsigned long height
        ) : width(width), height(height) {}
        unsigned long width = 0;
        unsigned long height = 0;
    };

    template <
        typename T, typename traits,
        typename alloc
    >
    string_dims compute_string_dims (
        const std::basic_string<T, traits, alloc>& str,
        const std::shared_ptr<font>& f_ptr = default_font::get_font()
    )
    {
        using string = std::basic_string<T, traits, alloc>;

        const font& f = *f_ptr;

        long height = f.height();
        long width = 0;
        for (typename string::size_type i = 0; i < str.size(); ++i)
        {
            // ignore the '\r' character
            if (str[i] == '\r')
                continue;

            // A combining character should be applied to the previous character, and we
            // therefore make one step back. If a combining comes right after a newline,
            // then there must be some kind of error in the string, and we don't combine.
            if (is_combining_char(str[i]))
            {
                width -= f[str[i]].width();
            }

            if (str[i] == '\n')
            {
                height += f.height();
                width = f.left_overflow();
                continue;
            }

            const letter& l = f[str[i]];
            width += l.width();
        }
        return string_dims(width, height);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename traits,
        typename alloc,
        typename image_type,
        typename pixel_type
    >
    void draw_string (
        image_type& image,
        const dlib::point& p,
        const std::basic_string<T,traits,alloc>& str,
        const pixel_type& color,
        const std::shared_ptr<font>& f_ptr = default_font::get_font(),
        typename std::basic_string<T,traits,alloc>::size_type first = 0,
        typename std::basic_string<T,traits,alloc>::size_type last = (std::basic_string<T,traits,alloc>::npos)
    )
    {
        using string = std::basic_string<T,traits,alloc>;
        DLIB_ASSERT((last == string::npos) || (first <= last && last < str.size()),
                    "\tvoid dlib::draw_string()"
                    << "\n\tlast == string::npos: " << ((last == string::npos)?"true":"false")
                    << "\n\tfirst: " << (unsigned long)first
                    << "\n\tlast:  " << (unsigned long)last
                    << "\n\tstr.size():  " << (unsigned long)str.size());

        if (last == string::npos)
            last = str.size();

        const rectangle rect(p, p);
        const font& f = *f_ptr;
        image_view<image_type> img(image);

        long y_offset = rect.top() + f.ascender() - 1;

        long pos = rect.left()+f.left_overflow();
        convert_to_utf32(str.begin() + first, str.begin() + last, [&](unichar ch)
        {
            // ignore the '\r' character
            if (ch == '\r')
                return;

            // A combining character should be applied to the previous character, and we
            // therefore make one step back. If a combining comes right after a newline, 
            // then there must be some kind of error in the string, and we don't combine.
            if(is_combining_char(ch) &&
               pos > rect.left() + static_cast<long>(f.left_overflow()))
            {
                pos -= f[ch].width();
            }

            if (ch == '\n')
            {
                y_offset += f.height();
                pos = rect.left()+f.left_overflow();
                return;
            }

            // only look at letters in the intersection area
            if (img.nr() + static_cast<long>(f.height()) < y_offset)
            {
                // the string is now below our rectangle so we are done
                return;
            }
            else if (0 > pos - static_cast<long>(f.left_overflow()) && 
                pos + static_cast<long>(f[ch].width() + f.right_overflow()) < 0)
            {
                pos += f[ch].width();
                return;
            }
            else if (img.nc() + static_cast<long>(f.right_overflow()) < pos)
            {
                // keep looking because there might be a '\n' in the string that
                // will wrap us around and put us back into our rectangle.
                return;
            }

            // at this point in the loop we know that f[str[i]] overlaps 
            // horizontally with the intersection rectangle area.

            const letter& l = f[ch];
            for (unsigned short i = 0; i < l.num_of_points(); ++i)
            {
                const long x = l[i].x + pos;
                const long y = l[i].y + y_offset;
                // draw each pixel of the letter if it is inside the intersection
                // rectangle
                if (x >= 0 && x < img.nc() && y >= 0 && y < img.nr())
                {
                    assign_pixel(img[y][x], color);
                }
            }

            pos += l.width();
        });

    }

    template <
        typename image_type,
        typename pixel_type
    >
    void draw_string (
        image_type& image,
        const dlib::point& p,
        const char* str,
        const pixel_type& color,
        const std::shared_ptr<font>& f_ptr = default_font::get_font(),
        typename std::string::size_type first = 0,
        typename std::string::size_type last = std::string::npos
    )
    {
        draw_string(image, p, std::string(str), color, f_ptr, first, last);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void fill_rect (
        image_type& img_,
        const rectangle& rect,
        const pixel_type& pixel
    )
    {
        image_view<image_type> img(img_);
        rectangle area = rect.intersect(get_rect(img));

        for (long r = area.top(); r <= area.bottom(); ++r)
        {
            for (long c = area.left(); c <= area.right(); ++c)
            {
                assign_pixel(img[r][c], pixel);
            }
        }
    }

// ---------------------------------------------------------------------------------------

    template <typename image_type, typename pixel_type>
    void draw_solid_convex_polygon (
        image_type& image,
        const polygon& poly,
        const pixel_type& color,
        const bool antialias = true,
        const rectangle& area = rectangle(std::numeric_limits<long>::min(), std::numeric_limits<long>::min(),
                                          std::numeric_limits<long>::max(), std::numeric_limits<long>::max())
    )
    {
        const rectangle image_rect = get_rect(image);
        const rectangle bounding_box = poly.get_rect();
        rectangle valid_area(image_rect.intersect(area));

        // Don't do anything if the polygon is totally outside the area we can draw in
        // right now.
        if (bounding_box.intersect(valid_area).is_empty())
            return;

        image_view<image_type> img(image);
        rgb_alpha_pixel alpha_pixel;
        assign_pixel(alpha_pixel, color);
        const unsigned char max_alpha = alpha_pixel.alpha;

        // we will only want to loop over the part of left_boundary that is part of the
        // valid_area.
        valid_area = shrink_rect(valid_area.intersect(bounding_box), 1);

        // Since we look at the adjacent rows of boundary information when doing the alpha
        // blending, we want to make sure we always have some boundary information unless
        // we are at the absolute edge of the polygon.
        const long left_offset = (antialias && valid_area.left() == bounding_box.left()) ? 0 : 1;
        const long top_offset = (antialias && valid_area.top() == bounding_box.top()) ? 0 : 1;
        const long right_offset = (antialias && valid_area.right() == bounding_box.right()) ? 0 : 1;
        const long bottom_offset = (antialias && valid_area.bottom() == bounding_box.bottom()) ? 0 : 1;
        if (antialias)
            valid_area = grow_rect(valid_area, 1).intersect(image_rect);

        std::vector<double> left_boundary;
        std::vector<double> right_boundary;
        poly.get_left_and_right_bounds(valid_area.top(), valid_area.bottom(), left_boundary, right_boundary);

        // draw the polygon row by row
        for (unsigned long i = top_offset; i < left_boundary.size(); ++i)
        {
            long left_x = static_cast<long>(std::ceil(left_boundary[i]));
            long right_x = static_cast<long>(std::floor(right_boundary[i]));

            left_x = std::max(left_x, valid_area.left());
            right_x = std::min(right_x, valid_area.right());

            if (i < left_boundary.size() - bottom_offset)
            {
                // draw the main body of the polygon
                for (long x = left_x + left_offset; x <= right_x - right_offset; ++x)
                {
                    const long y = i + valid_area.top();
                    assign_pixel(img[y][x], color);
                }
            }

            if (i == 0 || !antialias)
                continue;

            // Now draw anti-aliased edges so they don't look all pixely.

            // Alpha blend the edges on the left side.
            double delta = left_boundary[i-1] - left_boundary[i];
            if (std::abs(delta) <= 1)
            {
                if (std::floor(left_boundary[i]) != left_x)
                {
                    const point p(static_cast<long>(std::floor(left_boundary[i])), i + valid_area.top());
                    rgb_alpha_pixel temp = alpha_pixel;
                    temp.alpha = max_alpha-static_cast<unsigned char>((left_boundary[i]-p.x())*max_alpha);
                    if (valid_area.contains(p))
                        assign_pixel(img[p.y()][p.x()], temp);
                }
            }
            else if (delta < 0)  // on the bottom side
            {
                for (long x = static_cast<long>(std::ceil(left_boundary[i-1])); x < left_x; ++x)
                {
                    const point p(x, i+valid_area.top());
                    rgb_alpha_pixel temp = alpha_pixel;
                    temp.alpha = static_cast<unsigned char>((x-left_boundary[i-1])/std::abs(delta)*max_alpha);
                    if (valid_area.contains(p))
                        assign_pixel(img[p.y()][p.x()], temp);
                }
            }
            else // on the top side
            {
                const long old_left_x = static_cast<long>(std::ceil(left_boundary[i-1]));
                for (long x = left_x; x < old_left_x; ++x)
                {
                    const point p(x, i + valid_area.top()-1);
                    rgb_alpha_pixel temp = alpha_pixel;
                    temp.alpha = static_cast<unsigned char>((x-left_boundary[i])/delta*max_alpha);
                    if (valid_area.contains(p))
                        assign_pixel(img[p.y()][p.x()],temp);
                }
            }


            // Alpha blend the edges on the right side
            delta = right_boundary[i-1] - right_boundary[i];
            if (std::abs(delta) <= 1)
            {
                if (std::ceil(right_boundary[i]) != right_x)
                {
                    const point p(static_cast<long>(std::ceil(right_boundary[i])), i + valid_area.top());
                    rgb_alpha_pixel temp = alpha_pixel;
                    temp.alpha = max_alpha-static_cast<unsigned char>((p.x()-right_boundary[i])*max_alpha);
                    if (valid_area.contains(p))
                        assign_pixel(img[p.y()][p.x()],temp);
                }
            }
            else if (delta < 0) // on the top side
            {
                for (long x = static_cast<long>(std::floor(right_boundary[i-1]))+1; x <= right_x; ++x)
                {
                    const point p(x, i + valid_area.top() - 1);
                    rgb_alpha_pixel temp = alpha_pixel;
                    temp.alpha = static_cast<unsigned char>((right_boundary[i]-x)/std::abs(delta)*max_alpha);
                    if (valid_area.contains(p))
                        assign_pixel(img[p.y()][p.x()],temp);
                }
            }
            else // on the bottom side
            {
                const long old_right_x = static_cast<long>(std::floor(right_boundary[i-1]));
                for (long x = right_x+1; x <= old_right_x; ++x)
                {
                    const point p(x, i + valid_area.top());
                    rgb_alpha_pixel temp = alpha_pixel;
                    temp.alpha = static_cast<unsigned char>((right_boundary[i-1]-x)/delta*max_alpha);
                    if (valid_area.contains(p))
                        assign_pixel(img[p.y()][p.x()],temp);
                }
            }
        }
    }

// ---------------------------------------------------------------------------------------

    template <typename image_type, typename pixel_type>
    void draw_solid_polygon (
        image_type& image,
        const polygon& poly,
        const pixel_type& color,
        const rectangle& area = rectangle(std::numeric_limits<long>::min(), std::numeric_limits<long>::min(),
                                          std::numeric_limits<long>::max(), std::numeric_limits<long>::max())
    )
    {
        image_view<image_type> img(image);
        rectangle valid_area(get_rect(image).intersect(area));
        const rectangle bounding_box = poly.get_rect();
        if (bounding_box.intersect(valid_area).is_empty())
            return;

        valid_area = shrink_rect(valid_area.intersect(bounding_box), 1);

        std::vector<double> intersections;
        for (long y = valid_area.top(); y <= valid_area.bottom(); ++y)
        {
            // Compute the intersections with the scanline.
            intersections.clear();
            for (size_t i = 0; i < poly.size(); ++i)
            {
                const auto& p1 = poly[i];
                const auto& p2 = poly[(i + 1) % poly.size()];
                // Skip horizontal edges
                if (p1.y() == p2.y())
                    continue;

                // Skip if there are no intersections at this position
                if ((y <= p1.y() && y <= p2.y()) || (y > p1.y() && y > p2.y()))
                    continue;

                // Add x coordinates of intersecting points
                intersections.push_back(p1.x() + (y - p1.y()) * (p2.x() - p1.x()) / static_cast<double>(p2.y() - p1.y()));
            }

            std::sort(intersections.begin(), intersections.end());

            for (size_t i = 0; i < intersections.size(); i += 2)
            {
                long left_x = static_cast<long>(std::ceil(intersections[i]));
                long right_x = static_cast<long>(std::floor(intersections[i + 1]));
                left_x = std::max(left_x, valid_area.left());
                right_x = std::min(right_x, valid_area.right());

                // Draw the main body of the polygon
                for (long x = left_x; x <= right_x; ++x)
                {
                    assign_pixel(img[y][x], color);
                }
            }
        }
    }

// ---------------------------------------------------------------------------------------

    template <
        typename image_array_type
        >
    matrix<typename image_traits<typename image_array_type::value_type>::pixel_type> tile_images (
        const image_array_type& images
    )
    {
        typedef typename image_traits<typename image_array_type::value_type>::pixel_type T;

        if (images.size() == 0)
            return matrix<T>();

        const unsigned long size_nc = square_root(images.size());
        const unsigned long size_nr = (size_nc*(size_nc-1)>=images.size())? size_nc-1 : size_nc;
        // Figure out the size we have to use for each chip in the big main image.  We will
        // use the largest dimensions seen across all the chips.
        long nr = 0;
        long nc = 0;
        for (unsigned long i = 0; i < images.size(); ++i)
        {
            nr = std::max(num_rows(images[i]), nr);
            nc = std::max(num_columns(images[i]), nc);
        }

        matrix<T> temp(size_nr*nr, size_nc*nc);
        T background_color;
        assign_pixel(background_color, 0);
        temp = background_color;
        unsigned long idx = 0;
        for (unsigned long r = 0; r < size_nr; ++r)
        {
            for (unsigned long c = 0; c < size_nc; ++c)
            {
                if (idx < images.size())
                {
                    set_subm(temp, r*nr, c*nc, nr, nc) = mat(images[idx]);
                }
                ++idx;
            }
        }
        return temp;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_solid_circle (
        image_type& img_,
        const dpoint& center_point,
        double radius,
        const pixel_type& pixel
    )
    {
        image_view<image_type> img(img_);
        using std::sqrt;
        const rectangle valid_area(get_rect(img));
        const double x = center_point.x();
        const double y = center_point.y();
        const point cp(center_point);
        if (radius > 1)
        {
            long first_x = static_cast<long>(x - radius + 0.5);
            long last_x = static_cast<long>(x + radius + 0.5);
            const double rs = radius*radius;

            // ensure that we only loop over the part of the x dimension that this
            // image contains.
            if (first_x < valid_area.left())
                first_x = valid_area.left();
            if (last_x > valid_area.right())
                last_x = valid_area.right();

            long top, bottom;

            top = std::lround(sqrt(std::max(rs - (first_x-x-0.5)*(first_x-x-0.5),0.0)));
            top += y;
            long last = top;

            // draw the left half of the circle
            long middle = std::min(cp.x()-1,last_x);
            for (long i = first_x; i <= middle; ++i)
            {
                double a = i - x + 0.5;
                // find the top of the arc
                top = std::lround(sqrt(std::max(rs - a*a,0.0)));
                top += y;
                long temp = top;

                while(top >= last) 
                {
                    bottom = y - top + y;
                    draw_line(img_, point(i,top),point(i,bottom),pixel);
                    --top;
                }

                last = temp;
            }

            middle = std::max(cp.x(),first_x);
            top = std::lround(sqrt(std::max(rs - (last_x-x+0.5)*(last_x-x+0.5),0.0)));
            top += y;
            last = top;
            // draw the right half of the circle
            for (long i = last_x; i >= middle; --i)
            {
                double a = i - x - 0.5;
                // find the top of the arc
                top = std::lround(sqrt(std::max(rs - a*a,0.0)));
                top += y;
                long temp = top;

                while(top >= last) 
                {
                    bottom = y - top + y;
                    draw_line(img_, point(i,top),point(i,bottom),pixel);
                    --top;
                }

                last = temp;
            }
        }
        else if (valid_area.contains(cp))
        {
            // For circles smaller than a pixel we will just alpha blend them in proportion
            // to how small they are.
            rgb_alpha_pixel temp;
            assign_pixel(temp, pixel);
            temp.alpha = static_cast<unsigned char>(255*radius + 0.5);
            assign_pixel(img[cp.y()][cp.x()], temp);
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DRAW_IMAGe_




