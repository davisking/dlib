// Copyright (C) 2022  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_POLYGOn_
#define DLIB_POLYGOn_

#include "polygon_abstract.h"
#include "rectangle.h"
#include "vector.h"

namespace dlib
{
    class polygon
    {
    public:
        using size_type = std::vector<point>::size_type;

        polygon(std::vector<point> points) : points(std::move(points)) {}

        size_type size() const { return points.size(); }

        point& operator[](const size_type idx) { return points[idx]; }
        const point& operator[](const size_type idx) const { return points[idx]; }
        const point& at(const size_type idx) const { return points.at(idx); }

        std::vector<point>::iterator begin() { return points.begin(); }
        std::vector<point>::iterator end() { return points.end(); }
        const std::vector<point>::const_iterator begin() const { return points.begin(); }
        const std::vector<point>::const_iterator end() const { return points.end(); }

        rectangle get_rect() const
        {
            rectangle rect;
            for (const auto& p : points)
                rect += p;
            return rect;
        }

        double area() const { return polygon_area(points); }

        template <typename alloc>
        void get_left_and_right_bounds (
            const long top,
            const long bottom,
            std::vector<double, alloc>& left_boundary,
            std::vector<double, alloc>& right_boundary
        ) const
        {
            using std::min;
            using std::max;

            left_boundary.assign(bottom-top+1, std::numeric_limits<double>::infinity());
            right_boundary.assign(bottom-top+1, -std::numeric_limits<double>::infinity());

            // trace out the points along the edge of the polynomial and record them
            for (unsigned long i = 0; i < points.size(); ++i)
            {
                const point p1 = points[i];
                const point p2 = points[(i+1)%points.size()];

                if (p1.y() == p2.y())
                {
                    if (top <= p1.y() && p1.y() <= bottom)
                    {
                        const long y = p1.y() - top;
                        const double xmin = min(p1.x(), p2.x());
                        const double xmax = min(p1.x(), p2.x());
                        left_boundary[y]  = min(left_boundary[y], xmin);
                        right_boundary[y] = max(right_boundary[y], xmax);
                    }
                }
                else
                {
                    // Here we trace out the line from p1 to p2 and record where it hits.  

                    // x = m*y + b
                    const double m = (p2.x() - p1.x())/(double)(p2.y()-p1.y());
                    const double b = p1.x() - m*p1.y(); // because: x1 = m*y1 + b

                    const long ymin = max(top,min(p1.y(), p2.y()));
                    const long ymax = min(bottom,max(p1.y(), p2.y()));
                    for (long y = ymin; y <= ymax; ++y)
                    {
                        const double x = m*y + b;
                        const unsigned long idx = y-top;
                        left_boundary[idx]  = min(left_boundary[idx], x);
                        right_boundary[idx] = max(right_boundary[idx], x);
                    }
                }
            }
        }

    private:
        std::vector<point> points;
    };
}

#endif  // polygon_h_INCLUDED
