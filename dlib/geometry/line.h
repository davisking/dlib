// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LInE_H_
#define DLIB_LInE_H_

#include "line_abstract.h"
#include "vector.h"
#include <utility>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class line
    {
    public:

        line() = default;

        line(const dpoint& a, const dpoint& b) : end1(a), end2(b)
        {
            normal_vector = (end1-end2).cross(dlib::vector<double,3>(0,0,1)).normalize();
        }

        template <typename T>
        line(const std::pair<vector<T,2>,vector<T,2>>& l) : line(l.first, l.second) {}

        const dpoint& p1() const { return end1; }
        const dpoint& p2() const { return end2; }

        const dpoint& normal() const { return normal_vector; }

    private:

        dpoint end1;
        dpoint end2;

        dpoint normal_vector;
    };

// ----------------------------------------------------------------------------------------

    template <typename U>
    double signed_distance_to_line (
        const line& l,
        const vector<U,2>& p
    )
    {
        return dot(p-l.p1(), l.normal());
    }

    template <typename T, typename U>
    double signed_distance_to_line (
        const std::pair<vector<T,2>,vector<T,2> >& l,
        const vector<U,2>& p
    )
    {
        return signed_distance_to_line(line(l),p);
    }

    template <typename T, typename U>
    double distance_to_line (
        const std::pair<vector<T,2>,vector<T,2> >& l,
        const vector<U,2>& p
    )
    {
        return std::abs(signed_distance_to_line(l,p));
    }

    template <typename U>
    double distance_to_line (
        const line& l,
        const vector<U,2>& p
    )
    {
        return std::abs(signed_distance_to_line(l,p));
    }

// ----------------------------------------------------------------------------------------

    inline line reverse(const line& l)
    {
        return line(l.p2(), l.p1());
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    inline dpoint intersect(
        const std::pair<vector<T,2>,vector<T,2>>& a,
        const std::pair<vector<T,2>,vector<T,2>>& b
    )
    {
        // convert to homogeneous coordinates
        dlib::vector<double,3> a1 = a.first;
        dlib::vector<double,3> a2 = a.second;
        dlib::vector<double,3> b1 = b.first;
        dlib::vector<double,3> b2 = b.second;
        a1.z() = 1;
        a2.z() = 1;
        b1.z() = 1;
        b2.z() = 1;

        // find lines between pairs of points.
        auto l1 = a1.cross(a2);
        auto l2 = b1.cross(b2);

        // find intersection of the lines.
        auto p = l1.cross(l2);

        if (p.z() != 0)
            return p/p.z();
        else
            return dpoint(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
    }

// ----------------------------------------------------------------------------------------

    inline dpoint intersect(
        const line& a,
        const line& b
    )
    {
        return intersect(std::make_pair(a.p1(),a.p2()), std::make_pair(b.p1(), b.p2()));
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    inline size_t count_points_on_side_of_line(
        line l,
        const dpoint& reference_point,
        const std::vector<vector<T,2>>& pts,
        const double& dist_thresh
    )
    {
        if (signed_distance_to_line(l,reference_point) < 0)
            l = reverse(l);

        size_t cnt = 0;
        for (auto& p : pts)
        {
            double dist = signed_distance_to_line(l,p);
            if (0 <= dist && dist <= dist_thresh)
                ++cnt;
        }
        return cnt;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    inline double count_points_between_lines(
        line l1,
        line l2,
        const dpoint& reference_point,
        const std::vector<vector<T,2>>& pts
    )
    {
        if (signed_distance_to_line(l1,reference_point) < 0)
            l1 = reverse(l1);
        if (signed_distance_to_line(l2,reference_point) < 0)
            l2 = reverse(l2);

        size_t cnt = 0;
        for (auto& p : pts)
        {
            if (signed_distance_to_line(l1,p) > 0 && signed_distance_to_line(l2,p) > 0)
                ++cnt;
        }
        return cnt;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LInE_H_

