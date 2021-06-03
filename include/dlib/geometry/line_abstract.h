// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_LInE_ABSTRACT_H_
#ifdef DLIB_LInE_ABSTRACT_H_

#include "../vector_abstract.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class line
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a line in the 2D plane.  The line is defined by two
                points running through it, p1() and p2().  This object also includes a
                unit normal vector that is perpendicular to the line.
        !*/

    public:

        line(
        );
        /*!
            ensures
                - p1(), p2(), and normal() are all the 0 vector.
        !*/

        line(
            const dpoint& a, 
            const dpoint& b
        );
        /*!
            ensures
                - #p1() == a
                - #p2() == b
                - #normal() == A vector normal to the line passing through points a and b.
                  In particular, it is given by: (a-b).cross(dlib::vector<double,3>(0,0,1)).normalize().
                  Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        !*/

        template <typename T>
        line(
            const std::pair<vector<T,2>,vector<T,2>>& l
        );
        /*!
            ensures
                - #*this == line(l.first, l.second)
        !*/

        const dpoint& p1(
        ) const; 
        /*!
            ensures
                - returns the first endpoint of the line.
        !*/

        const dpoint& p2(
        ) const;
        /*!
            ensures
                - returns the second endpoint of the line.
        !*/

        const dpoint& normal(
        ) const; 
        /*!
            ensures
                - returns a unit vector that is normal to the line passing through p1() and p2().
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <typename U>
    double signed_distance_to_line (
        const line& l,
        const vector<U,2>& p
    );
    /*!
        ensures
            - returns how far p is from the line l.  This is a signed distance.  The sign
              indicates which side of the line the point is on and the magnitude is the
              distance. Moreover, the direction of positive sign is pointed to by the
              vector l.normal().
            - To be specific, this routine returns dot(p-l.p1(), l.normal())
    !*/

    template <typename T, typename U>
    double signed_distance_to_line (
        const std::pair<vector<T,2>,vector<T,2> >& l,
        const vector<U,2>& p
    );
    /*!
        ensures
            - returns signed_distance_to_line(line(l),p);
    !*/

    template <typename T, typename U>
    double distance_to_line (
        const std::pair<vector<T,2>,vector<T,2> >& l,
        const vector<U,2>& p
    );
    /*!
        ensures
            - returns abs(signed_distance_to_line(l,p))
    !*/

    template <typename U>
    double distance_to_line (
        const line& l,
        const vector<U,2>& p
    );
    /*!
        ensures
            - returns abs(signed_distance_to_line(l,p))
    !*/

// ----------------------------------------------------------------------------------------

    line reverse(
        const line& l
    );
    /*!
        ensures
            - returns line(l.p2(), l.p1())
              (i.e. returns a line object that represents the same line as l but with the
              endpoints, and therefore, the normal vector flipped.  This means that the
              signed distance of operator() is also flipped).
    !*/

// ----------------------------------------------------------------------------------------

    dpoint intersect(
        const line& a,
        const line& b
    );
    /*!
        ensures
            - returns the point of intersection between lines a and b.  If no such point
              exists then this function returns a point with Inf values in it.
    !*/

// ----------------------------------------------------------------------------------------

    double angle_between_lines (
        const line& a,
        const line& b
    );
    /*!
        ensures
            - returns the angle, in degrees, between the given lines.  This is a number in
              the range [0 90].
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T>
    dpoint intersect(
        const std::pair<vector<T,2>,vector<T,2>>& a,
        const std::pair<vector<T,2>,vector<T,2>>& b
    );
    /*!
        ensures
            - returns intersect(line(a), line(b))
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T>
    size_t count_points_on_side_of_line(
        const line& l,
        const dpoint& reference_point,
        const std::vector<vector<T,2>>& pts,
        const double& dist_thresh_min = 0,
        const double& dist_thresh_max = std::numeric_limits<double>::infinity()
    );
    /*!
        ensures
            - Returns a count of how many points in pts have a distance from the line l
              that is in the range [dist_thresh_min, dist_thresh_max].  This distance is a
              signed value that indicates how far a point is from the line. Moreover, if
              the point is on the same side as reference_point then the distance is
              positive, otherwise it is negative.  So for example, If this range is [0,
              infinity] then this function counts how many points are on the same side of l
              as reference_point.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T>
    double count_points_between_lines(
        const line& l1,
        const line& l2,
        const dpoint& reference_point,
        const std::vector<vector<T,2>>& pts
    );
    /*!
        ensures
            - Counts and returns the number of points in pts that are between lines l1 and
              l2.  Since a pair of lines will, in the general case, divide the plane into 4
              regions, we identify the region of interest as the one that contains the
              reference_point.  Therefore, this function counts the number of points in pts
              that appear in the same region as reference_point.
    !*/

// ----------------------------------------------------------------------------------------

    struct no_convex_quadrilateral : dlib::error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is the exception thrown by find_convex_quadrilateral() if the inputs
                can't form a convex quadrilateral.
        !*/
        no_convex_quadrilateral(
        ) : dlib::error("Lines given to find_convex_quadrilateral() don't form any convex quadrilateral.") 
        {}
    };

    std::array<dpoint,4> find_convex_quadrilateral (
        const std::array<line,4>& lines
    );
    /*!
        ensures
            - Is there a set of 4 points, made up of the intersections of the given lines,
              that forms a convex quadrilateral?  If yes then this routine returns those 4
              points and if not throws no_convex_quadrilateral.
        throws
            - no_convex_quadrilateral
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LInE_ABSTRACT_H_

