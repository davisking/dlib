// Copyright (C) 2022  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_POLYGOn_ABSTRACT_H_
#ifdef DLIB_POLYGOn_ABSTRACT_H_

#include "rectangle.h"
#include "vector.h"

namespace dlib
{
    class polygon
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a polygon inside a Cartesian coordinate system.
                It is just a wrapper class for a std::vector<point> with some added
                methods.

                Note that the origin of the coordinate system, i.e. (0,0), is located
                at the upper left corner.  That is, points such as (1,1) or (3,5)
                represent locations that are below and to the right of the origin.
        !*/

    public:
        using size_type = std::vector<point>::size_type;

        polygon(std::vector<point> points);
        /*!
            ensures
                - #*this represents a polygon defined by points.
        !*/

        size_type size() const;
        /*!
            ensures
                - returns the number of points in the polygon.
        !*/

        point& operator[](const size_type idx);
        /*!
            requires
                - idx < size()
            ensures
                - returns the point of the polygon at index idx.
        !*/

        const point& operator[](const size_type idx) const;
        /*!
            requires
                - idx < size()
            ensures
                - returns the point of the polygon at index idx.
        !*/

        const point& at(const size_type idx) const;
        /*!
            ensures
                - returns the point of the polygon at index idx.
            throws
                - std::out_of_range if idx >= size()
        !*/

        rectangle get_rect() const;
        /*!
            ensures
                - returns smallest rectangle that contains all points in the polygon.
        !*/

        double area() const;
        /*!
            ensures
                - If you walk the points of #*this in order to make a closed polygon, what is its
                  area?  This function returns that area.  It uses the shoelace formula to
                  compute the result and so works for general non-self-intersecting polygons.
        !*/

        template <typename alloc>
        void get_convex_shape (
            const long top,
            const long bottom,
            std::vector<double, alloc>& left_boundary,
            std::vector<double, alloc>& right_boundary
        ) const;
        /*!                                                                                
            requires                                                                       
                - 0 <= top <= bottom                                                       
            ensures                                                                        
                - interprets points as the coordinates defining a convex polygon.  In      
                  particular, we interpret points as a list of the vertices of the polygon 
                  and assume they are ordered in clockwise order.                          
                - #left_boundary.size() == bottom-top+1                                    
                - #right_boundary.size() == bottom-top+1                                   
                - for all top <= y <= bottom:                                              
                    - #left_boundary[y-top] == the x coordinate for the left most side of  
                      the polygon at coordinate y.                                         
                    - #right_boundary[y-top] == the x coordinate for the right most side of
                      the polygon at coordinate y.                                         

              Note that it isn't illegal to call this method if the polygon defined by
              points isn't convex.  In that case, we won't return a convex shape.
        !*/

    private:
        std::vector<point> points;
    };
}

#endif  // DLIB_POLYGOn_ABSTRACT_H_
