// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BORDER_EnUMERATOR_H_
#define DLIB_BORDER_EnUMERATOR_H_

#include "border_enumerator_abstract.h"
#include "rectangle.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class border_enumerator
    {
    public:
        border_enumerator(
        ) 
        {
        }

        border_enumerator(
            const rectangle& rect_,
            unsigned long border_size
        ) : 
            rect(rect_),
            inner_rect(shrink_rect(rect_, border_size))
        {
            reset();
        }

        void reset (
        ) 
        {
            p = rect.tl_corner();
            p.x() -= 1;
        }

        bool at_start (
        ) const
        {
            point temp = rect.tl_corner();
            temp.x() -=1;
            return temp == p;
        }

        bool current_element_valid(
        ) const
        {
            return rect.contains(p) && !inner_rect.contains(p);
        }

        bool move_next()
        {
            p.x() += 1;
            if (p.x() > rect.right())
            {
                p.y() += 1;
                p.x() = rect.left();
            }
            else if (inner_rect.contains(p))
            {
                p.x() = inner_rect.right()+1;
            }

            return current_element_valid();
        }

        unsigned long size (
        ) const
        {
            return rect.area() - inner_rect.area();
        }

        const point& element (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_element_valid(),
                "\t point border_enumerator::element()"
                << "\n\t This function can't be called unless the element is valid."
                << "\n\t this: " << this
                );

            return p;
        }

    private:

        point p;
        rectangle rect;
        rectangle inner_rect;  // the non-border regions of rect
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BORDER_EnUMERATOR_H_

