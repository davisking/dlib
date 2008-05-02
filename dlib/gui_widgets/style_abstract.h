// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net), and Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_WIDGETs_STYLE_ABSTRACT_
#ifdef DLIB_WIDGETs_STYLE_ABSTRACT_

#include "../algs.h"
#include "../gui_core.h"
#include "widgets_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // button styles  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class button_style
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is an abstract class that defines the interface a
                button style object must implement.

                Note that derived classes must be copyable via
                their copy constructors.
        !*/

    public:

        virtual ~button_style() {}

        virtual bool redraw_on_mouse_over (
        ) const { return false; }
        /*!
            ensures
                - if (this style draws buttons differently when a mouse is over them) then
                    - returns true
                - else
                    - returns false
        !*/

        virtual rectangle get_min_size (
            const std::string& name,
            const font& mfont 
        ) const = 0;
        /*!
            requires
                - the mutex drawable::m is locked
            ensures
                - returns a rectangle that represents the minimum size of the button
                  given the name and font.
        !*/

        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const std::string& name,
            const bool is_depressed
        ) const = 0;
        /*!
            requires
                - the mutex drawable::m is locked
                - c == the canvas to draw on
                - rect, hidden, enabled, mfont, lastx, and lasty are the variables
                  defined in the protected section of the drawable class.
                - name == the name of the button to be drawn
                - is_depressed == true if the button is to be drawn in a depressed state
            ensures
                - draws the button on the canvas c at the location given by rect.  
        !*/
    };

// ----------------------------------------------------------------------------------------

    class button_style_default : public button_style
    {
        /*!
            This is the default style for button objects.  It will cause
            a button to appear as the simple MS Windows 2000 button style.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class button_style_toolbar1 : public button_style
    {
        /*!
            This draws a simple toolbar style button that displays its name in the
            middle of itself.  When the mouse moves over it it will light up.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class button_style_toolbar_icon1 : public button_style
    {
        /*!
            This draws a simple toolbar style button that displays an image in the 
            middle of itself.  When the mouse moves over it it will light up.
        !*/
        template <typename image_type>
        button_style_toolbar_icon1 (
            const image_type& img, 
            unsigned long border_size = 6
        );
        /*!
            requires
                - image_type == an implementation of array2d/array2d_kernel_abstract.h
                - pixel_traits<typename image_type::type> is defined
            ensures
                - displays image img in the middle of the button
                - the distance between the edge of the button and the image
                  will be border_size pixels
        !*/
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // toggle button styles  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class toggle_button_style
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is an abstract class that defines the interface a
                toggle button style object must implement.

                Note that derived classes must be copyable via
                their copy constructors.
        !*/

    public:

        virtual ~toggle_button_style() {}

        virtual bool redraw_on_mouse_over (
        ) const { return false; }
        /*!
            ensures
                - if (this style draws buttons differently when a mouse is over them) then
                    - returns true
                - else
                    - returns false
        !*/

        virtual rectangle get_min_size (
            const std::string& name,
            const font& mfont 
        ) const = 0;
        /*!
            requires
                - the mutex drawable::m is locked
            ensures
                - returns a rectangle that represents the minimum size of the button
                  given the name and font.
        !*/

        virtual void draw_toggle_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const std::string& name,
            const bool is_depressed,
            const bool is_checked 
        ) const = 0;
        /*!
            requires
                - the mutex drawable::m is locked
                - c == the canvas to draw on
                - rect, hidden, enabled, mfont, lastx, and lasty are the variables
                  defined in the protected section of the drawable class.
                - name == the name of the button to be drawn
                - is_depressed == true if the button is to be drawn in a depressed state
                - is_checked == true if the toggle_button is in the checked state 
            ensures
                - draws the button on the canvas c at the location given by rect.  
        !*/
    };

// ----------------------------------------------------------------------------------------

    class toggle_button_style_default : public toggle_button_style
    {
        /*!
            This is the default style for toggle_button objects.  It will cause
            a button to appear as the simple MS Windows 2000 button style.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class toggle_button_style_check_box : public toggle_button_style
    {
        /*!
            This draws a simple check box style toggle button that displays its 
            name to the right of a check box. 
        !*/
    };

// ----------------------------------------------------------------------------------------

    class toggle_button_style_radio_button : public toggle_button_style
    {
        /*!
            This draws a simple radio button style toggle button that displays its 
            name to the right of a circular radio button. 
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WIDGETs_STYLE_ABSTRACT_



