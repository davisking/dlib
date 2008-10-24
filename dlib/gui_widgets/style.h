// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net), and Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_WIDGETs_STYLE_
#define DLIB_WIDGETs_STYLE_

#include "../algs.h"
#include "style_abstract.h"
#include "../gui_core.h"
#include "canvas_drawing.h"
#include <string>
#include <sstream>
#include "../unicode.h"
#include "../array2d.h"
#include "../pixel.h"
#include "fonts.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // button styles  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class button_style
    {
    public:

        button_style()
        {
        }

        virtual ~button_style() 
        {}

        virtual bool redraw_on_mouse_over (
        ) const { return false; }

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const = 0;

        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const = 0;
    };

// ----------------------------------------------------------------------------------------

    class button_style_default : public button_style
    {
    public:
        button_style_default () : padding(4), name_width(0) {}

        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const;

    private:

        // this is the minimum amount of padding that can separate the name from the
        // edge of the button
        const unsigned long padding;
        // this is the width of the name string
        mutable unsigned long name_width;

    };

// ----------------------------------------------------------------------------------------

    class button_style_toolbar1 : public button_style
    {
    public:
        button_style_toolbar1 () : padding(4), name_width(0) {}

        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual bool redraw_on_mouse_over (
        ) const { return true; }

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const;

    private:

        // this is the minimum amount of padding that can separate the name from the
        // edge of the button
        const unsigned long padding;
        // this is the width of the name string
        mutable unsigned long name_width;

    };

// ----------------------------------------------------------------------------------------

    class button_style_toolbar_icon1 : public button_style
    {
    public:
        template <typename image_type>
        button_style_toolbar_icon1 (const image_type& img_, unsigned long pad = 6) : padding(pad) 
        { 
            assign_image(img_mouseover,img_); 
            make_images();  
        }

        button_style_toolbar_icon1( const button_style_toolbar_icon1& item): padding(item.padding) 
        {
            assign_image(img_mouseover, item.img_mouseover);
            assign_image(img_normal, item.img_normal);
            assign_image(img_disabled, item.img_disabled);
        }

        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual bool redraw_on_mouse_over (
        ) const { return true; }

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const;

    private:

        void make_images (
        )
        {
            // make the disabled image grayscale and make both non-mouseover images have weaker alpha channels
            img_disabled.set_size(img_mouseover.nr(), img_mouseover.nc());
            img_normal.set_size(img_mouseover.nr(), img_mouseover.nc());

            for (long r = 0; r < img_mouseover.nr(); ++r)
            {
                for (long c = 0; c < img_mouseover.nc(); ++c)
                {
                    rgb_alpha_pixel p = img_mouseover[r][c];
                    long avg = p.red;
                    avg += p.green;
                    avg += p.blue;
                    avg /= 3;

                    if (p.alpha > 40)
                        p.alpha -= 40;
                    else
                        p.alpha = 0;

                    img_normal[r][c] = p;

                    if (p.alpha > 80)
                        p.alpha -= 80;
                    else
                        p.alpha = 0;

                    p.red = avg;
                    p.green = avg;
                    p.blue = avg;
                    img_disabled[r][c] = p;
                }
            }
        }

        array2d<rgb_alpha_pixel>::kernel_1a img_mouseover;
        array2d<rgb_alpha_pixel>::kernel_1a img_normal;
        array2d<rgb_alpha_pixel>::kernel_1a img_disabled;

        // this is the minimum amount of padding that can separate the name from the
        // edge of the button
        const unsigned long padding;

    };

// ----------------------------------------------------------------------------------------

    class button_style_left_arrow : public button_style
    {
    public:
        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const { return rectangle(); }
    };

// ----------------------------------------------------------------------------------------

    class button_style_right_arrow : public button_style
    {
    public:
        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const { return rectangle(); }
    };

// ----------------------------------------------------------------------------------------

    class button_style_up_arrow : public button_style
    {
    public:
        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const { return rectangle(); }
    };

// ----------------------------------------------------------------------------------------

    class button_style_down_arrow : public button_style
    {
    public:
        virtual void draw_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const { return rectangle(); }
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // toggle button styles  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class toggle_button_style
    {
    public:

        toggle_button_style()
        {
        }

        virtual ~toggle_button_style() 
        {}

        virtual bool redraw_on_mouse_over (
        ) const { return false; }

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const = 0;

        virtual void draw_toggle_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed,
            const bool is_checked
        ) const = 0;
    };

// ----------------------------------------------------------------------------------------

    class toggle_button_style_default : public toggle_button_style
    {
    public:
        toggle_button_style_default () : padding(4), name_width(0) {}

        virtual void draw_toggle_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed,
            const bool is_checked
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const;

    private:

        // this is the minimum amount of padding that can separate the name from the
        // edge of the button
        const unsigned long padding;
        // this is the width of the name string
        mutable unsigned long name_width;

    };

// ----------------------------------------------------------------------------------------

    class toggle_button_style_check_box : public toggle_button_style
    {
    public:
        virtual void draw_toggle_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed,
            const bool is_checked
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const;

    };

// ----------------------------------------------------------------------------------------

    class toggle_button_style_radio_button : public toggle_button_style
    {
    public:
        virtual void draw_toggle_button (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const font& mfont,
            const long lastx,
            const long lasty,
            const ustring& name,
            const bool is_depressed,
            const bool is_checked
        ) const;

        virtual rectangle get_min_size (
            const ustring& name,
            const font& mfont 
        ) const;

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // scroll_bar styles  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class scroll_bar_style
    {
    public:

        virtual ~scroll_bar_style() {}

        virtual bool redraw_on_mouse_over_slider (
        ) const { return false; }

        virtual long get_width (
        ) const = 0;

        virtual long get_slider_length (
            long total_length,
            long max_pos
        ) const = 0;

        virtual long get_button_length (
            long total_length,
            long max_pos
        ) const = 0;

        virtual void draw_scroll_bar_background (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const long lastx,
            const long lasty,
            const bool is_depressed
        ) const = 0;

        virtual void draw_scroll_bar_slider (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const long lastx,
            const long lasty,
            const bool is_being_dragged
        ) const = 0;

    };

// ----------------------------------------------------------------------------------------

    class scroll_bar_style_default : public scroll_bar_style
    {
    public:
        button_style_up_arrow get_up_button_style (
        ) const { return button_style_up_arrow(); }

        button_style_down_arrow get_down_button_style (
        ) const { return button_style_down_arrow(); }

        button_style_left_arrow get_left_button_style (
        ) const { return button_style_left_arrow(); }

        button_style_right_arrow get_right_button_style (
        ) const { return button_style_right_arrow(); }

        virtual long get_width (
        ) const  { return 16; }

        virtual long get_slider_length (
            long total_length,
            long max_pos
        ) const;

        virtual long get_button_length (
            long total_length,
            long max_pos
        ) const;

        virtual void draw_scroll_bar_background (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const long lastx,
            const long lasty,
            const bool is_depressed
        ) const;

        virtual void draw_scroll_bar_slider (
            const canvas& c,
            const rectangle& rect,
            const bool hidden,
            const bool enabled,
            const long lastx,
            const long lasty,
            const bool is_being_dragged
        ) const;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "style.cpp"
#endif

#endif // DLIB_WIDGETs_STYLE_


