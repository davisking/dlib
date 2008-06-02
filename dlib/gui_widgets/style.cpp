// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net), and Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_WIDGETs_STYLE_CPP_
#define DLIB_WIDGETs_STYLE_CPP_

#include "style.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // button style stuff 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void button_style_default::draw_button (
        const canvas& c,
        const rectangle& rect,
        const bool hidden,
        const bool enabled,
        const font& mfont,
        const long lastx,
        const long lasty,
        const std::string& name,
        const bool is_depressed
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        fill_rect(c,rect,rgb_pixel(212,208,200));

        unsigned char red, green, blue;
        if (enabled)
        {
            red = 0;
            green = 0;
            blue = 0;
        }
        else
        {
            red = 128;
            green = 128;
            blue = 128;
        }

        // compute the name length if it hasn't already been computed
        if (name_width == 0)
        {
            unsigned long height;
            mfont.compute_size(name,name_width,height);
        }

        // figure out where the name string should appear
        rectangle name_rect;
        const unsigned long width = name_width;
        const unsigned long height = mfont.height();
        name_rect.set_left((rect.right() + rect.left() - width)/2);
        name_rect.set_top((rect.bottom() + rect.top() - height)/2 + 1);
        name_rect.set_right(name_rect.left()+width-1);
        name_rect.set_bottom(name_rect.top()+height);


        if (is_depressed)
        {
            name_rect.set_left(name_rect.left()+1);
            name_rect.set_right(name_rect.right()+1);
            name_rect.set_top(name_rect.top()+1);
            name_rect.set_bottom(name_rect.bottom()+1);

            mfont.draw_string(c,name_rect,name,rgb_pixel(red,green,blue));

            draw_button_down(c,rect); 
        }
        else
        {
            mfont.draw_string(c,name_rect,name,rgb_pixel(red,green,blue));

            // now draw the edge of the button
            draw_button_up(c,rect);
        }
    }

// ----------------------------------------------------------------------------------------

    rectangle button_style_default::
    get_min_size (
        const std::string& name,
        const font& mfont 
    ) const 
    {

        unsigned long width; 
        unsigned long height;
        mfont.compute_size(name,width,height);
        name_width = width;

        return rectangle(width+2*padding, height+2*padding);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void button_style_toolbar1::draw_button (
        const canvas& c,
        const rectangle& rect,
        const bool hidden,
        const bool enabled,
        const font& mfont,
        const long lastx,
        const long lasty,
        const std::string& name,
        const bool is_depressed
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        const long radius = 7;

        unsigned char red, green, blue;
        if (enabled)
        {
            red = 0;
            green = 0;
            blue = 0;

            long d = 0;
            if (rect.contains(lastx,lasty))
                d = -40; 

            if (is_depressed)
                d = 50;

            fill_gradient_rounded(c,rect,radius,rgb_alpha_pixel(180-d,180-d,200-d,150), 
                                  rgb_alpha_pixel(130-d,130-d,150-d,100));
            draw_rounded_rectangle(c,rect,radius, rgb_alpha_pixel(80,80,100,190));
        }
        else
        {
            red = 128;
            green = 128;
            blue = 128;
            draw_rounded_rectangle(c,rect,radius, rgb_alpha_pixel(red,green,blue,210));
        }


        // compute the name length if it hasn't already been computed
        if (name_width == 0)
        {
            unsigned long height;
            mfont.compute_size(name,name_width,height);
        }

        // figure out where the name string should appear
        rectangle name_rect;
        const unsigned long width = name_width;
        const unsigned long height = mfont.height();
        name_rect.set_left((rect.right() + rect.left() - width)/2);
        name_rect.set_top((rect.bottom() + rect.top() - height)/2 + 1);
        name_rect.set_right(name_rect.left()+width-1);
        name_rect.set_bottom(name_rect.top()+height);


        if (is_depressed)
        {
            name_rect.set_left(name_rect.left()+1);
            name_rect.set_right(name_rect.right()+1);
            name_rect.set_top(name_rect.top()+1);
            name_rect.set_bottom(name_rect.bottom()+1);

            mfont.draw_string(c,name_rect,name,rgb_pixel(red,green,blue));

        }
        else
        {
            mfont.draw_string(c,name_rect,name,rgb_pixel(red,green,blue));
        }
    }

// ----------------------------------------------------------------------------------------

    rectangle button_style_toolbar1::
    get_min_size (
        const std::string& name,
        const font& mfont 
    ) const 
    {

        unsigned long width; 
        unsigned long height;
        mfont.compute_size(name,width,height);
        name_width = width;

        return rectangle(width+2*padding, height+2*padding);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void button_style_toolbar_icon1::draw_button (
        const canvas& c,
        const rectangle& rect,
        const bool hidden,
        const bool enabled,
        const font& mfont,
        const long lastx,
        const long lasty,
        const std::string& name,
        const bool is_depressed
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        const long radius = padding;

        if (enabled)
        {
            if (rect.contains(lastx,lasty))
            {
                if (is_depressed)
                {
                    fill_gradient_rounded(c,rect,radius,rgb_alpha_pixel(100,100,200,150), 
                                                        rgb_alpha_pixel(50,50,100,100));
                    draw_rounded_rectangle(c,rect,radius, rgb_alpha_pixel(150,150,30,200));
                }
                else
                {
                    fill_gradient_rounded(c,rect,radius,rgb_alpha_pixel(150,150,250,130), 
                                                        rgb_alpha_pixel(100,100,150,90));
                    draw_rounded_rectangle(c,rect,radius, rgb_alpha_pixel(150,150,30,200));
                }
            }

            if (is_depressed)
            {
                rectangle img_rect(translate_rect(centered_rect(rect,img_mouseover.nc(),img_mouseover.nr()),1,1));
                point p(img_rect.left(),img_rect.top());
                draw_image(c,p,img_mouseover);
            }
            else
            {
                rectangle img_rect(centered_rect(rect,img_normal.nc(),img_normal.nr()));
                point p(img_rect.left(),img_rect.top());
                if (rect.contains(lastx,lasty))
                    draw_image(c,p,img_mouseover);
                else
                    draw_image(c,p,img_normal);
            }

        }
        else
        {
            rectangle img_rect(centered_rect(rect,img_normal.nc(),img_normal.nr()));
            point p(img_rect.left(),img_rect.top());
            draw_image(c,p,img_disabled);
        }
    }

// ----------------------------------------------------------------------------------------

    rectangle button_style_toolbar_icon1::
    get_min_size (
        const std::string& name,
        const font& mfont 
    ) const 
    {
        return rectangle(img_normal.nc()+2*padding, img_normal.nr()+2*padding);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // toggle button style stuff 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void toggle_button_style_default::draw_toggle_button (
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
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        fill_rect(c,rect,rgb_pixel(212,208,200));

        unsigned char red, green, blue;
        if (enabled)
        {
            red = 0;
            green = 0;
            blue = 0;
        }
        else
        {
            red = 128;
            green = 128;
            blue = 128;
        }

        // compute the name length if it hasn't already been computed
        if (name_width == 0)
        {
            unsigned long height;
            mfont.compute_size(name,name_width,height);
        }

        // figure out where the name string should appear
        rectangle name_rect;
        const unsigned long width = name_width;
        const unsigned long height = mfont.height();
        name_rect.set_left((rect.right() + rect.left() - width)/2);
        name_rect.set_top((rect.bottom() + rect.top() - height)/2 + 1);
        name_rect.set_right(name_rect.left()+width-1);
        name_rect.set_bottom(name_rect.top()+height);

        long d = 0;
        if (is_checked)
            d = 1;

        if (is_depressed)
            d = 2;

        name_rect.set_left(name_rect.left()+d);
        name_rect.set_right(name_rect.right()+d);
        name_rect.set_top(name_rect.top()+d);
        name_rect.set_bottom(name_rect.bottom()+d);

        mfont.draw_string(c,name_rect,name,rgb_pixel(red,green,blue));

        // now draw the edge of the button
        if (is_checked || is_depressed)
            draw_button_down(c,rect);
        else
            draw_button_up(c,rect);
    }

// ----------------------------------------------------------------------------------------

    rectangle toggle_button_style_default::
    get_min_size (
        const std::string& name,
        const font& mfont 
    ) const 
    {

        unsigned long width; 
        unsigned long height;
        mfont.compute_size(name,width,height);
        name_width = width;

        return rectangle(width+2*padding, height+2*padding);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void toggle_button_style_check_box::draw_toggle_button (
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
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;


        rgb_pixel color;
        if (enabled)
        {
            color.red = 0;
            color.green = 0;
            color.blue = 0;
        }
        else
        {
            color.red = 128;
            color.green = 128;
            color.blue = 128;
        }


        // figure out where the name string should appear
        rectangle name_rect, box_rect;
        unsigned long padding = 0;
        if (mfont.height() < 13)
            padding = (rect.height() - mfont.height())/2;

        name_rect = rect;
        name_rect.set_left(rect.left() + 17-1);
        name_rect.set_top(rect.top() + padding);
        name_rect.set_bottom(rect.bottom() - padding);
            
        box_rect = rect;
        box_rect.set_right(rect.left() + 12);
        box_rect.set_bottom(rect.top() + 12);

        mfont.draw_string(c,name_rect,name,color);

        if (enabled && is_depressed == false)
            fill_rect(c, box_rect,rgb_pixel(255,255,255));
        else
            fill_rect(c, box_rect,rgb_pixel(212,208,200));

        draw_sunken_rectangle(c, box_rect);


        if (is_checked)
        {
            const long x = box_rect.left();
            const long y = box_rect.top();
            draw_line(c,point(3+x,5+y),point(6+x,8+y),color);
            draw_line(c,point(3+x,6+y),point(5+x,8+y),color);
            draw_line(c,point(3+x,7+y),point(5+x,9+y),color);
            draw_line(c,point(6+x,6+y),point(9+x,3+y),color);
            draw_line(c,point(6+x,7+y),point(9+x,4+y),color);
            draw_line(c,point(6+x,8+y),point(9+x,5+y),color);
        }
    }

// ----------------------------------------------------------------------------------------

    rectangle toggle_button_style_check_box::
    get_min_size (
        const std::string& name,
        const font& mfont 
    ) const 
    {
        unsigned long width;
        unsigned long height;
        mfont.compute_size(name,width,height);

        if (height < 13)
            height = 13;

        return rectangle(width + 17 -1, height -1);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void toggle_button_style_radio_button::draw_toggle_button (
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
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;
        

        rgb_pixel color;

        // figure out where the name string should appear
        rectangle name_rect, box_rect;
        unsigned long padding = 0;
        if (mfont.height() < 13)
            padding = (rect.height() - mfont.height())/2;

        name_rect = rect;
        name_rect.set_left(rect.left() + 17-1);
        name_rect.set_top(rect.top() + padding);
        name_rect.set_bottom(rect.bottom() - padding);
            
        box_rect = rect;
        box_rect.set_right(rect.left() + 12);
        box_rect.set_bottom(rect.top() + 12);

        
        const long x = box_rect.left();
        const long y = box_rect.top();

        if (enabled && is_depressed == false)
            draw_solid_circle(c,point(rect.left()+5,rect.top()+5),4.5,rgb_pixel(255,255,255));
        else
            draw_solid_circle(c,point(rect.left()+5,rect.top()+5),4.5,rgb_pixel(212,208,200));


        color = rgb_pixel(128,128,128);
        draw_line(c,point(0+x,4+y),point(0+x,7+y),color);
        draw_line(c,point(1+x,2+y),point(1+x,9+y),color);
        draw_line(c,point(2+x,1+y),point(9+x,1+y),color);
        draw_line(c,point(4+x,0+y),point(7+x,0+y),color);

        color = rgb_pixel(255,255,255);
        draw_line(c,point(4+x,11+y),point(7+x,11+y),color);
        draw_line(c,point(2+x,10+y),point(9+x,10+y),color);
        draw_line(c,point(10+x,2+y),point(10+x,9+y),color);
        draw_line(c,point(11+x,4+y),point(11+x,7+y),color);

        color = rgb_pixel(64,64,64);
        draw_line(c,point(1+x,4+y),point(1+x,7+y),color);
        draw_line(c,point(4+x,1+y),point(7+x,1+y),color);
        draw_pixel(c,point(2+x,3+y),color);
        draw_pixel(c,point(3+x,2+y),color);
        draw_pixel(c,point(2+x,2+y),color);
        draw_pixel(c,point(2+x,8+y),color);
        draw_pixel(c,point(8+x,2+y),color);
        draw_pixel(c,point(9+x,2+y),color);

        color = rgb_pixel(212,208,200);
        draw_line(c,point(4+x,10+y),point(7+x,10+y),color);
        draw_line(c,point(10+x,4+y),point(10+x,7+y),color);
        draw_pixel(c,point(3+x,9+y),color);
        draw_pixel(c,point(9+x,3+y),color);

        if (enabled)
        {
            color.red = 0;
            color.green = 0;
            color.blue = 0;
        }
        else
        {
            color.red = 128;
            color.green = 128;
            color.blue = 128;
        }

        mfont.draw_string(c,name_rect,name,color);

        if (is_checked)
        {
            draw_line(c,point(5+x,4+y),point(6+x,4+y),color);
            draw_line(c,point(4+x,5+y),point(7+x,5+y),color);
            draw_line(c,point(4+x,6+y),point(7+x,6+y),color);
            draw_line(c,point(5+x,7+y),point(6+x,7+y),color);
        }

    }

// ----------------------------------------------------------------------------------------

    rectangle toggle_button_style_radio_button::
    get_min_size (
        const std::string& name,
        const font& mfont 
    ) const 
    {
        unsigned long width;
        unsigned long height;
        mfont.compute_size(name,width,height);

        if (height < 13)
            height = 13;

        return rectangle(width + 17 -1, height -1);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WIDGETs_STYLE_CPP_

