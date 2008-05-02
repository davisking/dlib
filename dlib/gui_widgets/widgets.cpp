// Copyright (C) 2005  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_WIDGETs_CPP_
#define DLIB_WIDGETs_CPP_

#include "widgets.h"
#include <algorithm>

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // button object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void button::
    set_size (
        unsigned long width,
        unsigned long height
    )
    {
        auto_mutex M(m);
        rectangle min_rect = style->get_min_size(name_,*mfont);
        // only change the size if it isn't going to be too small to fit the name
        if (height >= min_rect.height() &&
            width >= min_rect.width())
        {
            rectangle old(rect);
            rect = resize_rect(rect,width,height);
            parent.invalidate_rectangle(rect+old);
            btn_tooltip.set_size(width,height);
        }
    }

// ----------------------------------------------------------------------------------------

    void button::
    show (
    )
    {
        button_action::show();
        btn_tooltip.show();
    }

// ----------------------------------------------------------------------------------------

    void button::
    hide (
    )
    {
        button_action::hide();
        btn_tooltip.hide();
    }

// ----------------------------------------------------------------------------------------

    void button::
    enable (
    )
    {
        button_action::enable();
        btn_tooltip.enable();
    }

// ----------------------------------------------------------------------------------------

    void button::
    disable (
    )
    {
        button_action::disable();
        btn_tooltip.disable();
    }

// ----------------------------------------------------------------------------------------

    void button::
    set_tooltip_text (
        const std::string& text
    )
    {
        btn_tooltip.set_text(text);
    }

// ----------------------------------------------------------------------------------------

    const std::string button::
    tooltip_text (
    ) const
    {
        return btn_tooltip.text();
    }

// ----------------------------------------------------------------------------------------

    void button::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;
        set_name(name_);
    }

// ----------------------------------------------------------------------------------------

    void button::
    set_pos (
        long x,
        long y
    )
    {
        auto_mutex M(m);
        button_action::set_pos(x,y);
        btn_tooltip.set_pos(x,y);
    }

// ----------------------------------------------------------------------------------------

    void button::
    set_name (
        const std::string& name
    )
    {
        auto_mutex M(m);
        name_ = name;
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        name_[0] = name_[0];

        rectangle old(rect);
        rect = move_rect(style->get_min_size(name,*mfont),rect.left(),rect.top());
        btn_tooltip.set_size(rect.width(),rect.height());
        
        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    const std::string button::
    name (
    ) const
    {
        auto_mutex M(m);
        std::string temp = name_;
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        temp[0] = name_[0];
        return temp;
    }

// ----------------------------------------------------------------------------------------

    void button::
    on_button_up (
        bool mouse_over
    )
    {
        if (mouse_over)                
        {
            // this is a valid button click
            if (event_handler.is_set())
                event_handler();
            else if (event_handler_self.is_set())
                event_handler_self(*this);
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // toggle_button object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_size (
        unsigned long width,
        unsigned long height
    )
    {
        auto_mutex M(m);
        rectangle min_rect = style->get_min_size(name_,*mfont);
        // only change the size if it isn't going to be too small to fit the name
        if (height >= min_rect.height() &&
            width >= min_rect.width())
        {
            rectangle old(rect);
            rect = resize_rect(rect,width,height);
            parent.invalidate_rectangle(rect+old);
            btn_tooltip.set_size(width,height);
        }
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_checked (
    )
    {
        auto_mutex M(m);
        checked = true;
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_unchecked (
    )
    {
        auto_mutex M(m);
        checked = false;
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    bool toggle_button::
    is_checked (
    ) const
    {
        auto_mutex M(m);
        return checked;
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    show (
    )
    {
        button_action::show();
        btn_tooltip.show();
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    hide (
    )
    {
        button_action::hide();
        btn_tooltip.hide();
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    enable (
    )
    {
        button_action::enable();
        btn_tooltip.enable();
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    disable (
    )
    {
        button_action::disable();
        btn_tooltip.disable();
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_tooltip_text (
        const std::string& text
    )
    {
        btn_tooltip.set_text(text);
    }

// ----------------------------------------------------------------------------------------

    const std::string toggle_button::
    tooltip_text (
    ) const
    {
        return btn_tooltip.text();
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;
        set_name(name_);
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_pos (
        long x,
        long y
    )
    {
        auto_mutex M(m);
        button_action::set_pos(x,y);
        btn_tooltip.set_pos(x,y);
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    set_name (
        const std::string& name
    )
    {
        auto_mutex M(m);
        name_ = name;
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        name_[0] = name_[0];

        rectangle old(rect);
        rect = move_rect(style->get_min_size(name,*mfont),rect.left(),rect.top());
        btn_tooltip.set_size(rect.width(),rect.height());
        
        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    const std::string toggle_button::
    name (
    ) const
    {
        auto_mutex M(m);
        std::string temp = name_;
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        temp[0] = name_[0];
        return temp;
    }

// ----------------------------------------------------------------------------------------

    void toggle_button::
    on_button_up (
        bool mouse_over
    )
    {
        if (mouse_over)                
        {
            checked = !checked;
            // this is a valid toggle_button click
            if (event_handler.is_set())
                event_handler();
            else if (event_handler_self.is_set())
                event_handler_self(*this);
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // label object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void label::
    draw (
        const canvas& c
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty() || text_.size() == 0)
            return;

        using namespace std;
        unsigned char r = text_color_.red;
        unsigned char g = text_color_.green;
        unsigned char b = text_color_.blue;
        if (!enabled)
        {
            r = 128;
            g = 128;
            b = 128;
        }

        rectangle text_rect(rect);

        string::size_type first, last;
        first = 0;
        last = text_.find_first_of('\n');
        mfont->draw_string(c,text_rect,text_,rgb_pixel(r,g,b),first,last);

        while (last != string::npos)
        {
            first = last+1;
            last = text_.find_first_of('\n',first);
            text_rect.set_top(text_rect.top()+mfont->height());
            mfont->draw_string(c,text_rect,text_,rgb_pixel(r,g,b),first,last);
        }
    }

// ----------------------------------------------------------------------------------------

    void label::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;
        set_text(text_);
    }

// ----------------------------------------------------------------------------------------

    void label::
    set_text (
        const std::string& text
    )
    {
        using namespace std;
        auto_mutex M(m);
        text_ = text;
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        text_[0] = text[0];

        rectangle old(rect);

        unsigned long width; 
        unsigned long height;
        mfont->compute_size(text,width,height);

        rect.set_right(rect.left() + width - 1); 
        rect.set_bottom(rect.top() + height - 1);

        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    const std::string label::
    text (
    ) const
    {
        auto_mutex M(m);
        std::string temp = text_;
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        temp[0] = text_[0];
        return temp;
    }

// ----------------------------------------------------------------------------------------

    void label::
    set_text_color (
        const rgb_pixel color
    )
    {
        m.lock();
        text_color_ = color;
        parent.invalidate_rectangle(rect);
        m.unlock();
    }

// ----------------------------------------------------------------------------------------

    const rgb_pixel label::
    text_color (
    ) const
    {
        auto_mutex M(m);
        return text_color_;
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // text_field object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    rectangle text_field::
    get_text_rect (
    ) const
    {
        // figure out where the text string should appear        
        unsigned long vertical_pad = (rect.height() - mfont->height())/2+1;

        rectangle text_rect;
        text_rect.set_left(rect.left()+(mfont->height()-mfont->ascender()));
        text_rect.set_top(rect.top()+vertical_pad);
        text_rect.set_right(rect.right()-(mfont->height()-mfont->ascender()));
        text_rect.set_bottom(text_rect.top()+mfont->height()-1);
        return text_rect;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    disable (
    )
    {
        auto_mutex M(m);
        drawable::disable();
        t.stop();
        has_focus = false;
        cursor_visible = false;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    hide (
    )
    {
        auto_mutex M(m);
        drawable::hide();
        t.stop();
        has_focus = false;
        cursor_visible = false;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;
        // adjust the height of this text field so that it is appropriate for the current
        // font size
        rect.set_bottom(rect.top() + mfont->height()+ (mfont->height()-mfont->ascender())*2);
        set_text(text_);
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    draw (
        const canvas& c
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;
       
        if (enabled)
        {
            // first fill our area with the bg_color_ 
            fill_rect(c, area,bg_color_);
        }
        else
        {
            // first fill our area with gray 
            fill_rect(c, area,rgb_pixel(212,208,200));
        }

        const rectangle text_rect = get_text_rect();

        if (enabled)
            mfont->draw_string(c,text_rect,text_,text_color_,text_pos);
        else
            mfont->draw_string(c,text_rect,text_,rgb_pixel(128,128,128),text_pos);

        // now draw the edge of the text_field
        draw_sunken_rectangle(c, rect);

        if (highlight_start <= highlight_end && enabled)
        {
            rectangle highlight_rect = text_rect;
            unsigned long left_pad = 0, right_pad = mfont->left_overflow();
            
            long i;
            for (i = text_pos; i <= highlight_end; ++i)
            {
                if (i == highlight_start)
                    left_pad = right_pad;

                right_pad += (*mfont)[text_[i]].width();
            }
            
            highlight_rect.set_left(text_rect.left()+left_pad);
            highlight_rect.set_right(text_rect.left()+right_pad);

            // highlight the highlight_rect area
            highlight_rect = highlight_rect.intersect(c);
            for (long row = highlight_rect.top(); row <= highlight_rect.bottom(); ++row)
            {
                for (long col = highlight_rect.left(); col <= highlight_rect.right(); ++col)
                {
                    canvas::pixel& pixel = c[row-c.top()][col-c.left()];
                    if (pixel.red == 255 && pixel.green == 255 && pixel.blue == 255)
                    {
                        // this is a background (and white) pixel so set it to a dark 
                        // blueish color.
                        pixel.red = 10;
                        pixel.green = 36;
                        pixel.blue = 106;
                    }
                    else
                    {
                        // this should be a pixel that is part of a letter so set it to white
                        pixel.red = 255;
                        pixel.green = 255;
                        pixel.blue = 255;
                    }
                }
            }
        }

        // now draw the cursor if we need to
        if (cursor_visible && has_focus && enabled && !hidden)
        {
            const unsigned long top = rect.top()+3;
            const unsigned long bottom = rect.bottom()-3;
            draw_line(c, point(rect.left()+cursor_x,top),point(rect.left()+cursor_x,bottom));
        }
        
        
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    set_text (
        const std::string& text
    )
    {
        DLIB_ASSERT ( text.find_first_of('\n') == std::string::npos ,
                "\tvoid text_field::set_text()"
                << "\n\ttext:  " << text );
        auto_mutex M(m);
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        text_ = text.c_str();
                
        move_cursor(0);

        highlight_start = 0;
        highlight_end = -1;
        
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    const std::string text_field::
    text (
    ) const
    {
        auto_mutex M(m);
        // do this to get rid of any reference counting that may be present in 
        // the std::string implementation.
        std::string temp = text_.c_str();
        return temp;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    set_width (
        unsigned long width
    )
    {        
        if (width < 10)
            return;

        m.lock();        
        rectangle old(rect);

        rect.set_right(rect.left() + width - 1); 

        parent.invalidate_rectangle(rect+old);
        m.unlock();
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    set_background_color (
        const rgb_pixel color
    )
    {
        auto_mutex M(m);
        bg_color_ = color;
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    const rgb_pixel text_field::
    background_color (
    ) const
    {
        auto_mutex M(m);
        return bg_color_;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    set_text_color (
        const rgb_pixel color
    )
    {
        auto_mutex M(m);
        text_color_ = color;
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    const rgb_pixel text_field::
    text_color (
    ) const
    {
        auto_mutex M(m);
        return text_color_;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    on_mouse_move (
        unsigned long state,
        long x,
        long y
    )
    {
        if (!enabled || hidden || !has_focus)
        {
            return;
        }

        if (state & base_window::LEFT)
        {
            if (highlight_start <= highlight_end)
            {
                if (highlight_start == cursor_pos)
                    shift_pos = highlight_end + 1;
                else
                    shift_pos = highlight_start;
            }

            unsigned long new_pos = mfont->compute_cursor_pos(get_text_rect(),text_,x,y,text_pos);
            if (static_cast<long>(new_pos) != cursor_pos)
            {
                move_cursor(new_pos);
                parent.invalidate_rectangle(rect);
            }
        }
        else if (shift_pos != -1)
        {
            shift_pos = -1;
        }

    }

// ----------------------------------------------------------------------------------------

    void text_field::
    on_mouse_up (
        unsigned long btn,
        unsigned long,
        long ,
        long 
    )
    {
        if (!enabled || hidden)
            return;

        if (btn == base_window::LEFT)
            shift_pos = -1;
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    on_mouse_down (
        unsigned long btn,
        unsigned long state,
        long x,
        long y,
        bool double_clicked 
    )
    {
        using namespace std;
        if (!enabled || hidden || btn != (unsigned long)base_window::LEFT)
            return;

        if (rect.contains(x,y))
        {
            has_focus = true;
            cursor_visible = true;
            parent.invalidate_rectangle(rect);
            t.start();

            if (double_clicked)
            {
                // highlight the double clicked word
                string::size_type first, last;
                first = text_.substr(0,cursor_pos).find_last_of(" \t\n");
                last = text_.find_first_of(" \t\n",cursor_pos);
                long f = static_cast<long>(first);
                long l = static_cast<long>(last);
                if (first == string::npos)
                    f = -1;
                if (last == string::npos)
                    l = static_cast<long>(text_.size());

                ++f;
                --l;

                move_cursor(l+1);
                highlight_start = f;
                highlight_end = l;
            }
            else
            {
                if (state & base_window::SHIFT)
                {
                    if (highlight_start <= highlight_end)
                    {
                        if (highlight_start == cursor_pos)
                            shift_pos = highlight_end + 1;
                        else
                            shift_pos = highlight_start;
                    }
                    else
                    {
                        shift_pos = cursor_pos;
                    }
                }

                bool at_end = false;
                if (cursor_pos == 0 || cursor_pos == static_cast<long>(text_.size()))
                    at_end = true;
                const long old_pos = cursor_pos;

                unsigned long new_pos = mfont->compute_cursor_pos(get_text_rect(),text_,x,y,text_pos);
                if (static_cast<long>(new_pos) != cursor_pos)
                {
                    move_cursor(new_pos);
                    parent.invalidate_rectangle(rect);
                }
                shift_pos = cursor_pos;

                if (at_end && cursor_pos == old_pos)
                {
                    highlight_start = 0;
                    highlight_end = -1;
                    parent.invalidate_rectangle(rect);
                }
            }

        }
        else if (has_focus)
        {
            t.stop();
            has_focus = false;
            cursor_visible = false;
            shift_pos = -1;
            highlight_start = 0;
            highlight_end = -1;

            parent.invalidate_rectangle(rect);
        }
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    on_keydown (
        unsigned long key,
        bool is_printable,
        unsigned long state
    )
    {
        const bool shift = (state&base_window::KBD_MOD_SHIFT) != 0;
        const bool ctrl = (state&base_window::KBD_MOD_CONTROL) != 0;
        if (has_focus && enabled && !hidden)
        {
            if (shift && is_printable == false)
            {
                if (shift_pos == -1)
                {
                    if (highlight_start <= highlight_end)
                    {
                        if (highlight_start == cursor_pos)
                            shift_pos = highlight_end + 1;
                        else
                            shift_pos = highlight_start;
                    }
                    else
                    {
                        shift_pos = cursor_pos;
                    }
                }
            }
            else
            {
                shift_pos = -1;
            }

            if (key == base_window::KEY_LEFT ||
                key == base_window::KEY_UP)
            {
                if (cursor_pos != 0)
                {
                    unsigned long new_pos;
                    if (ctrl)
                    {
                        // find the first non-whitespace to our left
                        std::string::size_type pos = text_.find_last_not_of(" \t\n",cursor_pos);
                        if (pos != std::string::npos)
                        {
                            pos = text_.find_last_of(" \n\t",pos);
                            if (pos != std::string::npos)
                                new_pos = static_cast<unsigned long>(pos);
                            else
                                new_pos = 0;
                        }
                        else
                        {
                            new_pos = 0;
                        }
                    }
                    else
                    {
                        new_pos = cursor_pos-1;
                    }

                    move_cursor(new_pos);
                }
                else if (shift_pos == -1)
                {
                    highlight_start = 0;
                    highlight_end = -1;
                    parent.invalidate_rectangle(rect);
                }

            }
            else if (key == base_window::KEY_RIGHT ||
                key == base_window::KEY_DOWN)
            {
                if (cursor_pos != static_cast<long>(text_.size()))
                {
                    unsigned long new_pos;
                    if (ctrl)
                    {
                        // find the first non-whitespace to our left
                        std::string::size_type pos = text_.find_first_not_of(" \t\n",cursor_pos);
                        if (pos != std::string::npos)
                        {
                            pos = text_.find_first_of(" \n\t",pos);
                            if (pos != std::string::npos)
                                new_pos = static_cast<unsigned long>(pos+1);
                            else
                                new_pos = static_cast<unsigned long>(text_.size());
                        }
                        else
                        {
                            new_pos = static_cast<unsigned long>(text_.size());
                        }
                    }
                    else
                    {
                        new_pos = cursor_pos+1;
                    }

                    move_cursor(new_pos);
                }
                else if (shift_pos == -1)
                {
                    highlight_start = 0;
                    highlight_end = -1;
                    parent.invalidate_rectangle(rect);
                }
            }
            else if (is_printable)
            {
                if (ctrl)
                {
                    if (key == 'a')
                    {
                        move_cursor(static_cast<long>(text_.size()));
                        highlight_start = 0;
                        highlight_end = static_cast<long>(text_.size()-1);
                        parent.invalidate_rectangle(rect);
                    }
                }
                else if (key != '\n')
                {
                    if (highlight_start <= highlight_end)
                    {
                        text_ = text_.substr(0,highlight_start) + static_cast<char>(key) +
                            text_.substr(highlight_end+1,text_.size()-highlight_end-1);
                        move_cursor(highlight_start+1);
                        highlight_start = 0;
                        highlight_end = -1;
                        parent.invalidate_rectangle(rect);
                    }
                    else
                    {
                        text_ = text_.substr(0,cursor_pos) + static_cast<char>(key) +
                            text_.substr(cursor_pos,text_.size()-cursor_pos);
                        move_cursor(cursor_pos+1);
                    }
                    unsigned long height;
                    mfont->compute_size(text_,text_width,height,text_pos);

                    // send out the text modified event
                    if (text_modified_handler.is_set())
                        text_modified_handler();
                }
            }
            else if (key == base_window::KEY_BACKSPACE)
            {                
                // if something is highlighted then delete that
                if (highlight_start <= highlight_end)
                {
                    text_ = text_.erase(highlight_start,highlight_end-highlight_start+1);
                    move_cursor(highlight_start);
                    highlight_start = 0;
                    highlight_end = -1;

                    // send out the text modified event
                    if (text_modified_handler.is_set())
                        text_modified_handler();
                }
                else if (cursor_pos != 0)
                {
                    text_ = text_.erase(cursor_pos-1,1);
                    move_cursor(cursor_pos-1);

                    // send out the text modified event
                    if (text_modified_handler.is_set())
                        text_modified_handler();
                }
                else
                {
                    // do this just so it repaints itself right
                    move_cursor(cursor_pos);
                }
                unsigned long height;
                mfont->compute_size(text_,text_width,height,text_pos);
                parent.invalidate_rectangle(rect);
            }
            else if (key == base_window::KEY_DELETE)
            {
                // if something is highlighted then delete that
                if (highlight_start <= highlight_end)
                {
                    text_ = text_.erase(highlight_start,highlight_end-highlight_start+1);
                    move_cursor(highlight_start);
                    highlight_start = 0;
                    highlight_end = -1;

                    // send out the text modified event
                    if (text_modified_handler.is_set())
                        text_modified_handler();
                }
                else if (cursor_pos != static_cast<long>(text_.size()))
                {
                    text_ = text_.erase(cursor_pos,1);

                    // send out the text modified event
                    if (text_modified_handler.is_set())
                        text_modified_handler();
                }
                else
                {
                    // do this just so it repaints itself right
                    move_cursor(cursor_pos);
                }
                parent.invalidate_rectangle(rect);

                unsigned long height;
                mfont->compute_size(text_,text_width,height,text_pos);
            }
            else if (key == base_window::KEY_HOME)
            {
                move_cursor(0);
                if (shift_pos == -1)
                {
                    highlight_start = 0;
                    highlight_end = -1;
                    parent.invalidate_rectangle(rect);
                }
            }
            else if (key == base_window::KEY_END)
            {
                move_cursor(static_cast<unsigned long>(text_.size()));
                if (shift_pos == -1)
                {
                    highlight_start = 0;
                    highlight_end = -1;
                    parent.invalidate_rectangle(rect);
                }
            }
            cursor_visible = true;
            recent_movement = true;

        }
    }

// ----------------------------------------------------------------------------------------

    void text_field::
    move_cursor (
        unsigned long pos
    )
    {
        using namespace std;
        const long old_cursor_pos = cursor_pos;

        if (text_pos >= pos)
        {
            // the cursor should go all the way to the left side of the text
            if (pos >= 6)
                text_pos = pos-6;
            else
                text_pos = 0;

            cursor_pos = pos;    
            unsigned long height;
            mfont->compute_size(text_,text_width,height,text_pos);

            unsigned long width;
            unsigned long new_x = (mfont->height()-mfont->ascender());
            if (static_cast<long>(cursor_pos)-1 >= static_cast<long>(text_pos))
            {
                mfont->compute_size(text_,width,height,text_pos,cursor_pos-1);
                if (cursor_pos != 0)
                    new_x += width - mfont->right_overflow();
            }

            cursor_x = new_x;
        }
        else
        {
            unsigned long height;
            unsigned long width;
            mfont->compute_size(text_,width,height,text_pos,pos-1);

            unsigned long new_x = (mfont->height()-mfont->ascender()) + 
                width - mfont->right_overflow();

            // move the text to the left if necessary
            if (new_x + 4 > rect.width())
            {
                while (new_x > rect.width() - rect.width()/5)
                {
                    new_x -= (*mfont)[text_[text_pos]].width();
                    ++text_pos;
                }
            }

            cursor_x = new_x;
            cursor_pos = pos;     
            mfont->compute_size(text_,text_width,height,text_pos);
        }

        if (old_cursor_pos != cursor_pos)
        {
            if (shift_pos != -1)
            {
                highlight_start = std::min(shift_pos,cursor_pos);
                highlight_end = std::max(shift_pos,cursor_pos)-1;
            }
            else
            {
                highlight_start = 0;
                highlight_end = -1;
            }

            recent_movement = true;
            cursor_visible = true;
            parent.invalidate_rectangle(rect);
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//             tabbed_display object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    tabbed_display::
    tabbed_display(  
        drawable_window& w
    ) : 
        drawable(w,MOUSE_CLICK),
        selected_tab_(0),
        left_pad(6),
        right_pad(4),
        top_pad(3),
        bottom_pad(3)
    {
        rect = rectangle(0,0,40,mfont->height()+top_pad+bottom_pad);
        enable_events();
        tabs.set_max_size(1);
        tabs.set_size(1);
    }

// ----------------------------------------------------------------------------------------

    tabbed_display::
    ~tabbed_display(
    )
    {
        disable_events();
        parent.invalidate_rectangle(rect); 
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    set_pos (
        long x,
        long y
    )
    {
        auto_mutex M(m);
        // we have to adjust the positions of all the tab rectangles
        const long xdelta = rect.left() - x;
        const long ydelta = rect.top() - y;
        for (unsigned long i = 0; i < tabs.size(); ++i)
        {
            tabs[i].rect.set_left(tabs[i].rect.left()+xdelta);
            tabs[i].rect.set_right(tabs[i].rect.right()+xdelta);

            tabs[i].rect.set_top(tabs[i].rect.top()+ydelta);
            tabs[i].rect.set_bottom(tabs[i].rect.bottom()+ydelta);


            // adjust the position of the group associated with this tab if it exists
            if (tabs[i].group)
                tabs[i].group->set_pos(x+3, y+mfont->height()+top_pad+bottom_pad+3);
        }
        drawable::set_pos(x,y);
        recompute_tabs();
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    fit_to_contents (
    )
    {
        auto_mutex M(m);
        rectangle new_rect;
        point p(rect.left(),rect.top());
        new_rect += p;

        for (unsigned long i = 0; i < tabs.size(); ++i)
        {
            if (tabs[i].group)
            {
                tabs[i].group->fit_to_contents();
                new_rect += tabs[i].group->get_rect();
            }
        }

        // and give the new rect an additional 4 pixels on the bottom and right sides
        // so that the contents to hit the edge of the tabbed display
        new_rect = resize_rect(new_rect, new_rect.width()+4, new_rect.height()+4);

        parent.invalidate_rectangle(new_rect+rect);
        rect = new_rect;
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    set_size (
        unsigned long width,
        unsigned long height
    )
    {
        auto_mutex M(m);
        rectangle old(rect);
        const long x = rect.left();
        const long y = rect.top();
        rect.set_right(x+width-1);
        rect.set_bottom(y+height-1);

        recompute_tabs();

        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    set_number_of_tabs (
        unsigned long num
    )
    {
        auto_mutex M(m);

        DLIB_ASSERT ( num > 0 ,
                "\tvoid tabbed_display::set_number_of_tabs()"
                << "\n\tnum:  " << num );

        tabs.set_max_size(num);
        tabs.set_size(num);

        selected_tab_ = 0;

        recompute_tabs();
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    unsigned long tabbed_display::
    number_of_tabs (
    ) const
    {
        auto_mutex M(m);
        return tabs.size();
    }

// ----------------------------------------------------------------------------------------

    const std::string& tabbed_display::
    tab_name (
        unsigned long idx
    ) const
    {
        auto_mutex M(m);

        DLIB_ASSERT ( idx < number_of_tabs() ,
                "\tvoid tabbed_display::tab_name()"
                << "\n\tidx:              " << idx 
                << "\n\tnumber_of_tabs(): " << number_of_tabs() );

        return tabs[idx].name;
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    set_tab_name (
        unsigned long idx,
        const std::string& new_name
    )
    {
        auto_mutex M(m);


        DLIB_ASSERT ( idx < number_of_tabs() ,
                "\tvoid tabbed_display::set_tab_name()"
                << "\n\tidx:              " << idx 
                << "\n\tnumber_of_tabs(): " << number_of_tabs() );


        tabs[idx].name = new_name;
        // do this so that there isn't any reference counting going on
        tabs[idx].name[0] = tabs[idx].name[0];
        unsigned long height;
        mfont->compute_size(new_name,tabs[idx].width,height);


        recompute_tabs();

        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    on_mouse_down (
        unsigned long btn,
        unsigned long,
        long x,
        long y,
        bool 
    )
    {
        if (rect.contains(x,y) && btn == base_window::LEFT && enabled && !hidden)
        {
            rectangle temp = rect;
            const long offset = mfont->height() + bottom_pad + top_pad;
            temp.set_bottom(rect.top()+offset);
            if (temp.contains(x,y))
            {
                // now we have to figure out which tab was clicked
                for (unsigned long i = 0; i < tabs.size(); ++i)
                {
                    if (selected_tab_ != i && tabs[i].rect.contains(x,y) &&
                        tabs[selected_tab_].rect.contains(x,y) == false)
                    {
                        unsigned long old_idx = selected_tab_;
                        selected_tab_ = i;
                        recompute_tabs();
                        parent.invalidate_rectangle(temp);

                        // adjust the widget_group objects for these tabs if they exist
                        if (tabs[i].group)
                            tabs[i].group->show();
                        if (tabs[old_idx].group)
                            tabs[old_idx].group->hide();

                        if (event_handler.is_set())
                            event_handler(i,old_idx);
                        break;
                    }
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    set_tab_group (
        unsigned long idx,
        widget_group& group
    )
    {
        auto_mutex M(m);

        DLIB_ASSERT ( idx < number_of_tabs() ,
                "\tvoid tabbed_display::set_tab_group()"
                << "\n\tidx:              " << idx 
                << "\n\tnumber_of_tabs(): " << number_of_tabs() );


        tabs[idx].group = &group;
        group.set_pos(rect.left()+3,rect.top()+mfont->height()+top_pad+bottom_pad+2);
        if (idx == selected_tab_)
            group.show();
        else
            group.hide();
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    disable (
    )
    {
        auto_mutex M(m);
        if (tabs[selected_tab_].group)
            tabs[selected_tab_].group->disable();
        drawable::disable();
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    enable (
    )
    {
        auto_mutex M(m);
        if (tabs[selected_tab_].group)
            tabs[selected_tab_].group->enable();
        drawable::enable();
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    hide (
    )
    {
        auto_mutex M(m);
        if (tabs[selected_tab_].group)
            tabs[selected_tab_].group->hide();
        drawable::hide();
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    show (
    )
    {
        auto_mutex M(m);
        if (tabs[selected_tab_].group)
            tabs[selected_tab_].group->show();
        drawable::show();
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    draw (
        const canvas& c
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        // draw the main border first
        rectangle main_box(rect.left(),rect.top()+mfont->height()+top_pad+bottom_pad,rect.right(),rect.bottom());
        draw_button_up(c,main_box);
        draw_pixel(c,point(main_box.right()-1,main_box.top()),rgb_pixel(128,128,128));

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

        // draw the tabs
        for (unsigned long i = 0; i < tabs.size(); ++i)
        {
            if (selected_tab_ != i)
                draw_tab(tabs[i].rect,c);

            // draw the name string
            rectangle temp = tabs[i].rect;
            temp.set_top(temp.top()+top_pad);
            temp.set_bottom(temp.bottom()+bottom_pad);
            temp.set_left(temp.left()+left_pad);
            temp.set_right(temp.right()+right_pad);
            mfont->draw_string(c,temp,tabs[i].name,color);
        }
        draw_tab(tabs[selected_tab_].rect,c);
        draw_line(c,
            point(tabs[selected_tab_].rect.left()+1,
            tabs[selected_tab_].rect.bottom()),
            point(tabs[selected_tab_].rect.right()-2,
            tabs[selected_tab_].rect.bottom()),
            rgb_pixel(212,208,200));
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    draw_tab (
        const rectangle& tab,
        const canvas& c
    ) const
    {
        const rgb_pixel white(255,255,255);
        const rgb_pixel background(212,208,200);
        const rgb_pixel dark_gray(64,64,64);
        const rgb_pixel gray(128,128,128);
        draw_line(c,point(tab.left(),tab.top()+2),point(tab.left(),tab.bottom()),white);
        draw_line(c,point(tab.left()+1,tab.top()+2),point(tab.left()+1,tab.bottom()),background);
        draw_line(c,point(tab.right(),tab.top()+2),point(tab.right(),tab.bottom()),dark_gray);
        draw_line(c,point(tab.right()-1,tab.top()+2),point(tab.right()-1,tab.bottom()),gray);
        draw_line(c,point(tab.left()+2,tab.top()),point(tab.right()-2,tab.top()),white);
        draw_pixel(c,point(tab.left()+1,tab.top()+1),white);
        draw_pixel(c,point(tab.right()-1,tab.top()+1),dark_gray);
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;

        for (unsigned long i = 0; i < tabs.size(); ++i)
        {
            unsigned long height;
            mfont->compute_size(tabs[i].name,tabs[i].width,height);
        }

        recompute_tabs();
        set_pos(rect.left(), rect.top());

        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void tabbed_display::
    recompute_tabs (
    )
    {
        const long offset = mfont->height() + bottom_pad + top_pad;


        // figure out the size and position of all the tabs
        rectangle sel_tab_rect, other_tab;
        sel_tab_rect.set_top(rect.top());
        sel_tab_rect.set_bottom(rect.top()+offset);

        other_tab.set_top(rect.top()+2);
        other_tab.set_bottom(rect.top()+offset-1);

        long cur_x = rect.left();
        for (unsigned long i = 0; i < tabs.size(); ++i)
        {
            const unsigned long str_width = tabs[i].width;
            if (selected_tab_ != i)
            {
                other_tab.set_left(cur_x);
                cur_x += left_pad + str_width + right_pad;
                other_tab.set_right(cur_x);
                tabs[i].rect = other_tab;
                ++cur_x;

            }
            else
            {
                if (i != 0)
                    sel_tab_rect.set_left(cur_x-2);
                else
                    sel_tab_rect.set_left(cur_x);

                cur_x += left_pad + str_width + right_pad;

                if (i != tabs.size()-1)
                    sel_tab_rect.set_right(cur_x+2);
                else
                    sel_tab_rect.set_right(cur_x);
                ++cur_x;

                tabs[i].rect = sel_tab_rect;
            }
        }

        // make sure this object is wide enough
        const rectangle& last = tabs[tabs.size()-1].rect;
        const rectangle& first = tabs[0].rect;
        rect = last + rect + first;

    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//             named_rectangle object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    named_rectangle::
    named_rectangle(  
        drawable_window& w
    ) :
        drawable(w),
        name_width(0),
        name_height(0)
    {
        make_name_fit_in_rect();
        enable_events();
    }

// ----------------------------------------------------------------------------------------

    named_rectangle::
    ~named_rectangle(
    )
    {
        disable_events();
        parent.invalidate_rectangle(rect); 
    }

// ----------------------------------------------------------------------------------------

    void named_rectangle::
    set_size (
        unsigned long width,
        unsigned long height
    )
    {
        auto_mutex M(m);
        rectangle old(rect);
        const long x = rect.left();
        const long y = rect.top();
        rect.set_right(x+width-1);
        rect.set_bottom(y+height-1);

        make_name_fit_in_rect();
        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    void named_rectangle::
    wrap_around (
        const rectangle& r
    )
    {
        auto_mutex M(m);
        rectangle old(rect);
        const unsigned long pad = name_height/2;

        rect = rectangle(r.left()-pad, r.top()-name_height*4/3, r.right()+pad, r.bottom()+pad);

        make_name_fit_in_rect();
        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    void named_rectangle::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;
        mfont->compute_size(name_,name_width,name_height);
        make_name_fit_in_rect();
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void named_rectangle::
    make_name_fit_in_rect (
    )
    {
        // make sure the named rectangle is big enough to contain the name
        const unsigned long wtemp = mfont->height() + name_width;
        const unsigned long htemp = mfont->height() + name_height;
        if (rect.width() < wtemp)
            rect.set_right(rect.left() + wtemp - 1 );
        if (rect.height() < htemp)
            rect.set_bottom(rect.bottom() + htemp - 1 );
    }

// ----------------------------------------------------------------------------------------

    void named_rectangle::
    set_name (
        const std::string& name
    )
    {
        auto_mutex M(m);
        name_ = name.c_str();
        mfont->compute_size(name_,name_width,name_height);

        make_name_fit_in_rect();
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    const std::string named_rectangle::
    name (
    ) const
    {
        auto_mutex M(m);
        return std::string(name_.c_str());
    }

// ----------------------------------------------------------------------------------------

    void named_rectangle::
    draw (
        const canvas& c
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;
        
        const unsigned long gap = mfont->height()/2;
        rectangle strrect = rect;
        strrect.set_left(rect.left() + gap);

        const unsigned long rtop = rect.top() + name_height/2;

        const rgb_pixel white(255,255,255);
        const rgb_pixel gray(128,128,128);

        mfont->draw_string(c,strrect,name_);
        draw_line(c,point(rect.left(), rtop),             
                  point(rect.left()+gap/2, rtop), gray);
        draw_line(c,point(rect.left(), rtop),             
                  point(rect.left(), rect.bottom()-1), gray);
        draw_line(c,point(rect.left(), rect.bottom()-1),  
                  point(rect.right()-1, rect.bottom()-1), gray);
        draw_line(c,point(rect.right()-1, rtop),          
                  point(rect.right()-1, rect.bottom()-2), gray);
        draw_line(c,point(strrect.left() + name_width + 2, rtop), 
                  point(rect.right()-1, rtop), gray);

        draw_line(c,point(strrect.left() + name_width + 2, rtop+1),   
                  point( rect.right()-2, rtop+1), white);
        draw_line(c,point(rect.right(), rtop), 
                  point(rect.right(), rect.bottom()), white);
        draw_line(c,point(rect.left(), rect.bottom()),                
                  point(rect.right(), rect.bottom()), white);
        draw_line(c,point(rect.left()+1, rtop+1), 
                  point(rect.left()+1, rect.bottom()-2), white);
        draw_line(c,point(rect.left()+1, rtop+1), 
                  point(rect.left()+gap/2, rtop+1), white);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class mouse_tracker
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    mouse_tracker::
    mouse_tracker(  
        drawable_window& w
    ) :
        dragable(w),
        offset(18),
        nr(w),
        x_label(w),
        y_label(w),
        click_x(-1),
        click_y(-1)
    {
        set_dragable_area(rectangle(0,0,500,500));


        x_label.set_text("x: ");
        y_label.set_text("y: ");
        nr.set_name("mouse position");


        x_label.set_pos(offset,offset);
        y_label.set_pos(x_label.get_rect().left(), x_label.get_rect().bottom()+3);

        nr.wrap_around(x_label.get_rect() + y_label.get_rect());
        rect = nr.get_rect();

        set_z_order(2000000000);
        x_label.set_z_order(2000000001);
        y_label.set_z_order(2000000001);
        nr.set_z_order(2000000001);

        enable_events();
    }

// ----------------------------------------------------------------------------------------

    mouse_tracker::
    ~mouse_tracker(
    )
    { 
        disable_events(); 
        parent.invalidate_rectangle(rect); 
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        nr.set_main_font(f);
        x_label.set_main_font(f);
        y_label.set_main_font(f);
        mfont = f;
        nr.wrap_around(x_label.get_rect() + y_label.get_rect());
        rect = nr.get_rect();
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    set_pos (
        long x,
        long y
    )
    {
        dragable::set_pos(x,y);
        nr.set_pos(x,y);
        x_label.set_pos(rect.left()+offset,rect.top()+offset);
        y_label.set_pos(x_label.get_rect().left(), x_label.get_rect().bottom()+3);
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    show (
    )
    {
        dragable::show();
        nr.show();
        x_label.show();
        y_label.show();
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    hide (
    )
    {
        dragable::hide();
        nr.hide();
        x_label.hide();
        y_label.hide();
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    enable (
    )
    {
        dragable::enable();
        nr.enable();
        x_label.enable();
        y_label.enable();
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    disable (
    )
    {
        dragable::disable();
        nr.disable();
        x_label.disable();
        y_label.disable();
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    on_mouse_down (
        unsigned long btn,
        unsigned long state,
        long x,
        long y,
        bool double_clicked 
    )
    {
        dragable::on_mouse_down(btn,state,x,y,double_clicked);
        if ((state & base_window::SHIFT) && (btn == base_window::LEFT) && enabled && !hidden)
        {
            parent.invalidate_rectangle(rectangle(x,y,x,y));
            parent.invalidate_rectangle(rectangle(click_x,click_y,click_x,click_y));
            click_x = x;
            click_y = y;

            y_label.set_text("y: 0");
            x_label.set_text("x: 0");
        }
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    on_mouse_move (
        unsigned long state,
        long x,
        long y
    )
    {
        if (!hidden && enabled)
        {
            parent.invalidate_rectangle(rect);
            dragable::on_mouse_move(state,x,y);

            long dx = 0;
            long dy = 0;
            if (click_x != -1)
                dx = click_x;
            if (click_y != -1)
                dy = click_y;

            sout.str("");
            sout << "y: " << y - dy;
            y_label.set_text(sout.str());

            sout.str("");
            sout << "x: " << x - dx;
            x_label.set_text(sout.str());
        }
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    on_drag (
    )
    {
        nr.set_pos(rect.left(),rect.top());
        x_label.set_pos(rect.left()+offset,rect.top()+offset);
        y_label.set_pos(x_label.get_rect().left(), x_label.get_rect().bottom()+3);

        long x = 0;
        long y = 0;
        if (click_x != -1)
            x = click_x;
        if (click_y != -1)
            y = click_y;

        sout.str("");
        sout << "y: " << lasty - y;
        y_label.set_text(sout.str());

        sout.str("");
        sout << "x: " << lastx - x;
        x_label.set_text(sout.str());
    }

// ----------------------------------------------------------------------------------------

    void mouse_tracker::
    draw (
        const canvas& c
    ) const 
    { 
        fill_rect(c, rect,rgb_pixel(212,208,200));
        draw_pixel(c, point(click_x,click_y),rgb_pixel(255,0,0));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class list_box
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    list_box::
    list_box(  
        drawable_window& w
    ) : 
        drawable(w,MOUSE_WHEEL|MOUSE_CLICK),
        ms_enabled(false),
        pos(0),
        text_start(0),
        last_selected(0),
        sbv(w,scroll_bar::VERTICAL),
        sbh(w,scroll_bar::HORIZONTAL)
    {
        adjust_sliders();
        sbv.set_scroll_handler(*this,&list_box::sbv_handler);
        sbh.set_scroll_handler(*this,&list_box::sbh_handler);
        enable_events();
    }

// ----------------------------------------------------------------------------------------

    list_box::
    ~list_box(
    )
    {
        disable_events();
        parent.invalidate_rectangle(rect); 
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    set_main_font (
        const font* f
    )
    {
        auto_mutex M(m);
        mfont = f;
        // recompute the sizes of all the items
        for (unsigned long i = 0; i < items.size(); ++i)
        {
            mfont->compute_size(items[i].name,items[i].width, items[i].height);
        }
        adjust_sliders();
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    set_size (
        unsigned long width_,
        unsigned long height_
    )
    {
        auto_mutex M(m);
        rectangle old(rect);
        const long x = rect.left();
        const long y = rect.top();
        rect.set_right(x+width_-1);
        rect.set_bottom(y+height_-1);

        adjust_sliders();
        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    bool list_box::
    is_selected (
        unsigned long index
    ) const
    {
        auto_mutex M(m);
        DLIB_ASSERT ( index < size() ,
                "\tbool list_box::is_selected(index)"
                << "\n\tindex:  " << index 
                << "\n\tsize(): " << size() );

        return items[index].is_selected;
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    select (
        unsigned long index 
    )
    {
        auto_mutex M(m);
        DLIB_ASSERT ( index < size() ,
                "\tvoid list_box::select(index)"
                << "\n\tindex:  " << index 
                << "\n\tsize(): " << size() );

        items[index].is_selected = true;
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    unselect (
        unsigned long index 
    )
    {
        auto_mutex M(m);
        DLIB_ASSERT ( index < size() ,
                "\tvoid list_box::unselect(index)"
                << "\n\tindex:  " << index 
                << "\n\tsize(): " << size() );
        items[index].is_selected = false;
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    const std::string& list_box::operator [] (
        unsigned long index
    ) const
    {
        auto_mutex M(m);
        DLIB_ASSERT ( index < size() ,
                "\tconst std::string& list_box::operator[](index)"
                << "\n\tindex:  " << index 
                << "\n\tsize(): " << size() );
        return items[index].name;
    }

// ----------------------------------------------------------------------------------------

    bool list_box::
    multiple_select_enabled (
    ) const
    {
        auto_mutex M(m);
        return ms_enabled;
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    enable_multiple_select (
    ) 
    {
        auto_mutex M(m);
        ms_enabled = true;
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    disable_multiple_select (
    )
    {
        auto_mutex M(m);
        ms_enabled = false;
    }

// ----------------------------------------------------------------------------------------

    bool list_box::
    at_start (
    ) const
    {
        auto_mutex M(m);
        return items.at_start();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    reset (
    ) const
    {
        auto_mutex M(m);
        items.reset();
    }

// ----------------------------------------------------------------------------------------

    bool list_box::
    current_element_valid (
    ) const
    {
        auto_mutex M(m);
        return items.current_element_valid();
    }

// ----------------------------------------------------------------------------------------

    const std::string& list_box::
    element (
    ) const
    {
        auto_mutex M(m);
        DLIB_ASSERT ( current_element_valid() ,
                "\tconst std::string& list_box::element()"
                 );
        return items.element().name;
    }

// ----------------------------------------------------------------------------------------

    const std::string& list_box::
    element (
    )
    {
        auto_mutex M(m);
        DLIB_ASSERT ( current_element_valid() ,
                "\tconst std::string& list_box::element()"
                 );
        return items.element().name;
    }

// ----------------------------------------------------------------------------------------

    bool list_box::
    move_next (
    ) const
    {
        auto_mutex M(m);
        return items.move_next();
    }

// ----------------------------------------------------------------------------------------

    unsigned long list_box::
    size (
    ) const
    {
        auto_mutex M(m);
        return items.size();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    draw (
        const canvas& c
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        if (enabled)
        {
            // first fill our area with white
            fill_rect(c, area,rgb_pixel(255,255,255));
        }
        else
        {
            // first fill our area with gray 
            fill_rect(c, area,rgb_pixel(212,208,200));
        }

        draw_sunken_rectangle(c, rect);

        long y = text_area.top();
        long x = text_area.left();
        for (unsigned long i = pos; i < items.size(); ++i)
        {
            rectangle r(x-(long)text_start,y,text_area.right(),y+items[i].height);
            rectangle draw_area(x,y,text_area.right(),y+items[i].height);
            draw_area = draw_area.intersect(text_area);
            if (draw_area.is_empty())
                break;

            if (items[i].is_selected)
            {
                if (enabled)
                    fill_rect_with_vertical_gradient(c,draw_area,rgb_pixel(110,160,255),  rgb_pixel(100,130,250));
                else
                    fill_rect_with_vertical_gradient(c,draw_area,rgb_pixel(140,190,255),  rgb_pixel(130,160,250));
            }

            if (enabled)
                mfont->draw_string(c,r,items[i].name,rgb_pixel(0,0,0),0,std::string::npos,draw_area);
            else
                mfont->draw_string(c,r,items[i].name,rgb_pixel(128,128,128),0,std::string::npos,draw_area);
            y += items[i].height;
            if (y > area.bottom())
                break;
        }
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    hide (
    )
    {
        auto_mutex M(m);
        sbv.hide();
        sbh.hide();
        drawable::hide();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    show (
    )
    {
        auto_mutex M(m);
        hidden = false;
        adjust_sliders();
        drawable::show();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    disable (
    ) 
    {
        sbv.disable();
        sbh.disable();
        drawable::disable();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    enable (
    ) 
    {
        sbv.enable();
        sbh.enable();
        drawable::enable();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    set_pos (
        long x,
        long y
    )
    {
        auto_mutex M(m);
        drawable::set_pos(x,y);
        adjust_sliders();
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    adjust_sliders (
    ) 
    {
        text_area = rectangle(rect.left()+2,rect.top()+2,rect.right()-1,rect.bottom()-1);
        
        int extra_count = 0;
        // find the max width and height of our text
        unsigned long maxw = 0, maxh = 0;
        for (unsigned long i = 0; i < items.size(); ++i)
        {
            maxh += items[i].height;
            if (maxh > text_area.height())
                ++extra_count;
            if (items[i].width > maxw )
                maxw = items[i].width;
        }

        if (maxh > text_area.height() && text_area.is_empty() == false)
        {
            if (!hidden)
                sbv.show();
            sbv.set_pos(rect.right()-sbv.width()-pad+1,rect.top()+pad);
            sbv.set_length(rect.height()-pad*2);

            text_area.set_right(text_area.right()-sbv.width()-1);

            if (maxw > text_area.width())
            {
                if (!hidden)
                    sbh.show();
                sbh.set_pos(rect.left()+pad,rect.bottom()-sbh.height()-pad+1);
                sbh.set_length(rect.width()-pad*2 - sbv.width());
                text_area.set_bottom(text_area.bottom()-sbh.height()-1);
                sbh.set_max_slider_pos(maxw - text_area.width());

                ++extra_count;
            }
            else
            {
                sbh.hide();
            }
            sbv.set_max_slider_pos(extra_count);
        }
        else
        {
            sbv.hide();
            pos = 0;
            sbv.set_max_slider_pos(0);
            if (maxw > text_area.width() && text_area.is_empty() == false)
            {
                if (!hidden)
                    sbh.show();
                sbh.set_pos(rect.left()+pad,rect.bottom()-sbh.height()-pad+1);
                sbh.set_length(rect.width()-pad*2);
                text_area.set_bottom(text_area.bottom()-sbh.height()-1);
                sbh.set_max_slider_pos(maxw - text_area.width());
            }
            else
            {
                sbh.hide();
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    sbh_handler (
    )
    {
        text_start = sbh.slider_pos();
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    sbv_handler (
    )
    {
        pos = sbv.slider_pos();
        parent.invalidate_rectangle(rect);
    }

// ----------------------------------------------------------------------------------------
    
    void list_box::
    on_wheel_up (
    )
    {
        if (rect.contains(lastx,lasty) && enabled && !hidden)
        {
            long new_pos = sbv.slider_pos();
            if (new_pos > 0)
            {
                pos = new_pos-1;
                sbv.set_slider_pos(pos);
                parent.invalidate_rectangle(rect);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    on_wheel_down (
    )
    {
        if (rect.contains(lastx,lasty) && enabled && !hidden)
        {
            long new_pos = sbv.slider_pos();
            if (new_pos < sbv.max_slider_pos())
            {
                pos = new_pos+1;
                sbv.set_slider_pos(pos);
                parent.invalidate_rectangle(rect);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    on_mouse_down (
        unsigned long btn,
        unsigned long state,
        long x,
        long y,
        bool is_double_click
    )
    {
        if (text_area.contains(x,y) && btn == base_window::LEFT && enabled && !hidden )
        {
            if ( ms_enabled == false || 
                 (!(state&base_window::CONTROL)) && !(state&base_window::SHIFT))
            {
                items.reset();
                while (items.move_next())
                {
                    items.element().is_selected = false;
                }
            }

            y -= text_area.top();
            long h = 0;
            for (unsigned long i = pos; i < items.size(); ++i)
            {
                h += items[i].height;
                if (h >= y)
                {
                    if (ms_enabled)
                    {
                        if (state&base_window::CONTROL)
                        {
                            items[i].is_selected = !items[i].is_selected;
                            if (items[i].is_selected)
                                last_selected = i;
                        }
                        else if (state&base_window::SHIFT)
                        {
                            // we want to select everything between (and including) the
                            // current thing clicked and last_selected.
                            const unsigned long first = std::min(i,last_selected);
                            const unsigned long last = std::max(i,last_selected);
                            for (unsigned long j = first; j <= last; ++j)
                                items[j].is_selected = true;
                        }
                        else
                        {
                            items[i].is_selected = true;
                            last_selected = i;
                            if (is_double_click && event_handler.is_set())
                                event_handler(i);
                            else if (single_click_event_handler.is_set())
                                single_click_event_handler(i);
                        }
                    }
                    else
                    {
                        items[i].is_selected = true;
                        last_selected = i;
                        if (is_double_click && event_handler.is_set())
                            event_handler(i);
                        else if (single_click_event_handler.is_set())
                            single_click_event_handler(i);
                    }

                    break;
                }
            }

            parent.invalidate_rectangle(rect);
        }
    }

// ----------------------------------------------------------------------------------------

    void list_box::
    set_z_order (
        long order
    )
    {
        sbv.set_z_order(order);
        sbh.set_z_order(order);
        drawable::set_z_order(order);
    }

// ----------------------------------------------------------------------------------------

    unsigned long list_box::
    get_selected (
    ) const
    {
        auto_mutex M(m);
        DLIB_ASSERT ( multiple_select_enabled() == false,
                "\tunsigned long list_box::get_selected()"
                 );
        for (unsigned long i = 0; i < items.size(); ++i)
        {
            if (items[i].is_selected)
                return i;
        }
        return items.size();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WIDGETs_CPP_

