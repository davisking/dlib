// Copyright (C) 2005  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BASE_WIDGETs_CPP_
#define DLIB_BASE_WIDGETs_CPP_

#include "base_widgets.h"
#include "../assert.h"
#include <iostream>


namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // dragable object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    dragable::~dragable() {}

// ----------------------------------------------------------------------------------------

    void dragable::
    on_mouse_move (
        unsigned long state,
        long x,
        long y
    )
    {
        if (drag && (state & base_window::LEFT) && enabled && !hidden)
        {
            // the user is trying to drag this object.  we should calculate the new
            // x and y positions for the upper left corner of this object's rectangle

            long new_x = x - this->x;
            long new_y = y - this->y;

            // make sure these points are inside the dragable area.  
            if (new_x < area.left())
                new_x = area.left();
            if (new_x + static_cast<long>(rect.width()) - 1 > area.right())
                new_x = area.right() - rect.width() + 1;

            if (new_y + static_cast<long>(rect.height()) - 1 > area.bottom())
                new_y = area.bottom() - rect.height() + 1;
            if (new_y < area.top())
                new_y = area.top();

            // now make the new rectangle for this object
            rectangle new_rect(
                new_x,
                new_y,
                new_x + rect.width() - 1,
                new_y + rect.height() - 1
            );

            // only do anything if this is a new rectangle and it is inside area
            if (new_rect != rect && area.intersect(new_rect) == new_rect)
            {
                parent.invalidate_rectangle(new_rect + rect);
                rect = new_rect;

                // call the on_drag() event handler
                on_drag();
            }
        }
        else
        {
            drag = false;
        }
    }

// ----------------------------------------------------------------------------------------

    void dragable::
    on_mouse_down (
        unsigned long btn,
        unsigned long ,
        long x,
        long y,
        bool 
    )
    {
        if (enabled && !hidden && rect.contains(x,y) && btn == base_window::LEFT)
        {
            drag = true;
            this->x = x - rect.left();
            this->y = y - rect.top();
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // mouse_over_event object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    mouse_over_event::~mouse_over_event() {}

// ----------------------------------------------------------------------------------------

    void mouse_over_event::
    on_mouse_leave (
    )
    {
        if (enabled && !hidden && is_mouse_over_)
        {
            is_mouse_over_ = false;
            on_mouse_not_over();
        }
    }

// ----------------------------------------------------------------------------------------

    void mouse_over_event::
    on_mouse_move (
        unsigned long state,
        long x,
        long y
    )
    {
        if (enabled == false || hidden == true)
            return;

        if (rect.contains(x,y) == false)
        {
            if (is_mouse_over_)
            {
                is_mouse_over_ = false;
                on_mouse_not_over();
            }
        }
        else if (is_mouse_over_ == false)
        {
            is_mouse_over_ = true;
            on_mouse_over();
        }
    }

// ----------------------------------------------------------------------------------------

    bool mouse_over_event::
    is_mouse_over (
    ) const
    {
        // check if the mouse is still really over this button
        if (enabled && !hidden && is_mouse_over_ && rect.contains(lastx,lasty) == false)
        {
            // trigger a user event to call on_mouse_not_over() and repaint this object.
            // we must do this in another event because someone might call is_mouse_over()
            // from draw() and you don't want this function to end up calling 
            // parent.invalidate_rectangle().  It would lead to draw() being called over
            // and over.
            parent.trigger_user_event((void*)this,drawable::next_free_user_event_number());
            return false;
        }

        return is_mouse_over_;
    }

// ----------------------------------------------------------------------------------------

    void mouse_over_event::
    on_user_event (
        int num 
    )
    {
        if (is_mouse_over_ && num == drawable::next_free_user_event_number())
        {
            is_mouse_over_ = false;
            on_mouse_not_over();
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // button_action object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    button_action::~button_action() {}

// ----------------------------------------------------------------------------------------

    void button_action::
    on_mouse_down (
        unsigned long btn,
        unsigned long ,
        long x,
        long y,
        bool
    )
    {
        if (enabled && !hidden && btn == base_window::LEFT && rect.contains(x,y))
        {
            is_depressed_ = true;
            seen_click = true;
            parent.invalidate_rectangle(rect);
            on_button_down();
        }
    }

// ----------------------------------------------------------------------------------------

    void button_action::
    on_mouse_not_over (
    )
    {
        if (is_depressed_)
        {
            is_depressed_ = false;
            parent.invalidate_rectangle(rect);
            on_button_up(false);
        }
    }

// ----------------------------------------------------------------------------------------

    void button_action::
    on_mouse_move (
        unsigned long state,
        long x,
        long y
    )
    {
        // forward event to the parent class so it can do it's thing as well as us
        mouse_over_event::on_mouse_move(state,x,y);

        if (enabled == false || hidden == true)
            return;


        if ((state & base_window::LEFT) == 0)
        {
            seen_click = false;
            if (is_depressed_)
            {
                is_depressed_ = false;
                parent.invalidate_rectangle(rect);
                on_button_up(false);
            }

            // the left button isn't down so we don't care about anything else
            return;
        }

        if (rect.contains(x,y) == false)
        {
            if (is_depressed_)
            {
                is_depressed_ = false;
                parent.invalidate_rectangle(rect);
                on_button_up(false);
            }
        }
        else if (is_depressed_ == false && seen_click)
        {
            is_depressed_ = true;
            parent.invalidate_rectangle(rect);
            on_button_down();
        }
    }

// ----------------------------------------------------------------------------------------

    void button_action::
    on_mouse_up (
        unsigned long btn,
        unsigned long,
        long x,
        long y
    )
    {
        if (enabled && !hidden && btn == base_window::LEFT)
        {
            if (is_depressed_)
            {
                is_depressed_ = false;
                parent.invalidate_rectangle(rect);

                if (rect.contains(x,y))                
                {
                    on_button_up(true);
                }
                else
                {
                    on_button_up(false);
                }
            }
            else if (seen_click && rect.contains(x,y))
            {
                // this case here covers the unlikly event that you click on a button,
                // move the mouse off the button and then move it back very quickly and
                // release the mouse button.   It is possible that this mouse up event
                // will occurr before any mouse move event so you might not have set
                // that the button is depressed yet.
                
                // So we should say that this triggers an on_button_down() event and
                // then an on_button_up(true) event.

                parent.invalidate_rectangle(rect);

                on_button_down();
                on_button_up(true);
            }

            seen_click = false;
        }
    }

// ----------------------------------------------------------------------------------------

    bool button_action::
    is_depressed (
    ) const
    {
        // check if the mouse is still really over this button
        if (enabled && !hidden && is_depressed_ && rect.contains(lastx,lasty) == false)
        {
            // trigger a user event to call on_button_up() and repaint this object.
            // we must do this in another event because someone might call is_depressed()
            // from draw() and you don't want this function to end up calling 
            // parent.invalidate_rectangle().  It would lead to draw() being called over
            // and over.
            parent.trigger_user_event((void*)this,mouse_over_event::next_free_user_event_number());
            return false;
        }

        return is_depressed_;
    }

// ----------------------------------------------------------------------------------------

    void button_action::
    on_user_event (
        int num
    )
    {
        // forward event to the parent class so it can do it's thing as well as us
        mouse_over_event::on_user_event(num);

        if (is_depressed_ && num == mouse_over_event::next_free_user_event_number())
        {
            is_depressed_ = false;
            parent.invalidate_rectangle(rect);
            on_button_up(false);
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // arrow_button object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void arrow_button::
    set_size (
        unsigned long width,
        unsigned long height
    )
    {
        auto_mutex M(m);

        rectangle old(rect);
        const unsigned long x = rect.left();
        const unsigned long y = rect.top();
        rect.set_right(x+width-1);
        rect.set_bottom(y+height-1);

        parent.invalidate_rectangle(rect+old);
    }

// ----------------------------------------------------------------------------------------

    void arrow_button::
    draw (
        const canvas& c
    ) const
    {
        rectangle area = rect.intersect(c);
        if (area.is_empty())
            return;

        fill_rect(c,rect,rgb_pixel(212,208,200));

        const long height = rect.height();
        const long width = rect.width();

        const long smallest = (width < height) ? width : height; 

        const long rows = (smallest+3)/4;
        const long start = rows + rows/2-1;
        long dep;

        long tip_x = 0;
        long tip_y = 0;
        long wy = 0;
        long hy = 0;
        long wx = 0; 
        long hx = 0;

        if (button_action::is_depressed())
        {
            dep = 0;

            // draw the button's border
            draw_button_down(c,rect); 
        }
        else
        {
            dep = -1;

            // draw the button's border
            draw_button_up(c,rect);
        }


        switch (dir)
        {
            case UP:
                tip_x = width/2 + rect.left() + dep;
                tip_y = (height - start)/2 + rect.top() + dep + 1;
                wy = 0;
                hy = 1;
                wx = 1;
                hx = 0;
                break;

            case DOWN:
                tip_x = width/2 + rect.left() + dep;
                tip_y = rect.bottom() - (height - start)/2 + dep;
                wy = 0;
                hy = -1;
                wx = 1;
                hx = 0;
                break;

            case LEFT:
                tip_x = rect.left() + (width - start)/2 + dep + 1;
                tip_y = height/2 + rect.top() + dep;
                wy = 1;
                hy = 0;
                wx = 0;
                hx = 1;
                break;

            case RIGHT:
                tip_x = rect.right() - (width - start)/2 + dep;
                tip_y = height/2 + rect.top() + dep;
                wy = 1;
                hy = 0;
                wx = 0;
                hx = -1;
                break;
        }


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



        for (long i = 0; i < rows; ++i)
        {
            draw_line(c,point(tip_x + wx*i + hx*i, tip_y + wy*i + hy*i), 
                      point(tip_x + wx*i*-1 + hx*i, tip_y + wy*i*-1 + hy*i), 
                       color);
        }

    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // scroll_bar object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    scroll_bar::
    scroll_bar(  
        drawable_window& w,
        bar_orientation orientation 
    ) :
        drawable(w),
        width_(16),
        b1(w,arrow_button::UP),
        b2(w,arrow_button::DOWN),
        slider(w,*this,&scroll_bar::on_slider_drag),
        ori(orientation),
        top_filler(w,*this,&scroll_bar::top_filler_down,&scroll_bar::top_filler_up),
        bottom_filler(w,*this,&scroll_bar::bottom_filler_down,&scroll_bar::bottom_filler_up),
        pos(0),
        max_pos(0),
        js(10),
        b1_timer(*this,&scroll_bar::b1_down_t),
        b2_timer(*this,&scroll_bar::b2_down_t),
        top_filler_timer(*this,&scroll_bar::top_filler_down_t),
        bottom_filler_timer(*this,&scroll_bar::bottom_filler_down_t)
    {
        if (ori == HORIZONTAL)
        {
            b1.set_direction(arrow_button::LEFT);
            b2.set_direction(arrow_button::RIGHT);
        }

        // don't show the slider when there is no place it can move.
        slider.hide();

        set_length(100);

        b1.set_button_down_handler(*this,&scroll_bar::b1_down);
        b2.set_button_down_handler(*this,&scroll_bar::b2_down);

        b1.set_button_up_handler(*this,&scroll_bar::b1_up);
        b2.set_button_up_handler(*this,&scroll_bar::b2_up);
        b1.disable();
        b2.disable();
        enable_events();
    }

// ----------------------------------------------------------------------------------------

    scroll_bar::
    ~scroll_bar(
    )
    {
        disable_events();
        parent.invalidate_rectangle(rect); 
        // wait for all the timers to be stopped
        b1_timer.stop_and_wait();
        b2_timer.stop_and_wait();
        top_filler_timer.stop_and_wait();
        bottom_filler_timer.stop_and_wait();
    }

// ----------------------------------------------------------------------------------------

    scroll_bar::bar_orientation scroll_bar::
    orientation (
    ) const
    {
        auto_mutex M(m);
        return ori;
    }

// ----------------------------------------------------------------------------------------
    
    void scroll_bar::
    set_orientation (
        bar_orientation new_orientation   
    )
    {
        auto_mutex M(m);
        unsigned long length;

        if (ori == HORIZONTAL)
            length = rect.width();
        else
            length = rect.height();

        rectangle old_rect = rect;

        if (new_orientation == HORIZONTAL)
        {
            rect.set_right(rect.left() + length - 1 );
            rect.set_bottom(rect.top() + width_ - 1 );
            b1.set_direction(arrow_button::LEFT);
            b2.set_direction(arrow_button::RIGHT);

        }
        else
        {
            rect.set_right(rect.left() + width_ - 1);
            rect.set_bottom(rect.top() + length - 1);
            b1.set_direction(arrow_button::UP);
            b2.set_direction(arrow_button::DOWN);
        }
        ori = new_orientation;

        parent.invalidate_rectangle(old_rect);
        parent.invalidate_rectangle(rect);

        // call this to put everything is in the right spot.
        set_pos (rect.left(),rect.top());
    }

// ----------------------------------------------------------------------------------------
    
    void scroll_bar::
    set_length (
        unsigned long length
    )
    {
        // make the min length be at least 1
        if (length == 0)
        {
            length = 1;
        }

        auto_mutex M(m);

        parent.invalidate_rectangle(rect);

        if (ori == HORIZONTAL)
        {
            rect.set_right(rect.left() + length - 1);
            rect.set_bottom(rect.top() + width_ - 1);

            // if the length is too small then we have to smash up the arrow buttons
            // and hide the slider.
            if (length <= width_*2)
            {
                b1.set_size(length/2,width_);
                b2.set_size(length/2,width_);
            }
            else
            {
                b1.set_size(width_,width_);
                b2.set_size(width_,width_);

                // now adjust the slider
                if (max_pos != 0)
                {
                    slider.set_size(get_slider_size(),width_);
                }
            }

        }
        else
        {
            rect.set_right(rect.left() + width_ - 1);
            rect.set_bottom(rect.top() + length - 1);

            // if the length is too small then we have to smush up the arrow buttons
            // and hide the slider.
            if (length <= width_*2)
            {
                b1.set_size(width_,length/2);
                b2.set_size(width_,length/2);
            }
            else
            {

                b1.set_size(width_,width_);
                b2.set_size(width_,width_);

                // now adjust the slider
                if (max_pos != 0)
                {
                    slider.set_size(width_,get_slider_size());
                }
            }


        }

        // call this to put everything is in the right spot.
        set_pos (rect.left(),rect.top());

        if ((b2.get_rect().top() - b1.get_rect().bottom() - 1 <= 8 && ori == VERTICAL) || 
            (b2.get_rect().left() - b1.get_rect().right() - 1 <= 8 && ori == HORIZONTAL) || 
            max_pos == 0)
        {
            hide_slider();
        }
        else if (enabled && !hidden)
        {
            show_slider();
        }
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    set_pos (
        long x,
        long y
    )
    {
        auto_mutex M(m);
        drawable::set_pos(x,y);

        b1.set_pos(rect.left(),rect.top());
        if (ori == HORIZONTAL)
        {
            // make the b2 button appear at the end of the scroll_bar
            b2.set_pos(rect.right()-b2.get_rect().width() + 1,rect.top());

            if (max_pos != 0)
            {
                double range = b2.get_rect().left() - b1.get_rect().right() - slider.get_rect().width() - 1;
                double slider_pos = pos;
                slider_pos /= max_pos;
                slider_pos *= range;
                slider.set_pos(
                    static_cast<long>(slider_pos)+rect.left() + b1.get_rect().width(),
                    rect.top()
                    );

                // move the dragable area for the slider to the new location
                rectangle area = rect;
                area.set_left(area.left() + width_);
                area.set_right(area.right() - width_);
                slider.set_dragable_area(area);

            }

            
        }
        else
        {
            // make the b2 button appear at the end of the scroll_bar
            b2.set_pos(rect.left(), rect.bottom() - b2.get_rect().height() + 1);

            if (max_pos != 0)
            {
                double range = b2.get_rect().top() - b1.get_rect().bottom() - slider.get_rect().height() - 1;
                double slider_pos = pos;
                slider_pos /= max_pos;
                slider_pos *= range;
                slider.set_pos(
                    rect.left(), 
                    static_cast<long>(slider_pos) + rect.top() + b1.get_rect().height()
                    );

                // move the dragable area for the slider to the new location
                rectangle area = rect;
                area.set_top(area.top() + width_);
                area.set_bottom(area.bottom() - width_);
                slider.set_dragable_area(area);
            }
        }

        adjust_fillers();
    }

// ----------------------------------------------------------------------------------------

    unsigned long scroll_bar::
    get_slider_size (
    ) const
    {
        double range;
        if (ori == HORIZONTAL)
        {
            range = rect.width() - b2.get_rect().width() - b1.get_rect().width();
        }
        else
        {
            range = rect.height() - b2.get_rect().height() - b1.get_rect().height(); 
        }

        double scale_factor = 30.0/(max_pos + 30.0);

        if (scale_factor < 0.1)
            scale_factor = 0.1;


        double fraction = range/(max_pos + range)*scale_factor;
        double result = fraction * range;
        unsigned long res = static_cast<unsigned long>(result);
        if (res < 8)
            res = 8;
        return res;
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    adjust_fillers (
    )
    {
        rectangle top(rect), bottom(rect);

        if (ori == HORIZONTAL)
        {
            if (slider.is_hidden())
            {
                top.set_left(b1.get_rect().right()+1);
                top.set_right(b2.get_rect().left()-1);
                bottom.set_left(1);
                bottom.set_right(-1);
            }
            else
            {
                top.set_left(b1.get_rect().right()+1);
                top.set_right(slider.get_rect().left()-1);
                bottom.set_left(slider.get_rect().right()+1);
                bottom.set_right(b2.get_rect().left()-1);
            }
        }
        else
        {
            if (slider.is_hidden())
            {
                top.set_top(b1.get_rect().bottom()+1);
                top.set_bottom(b2.get_rect().top()-1);
                bottom.set_top(1);
                bottom.set_bottom(-1);
            }
            else
            {
                top.set_top(b1.get_rect().bottom()+1);
                top.set_bottom(slider.get_rect().top()-1);
                bottom.set_top(slider.get_rect().bottom()+1);
                bottom.set_bottom(b2.get_rect().top()-1);
            }
        }

        top_filler.rect = top;
        bottom_filler.rect = bottom;
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    hide_slider (
    )
    {
        rectangle top(rect), bottom(rect);
        slider.hide();
        top_filler.disable();
        bottom_filler.disable();
        bottom_filler.hide();
        if (ori == HORIZONTAL)
        {
            top.set_left(b1.get_rect().right()+1);
            top.set_right(b2.get_rect().left()-1);
        }
        else
        {
            top.set_top(b1.get_rect().bottom()+1);
            top.set_bottom(b2.get_rect().top()-1);
        }
        top_filler.rect = top;
        bottom_filler.rect = bottom;
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    show_slider (
    )
    {
        if ((b2.get_rect().top() - b1.get_rect().bottom() - 1 <= 8 && ori == VERTICAL) || 
            (b2.get_rect().left() - b1.get_rect().right() - 1 <= 8 && ori == HORIZONTAL) || 
            max_pos == 0)
            return;

        rectangle top(rect), bottom(rect);
        slider.show();
        top_filler.enable();
        bottom_filler.enable();
        bottom_filler.show();
        if (ori == HORIZONTAL)
        {
            top.set_left(b1.get_rect().right()+1);
            top.set_right(slider.get_rect().left()-1);
            bottom.set_left(slider.get_rect().right()+1);
            bottom.set_right(b2.get_rect().left()-1);
        }
        else
        {
            top.set_top(b1.get_rect().bottom()+1);
            top.set_bottom(slider.get_rect().top()-1);
            bottom.set_top(slider.get_rect().bottom()+1);
            bottom.set_bottom(b2.get_rect().top()-1);
        }
        top_filler.rect = top;
        bottom_filler.rect = bottom;
    }

// ----------------------------------------------------------------------------------------
    long scroll_bar::
    max_slider_pos (
    ) const
    {
        auto_mutex M(m);
        return max_pos;
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    set_max_slider_pos (
        long mpos
    )
    {
        auto_mutex M(m);
        max_pos = mpos;
        if (pos > mpos)
            pos = mpos;

        if (ori == HORIZONTAL)
            set_length(rect.width());
        else
            set_length(rect.height());

        if (mpos != 0 && enabled)
        {
            b1.enable();
            b2.enable();
        }
        else
        {
            b1.disable();
            b2.disable();
        }
             
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    set_slider_pos (
        long pos
    )
    {
        auto_mutex M(m);
        if (pos < 0)
            pos = 0;
        if (pos > max_pos)
            pos = max_pos;

        this->pos = pos;

        // move the slider object to its new position
        set_pos(rect.left(),rect.top());
    }

// ----------------------------------------------------------------------------------------

    long scroll_bar::
    slider_pos (
    ) const
    {
        auto_mutex M(m);
        return pos;
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    on_slider_drag (
    )
    {
        if (ori == HORIZONTAL)
        {
            double slider_pos = slider.get_rect().left() - b1.get_rect().right() - 1;
            double range = b2.get_rect().left() - b1.get_rect().right() - slider.get_rect().width() - 1;
            slider_pos /= range;
            slider_pos *= max_pos;
            pos = static_cast<unsigned long>(slider_pos);
        }
        else
        {
            double slider_pos = slider.get_rect().top() - b1.get_rect().bottom() - 1;
            double range = b2.get_rect().top() - b1.get_rect().bottom() - slider.get_rect().height() - 1;
            slider_pos /= range;
            slider_pos *= max_pos;
            pos = static_cast<unsigned long>(slider_pos);
        }

        adjust_fillers();
        
        if (scroll_handler.is_set())
            scroll_handler();
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    draw (
        const canvas& 
    ) const
    {
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    b1_down (
    )
    {
        if (pos != 0)
        {
            set_slider_pos(pos-1);
            if (scroll_handler.is_set())
                scroll_handler();

            if (b1_timer.delay_time() == 1000)
                b1_timer.set_delay_time(500);
            else
                b1_timer.set_delay_time(50);
            b1_timer.start();
        }
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    b1_up (
        bool 
    )
    {
        b1_timer.stop();
        b1_timer.set_delay_time(1000);
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    b2_down (
    )
    {
        if (pos != max_pos)
        {
            set_slider_pos(pos+1);
            if (scroll_handler.is_set())
                scroll_handler();

            if (b2_timer.delay_time() == 1000)
                b2_timer.set_delay_time(500);
            else
                b2_timer.set_delay_time(50);
            b2_timer.start();
        }
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    b2_up (
        bool 
    )
    {
        b2_timer.stop();
        b2_timer.set_delay_time(1000);
    }
        
// ----------------------------------------------------------------------------------------

    void scroll_bar::
    top_filler_down (
    )
    {
        // ignore this if the mouse is now outside this object.  This could happen
        // since the timers are also calling this function.
        if (top_filler.rect.contains(lastx,lasty) == false)
        {
            top_filler_up(false);
            return;
        }

        if (pos != 0)
        {
            if (pos < js)
            {
                // if there is less than jump_size() space left then jump the remaining
                // amount.
                delayed_set_slider_pos(0);
            }
            else
            {
                delayed_set_slider_pos(pos-js);
            }

            if (top_filler_timer.delay_time() == 1000)
                top_filler_timer.set_delay_time(500);
            else
                top_filler_timer.set_delay_time(50);
            top_filler_timer.start();
        }
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    top_filler_up (
        bool 
    )
    {
        top_filler_timer.stop();
        top_filler_timer.set_delay_time(1000);
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    bottom_filler_down (
    )
    {
        // ignore this if the mouse is now outside this object.  This could happen
        // since the timers are also calling this function.
        if (bottom_filler.rect.contains(lastx,lasty) == false)
        {
            bottom_filler_up(false);
            return;
        }

        if (pos != max_pos)
        {
            if (max_pos - pos < js)
            {
                // if there is less than jump_size() space left then jump the remaining
                // amount.
                delayed_set_slider_pos(max_pos);
            }
            else
            {
                delayed_set_slider_pos(pos+js);
            }

            if (bottom_filler_timer.delay_time() == 1000)
                bottom_filler_timer.set_delay_time(500);
            else
                bottom_filler_timer.set_delay_time(50);
            bottom_filler_timer.start();
        }
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    bottom_filler_up (
        bool 
    )
    {
        bottom_filler_timer.stop();
        bottom_filler_timer.set_delay_time(1000);
    }

// ----------------------------------------------------------------------------------------

    void scroll_bar::
    set_jump_size (
        long js_
    )
    {
        auto_mutex M(m);
        if (js_ < 1)
            js = 1;
        else
            js = js_;
    }

// ----------------------------------------------------------------------------------------

    long scroll_bar::
    jump_size (
    ) const
    {
        auto_mutex M(m);
        return js;
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                  widget_group object methods  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void widget_group::
    empty (
    ) 
    {  
        auto_mutex M(m); 
        widgets.clear(); 
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    add (
        drawable& widget,
        unsigned long x,
        unsigned long y
    )
    {
        auto_mutex M(m); 
        drawable* w = &widget;
        relpos rp;
        rp.x = x;
        rp.y = y;
        if (widgets.is_in_domain(w))
        {
            widgets[w].x = x;
            widgets[w].y = y;
        }
        else
        {
            widgets.add(w,rp);
        }
        if (is_hidden())
            widget.hide();
        else
            widget.show();

        if (is_enabled())
            widget.enable();
        else
            widget.disable();

        widget.set_z_order(z_order());
        widget.set_pos(x+rect.left(),y+rect.top());
    }

// ----------------------------------------------------------------------------------------

    bool widget_group::
    is_member (
        const drawable& widget
    ) const 
    { 
        auto_mutex M(m); 
        drawable* w = const_cast<drawable*>(&widget);
        return widgets.is_in_domain(w); 
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    remove (
        const drawable& widget
    )
    {
        auto_mutex M(m); 
        drawable* w = const_cast<drawable*>(&widget);
        if (widgets.is_in_domain(w))
        {
            relpos junk;
            drawable* junk2;
            widgets.remove(w,junk2,junk);
        }
    }

// ----------------------------------------------------------------------------------------

    unsigned long widget_group::
    size (
    ) const 
    {  
        auto_mutex M(m); 
        return widgets.size(); 
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    set_pos (
        long x,
        long y
    )
    {
        auto_mutex M(m);
        widgets.reset();
        while (widgets.move_next())
        {
            const unsigned long rx = widgets.element().value().x;
            const unsigned long ry = widgets.element().value().y;
            widgets.element().key()->set_pos(x+rx,y+ry);
        }
        drawable::set_pos(x,y);
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    set_z_order (
        long order
    )
    {
        auto_mutex M(m);
        widgets.reset();
        while (widgets.move_next())
            widgets.element().key()->set_z_order(order);
        drawable::set_z_order(order);
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    show (
    )
    {
        auto_mutex M(m);
        widgets.reset();
        while (widgets.move_next())
            widgets.element().key()->show();
        drawable::show();
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    hide (
    )
    {
        auto_mutex M(m);
        widgets.reset();
        while (widgets.move_next())
            widgets.element().key()->hide();
        drawable::hide();
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    enable (
    )
    {
        auto_mutex M(m);
        widgets.reset();
        while (widgets.move_next())
            widgets.element().key()->enable();
        drawable::enable();
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    disable (
    )
    {
        auto_mutex M(m);
        widgets.reset();
        while (widgets.move_next())
            widgets.element().key()->disable();
        drawable::disable();
    }

// ----------------------------------------------------------------------------------------

    void widget_group::
    fit_to_contents (
    )
    {
        auto_mutex M(m);
        rectangle r;
        widgets.reset();
        while (widgets.move_next())
            r = r + widgets.element().key()->get_rect();

        if (r.is_empty())
        {
            // make sure it is still empty after we set it at the correct position 
            r.set_right(rect.left()-1);
            r.set_bottom(rect.top()-1);
        }

        r.set_left(rect.left());
        r.set_top(rect.top());
        rect = r;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BASE_WIDGETs_CPP_ 

