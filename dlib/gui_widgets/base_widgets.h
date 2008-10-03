// Copyright (C) 2005  Davis E. King (davisking@users.sourceforge.net), Keita Mochizuki
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_BASE_WIDGETs_
#define DLIB_BASE_WIDGETs_

#include "base_widgets_abstract.h"
#include "drawable.h"
#include "../gui_core.h"
#include "../algs.h"
#include "../member_function_pointer.h"
#include "../timer.h"
#include "../map.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../image_transforms.h"
#include "../array.h" 
#include "../smart_pointers.h"
#include "../unicode.h"
#include <cctype>


namespace dlib
{


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class dragable
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class dragable : public drawable
    {
        /*!
            INITIAL VALUE
                - drag == false

            CONVENTION
                - if (the user is holding the left button down over this object) then
                    - drag == true
                    - x == the x position of the mouse relative to the upper left corner
                      of this object.
                    - y == the y position of the mouse relative to the upper left corner
                      of this object.
                - else
                    - drag == false
        !*/

    public:

        dragable(  
            drawable_window& w,
            unsigned long events = 0
        ) : 
            drawable(w,events | MOUSE_MOVE | MOUSE_CLICK),
            drag(false)
        {}

        virtual ~dragable(
        ) = 0;

        rectangle dragable_area (
        ) const { auto_mutex M(m); return area; }

        void set_dragable_area (
            const rectangle& area_ 
        ) { auto_mutex M(m); area = area_; } 

    protected:

        virtual void on_drag (
        ){}

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        );

        void on_mouse_down (
            unsigned long btn,
            unsigned long ,
            long x,
            long y,
            bool 
        );

    private:

        rectangle area;
        bool drag;
        long x, y;

        // restricted functions
        dragable(dragable&);        // copy constructor
        dragable& operator=(dragable&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class mouse_over_event 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class mouse_over_event : public drawable
    {
        /*!
            INITIAL VALUE
                - is_mouse_over_ == false

            CONVENTION
                - is_mouse_over_ == is_mouse_over()
        !*/

    public:

        mouse_over_event(  
            drawable_window& w,
            unsigned long events = 0
        ) :
            drawable(w,events | MOUSE_MOVE),
            is_mouse_over_(false)
        {}


        virtual ~mouse_over_event(
        ) = 0;

        int next_free_user_event_number() const
        {
            return drawable::next_free_user_event_number()+1;
        }

    protected:

        bool is_mouse_over (
        ) const;

        virtual void on_mouse_over (
        ){}

        virtual void on_mouse_not_over (
        ){}

        void on_mouse_leave (
        );

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        );

        void on_user_event (
            int num
        );

    private:
        mutable bool is_mouse_over_;

        // restricted functions
        mouse_over_event(mouse_over_event&);        // copy constructor
        mouse_over_event& operator=(mouse_over_event&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class button_action 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class button_action : public mouse_over_event 
    {
        /*!
            INITIAL VALUE
                - is_depressed_ == false
                - seen_click == false

            CONVENTION
                - is_depressed_ == is_depressed()
                - if (the user has clicked the button but hasn't yet released the
                      left mouse button) then
                    - seen_click == true
                - else 
                    - seen_click == false
        !*/

    public:

        button_action(  
            drawable_window& w,
            unsigned long events = 0
        ) :
            mouse_over_event(w,events | MOUSE_MOVE | MOUSE_CLICK),
            is_depressed_(false),
            seen_click(false)
        {}


        virtual ~button_action(
        ) = 0;

        int next_free_user_event_number() const
        {
            return mouse_over_event::next_free_user_event_number()+1;
        }

    protected:

        bool is_depressed (
        ) const;

        virtual void on_button_down (
        ){}

        virtual void on_button_up (
            bool mouse_over
        ){}

        void on_mouse_not_over (
        );

        void on_mouse_down (
            unsigned long btn,
            unsigned long ,
            long x,
            long y,
            bool
        );

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        );

        void on_mouse_up (
            unsigned long btn,
            unsigned long,
            long x,
            long y
        );


    private:
        mutable bool is_depressed_;
        bool seen_click;

        void on_user_event (
            int num
        );

        // restricted functions
        button_action(button_action&);        // copy constructor
        button_action& operator=(button_action&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class arrow_button 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class arrow_button : public button_action 
    {
        /*!
            INITIAL VALUE
                dir == whatever is given to the constructor

            CONVENTION
                - dir == direction()
        !*/

    public:
        enum arrow_direction 
        {
            UP,
            DOWN,
            LEFT,
            RIGHT
        };

        arrow_button(  
            drawable_window& w,
            arrow_direction dir_ 
        ) : 
            button_action(w),
            dir(dir_)
        {
            enable_events();
        }

        virtual ~arrow_button(
        ){ disable_events();  parent.invalidate_rectangle(rect); }

        arrow_direction direction (
        ) const 
        { 
            auto_mutex M(m); 
            return dir; 
        }

        void set_direction (
            arrow_direction new_direction
        )
        {
            auto_mutex M(m);
            dir = new_direction;
            parent.invalidate_rectangle(rect);
        }

        void set_size (
            unsigned long width,
            unsigned long height
        );

        bool is_depressed (
        ) const
        {
            auto_mutex M(m);
            return button_action::is_depressed();
        }

        template <
            typename T
            >
        void set_button_down_handler (
            T& object,
            void (T::*event_handler)()
        )
        {
            auto_mutex M(m);
            button_down_handler.set(object,event_handler);
        }

        template <
            typename T
            >
        void set_button_up_handler (
            T& object,
            void (T::*event_handler)(bool mouse_over)
        )
        {
            auto_mutex M(m);
            button_up_handler.set(object,event_handler);
        }

    protected:

        void draw (
            const canvas& c
        ) const;

        void on_button_down (
        )
        { 
            if (button_down_handler.is_set())
                button_down_handler(); 
        }

        void on_button_up (
            bool mouse_over
        )
        { 
            if (button_up_handler.is_set())
                button_up_handler(mouse_over); 
        }

    private:

        arrow_direction dir;
        member_function_pointer<>::kernel_1a button_down_handler;
        member_function_pointer<bool>::kernel_1a button_up_handler;

        // restricted functions
        arrow_button(arrow_button&);        // copy constructor
        arrow_button& operator=(arrow_button&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class scroll_bar 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class scroll_bar : public drawable 
    {
        /*!
            INITIAL VALUE
                - ori == a value given by the constructor
                - width_ == 16
                - pos == 0
                - max_pos == 0
                - js == 10

            CONVENTION
                - ori == orientation()
                - b1 == the button that is near the 0 end of the scroll bar
                - b2 == the button that is near the max_pos() end of the scroll bar

                - max_pos == max_slider_pos()
                - pos == slider_pos()
                - js == jump_size()
        !*/

    public:
        enum bar_orientation 
        {
            HORIZONTAL,
            VERTICAL
        };

        scroll_bar(  
            drawable_window& w,
            bar_orientation orientation_
        );

        virtual ~scroll_bar(
        );

        bar_orientation orientation (
        ) const;

        void set_orientation (
            bar_orientation new_orientation   
        );

        void set_length (
            unsigned long length
        );

        long max_slider_pos (
        ) const;

        void set_max_slider_pos (
            long mpos
        );

        void set_slider_pos (
            long pos
        );

        long slider_pos (
        ) const;

        template <
            typename T
            >
        void set_scroll_handler (
            T& object,
            void (T::*eh)()
        ) { scroll_handler.set(object,eh); }

        void set_pos (
            long x,
            long y
        );

        void enable (
        )
        {
            if (!hidden)
                show_slider();
            if (max_pos != 0)
            {
                b1.enable();
                b2.enable();
            }
            drawable::enable();
        }

        void disable (
        )
        {
            hide_slider();
            b1.disable();
            b2.disable();
            drawable::disable();
        }
            
        void hide (
        )
        {
            hide_slider();
            top_filler.hide();
            bottom_filler.hide();
            b1.hide();
            b2.hide();
            drawable::hide();
        }
            
        void show (
        )
        {
            b1.show();
            b2.show();
            drawable::show();
            top_filler.show();
            if (enabled)
                show_slider();
        }

        void set_z_order (
            long order
        )
        {
            slider.set_z_order(order);
            top_filler.set_z_order(order);
            bottom_filler.set_z_order(order);
            b1.set_z_order(order);
            b2.set_z_order(order);
            drawable::set_z_order(order);
        }

        void set_jump_size (
            long js
        );

        long jump_size (
        ) const;


    private:

        void hide_slider (
        );
        /*!
            ensures
                - hides the slider and makes any other changes needed so that the
                  scroll_bar still looks right.
        !*/

        void show_slider (
        );
        /*!
            ensures
                - shows the slider and makes any other changes needed so that the
                  scroll_bar still looks right.
        !*/


        void on_slider_drag (
        ); 
        /*!
            requires
                - is called whenever the user drags the slider
        !*/

        void draw (
            const canvas& c
        ) const;

        void b1_down (
        );

        void b1_up (
            bool mouse_over
        );

        void b2_down (
        );

        void b2_up (
            bool mouse_over
        );

        void top_filler_down (
        );

        void top_filler_up (
            bool mouse_over
        );

        void bottom_filler_down (
        );

        void bottom_filler_up (
            bool mouse_over
        );

        void on_user_event (
            int i
        )
        {
            switch (i)
            {
                case 0:
                    b1_down();
                    break;
                case 1:
                    b2_down();
                    break;
                case 2:
                    top_filler_down();
                    break;
                case 3:
                    bottom_filler_down();
                    break;
                case 4:
                    // if the position we are supposed to switch the slider too isn't 
                    // already set
                    if (delayed_pos != pos)
                    {
                        set_slider_pos(delayed_pos);
                        if (scroll_handler.is_set())
                            scroll_handler();
                    }
                    break;
                default:
                    break;
            }
        }

        void delayed_set_slider_pos (
            unsigned long dpos
        ) 
        {
            delayed_pos = dpos;
            parent.trigger_user_event(this,4); 
        }

        void b1_down_t (
        ) { parent.trigger_user_event(this,0); }

        void b2_down_t (
        ) { parent.trigger_user_event(this,1); }

        void top_filler_down_t (
        ) { parent.trigger_user_event(this,2); }

        void bottom_filler_down_t (
        ) { parent.trigger_user_event(this,3); }


        class filler : public button_action
        {
            friend class scroll_bar;
        public:
            filler (
                drawable_window& w,
                scroll_bar& object,
                void (scroll_bar::*down)(),
                void (scroll_bar::*up)(bool)
            ):
                button_action(w)
            {
                bup.set(object,up);
                bdown.set(object,down);

                enable_events();
            }

            ~filler (
            )
            {
               disable_events();
            }

            void set_size (
                unsigned long width,
                unsigned long height
            )
            {
                rectangle old(rect);
                const unsigned long x = rect.left();
                const unsigned long y = rect.top();
                rect.set_right(x+width-1);
                rect.set_bottom(y+height-1);

                parent.invalidate_rectangle(rect+old);
            }

        private:

            void draw (
                const canvas& c
            ) const
            {
                if (is_depressed())
                    draw_checkered(c, rect,rgb_pixel(0,0,0),rgb_pixel(43,47,55));
                else
                    draw_checkered(c, rect,rgb_pixel(255,255,255),rgb_pixel(212,208,200));
            }

            void on_button_down (
            ) { bdown(); } 

            void on_button_up (
                bool mouse_over
            ) { bup(mouse_over); } 

            member_function_pointer<>::kernel_1a bdown;
            member_function_pointer<bool>::kernel_1a bup;
        };

        class slider_class : public dragable
        {
            friend class scroll_bar;
        public:
            slider_class ( 
                drawable_window& w,
                scroll_bar& object,
                void (scroll_bar::*handler)()
            ) :
                dragable(w)
            {
                mfp.set(object,handler);
                enable_events();
            }

            ~slider_class (
            )
            {
               disable_events();
            }

            void set_size (
                unsigned long width,
                unsigned long height
            )
            {
                rectangle old(rect);
                const unsigned long x = rect.left();
                const unsigned long y = rect.top();
                rect.set_right(x+width-1);
                rect.set_bottom(y+height-1);

                parent.invalidate_rectangle(rect+old);
            }

        private:
            void on_drag (
            )
            {
                mfp();
            }

            void draw (
                const canvas& c
            ) const
            {
                fill_rect(c, rect, rgb_pixel(212,208,200));
                draw_button_up(c, rect);
            }

            member_function_pointer<>::kernel_1a mfp;
        };


        void adjust_fillers (
        );
        /*!
            ensures
                - top_filler and bottom_filler appear in their correct positions
                  relative to the current positions of the slider and the b1 and
                  b2 buttons
        !*/

        unsigned long get_slider_size (
        ) const;
        /*!
            ensures
                - returns the length in pixels the slider should have based on the current
                  state of this scroll bar
        !*/


        const unsigned long width_;
        arrow_button b1, b2;
        slider_class slider;
        bar_orientation ori; 
        filler top_filler, bottom_filler;
        member_function_pointer<>::kernel_1a scroll_handler;

        long pos;
        long max_pos; 
        long js;

        timer<scroll_bar>::kernel_2a b1_timer;
        timer<scroll_bar>::kernel_2a b2_timer;
        timer<scroll_bar>::kernel_2a top_filler_timer;
        timer<scroll_bar>::kernel_2a bottom_filler_timer;
        long delayed_pos;

        // restricted functions
        scroll_bar(scroll_bar&);        // copy constructor
        scroll_bar& operator=(scroll_bar&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class widget_group 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class widget_group : public drawable
    {
        /*!
            INITIAL VALUE
                widgets.size() == 0

            CONVENTION
                Widgets contains all the drawable objects and their relative positions
                that are in *this.
        !*/

        struct relpos
        {
            unsigned long x;
            unsigned long y;
        };

    public:
        widget_group(  
            drawable_window& w
        ) : drawable(w) { rect = rectangle(0,0,-1,-1); enable_events();}

        virtual ~widget_group(
        ){ disable_events(); }

        void empty (
        );

        void add (
            drawable& widget,
            unsigned long x,
            unsigned long y
        );

        bool is_member (
            const drawable& widget
        ) const;

        void remove (
            const drawable& widget
        );

        unsigned long size (
        ) const; 

        void set_pos (
            long x,
            long y
        );

        void set_z_order (
            long order
        );

        void show (
        );

        void hide (
        );

        void enable (
        );

        void disable (
        );

        void fit_to_contents (
        );

    protected:

        // this object doesn't draw anything but also isn't abstract
        void draw (
            const canvas& c
        ) const {}

    private:

        map<drawable*,relpos>::kernel_1a_c widgets;

        // restricted functions
        widget_group(widget_group&);        // copy constructor
        widget_group& operator=(widget_group&);    // assignment operator
    };


// ----------------------------------------------------------------------------------------

    class image_widget : public dragable
    {
        /*!
            INITIAL VALUE
                - img.size() == 0

            CONVENTION
                - img == the image this object displays
        !*/

    public:

        image_widget(  
            drawable_window& w
        ): dragable(w)  { enable_events(); }

        ~image_widget(
        )
        {
            disable_events();
            parent.invalidate_rectangle(rect); 
        }

        template <
            typename image_type
            >
        void set_image (
            const image_type& new_img
        )
        {
            auto_mutex M(m);
            assign_image(img,new_img);
            rectangle old(rect);
            rect.set_right(rect.left()+img.nc()-1); 
            rect.set_bottom(rect.top()+img.nr()-1);
            parent.invalidate_rectangle(rect+old);
        }

    private:

        void draw (
            const canvas& c
        ) const
        {
            rectangle area = rect.intersect(c);
            if (area.is_empty())
                return;

            draw_image(c, point(rect.left(),rect.top()), img);
        }

        array2d<rgb_alpha_pixel>::kernel_1a img;

        // restricted functions
        image_widget(image_widget&);        // copy constructor
        image_widget& operator=(image_widget&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class tooltip 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class tooltip : public mouse_over_event 
    {
        /*!
            INITIAL VALUE
                - stuff.get() == 0
                - events_are_enabled() == false

            CONVENTION
                - if (events_are_enabled() == true) then
                    - stuff.get() != 0
        !*/

    public:

        tooltip(  
            drawable_window& w
        ) : 
            mouse_over_event(w,MOUSE_CLICK)
        {}

        ~tooltip(
        ){ disable_events();}

        void set_size (
            long width, 
            long height 
        )
        {
            auto_mutex M(m);
            rect = resize_rect(rect,width,height);
        }


        void set_text (
            const std::string& str
        )
        {
            set_text(convert_mbstring_to_wstring(str));
        }

        void set_text (
            const std::wstring& str
        )
        {
            set_text(convert_wstring_to_utf32(str));
        }

        void set_text (
            const ustring& str
        )
        {
            auto_mutex M(m);
            if (!stuff)
            {
                stuff.reset(new data(*this));
                enable_events();
            }

            stuff->win.set_text(str);
        }

        const std::string text (
        ) const
        {
            return convert_wstring_to_mbstring(wtext());
        }

        const std::wstring wtext (
        ) const
        {
            return convert_utf32_to_wstring(utext());
        }

        const dlib::ustring utext (
        ) const
        {
            auto_mutex M(m);
            dlib::ustring temp;
            if (stuff)
            {
                temp = stuff->win.text;
            }
            return temp.c_str();
        }

        void hide (
        )
        {
            auto_mutex M(m);
            mouse_over_event::hide();
            if (stuff)
            {
                stuff->tt_timer.stop();
                stuff->win.hide();
            }
        }

        void disable (
        )
        {
            auto_mutex M(m);
            mouse_over_event::disable();
            if (stuff)
            {
                stuff->tt_timer.stop();
                stuff->win.hide();
            }
        }

    protected:

        void on_mouse_over()
        {
            stuff->x = lastx;
            stuff->y = lasty;
            stuff->tt_timer.start();
        }

        void on_mouse_not_over ()
        {
            stuff->tt_timer.stop();
            stuff->win.hide();
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        )
        {
            mouse_over_event::on_mouse_down(btn,state,x,y,is_double_click);
            stuff->tt_timer.stop();
            stuff->win.hide();
        }

        void draw (
            const canvas& 
        ) const{}

    private:

        class tooltip_window : public base_window 
        {
        public:
            tooltip_window (const shared_ptr_thread_safe<font>& f) : base_window(false,true), pad(3), mfont(f)
            {
            }

            ustring text;
            rectangle rect_all;
            rectangle rect_text;
            const unsigned long pad;
            const shared_ptr_thread_safe<font> mfont;
            
            void set_text (
                const std::string& str
            )
            {
                set_text(convert_mbstring_to_wstring(str));
            }

            void set_text (
                const std::wstring& str
            )
            {
                set_text(convert_wstring_to_utf32(str));
            }

            void set_text (
                const dlib::ustring& str
            )
            {
                text = str.c_str();

                unsigned long width, height;
                mfont->compute_size(text,width,height);

                set_size(width+pad*2, height+pad*2);
                rect_all.set_left(0);
                rect_all.set_top(0);
                rect_all.set_right(width+pad*2-1);
                rect_all.set_bottom(height+pad*2-1);

                rect_text = move_rect(rectangle(width,height),pad,pad);
            }

            void paint(const canvas& c)
            {
                c.fill(255,255,150);
                draw_rectangle(c, rect_all);
                mfont->draw_string(c,rect_text,text);
            }
        };

        void show_tooltip (
        )
        {
            auto_mutex M(m);
            long x, y;
            // if the mouse has moved since we started the timer then 
            // keep waiting until the user stops moving it
            if (lastx != stuff->x || lasty != stuff->y)
            {
                stuff->x = lastx;
                stuff->y = lasty;
                return;
            }

            unsigned long display_width, display_height;
            // stop the timer
            stuff->tt_timer.stop();
            parent.get_pos(x,y);
            x += lastx+15;
            y += lasty+15;

            // make sure the tooltip isn't going to be off the screen
            parent.get_display_size(display_width, display_height);
            rectangle wrect(move_rect(stuff->win.rect_all,x,y));
            rectangle srect(display_width, display_height); 
            if (srect.contains(wrect) == false)
            {
                rectangle temp(srect.intersect(wrect));
                x -= wrect.width()-temp.width();
                y -= wrect.height()-temp.height();
            }

            stuff->win.set_pos(x,y);
            stuff->win.show();
        }

        // put all this stuff in data so we can arrange to only
        // construct it when someone is actually using the tooltip widget 
        // rather than just instantiating it.
        struct data
        {
            data(
                tooltip& self
            ) : 
                x(-1), 
                y(-1),
                win(self.mfont),
                tt_timer(self,&tooltip::show_tooltip) 
            { 
                tt_timer.set_delay_time(400); 
            }

            long x, y;
            tooltip_window win;
            timer<tooltip>::kernel_2a tt_timer;

        };
        friend struct data;
        scoped_ptr<data> stuff;



        // restricted functions
        tooltip(tooltip&);        // copy constructor
        tooltip& operator=(tooltip&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class popup_menu 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class menu_item
    {
    public:
        virtual ~menu_item() {}

        virtual rectangle get_left_size (
        ) const { return rectangle(); }
        virtual rectangle get_middle_size (
        ) const = 0; 
        virtual rectangle get_right_size (
        ) const { return rectangle(); }

        virtual unichar get_hot_key (
        ) const { return 0; }

        virtual void draw_background (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const {}

        virtual void draw_left (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const {}

        virtual void draw_middle (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const = 0;

        virtual void draw_right (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const {}

        virtual void on_click (
        ) const {}

        virtual bool has_click_event (
        ) const { return false; }

    };

// ----------------------------------------------------------------------------------------

    class menu_item_submenu : public menu_item
    {
        void initialize (
            unichar hk
        )
        {
            const dlib::ustring &str = text;
            if (hk != 0)
            {
                std::string::size_type pos = str.find_first_of(hk);
                if (pos != std::string::npos)
                {
                    // now compute the location of the underline bar
                    rectangle r1 = f->compute_cursor_rect( rectangle(100000,100000), str, pos);
                    rectangle r2 = f->compute_cursor_rect( rectangle(100000,100000), str, pos+1);

                    underline_p1.x() = r1.left()+1;
                    underline_p2.x() = r2.left()-1;
                    underline_p1.y() = r1.bottom()-f->height()+f->ascender()+2;
                    underline_p2.y() = r2.bottom()-f->height()+f->ascender()+2;
                }
            }
        }
    public:
        menu_item_submenu (
            const std::string& str,
            unichar hk = 0
        ) : 
            text(convert_wstring_to_utf32(convert_mbstring_to_wstring(str))),
            f(default_font::get_font()),
            hotkey(hk)
        {
            initialize(hk);
        }

        menu_item_submenu (
            const std::wstring& str,
            unichar hk = 0
        ) : 
            text(convert_wstring_to_utf32(str)),
            f(default_font::get_font()),
            hotkey(hk)
        {
            initialize(hk);
        }

        menu_item_submenu (
            const dlib::ustring& str,
            unichar hk = 0
        ) : 
            text(str),
            f(default_font::get_font()),
            hotkey(hk)
        {
            initialize(hk);
        }

        virtual unichar get_hot_key (
        ) const { return hotkey; }

        virtual rectangle get_middle_size (
        ) const  
        {
            unsigned long width, height;
            f->compute_size(text,width,height);
            return rectangle(width+30,height);
        }

        virtual rectangle get_right_size (
        ) const  
        {
            return rectangle(15, 5);
        }

        virtual void draw_background (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
            if (c.intersect(rect).is_empty())
                return;

            if (enabled && is_selected)
            {
                fill_rect_with_vertical_gradient(c, rect,rgb_alpha_pixel(0,200,0,100), rgb_alpha_pixel(0,0,0,100));
                draw_rectangle(c, rect,rgb_alpha_pixel(0,0,0,100));
            }
        }

        virtual void draw_right (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
            if (c.intersect(rect).is_empty())
                return;

            unsigned char color = 0;

            if (enabled == false)
                color = 128;

            long x, y;
            x = rect.right() - 7;
            y = rect.top() + rect.height()/2;

            for ( unsigned long i = 0; i < 5; ++i)
                draw_line (c, point(x - i, y + i), point(x - i, y - i), color);
        }

        virtual void draw_middle (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
            if (c.intersect(rect).is_empty())
                return;

            if (enabled)
            {
                f->draw_string(c,rect,text);
            }
            else
            {
                f->draw_string(c,rect,text,128);
            }

            if (underline_p1 != underline_p2)
            {
                point base(rect.left(),rect.top());
                draw_line(c, base+underline_p1, base+underline_p2);
            }
        }

    private:
        dlib::ustring text;
        const shared_ptr_thread_safe<font> f;
        member_function_pointer<>::kernel_1a action;
        unichar hotkey;
        point underline_p1;
        point underline_p2;
    };

// ----------------------------------------------------------------------------------------

    class menu_item_text : public menu_item
    {
        template <typename T>
        void initialize (
            T &object,
            void (T::*event_handler_)(),
            unichar hk
        )
        {
            dlib::ustring &str = text;
            action.set(object,event_handler_);

            if (hk != 0)
            {
                std::string::size_type pos = str.find_first_of(hk);
                if (pos != std::string::npos)
                {
                    // now compute the location of the underline bar
                    rectangle r1 = f->compute_cursor_rect( rectangle(100000,100000), str, pos);
                    rectangle r2 = f->compute_cursor_rect( rectangle(100000,100000), str, pos+1);

                    underline_p1.x() = r1.left()+1;
                    underline_p2.x() = r2.left()-1;
                    underline_p1.y() = r1.bottom()-f->height()+f->ascender()+2;
                    underline_p2.y() = r2.bottom()-f->height()+f->ascender()+2;
                }
            }
        }

    public:
        template <typename T>
        menu_item_text (
            const std::string& str,
            T& object,
            void (T::*event_handler_)(),
            unichar hk = 0
        ) : 
            text(convert_wstring_to_utf32(convert_mbstring_to_wstring(str))),
            f(default_font::get_font()),
            hotkey(hk)
        {
            initialize(object, event_handler_, hk);
        }

        template <typename T>
        menu_item_text (
            const std::wstring& str,
            T& object,
            void (T::*event_handler_)(),
            unichar hk = 0
        ) : 
            text(convert_wstring_to_utf32(str)),
            f(default_font::get_font()),
            hotkey(hk)
        {
            initialize(object, event_handler_, hk);
        }

        template <typename T>
        menu_item_text (
            const dlib::ustring& str,
            T& object,
            void (T::*event_handler_)(),
            unichar hk = 0
        ) : 
            text(str),
            f(default_font::get_font()),
            hotkey(hk)
        {
            initialize(object, event_handler_, hk);
        }

        virtual unichar get_hot_key (
        ) const { return hotkey; }

        virtual rectangle get_middle_size (
        ) const  
        {
            unsigned long width, height;
            f->compute_size(text,width,height);
            return rectangle(width,height);
        }

        virtual void draw_background (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
            if (c.intersect(rect).is_empty())
                return;

            if (enabled && is_selected)
            {
                fill_rect_with_vertical_gradient(c, rect,rgb_alpha_pixel(0,200,0,100), rgb_alpha_pixel(0,0,0,100));
                draw_rectangle(c, rect,rgb_alpha_pixel(0,0,0,100));
            }
        }

        virtual void draw_middle (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
            if (c.intersect(rect).is_empty())
                return;

            if (enabled)
            {
                f->draw_string(c,rect,text);
            }
            else
            {
                f->draw_string(c,rect,text,128);
            }

            if (underline_p1 != underline_p2)
            {
                point base(rect.left(),rect.top());
                draw_line(c, base+underline_p1, base+underline_p2);
            }
        }

        virtual void on_click (
        ) const 
        {
            action();
        }

        virtual bool has_click_event (
        ) const { return true; }

    private:
        dlib::ustring text;
        const shared_ptr_thread_safe<font> f;
        member_function_pointer<>::kernel_1a action;
        unichar hotkey;
        point underline_p1;
        point underline_p2;
    };

// ----------------------------------------------------------------------------------------

    class menu_item_separator : public menu_item
    {
    public:
        virtual rectangle get_middle_size (
        ) const  
        {
            return rectangle(10,4);
        }

        virtual void draw_background (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
            if (c.intersect(rect).is_empty())
                return;

            point p1(rect.left(),rect.top()+rect.height()/2-1);
            point p2(rect.right(),rect.top()+rect.height()/2-1);

            point p3(rect.left(),rect.top()+rect.height()/2);
            point p4(rect.right(),rect.top()+rect.height()/2);
            draw_line(c, p1,p2,128);
            draw_line(c, p3,p4,255);
        }

        virtual void draw_middle (
            const canvas& c,
            const rectangle& rect,
            const bool enabled,
            const bool is_selected
        ) const 
        {
        }
    };

// ----------------------------------------------------------------------------------------

    class popup_menu : public base_window
    {
        /*!
            INITIAL VALUE
                - pad == 2
                - item_pad == 3
                - cur_rect == rectangle(pad,pad,pad-1,pad-1)
                - left_width == 0
                - middle_width == 0
                - selected_item == 0 
                - submenu_open == false
                - items.size() == 0
                - item_enabled.size() == 0
                - left_rects.size() == 0
                - middle_rects.size() == 0
                - right_rects.size() == 0
                - line_rects.size() == 0
                - submenus.size() == 0
                - hide_handlers.size() == 0

            CONVENTION
                - pad = 2
                - item_pad = 3
                - all of the following arrays have the same size:
                    - items.size() 
                    - item_enabled.size() 
                    - left_rects.size() 
                    - middle_rects.size() 
                    - right_rects.size() 
                    - line_rects.size() 
                    - submenus.size() 

                - win_rect == a rectangle that is the exact size of this window and with
                  its upper left corner at (0,0)
                - cur_rect == the rect inside which all the menu items are drawn

                - if (a menu_item is supposed to be selected) then
                    - selected_item == the index in menus of the menu_item
                - else
                    - selected_item == submenus.size()

                - if (there is a selected submenu and it is currently open) then
                    - submenu_open == true
                - else 
                    - submenu_open == false

                - for all valid i:
                    - items[i] == a pointer to the ith menu_item
                    - item_enabled[i] == true if the ith menu_item is enabled, false otherwise
                    - left_rects[i] == the left rectangle for the ith menu item
                    - middle_rects[i] == the middle rectangle for the ith menu item
                    - right_rects[i] == the right rectangle for the ith menu item
                    - line_rects[i] == the rectangle for the entire line on which the ith menu
                      item appears. 
                    - if (submenus[i] != 0) then
                        - the ith menu item has a submenu and it is pointed to by submenus[i]

                - hide_handlers == an array of all the on_hide events registered for
                  this popup_menu
        !*/

    public:

        popup_menu (
        ) :
            base_window(false,true),
            pad(2),
            item_pad(3),
            cur_rect(pad,pad,pad-1,pad-1),
            left_width(0),
            middle_width(0),
            selected_item(0),
            submenu_open(false)
        {
        }

        template <
            typename menu_item_type
            >
        unsigned long add_menu_item (
            const menu_item_type& new_item
        )
        {
            auto_mutex M(wm);
            bool t = true;
            scoped_ptr<menu_item> item(new menu_item_type(new_item));
            items.push_back(item);
            item_enabled.push_back(t);

            // figure out how big the window should be now and what not
            rectangle left = new_item.get_left_size();
            rectangle middle = new_item.get_middle_size();
            rectangle right = new_item.get_right_size();

            bool recalc_rect_positions = false;
            const rectangle all = left+middle+right;


            // make sure left_width contains the max of all the left rectangles
            if (left.width() > left_width)
            {
                left_width = left.width();
                recalc_rect_positions = true;
            }
            // make sure middle_width contains the max of all the middle rectangles
            if (middle.width() > middle_width)
            {
                middle_width = middle.width();
                recalc_rect_positions = true;
            }

            // make the current rectangle wider if necessary
            if (cur_rect.width() < left_width + middle_width + right.width() + 2*item_pad)
            {
                cur_rect = resize_rect_width(cur_rect, left_width + middle_width + right.width() + 2*item_pad);
                recalc_rect_positions = true;
            }

            const long y = cur_rect.bottom()+1 + item_pad;
            const long x = cur_rect.left() + item_pad;

            // make the current rectangle taller to account for this new menu item
            cur_rect.set_bottom(cur_rect.bottom()+all.height() + 2*item_pad);

            // adjust all the saved rectangles since the width of the window changed
            // or left_width changed
            if (recalc_rect_positions)
            {
                long y = cur_rect.top() + item_pad;
                for (unsigned long i = 0; i < left_rects.size(); ++i)
                {
                    middle_rects[i] = move_rect(middle_rects[i], x+left_width, y);
                    right_rects[i] = move_rect(right_rects[i], x+cur_rect.width()-right_rects[i].width()-item_pad, y);
                    line_rects[i] = resize_rect_width(line_rects[i], cur_rect.width());

                    y += line_rects[i].height();
                }
            }

            // save the rectangles for later use.  Also position them at the
            // right spots
            left = move_rect(left,x,y);
            middle = move_rect(middle,x+left_width,y);
            right = move_rect(right,x+cur_rect.width()-right.width()-item_pad,y);
            rectangle line(move_rect(rectangle(cur_rect.width(),all.height()+2*item_pad), x-item_pad, y-item_pad));

            // make sure the left, middle, and right rectangles are centered in the
            // line. 
            if (left.height() < all.height())
                left = translate_rect(left,0, (all.height()-left.height())/2);
            if (middle.height() < all.height())
                middle = translate_rect(middle,0, (all.height()-middle.height())/2);
            if (right.height() < all.height())
                right = translate_rect(right,0, (all.height()-right.height())/2);

            left_rects.push_back(left);
            middle_rects.push_back(middle);
            right_rects.push_back(right);
            line_rects.push_back(line);

            popup_menu* junk = 0;
            submenus.push_back(junk);

            win_rect.set_right(cur_rect.right()+pad);
            win_rect.set_bottom(cur_rect.bottom()+pad);
            set_size(win_rect.width(),win_rect.height());

            // make it so that nothing is selected
            selected_item = submenus.size();

            return items.size()-1;
        }
        
        template <
            typename menu_item_type
            >
        unsigned long add_submenu (
            const menu_item_type& new_item,
            popup_menu& submenu
        )
        {
            auto_mutex M(wm);

            submenus[add_menu_item(new_item)] = &submenu;

            submenu.set_on_hide_handler(*this,&popup_menu::on_submenu_hide);

            return items.size()-1;
        }

        void enable_menu_item (
            unsigned long idx
        )
        {
            DLIB_ASSERT ( idx < size() ,
                    "\tvoid popup_menu::enable_menu_item()"
                    << "\n\tidx:    " << idx
                    << "\n\tsize(): " << size() 
                    );
            auto_mutex M(wm);
            item_enabled[idx] = true;
            invalidate_rectangle(cur_rect);
        }

        void disable_menu_item (
            unsigned long idx
        )
        {
            DLIB_ASSERT ( idx < size() ,
                    "\tvoid popup_menu::enable_menu_item()"
                    << "\n\tidx:    " << idx
                    << "\n\tsize(): " << size() 
                    );
            auto_mutex M(wm);
            item_enabled[idx] = false;
            invalidate_rectangle(cur_rect);
        }

        unsigned long size (
        ) const
        { 
            auto_mutex M(wm);
            return items.size();
        }

        void clear (
        )
        {
            auto_mutex M(wm);
            hide();
            cur_rect = rectangle(pad,pad,pad-1,pad-1);
            win_rect = rectangle();
            left_width = 0;
            middle_width = 0;
            items.clear();
            item_enabled.clear();
            left_rects.clear();
            middle_rects.clear();
            right_rects.clear();
            line_rects.clear();
            submenus.clear();
            selected_item = 0;
            submenu_open = false;
        }
        
        void show (
        )
        {
            auto_mutex M(wm);
            selected_item = submenus.size();
            base_window::show();
        }

        void hide (
        )
        {
            auto_mutex M(wm);
            // hide ourselves
            close_submenu();
            selected_item = submenus.size();
            base_window::hide();
        }

        template <typename T>
        void set_on_hide_handler (
            T& object,
            void (T::*event_handler)()
        )
        {
            auto_mutex M(wm);

            member_function_pointer<>::kernel_1a temp;
            temp.set(object,event_handler);

            // if this handler isn't already registered then add it
            bool found_handler = false;
            for (unsigned long i = 0; i < hide_handlers.size(); ++i)
            {
                if (hide_handlers[i] == temp)
                {
                    found_handler = true;
                    break;
                }
            }

            if (found_handler == false)
            {
                hide_handlers.push_back(temp);
            }
        }

        void select_first_item (
        )
        {
            auto_mutex M(wm);
            close_submenu();
            selected_item = items.size();
            for (unsigned long i = 0; i < items.size(); ++i)
            {
                if ((items[i]->has_click_event() || submenus[i]) && item_enabled[i])
                {
                    selected_item = i;
                    break;
                }
            }
            invalidate_rectangle(cur_rect);
        }

        bool forwarded_on_keydown (
            unsigned long key,
            bool is_printable,
            unsigned long state
        )
        {
            auto_mutex M(wm);
            // do nothing if this popup menu is empty
            if (items.size() == 0)
                return false;

            
            // check if the selected item is a submenu
            if (selected_item != submenus.size() && submenus[selected_item] != 0 && submenu_open)
            {
                // send the key to the submenu and return if that menu used the key
                if (submenus[selected_item]->forwarded_on_keydown(key,is_printable,state) == true)
                    return true;
            }

            if (key == KEY_UP)
            {
                for (unsigned long i = 0; i < items.size(); ++i)
                {
                    selected_item = (selected_item + items.size() - 1)%items.size();
                    // only stop looking if this one is enabled and has a click event or is a submenu
                    if (item_enabled[selected_item] && (items[selected_item]->has_click_event() || submenus[selected_item]) )
                        break;
                }
                invalidate_rectangle(cur_rect);
                return true;
            }
            else if (key == KEY_DOWN)
            {
                for (unsigned long i = 0; i < items.size(); ++i)
                {
                    selected_item = (selected_item + 1)%items.size();
                    // only stop looking if this one is enabled and has a click event or is a submenu
                    if (item_enabled[selected_item] && (items[selected_item]->has_click_event() || submenus[selected_item]))
                        break;
                }
                invalidate_rectangle(cur_rect);
                return true;
            }
            else if (key == KEY_RIGHT && submenu_open == false && display_selected_submenu())
            {
                submenus[selected_item]->select_first_item();
                return true;
            }
            else if (key == KEY_LEFT && selected_item != submenus.size() && 
                     submenus[selected_item] != 0 && submenu_open)
            {
                close_submenu();
                return true;
            }
            else if (key == '\n')
            {
                if (selected_item != submenus.size() && (items[selected_item]->has_click_event() || submenus[selected_item]))
                {
                    const long idx = selected_item;
                    // only hide this popup window if this isn't a submenu
                    if (submenus[idx] == 0)
                    {
                        hide();
                        hide_handlers.reset();
                        while (hide_handlers.move_next())
                            hide_handlers.element()();
                    }
                    else
                    {
                        display_selected_submenu();
                        submenus[idx]->select_first_item();
                    }
                    items[idx]->on_click();
                    return true;
                }
            }
            else if (is_printable)
            {
                // check if there is a hotkey for this key
                for (unsigned long i = 0; i < items.size(); ++i)
                {
                    if (std::tolower(key) == std::tolower(items[i]->get_hot_key()) && 
                        (items[i]->has_click_event() || submenus[i]) )
                    {
                        // only hide this popup window if this isn't a submenu
                        if (submenus[i] == 0)
                        {
                            hide();
                            hide_handlers.reset();
                            while (hide_handlers.move_next())
                                hide_handlers.element()();
                        }
                        else
                        {
                            if (selected_item != items.size())
                                invalidate_rectangle(line_rects[selected_item]);

                            selected_item = i;
                            display_selected_submenu();
                            invalidate_rectangle(line_rects[i]);
                            submenus[i]->select_first_item();
                        }
                        items[i]->on_click();
                    }
                }

                // always say we use a printable key for hotkeys
                return true;
            }

            return false;
        }

    private:

        void on_submenu_hide (
        )
        {
            hide();
            hide_handlers.reset();
            while (hide_handlers.move_next())
                hide_handlers.element()();
        }

        void on_window_resized(
        )
        {
            invalidate_rectangle(win_rect);
        }

        void on_mouse_up (
            unsigned long btn,
            unsigned long,
            long x,
            long y
        )
        {
            if (cur_rect.contains(x,y) && btn == LEFT)
            {
                // figure out which item this was on 
                for (unsigned long i = 0; i < items.size(); ++i)
                {
                    if (line_rects[i].contains(x,y) && item_enabled[i] && items[i]->has_click_event())
                    {
                        // only hide this popup window if this isn't a submenu
                        if (submenus[i] == 0)
                        {
                            hide();
                            hide_handlers.reset();
                            while (hide_handlers.move_next())
                                hide_handlers.element()();
                        }
                        items[i]->on_click();
                        break;
                    }
                }
            }
        }

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        )
        {
            if (cur_rect.contains(x,y))
            {
                // check if the mouse is still in the same rect it was in last time
                rectangle last_rect;
                if (selected_item != submenus.size())
                {
                    last_rect = line_rects[selected_item];
                }

                // if the mouse isn't in the same rectangle any more
                if (last_rect.contains(x,y) == false)
                {
                    if (selected_item != submenus.size())
                    {
                        invalidate_rectangle(last_rect);
                        close_submenu();
                        selected_item = submenus.size();
                    }


                    // figure out if we should redraw any menu items 
                    for (unsigned long i = 0; i < items.size(); ++i)
                    {
                        if (items[i]->has_click_event() || submenus[i])
                        {
                            if (line_rects[i].contains(x,y))
                            {
                                selected_item = i;
                                break;
                            }
                        }
                    }

                    // if we found a rectangle that contains the mouse then
                    // tell it to redraw itself
                    if (selected_item != submenus.size())
                    {
                        display_selected_submenu();
                        invalidate_rectangle(line_rects[selected_item]);
                    }
                }
            }
        }

        void close_submenu (
        )
        {
            if (selected_item != submenus.size() && submenus[selected_item] && submenu_open)
            {
                submenus[selected_item]->hide();
                submenu_open = false;
            }
        }

        bool display_selected_submenu (
        )
        /*!
            ensures
                - if (submenus[selected_item] isn't null) then
                    - displays the selected submenu
                    - returns true
                - else
                    - returns false
        !*/
        {
            // show the submenu if one exists
            if (selected_item != submenus.size() && submenus[selected_item])
            {
                long wx, wy;
                get_pos(wx,wy);
                wx += line_rects[selected_item].right();
                wy += line_rects[selected_item].top();
                submenus[selected_item]->set_pos(wx+1,wy-2);
                submenus[selected_item]->show();
                submenu_open = true;
                return true;
            }
            return false;
        }

        void on_mouse_leave (
        )
        {
            if (selected_item != submenus.size())
            {
                // only unhighlight a menu item if it isn't a submenu item
                if (submenus[selected_item] == 0)
                {
                    invalidate_rectangle(line_rects[selected_item]);
                    selected_item = submenus.size();
                }
            }
        }

        void paint (
            const canvas& c
        )
        {
            c.fill(200,200,200);
            draw_rectangle(c, win_rect);
            for (unsigned long i = 0; i < items.size(); ++i)
            {
                bool is_selected = false;
                if (selected_item != submenus.size() && i == selected_item && 
                    item_enabled[i])
                    is_selected = true;

                items[i]->draw_background(c,line_rects[i], item_enabled[i], is_selected);
                items[i]->draw_left(c,left_rects[i], item_enabled[i], is_selected);
                items[i]->draw_middle(c,middle_rects[i], item_enabled[i], is_selected);
                items[i]->draw_right(c,right_rects[i], item_enabled[i], is_selected);
            }
        }

        const long pad;
        const long item_pad;
        rectangle cur_rect; 
        rectangle win_rect; 
        unsigned long left_width;    
        unsigned long middle_width;    
        array<scoped_ptr<menu_item> >::expand_1d_c items;
        array<bool>::expand_1d_c item_enabled;
        array<rectangle>::expand_1d_c left_rects;
        array<rectangle>::expand_1d_c middle_rects;
        array<rectangle>::expand_1d_c right_rects;
        array<rectangle>::expand_1d_c line_rects;
        array<popup_menu*>::expand_1d_c submenus;
        unsigned long selected_item;
        bool submenu_open;
        array<member_function_pointer<>::kernel_1a>::expand_1d_c hide_handlers;

        // restricted functions
        popup_menu(popup_menu&);        // copy constructor
        popup_menu& operator=(popup_menu&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------

    class zoomable_region : public drawable 
    {
        /*!
            INITIAL VALUE
                - min_scale == 0.15
                - max_scale == 1.0
                - zoom_increment_ == 0.02
                - scale == 1.0
                - mouse_drag_screen == false


            CONVENTION
                - zoom_increment() == zoom_increment_
                - min_zoom_scale() == min_scale
                - max_zoom_scale() == max_scale
                - zoom_scale() == scale
                - if (the user is currently dragging the graph around via the mouse) then 
                    - mouse_drag_screen == true
                - else
                    - mouse_drag_screen == false 

                - max_graph_point() == lr_point
                - display_rect() == display_rect_
                - gui_to_graph_space(point(display_rect.left(),display_rect.top())) == gr_orig
        !*/

    public:

        zoomable_region (
            drawable_window& w,
            unsigned long events = 0
        ) :
            drawable(w,MOUSE_CLICK | MOUSE_WHEEL | MOUSE_MOVE | events),
            min_scale(0.15),
            max_scale(1.0),
            zoom_increment_(0.02),
            vsb(w, scroll_bar::VERTICAL),
            hsb(w, scroll_bar::HORIZONTAL)
        {
            scale = 1;
            mouse_drag_screen = false;

            hsb.set_scroll_handler(*this,&zoomable_region::on_h_scroll);
            vsb.set_scroll_handler(*this,&zoomable_region::on_v_scroll);
        }

        inline ~zoomable_region (
        )= 0;

        void set_pos (
            long x,
            long y
        )
        {
            auto_mutex M(m);
            drawable::set_pos(x,y);
            vsb.set_pos(rect.right()-2-vsb.width(),rect.top()+2);
            hsb.set_pos(rect.left()+2,rect.bottom()-2-hsb.height());

            display_rect_ = rectangle(rect.left()+2,rect.top()+2,rect.right()-2-vsb.width(),rect.bottom()-2-hsb.height());

        }

        void set_zoom_increment (
            double zi
        )
        {
            auto_mutex M(m);
            zoom_increment_ = zi;
        }

        double zoom_increment (
        ) const
        {
            auto_mutex M(m);
            return zoom_increment_;
        }

        void set_max_zoom_scale (
            double ms 
        )
        {
            auto_mutex M(m);
            max_scale = ms;
            if (scale > ms)
            {
                scale = max_scale;
                lr_point = gui_to_graph_space(point(display_rect_.right(),display_rect_.bottom()));
                redraw_graph();
            }
        }

        void set_min_zoom_scale (
            double ms 
        )
        {
            auto_mutex M(m);
            min_scale = ms;
            if (scale < ms)
            {
                scale = min_scale;
                lr_point = gui_to_graph_space(point(display_rect_.right(),display_rect_.bottom()));
                redraw_graph();
            }
        }

        double min_zoom_scale (
        ) const
        {
            auto_mutex M(m);
            return min_scale;
        }

        double max_zoom_scale (
        ) const
        {
            auto_mutex M(m);
            return max_scale;
        }

        void set_size (
            long width,
            long height
        )
        {
            auto_mutex M(m);
            rectangle old(rect);
            rect = resize_rect(rect,width,height);
            vsb.set_pos(rect.right()-1-vsb.width(),  rect.top()+2);
            hsb.set_pos(rect.left()+2,  rect.bottom()-1-hsb.height());

            display_rect_ = rectangle(rect.left()+2,rect.top()+2,rect.right()-2-vsb.width(),rect.bottom()-2-hsb.height());
            vsb.set_length(display_rect_.height());
            hsb.set_length(display_rect_.width());
            parent.invalidate_rectangle(rect+old);

            const double old_scale = scale;
            scale = min_scale;
            lr_point = gui_to_graph_space(point(display_rect_.right(),display_rect_.bottom()));
            scale = old_scale;

            // call adjust_origin() so that the scroll bars get their max slider positions
            // setup right
            const point rect_corner(display_rect_.left(), display_rect_.top());
            const vector<double> rect_corner_graph(gui_to_graph_space(rect_corner));
            adjust_origin(rect_corner, rect_corner_graph);
        }

        void show (
        )
        {
            auto_mutex M(m);
            drawable::show();
            hsb.show();
            vsb.show();
        }

        void hide (
        )
        {
            auto_mutex M(m);
            drawable::hide();
            hsb.hide();
            vsb.hide();
        }

        void enable (
        )
        {
            auto_mutex M(m);
            drawable::enable();
            hsb.enable();
            vsb.enable();
        }

        void disable (
        )
        {
            auto_mutex M(m);
            drawable::disable();
            hsb.disable();
            vsb.disable();
        }

        void set_z_order (
            long order
        )
        {
            auto_mutex M(m);
            drawable::set_z_order(order);
            hsb.set_z_order(order);
            vsb.set_z_order(order);
        }

    protected:

        point graph_to_gui_space (
            const vector<double>& p
        ) const
        {
            const point rect_corner(display_rect_.left(), display_rect_.top());
            const dlib::vector<double> v(p);
            return (v - gr_orig)*scale + rect_corner;
        }

        vector<double> gui_to_graph_space (
            const point& p
        ) const
        {
            const point rect_corner(display_rect_.left(), display_rect_.top());
            const dlib::vector<double> v(p - rect_corner);
            return v/scale + gr_orig;
        }

        point max_graph_point (
        ) const
        {
            return lr_point;
        }

        rectangle display_rect (
        ) const 
        {
            return display_rect_;
        }

        double zoom_scale (
        ) const
        {
            return scale;
        }

        void set_zoom_scale (
            double new_scale
        )
        {
            if (min_scale <= new_scale && new_scale <= max_scale)
            {
                // find the point in the center of the graph area
                point center((display_rect_.left()+display_rect_.right())/2,  (display_rect_.top()+display_rect_.bottom())/2);
                point graph_p(gui_to_graph_space(center));
                scale = new_scale;
                adjust_origin(center, graph_p);
                redraw_graph();
            }
        }

        void center_display_at_graph_point (
            const vector<double>& p
        )
        {
            // find the point in the center of the graph area
            point center((display_rect_.left()+display_rect_.right())/2,  (display_rect_.top()+display_rect_.bottom())/2);
            adjust_origin(center, p);
            redraw_graph();
        }

    // ----------- event handlers ---------------

        void on_wheel_down (
            unsigned long state
        )
        {
            // zoom out
            if (enabled && !hidden && scale > min_scale && display_rect_.contains(lastx,lasty))
            {
                point gui_p(lastx,lasty);
                point graph_p(gui_to_graph_space(gui_p));
                scale -= zoom_increment_;
                if (scale < min_scale)
                    scale = min_scale;
                redraw_graph(); 
                adjust_origin(gui_p, graph_p);
            }
        }

        void on_wheel_up (
            unsigned long state
        )
        {
            // zoom in 
            if (enabled && !hidden && scale < max_scale  && display_rect_.contains(lastx,lasty))
            {
                point gui_p(lastx,lasty);
                point graph_p(gui_to_graph_space(gui_p));
                scale += zoom_increment_;
                if (scale > max_scale)
                    scale = max_scale;
                redraw_graph(); 
                adjust_origin(gui_p, graph_p);
            }
        }

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        )
        {
            if (enabled && !hidden && mouse_drag_screen)
            {
                adjust_origin(point(x,y), drag_screen_point);
                redraw_graph();
            }

            // check if the mouse isn't being dragged anymore
            if ((state & base_window::LEFT) == 0)
            {
                mouse_drag_screen = false;
            }
        }

        void on_mouse_up (
            unsigned long btn,
            unsigned long state,
            long x,
            long y
        )
        {
            mouse_drag_screen = false;
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        )
        {
            if (enabled && !hidden && display_rect_.contains(x,y) && btn == base_window::LEFT)
            {
                mouse_drag_screen = true;
                drag_screen_point = gui_to_graph_space(point(x,y));
            }
        }

        void draw (
            const canvas& c
        ) const
        {
            draw_sunken_rectangle(c,rect);
        }

    private:

        void on_h_scroll (
        )
        {
            gr_orig.x() = hsb.slider_pos();
            redraw_graph();
        }

        void on_v_scroll (
        )
        {
            gr_orig.y() = vsb.slider_pos();
            redraw_graph();
        }

        void redraw_graph (
        )
        {
            parent.invalidate_rectangle(display_rect_);
        }

        void adjust_origin (
            const point& gui_p,
            const vector<double>& graph_p
        )
        /*!
            ensures
                - adjusts gr_orig so that we are as close to the following as possible:
                    - graph_to_gui_space(graph_p) == gui_p
                    - gui_to_graph_space(gui_p) == graph_p
        !*/
        {
            const point rect_corner(display_rect_.left(), display_rect_.top());
            const dlib::vector<double> v(gui_p - rect_corner);
            gr_orig = graph_p - v/scale;


            // make sure the origin isn't outside the point (0,0)
            if (gr_orig.x() < 0)
                gr_orig.x() = 0;
            if (gr_orig.y() < 0)
                gr_orig.y() = 0;

            // make sure the lower right corner of the display_rect_ doesn't map to a point beyond lr_point
            point lr_rect_corner(display_rect_.right(), display_rect_.bottom());
            point p = graph_to_gui_space(lr_point);
            vector<double> lr_rect_corner_graph_space(gui_to_graph_space(lr_rect_corner));
            vector<double> delta(lr_point - lr_rect_corner_graph_space);
            if (lr_rect_corner.x() > p.x())
            {
                gr_orig.x() += delta.x();
            }

            if (lr_rect_corner.y() > p.y())
            {
                gr_orig.y() += delta.y();
            }


            const vector<double> ul_rect_corner_graph_space(gui_to_graph_space(rect_corner));
            lr_rect_corner_graph_space = gui_to_graph_space(lr_rect_corner);
            // now adjust the scroll bars

            hsb.set_max_slider_pos((unsigned long)std::max(lr_point.x()-(lr_rect_corner_graph_space.x()-ul_rect_corner_graph_space.x()),0.0));
            vsb.set_max_slider_pos((unsigned long)std::max(lr_point.y()-(lr_rect_corner_graph_space.y()-ul_rect_corner_graph_space.y()),0.0));
            // adjust slider position now.  
            hsb.set_slider_pos(static_cast<long>(ul_rect_corner_graph_space.x()));
            vsb.set_slider_pos(static_cast<long>(ul_rect_corner_graph_space.y()));

        }


        vector<double> gr_orig; // point in graph space such that it's gui space point is the upper left of display_rect_
        vector<double> lr_point; // point in graph space such that it is at the lower right corner of the screen at max zoom

        mutable std::ostringstream sout;

        double scale; // 0 < scale <= 1
        double min_scale;
        double max_scale;
        double zoom_increment_;
        rectangle display_rect_;

        bool mouse_drag_screen;  // true if the user is dragging the white background area
        point drag_screen_point; // the starting point the mouse was at in graph space for the background area drag

        scroll_bar vsb;
        scroll_bar hsb;

        // restricted functions
        zoomable_region(zoomable_region&);        // copy constructor
        zoomable_region& operator=(zoomable_region&);    // assignment operator

    };
    zoomable_region::~zoomable_region() {}

// ----------------------------------------------------------------------------------------

    class scrollable_region : public drawable 
    {
        /*!
            INITIAL VALUE
                - border_size == 2
                - hscroll_bar_inc == 1
                - vscroll_bar_inc == 1
                - h_wheel_scroll_bar_inc == 1
                - v_wheel_scroll_bar_inc == 1
                - mouse_drag_enabled_ == false
                - user_is_dragging_mouse == false

            CONVENTION
                - mouse_drag_enabled() == mouse_drag_enabled_
                - border_size == 2
                - horizontal_scroll_increment() == hscroll_bar_inc
                - vertical_scroll_increment() == vscroll_bar_inc
                - horizontal_mouse_wheel_scroll_increment() == h_wheel_scroll_bar_inc
                - vertical_mouse_wheel_scroll_increment() == v_wheel_scroll_bar_inc
                - vertical_scroll_pos() == vsb.slider_pos()
                - horizontal_scroll_pos() == hsb.slider_pos()
                - total_rect() == total_rect_
                - display_rect() == display_rect_

                - if (the user is currently dragging the total_rect around with a mouse drag) then
                    - user_is_dragging_mouse == true
                    - drag_origin == the point the mouse was at, with respect to total_rect, 
                      when the dragging started
                - else
                    - user_is_dragging_mouse == false 
        !*/

    public:

        scrollable_region (
            drawable_window& w,
            unsigned long events = 0
        ) :
            drawable(w, MOUSE_WHEEL|events|MOUSE_CLICK|MOUSE_MOVE),
            hsb(w,scroll_bar::HORIZONTAL),
            vsb(w,scroll_bar::VERTICAL),
            border_size(2),
            hscroll_bar_inc(1),
            vscroll_bar_inc(1),
            h_wheel_scroll_bar_inc(1),
            v_wheel_scroll_bar_inc(1),
            mouse_drag_enabled_(false),
            user_is_dragging_mouse(false)
        {
            hsb.set_scroll_handler(*this,&scrollable_region::on_h_scroll);
            vsb.set_scroll_handler(*this,&scrollable_region::on_v_scroll);
        }

        virtual inline ~scrollable_region (
        ) = 0;

        void show (
        )
        {
            auto_mutex M(m);
            drawable::show();
            if (need_h_scroll())
                hsb.show();
            if (need_v_scroll())
                vsb.show();
        }

        void hide (
        )
        {
            auto_mutex M(m);
            drawable::hide();
            hsb.hide();
            vsb.hide();
        }

        void enable (
        )
        {
            auto_mutex M(m);
            drawable::enable();
            hsb.enable();
            vsb.enable();
        }

        void disable (
        )
        {
            auto_mutex M(m);
            drawable::disable();
            hsb.disable();
            vsb.disable();
        }

        void set_z_order (
            long order
        )
        {
            auto_mutex M(m);
            drawable::set_z_order(order);
            hsb.set_z_order(order);
            vsb.set_z_order(order);
        }

        void set_size (
            unsigned long width,
            unsigned long height
        )
        {
            auto_mutex M(m);
            rectangle old(rect);
            rect = resize_rect(rect,width,height);
            vsb.set_pos(rect.right()-border_size-vsb.width()+1, rect.top()+border_size);
            hsb.set_pos(rect.left()+border_size, rect.bottom()-border_size-hsb.height()+1);

            // adjust the display_rect_
            if (need_h_scroll() && need_v_scroll())
            {
                // both scroll bars aren't hidden
                if (!hidden)
                {
                    vsb.show();
                    hsb.show();
                }
                display_rect_ = rectangle( rect.left()+border_size,
                                           rect.top()+border_size,
                                           rect.right()-border_size-vsb.width(),
                                           rect.bottom()-border_size-hsb.height());

                // figure out how many scroll bar positions there should be
                unsigned long hdelta = total_rect_.width()-display_rect_.width();
                unsigned long vdelta = total_rect_.height()-display_rect_.height();
                hdelta = (hdelta+hscroll_bar_inc-1)/hscroll_bar_inc;
                vdelta = (vdelta+vscroll_bar_inc-1)/vscroll_bar_inc;

                hsb.set_max_slider_pos(hdelta);
                vsb.set_max_slider_pos(vdelta);

                vsb.set_jump_size((display_rect_.height()+vscroll_bar_inc-1)/vscroll_bar_inc/2+1);
                hsb.set_jump_size((display_rect_.width()+hscroll_bar_inc-1)/hscroll_bar_inc/2+1);
            }
            else if (need_h_scroll())
            {
                // only hsb is hidden 
                if (!hidden)
                {
                    hsb.show();
                    vsb.hide();
                }
                display_rect_ = rectangle( rect.left()+border_size,
                                           rect.top()+border_size,
                                           rect.right()-border_size,
                                           rect.bottom()-border_size-hsb.height());

                // figure out how many scroll bar positions there should be
                unsigned long hdelta = total_rect_.width()-display_rect_.width();
                hdelta = (hdelta+hscroll_bar_inc-1)/hscroll_bar_inc;

                hsb.set_max_slider_pos(hdelta);
                vsb.set_max_slider_pos(0);

                hsb.set_jump_size((display_rect_.width()+hscroll_bar_inc-1)/hscroll_bar_inc/2+1);
            }
            else if (need_v_scroll())
            {
                // only vsb is hidden 
                if (!hidden)
                {
                    hsb.hide();
                    vsb.show();
                }
                display_rect_ = rectangle( rect.left()+border_size,
                                           rect.top()+border_size,
                                           rect.right()-border_size-vsb.width(),
                                           rect.bottom()-border_size);

                unsigned long vdelta = total_rect_.height()-display_rect_.height();
                vdelta = (vdelta+vscroll_bar_inc-1)/vscroll_bar_inc;

                hsb.set_max_slider_pos(0);
                vsb.set_max_slider_pos(vdelta);

                vsb.set_jump_size((display_rect_.height()+vscroll_bar_inc-1)/vscroll_bar_inc/2+1);
            }
            else
            {
                // both are hidden 
                if (!hidden)
                {
                    hsb.hide();
                    vsb.hide();
                }
                display_rect_ = rectangle( rect.left()+border_size,
                                           rect.top()+border_size,
                                           rect.right()-border_size,
                                           rect.bottom()-border_size);

                hsb.set_max_slider_pos(0);
                vsb.set_max_slider_pos(0);
            }

            vsb.set_length(display_rect_.height());
            hsb.set_length(display_rect_.width());

            // adjust the total_rect_ position by trigging the scroll events
            on_h_scroll();
            on_v_scroll();

            parent.invalidate_rectangle(rect+old);
        }

        unsigned long horizontal_mouse_wheel_scroll_increment (
        ) const
        {
            auto_mutex M(m);
            return h_wheel_scroll_bar_inc;
        }

        unsigned long vertical_mouse_wheel_scroll_increment (
        ) const
        {
            auto_mutex M(m);
            return v_wheel_scroll_bar_inc;
        }

        void set_horizontal_mouse_wheel_scroll_increment (
            unsigned long inc
        )
        {
            auto_mutex M(m);
            h_wheel_scroll_bar_inc = inc;
        }

        void set_vertical_mouse_wheel_scroll_increment (
            unsigned long inc
        )
        {
            auto_mutex M(m);
            v_wheel_scroll_bar_inc = inc;
        }


        unsigned long horizontal_scroll_increment (
        ) const
        {
            auto_mutex M(m);
            return hscroll_bar_inc;
        }

        unsigned long vertical_scroll_increment (
        ) const
        {
            auto_mutex M(m);
            return vscroll_bar_inc;
        }

        void set_horizontal_scroll_increment (
            unsigned long inc
        )
        {
            auto_mutex M(m);
            hscroll_bar_inc = inc;
            // call set_size to reset the scroll bars
            set_size(rect.width(),rect.height());
        }

        void set_vertical_scroll_increment (
            unsigned long inc
        )
        {
            auto_mutex M(m);
            vscroll_bar_inc = inc;
            // call set_size to reset the scroll bars
            set_size(rect.width(),rect.height());
        }

        long horizontal_scroll_pos (
        ) const
        {
            auto_mutex M(m);
            return hsb.slider_pos();
        }

        long vertical_scroll_pos (
        ) const
        {
            auto_mutex M(m);
            return vsb.slider_pos();
        }

        void set_horizontal_scroll_pos (
            long pos
        )
        {
            auto_mutex M(m);

            hsb.set_slider_pos(pos);
            on_h_scroll();
        }

        void set_vertical_scroll_pos (
            long pos
        )
        {
            auto_mutex M(m);

            vsb.set_slider_pos(pos);
            on_v_scroll();
        }

        void set_pos (
            long x,
            long y
        )
        {
            auto_mutex M(m);
            drawable::set_pos(x,y);
            vsb.set_pos(rect.right()-border_size-vsb.width()+1, rect.top()+border_size);
            hsb.set_pos(rect.left()+border_size, rect.bottom()-border_size-hsb.height()+1);

            const long delta_x = total_rect_.left() - display_rect_.left();
            const long delta_y = total_rect_.top() - display_rect_.top();

            display_rect_ = move_rect(display_rect_, rect.left()+border_size, rect.top()+border_size);

            total_rect_ = move_rect(total_rect_, display_rect_.left()+delta_x, display_rect_.top()+delta_y);
        }

        bool mouse_drag_enabled (
        ) const
        {
            auto_mutex M(m);
            return mouse_drag_enabled_;
        }

        void enable_mouse_drag (
        )
        {
            auto_mutex M(m);
            mouse_drag_enabled_ = true;
        }

        void disable_mouse_drag (
        )
        {
            auto_mutex M(m);
            mouse_drag_enabled_ = false;
        }

    protected:

        const rectangle& display_rect (
        ) const
        {
            return display_rect_;
        }

        void set_total_rect_size (
            unsigned long width,
            unsigned long height
        )
        {
            DLIB_ASSERT(width > 0 && height > 0 || width == 0 && height == 0,
                "\tvoid scrollable_region::set_total_rect_size(width,height)"
                << "\n\twidth and height must be > 0 or both == 0"
                << "\n\twidth:  " << width 
                << "\n\theight: " << height 
                << "\n\tthis: " << this
                );

            total_rect_ = move_rect(rectangle(width,height), 
                                    display_rect_.left()-static_cast<long>(hsb.slider_pos()),
                                    display_rect_.top()-static_cast<long>(vsb.slider_pos()));

            // call this just to reconfigure the scroll bars
            set_size(rect.width(),rect.height());
        }

        const rectangle& total_rect (
        ) const
        {
            return total_rect_;
        }

        void scroll_to_rect (
            const rectangle& r_
        )
        {
            const rectangle r(total_rect_.intersect(r_));
            const rectangle old(total_rect_);
            // adjust the horizontal scroll bar so that r fits as best as possible
            if (r.left() < display_rect_.left())
            {
                long distance = (r.left()-total_rect_.left())/hscroll_bar_inc;
                hsb.set_slider_pos(distance);
            }
            else if (r.right() > display_rect_.right())
            {
                long distance = (r.right()-total_rect_.left()-display_rect_.width()+hscroll_bar_inc)/hscroll_bar_inc;
                hsb.set_slider_pos(distance);
            }

            // adjust the vertical scroll bar so that r fits as best as possible
            if (r.top() < display_rect_.top())
            {
                long distance = (r.top()-total_rect_.top())/vscroll_bar_inc;
                vsb.set_slider_pos(distance);
            }
            else if (r.bottom() > display_rect_.bottom())
            {
                long distance = (r.bottom()-total_rect_.top()-display_rect_.height()+vscroll_bar_inc)/vscroll_bar_inc;
                vsb.set_slider_pos(distance);
            }
      

            // adjust total_rect_ so that it matches where the scroll bars are now
            total_rect_ = move_rect(total_rect_, 
                                    display_rect_.left()-hscroll_bar_inc*hsb.slider_pos(), 
                                    display_rect_.top()-vscroll_bar_inc*vsb.slider_pos());

            // only redraw if we actually changed something
            if (total_rect_ != old)
            {
                parent.invalidate_rectangle(display_rect_);
            }
        }

        void on_wheel_down (
            unsigned long state
        )
        {
            if (rect.contains(lastx,lasty) && enabled && !hidden)
            {
                if (need_v_scroll())
                {
                    long pos = vsb.slider_pos();
                    vsb.set_slider_pos(pos+(long)v_wheel_scroll_bar_inc);
                    on_v_scroll();
                }
                else if (need_h_scroll())
                {
                    long pos = hsb.slider_pos();
                    hsb.set_slider_pos(pos+(long)h_wheel_scroll_bar_inc);
                    on_h_scroll();
                }
            }
        }

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        )
        {
            if (enabled && !hidden && user_is_dragging_mouse && state==base_window::LEFT)
            {
                point current_delta = point(x,y) - point(total_rect().left(), total_rect().top());
                rectangle new_rect(translate_rect(display_rect(), drag_origin - current_delta));
                new_rect = centered_rect(new_rect, new_rect.width()-hscroll_bar_inc, new_rect.height()-vscroll_bar_inc);
                scroll_to_rect(new_rect);
            }
            else
            {
                user_is_dragging_mouse = false;
            }
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        )
        {
            if (mouse_drag_enabled_ && enabled && !hidden && display_rect().contains(x,y) && (btn==base_window::LEFT))
            {
                drag_origin = point(x,y) - point(total_rect().left(), total_rect().top());
                user_is_dragging_mouse = true;
            }
            else
            {
                user_is_dragging_mouse = false;
            }
        }

        void on_mouse_up   (
            unsigned long btn,
            unsigned long state,
            long x,
            long y
        )
        {
            user_is_dragging_mouse = false;
        }

        void on_wheel_up (
            unsigned long state
        )
        {
            if (rect.contains(lastx,lasty) && enabled && !hidden)
            {
                if (need_v_scroll())
                {
                    long pos = vsb.slider_pos();
                    vsb.set_slider_pos(pos-(long)v_wheel_scroll_bar_inc);
                    on_v_scroll();
                }
                else if (need_h_scroll())
                {
                    long pos = hsb.slider_pos();
                    hsb.set_slider_pos(pos-(long)h_wheel_scroll_bar_inc);
                    on_h_scroll();
                }
            }
        }

        void draw (
            const canvas& c
        ) const
        {
            rectangle area = c.intersect(rect);
            if (area.is_empty() == true)
                return;

            draw_sunken_rectangle(c,rect);
        }

    private:

        bool need_h_scroll (
        ) const
        {
            if (total_rect_.width() > rect.width()-border_size*2)
            {
                return true;
            }
            else
            {
                // check if we would need a vertical scroll bar and if adding one would make us need
                // a horizontal one
                if (total_rect_.height() > rect.height()-border_size*2 && 
                    total_rect_.width() > rect.width()-border_size*2-vsb.width())
                    return true;
                else
                    return false;
            }
        }

        bool need_v_scroll (
        ) const
        {
            if (total_rect_.height() > rect.height()-border_size*2)
            {
                return true;
            }
            else
            {
                // check if we would need a horizontal scroll bar and if adding one would make us need
                // a vertical_scroll_pos one
                if (total_rect_.width() > rect.width()-border_size*2 && 
                    total_rect_.height() > rect.height()-border_size*2-hsb.height())
                    return true;
                else
                    return false;
            }
        }

        void on_h_scroll (
        )
        {
            total_rect_ = move_rect(total_rect_, display_rect_.left()-hscroll_bar_inc*hsb.slider_pos(), total_rect_.top());
            parent.invalidate_rectangle(display_rect_);
        }

        void on_v_scroll (
        )
        {
            total_rect_ = move_rect(total_rect_, total_rect_.left(), display_rect_.top()-vscroll_bar_inc*vsb.slider_pos());
            parent.invalidate_rectangle(display_rect_);
        }

        rectangle total_rect_;
        rectangle display_rect_;
        scroll_bar hsb;
        scroll_bar vsb;
        const unsigned long border_size;
        unsigned long hscroll_bar_inc;
        unsigned long vscroll_bar_inc;
        unsigned long h_wheel_scroll_bar_inc;
        unsigned long v_wheel_scroll_bar_inc;
        bool mouse_drag_enabled_;
        bool user_is_dragging_mouse;
        point drag_origin;

    };
    scrollable_region::~scrollable_region(){}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class popup_menu_region 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class popup_menu_region : public drawable 
    {

    public:

        popup_menu_region(  
            drawable_window& w
        ) :
            drawable(w,MOUSE_CLICK | KEYBOARD_EVENTS | FOCUS_EVENTS | WINDOW_MOVED),
            popup_menu_shown(false)
        {
            enable_events();
        }

        virtual ~popup_menu_region(
        ){ disable_events();}

        void set_size (
            long width, 
            long height
        )
        {
            auto_mutex M(m);
            rect = resize_rect(rect,width,height);
        }

        popup_menu& menu (
        )
        {
            return menu_;
        }

        void hide (
        )
        {
            auto_mutex M(m);
            drawable::hide();
            menu_.hide();
            popup_menu_shown = false;
        }

        void disable (
        )
        {
            auto_mutex M(m);
            drawable::disable();
            menu_.hide();
            popup_menu_shown = false;
        }

    protected:

        void on_keydown (
            unsigned long key,
            bool is_printable,
            unsigned long state
        )
        {
            if (enabled && !hidden && popup_menu_shown)
            {
                menu_.forwarded_on_keydown(key, is_printable, state);
            }
            else if (popup_menu_shown)
            {
                menu_.hide();
                popup_menu_shown = false;
            }
        }

        void on_focus_lost (
        )
        {
            if (popup_menu_shown)
            {
                menu_.hide();
                popup_menu_shown = false;
            }
        }

        void on_focus_gained (
        )
        {
            if (popup_menu_shown)
            {
                menu_.hide();
                popup_menu_shown = false;
            }
        }

        void on_window_moved(
        )
        {
            if (popup_menu_shown)
            {
                menu_.hide();
                popup_menu_shown = false;
            }
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        )
        {
            if (enabled && !hidden && rect.contains(x,y) && btn == base_window::RIGHT)
            {
                long orig_x, orig_y;
                parent.get_pos(orig_x, orig_y);
                menu_.set_pos(orig_x+x, orig_y+y);
                menu_.show();
                popup_menu_shown = true;
            }
            else if (popup_menu_shown)
            {
                menu_.hide();
                popup_menu_shown = false;
            }
        }

        void draw (
            const canvas& 
        ) const{}

    private:

        popup_menu menu_;
        bool popup_menu_shown;

        // restricted functions
        popup_menu_region(popup_menu_region&);        // copy constructor
        popup_menu_region& operator=(popup_menu_region&);    // assignment operator
    };


// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "base_widgets.cpp"
#endif

#endif // DLIB_BASE_WIDGETs_

