// Copyright (C) 2005  Davis E. King (davisking@users.sourceforge.net), Keita Mochizuki
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_WIDGETs_
#define DLIB_WIDGETs_

#include "../algs.h"
#include "widgets_abstract.h"
#include "drawable.h"
#include "../gui_core.h"
#include "fonts.h"
#include <string>
#include <sstream>
#include "../timer.h"
#include "base_widgets.h"
#include "../member_function_pointer.h"
#include "../array.h"
#include "../sequence.h"
#include "../dir_nav.h"
#include "../queue.h"
#include "../smart_pointers.h"
#include "style.h"
#include "../string.h"
#include "../misc_api.h"
#include <cctype>

#ifdef _MSC_VER
// This #pragma directive is also located in the algs.h file but for whatever
// reason visual studio 9 just ignores it when it is only there. 

// this is to disable the "'this' : used in base member initializer list"
// warning you get from some of the GUI objects since all the objects
// require that their parent class be passed into their constructor. 
// In this case though it is totally safe so it is ok to disable this warning.
#pragma warning(disable : 4355)
#endif

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class label  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class label : public drawable
    {
    public:
        label(
            drawable_window& w
        ) : 
            drawable(w),
            text_color_(0,0,0)
        {
            enable_events();
        }

        ~label()
        { disable_events(); parent.invalidate_rectangle(rect); }

        void set_text (
            const std::string& text
        );

        void set_text (
            const std::wstring& text
        );

        void set_text (
            const dlib::ustring& text
        );

        const std::string text (
        ) const;

        const std::wstring wtext (
        ) const;

        const dlib::ustring utext (
        ) const;

        void set_text_color (
            const rgb_pixel color
        );

        const rgb_pixel text_color (
        ) const;

        void set_main_font (
            const font* f
        );

    private:
        dlib::ustring text_;
        rgb_pixel text_color_;


        // restricted functions
        label(label&);        // copy constructor
        label& operator=(label&);    // assignment operator

    protected:

        void draw (
            const canvas& c
        ) const;

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class button  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class button : public button_action 
    {
    public:
        button(
            drawable_window& w
        ) : 
            button_action(w),
            btn_tooltip(w)
        {
            style.reset(new button_style_default());
            enable_events();
        }
        
        ~button() { disable_events(); parent.invalidate_rectangle(rect); }

        void set_size (
            unsigned long width,
            unsigned long height
        );

        void set_name (
            const std::string& name_
        );

        void set_name (
            const std::wstring& name_
        );

        void set_name (
            const dlib::ustring& name_
        );

        const std::string name (
        ) const;

        const std::wstring wname (
        ) const;

        const dlib::ustring uname (
        ) const;

        void set_tooltip_text (
            const std::string& text
        );

        void set_tooltip_text (
            const std::wstring& text
        );

        void set_tooltip_text (
            const dlib::ustring& text
        );

        void set_pos(
            long x,
            long y
        );

        const std::string tooltip_text (
        ) const;

        const std::wstring tooltip_wtext (
        ) const;

        const dlib::ustring tooltip_utext (
        ) const;

        void set_main_font (
            const font* f
        );

        void show (
        );

        void hide (
        );

        void enable (
        );

        void disable (
        );

        template <
            typename style_type
            >
        void set_style (
            const style_type& style_
        )
        {
            auto_mutex M(m);
            style.reset(new style_type(style_));
            rect = move_rect(style->get_min_size(name_,*mfont), rect.left(), rect.top());
            parent.invalidate_rectangle(rect);
        }

        template <
            typename T
            >
        void set_click_handler (
            T& object,
            void (T::*event_handler_)()
        )
        {
            auto_mutex M(m);
            event_handler.set(object,event_handler_);
            event_handler_self.clear();
        }

        template <
            typename T
            >
        void set_click_handler (
            T& object,
            void (T::*event_handler_)(button&)
        )
        {
            auto_mutex M(m);
            event_handler_self.set(object,event_handler_);
            event_handler.clear();
        }

    private:

        // restricted functions
        button(button&);        // copy constructor
        button& operator=(button&);    // assignment operator

        dlib::ustring name_;
        tooltip btn_tooltip;

        member_function_pointer<>::kernel_1a event_handler;
        member_function_pointer<button&>::kernel_1a event_handler_self;

        scoped_ptr<button_style> style;

    protected:

        void draw (
            const canvas& c
        ) const { style->draw_button(c,rect,hidden,enabled,*mfont,lastx,lasty,name_,is_depressed()); }

        void on_button_up (
            bool mouse_over
        );

        void on_mouse_over (
        ){ if (style->redraw_on_mouse_over()) parent.invalidate_rectangle(rect); }

        void on_mouse_not_over (
        ){ if (style->redraw_on_mouse_over()) parent.invalidate_rectangle(rect); }
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class toggle_button
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class toggle_button : public button_action 
    {
        /*!
            INITIAL VALUE
                - checked == false

            CONVENTION
                - is_checked() == checked
        !*/

    public:

        toggle_button(
            drawable_window& w
        ) : 
            button_action(w),
            btn_tooltip(w),
            checked(false)
        {
            style.reset(new toggle_button_style_default());
            enable_events();
        }
        
        ~toggle_button() { disable_events(); parent.invalidate_rectangle(rect); }

        void set_name (
            const std::string& name
        );

        void set_name (
            const std::wstring& name
        );

        void set_name (
            const dlib::ustring& name
        );

        void set_size (
            unsigned long width_,
            unsigned long height_
        );

        void set_tooltip_text (
            const std::string& text
        );

        void set_tooltip_text (
            const std::wstring& text
        );

        void set_tooltip_text (
            const ustring& text
        );

        const std::string tooltip_text (
        ) const;

        const std::wstring tooltip_wtext (
        ) const;

        const dlib::ustring tooltip_utext (
        ) const;

        bool is_checked (
        ) const;

        const std::string name (
        ) const;

        const std::wstring wname (
        ) const;

        const dlib::ustring uname (
        ) const;

        void set_checked (
        );

        void set_unchecked (
        );

        void show (
        );

        void hide (
        );

        void enable (
        );

        void disable (
        );

        void set_main_font (
            const font* f
        );

        void set_pos (
            long x,
            long y
        );

        template <
            typename style_type
            >
        void set_style (
            const style_type& style_
        )
        {
            auto_mutex M(m);
            style.reset(new style_type(style_));
            rect = move_rect(style->get_min_size(name_,*mfont), rect.left(), rect.top());
            parent.invalidate_rectangle(rect);
        }

        template <
            typename T
            >
        void set_click_handler (
            T& object,
            void (T::*event_handler_)()
        )
        {
            auto_mutex M(m);
            event_handler.set(object,event_handler_);
            event_handler_self.clear();
        }

        template <
            typename T
            >
        void set_click_handler (
            T& object,
            void (T::*event_handler_)(toggle_button&)
        )
        {
            auto_mutex M(m);
            event_handler_self.set(object,event_handler_);
            event_handler.clear();
        }

    private:

        // restricted functions
        toggle_button(toggle_button&);        // copy constructor
        toggle_button& operator=(toggle_button&);    // assignment operator

        dlib::ustring name_;
        tooltip btn_tooltip;
        bool checked;

        member_function_pointer<>::kernel_1a event_handler;
        member_function_pointer<toggle_button&>::kernel_1a event_handler_self;

        scoped_ptr<toggle_button_style> style;

    protected:

        void draw (
            const canvas& c
        ) const { style->draw_toggle_button(c,rect,hidden,enabled,*mfont,lastx,lasty,name_,is_depressed(),checked); }

        void on_button_up (
            bool mouse_over
        );

        void on_mouse_over (
        ){ if (style->redraw_on_mouse_over()) parent.invalidate_rectangle(rect); }

        void on_mouse_not_over (
        ){ if (style->redraw_on_mouse_over()) parent.invalidate_rectangle(rect); }
    };
 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class text_field  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class text_field : public drawable
    {
        /*!
            INITIAL VALUE
                text_color_ == rgb_pixel(0,0,0)
                bg_color_ == rgb_pixel(255,255,255)
                cursor_pos == 0
                text_width == 0
                text_ == ""
                has_focus == false
                cursor_visible == false
                recent_movement == false
                highlight_start == 0
                highlight_end == -1
                shift_pos == -1
                text_pos == 0
    
            CONVENTION
                - cursor_pos == the position of the cursor in the string text_.  The 
                  cursor appears before the letter text_[cursor_pos]
                - cursor_x == the x coordinate of the cursor relative to the left side 
                  of rect.  i.e. the number of pixels that separate the cursor from the
                  left side of the text_field.
                - has_focus == true if this text field has keyboard input focus
                - cursor_visible == true if the cursor should be painted
                - text_ == text()
                - text_pos == the index of the first letter in text_ that appears in 
                  this text field.
                - text_width == the width of text_[text_pos] though text_[text.size()-1]

                - if (has_focus && the user has recently moved the cursor) then
                    - recent_movement == true
                - else
                    - recent_movement == false

                - if (highlight_start <= highlight_end) then
                    - text[highlight_start] though text[highlight_end] should be
                      highlighted

                - if (shift_pos != -1) then
                    - has_focus == true
                    - the shift key is being held down or the left mouse button is
                      being held down.
                    - shift_pos == the position of the cursor when the shift or mouse key
                      was first pressed.

                - text_color() == text_color_
                - background_color() == bg_color_
        !*/

    public:
        text_field(
            drawable_window& w
        ) : 
            drawable(w,MOUSE_CLICK | KEYBOARD_EVENTS | MOUSE_MOVE | STRING_PUT),
            text_color_(0,0,0),
            bg_color_(255,255,255),
            text_width(0),
            text_pos(0),
            recent_movement(false),
            has_focus(false),
            cursor_visible(false),
            cursor_pos(0),
            highlight_start(0),
            highlight_end(-1),
            shift_pos(-1),
            t(*this,&text_field::timer_action)
        {
            rect.set_bottom(mfont->height()+ (mfont->height()-mfont->ascender())*2);
            rect.set_right(9);
            cursor_x = (mfont->height()-mfont->ascender());
            enable_events();

            t.set_delay_time(500);
        }

        ~text_field (
        )
        {
            disable_events();
            parent.invalidate_rectangle(rect); 
            t.stop_and_wait();
        }

        void set_text (
            const std::string& text_
        );

        void set_text (
            const std::wstring& text_
        );

        void set_text (
            const dlib::ustring& text_
        );

        const std::string text (
        ) const;

        const std::wstring wtext (
        ) const;

        const dlib::ustring utext (
        ) const;

        void set_text_color (
            const rgb_pixel color
        );

        const rgb_pixel text_color (
        ) const;

        void set_background_color (
            const rgb_pixel color
        );

        const rgb_pixel background_color (
        ) const;

        void set_width (
            unsigned long width
        );

        void set_main_font (
            const font* f
        );

        int next_free_user_event_number (
        ) const
        {
            return drawable::next_free_user_event_number()+1;
        }

        void disable (
        );

        void hide (
        );

        template <
            typename T
            >
        void set_text_modified_handler (
            T& object,
            void (T::*event_handler)()
        )
        {
            auto_mutex M(m);
            text_modified_handler.set(object,event_handler);
        }

        template <
            typename T
            >
        void set_enter_key_handler (
            T& object,
            void (T::*event_handler)()
        )
        {
            auto_mutex M(m);
            enter_key_handler.set(object,event_handler);
        }


        template <
            typename T
            >
        void set_focus_lost_handler (
            T& object,
            void (T::*event_handler)()
        )
        {
            auto_mutex M(m);
            focus_lost_handler.set(object,event_handler);
        }

    private:

        void on_user_event (
            int num
        )
        {
            // ignore this user event if it isn't for us
            if (num != drawable::next_free_user_event_number())
                return;

            if (recent_movement == false)
            {
                cursor_visible = !cursor_visible; 
                parent.invalidate_rectangle(rect); 
            }
            else
            {
                if (cursor_visible == false)
                {
                    cursor_visible = true;
                    parent.invalidate_rectangle(rect); 
                }
                recent_movement = false;
            }
        }

        void timer_action (
        ) { parent.trigger_user_event(this,drawable::next_free_user_event_number()); }
        /*!
            ensures
                - flips the state of cursor_visible
        !*/

        void move_cursor (
            unsigned long pos
        );
        /*!
            requires
                - pos <= text_.size() 
            ensures
                - moves the cursor to the position given by pos and moves the text 
                  in the text box if necessary
                - if the position changes then the parent window will be updated
        !*/

        rectangle get_text_rect (
        ) const;
        /*!
            ensures
                - returns the rectangle that should contain the text in this widget
        !*/

        dlib::ustring text_;
        rgb_pixel text_color_;
        rgb_pixel bg_color_;

        unsigned long text_width;
        unsigned long text_pos;


        bool recent_movement;
        bool has_focus;
        bool cursor_visible;
        long cursor_pos;
        unsigned long cursor_x;

        // this tells you what part of the text is highlighted
        long highlight_start;
        long highlight_end;
        long shift_pos;
        member_function_pointer<>::kernel_1a_c text_modified_handler;
        member_function_pointer<>::kernel_1a_c enter_key_handler;
        member_function_pointer<>::kernel_1a_c focus_lost_handler;


        timer<text_field>::kernel_2a t;

        // restricted functions
        text_field(text_field&);        // copy constructor
        text_field& operator=(text_field&);    // assignment operator


    protected:

        void draw (
            const canvas& c
        ) const;


        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        );

        void on_mouse_up (
            unsigned long btn,
            unsigned long state,
            long x,
            long y
        );

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        );

        void on_keydown (
            unsigned long key,
            bool is_printable,
            unsigned long state
        );

        void on_string_put (
            const std::wstring &str
        );
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class check_box
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class check_box : public toggle_button 
    {
    public:
        check_box(  
            drawable_window& w
        ) : toggle_button(w)
        {
            set_style(toggle_button_style_check_box());
        }

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class radio_button
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class radio_button : public toggle_button 
    {
    public:
        radio_button (  
            drawable_window& w
        ) : toggle_button(w)
        {
            set_style(toggle_button_style_radio_button());
        }

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class tabbed_display
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class tabbed_display : public drawable
    {
        /*!
            INITIAL VALUE
                - tabs.size() == 0
                - selected_tab_ == 0

            CONVENTION
                - number_of_tabs() == tabs.size()
                - tab_name(idx) == tabs[idx]
                - if (tabs.size() > 0) then
                    - selected_tab_ == the index of the tab that is currently selected

                - for all valid i:
                    - tabs[i].width == mfont->compute_size(tabs[i].name)
                    - tabs[i].rect == the rectangle that defines where this tab is
                    - if (tabs[i].group != 0) then
                        - tabs[i].group == a pointer to the widget_group for this tab.

                - left_pad == the amount of padding in a tab to the left of the name string.
                - right_pad == the amount of padding in a tab to the right of the name string.
                - top_pad == the amount of padding in a tab to the top of the name string.
                - bottom_pad == the amount of padding in a tab to the bottom of the name string.

                - if (event_handler.is_set()) then
                    - event_handler() is what is called to process click events
                      on this object.
        !*/

    public:

        tabbed_display(  
            drawable_window& w
        );

        virtual ~tabbed_display(
        );

        void set_size (
            unsigned long width,
            unsigned long height
        );

        void set_number_of_tabs (
            unsigned long num
        );

        unsigned long number_of_tabs (
        ) const;

        const std::string tab_name (
            unsigned long idx
        ) const;

        const std::wstring tab_wname (
            unsigned long idx
        ) const;

        const dlib::ustring& tab_uname (
            unsigned long idx
        ) const;

        void set_tab_name (
            unsigned long idx,
            const std::string& new_name
        );

        void set_tab_name (
            unsigned long idx,
            const std::wstring& new_name
        );

        void set_tab_name (
            unsigned long idx,
            const dlib::ustring& new_name
        );

        void set_pos (
            long x,
            long y
        );

        template <
            typename T
            >
        void set_click_handler (
            T& object,
            void (T::*eh)(unsigned long new_idx,unsigned long old_idx)
        )
        {
            auto_mutex M(m);
            event_handler.set(object,eh);
        }

        void set_tab_group (
            unsigned long idx,
            widget_group& group
        );

        void show (
        );

        void hide (
        );

        void enable (
        );

        void disable (
        );

        void set_main_font (
            const font* f
        );

        void fit_to_contents (
        );

    protected:
        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        );

        void draw (
            const canvas& c
        ) const;

    private:
        void recompute_tabs (
        );
        /*!
            ensures
                - recomputes the rectangles for all the tabs and makes this object
                  wider if needed
        !*/

        void draw_tab (
            const rectangle& tab,
            const canvas& c
        ) const;
        /*!
            ensures
                - draws the outline of a tab as given by the rectangle onto c
        !*/

        struct tab_data
        {
            tab_data() : width(0), group(0) {}

            dlib::ustring name;
            unsigned long width;
            rectangle rect;
            widget_group* group;
        };

        unsigned long selected_tab_;

        array<tab_data>::kernel_2a_c tabs;

        const long left_pad;
        const long right_pad;
        const long top_pad;
        const long bottom_pad;

        member_function_pointer<unsigned long,unsigned long>::kernel_1a event_handler;

        // restricted functions
        tabbed_display(tabbed_display&);        // copy constructor
        tabbed_display& operator=(tabbed_display&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class named_rectangle
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class named_rectangle : public drawable 
    {
        /*!
            INITIAL VALUE
                name == ""

            CONVENTION
                name_ == name()
        !*/

    public:

        named_rectangle(  
            drawable_window& w
        );

        virtual ~named_rectangle(
        );

        void set_size (
            unsigned long width,
            unsigned long height
        );

        void set_name (
            const std::string& name
        );

        void set_name (
            const std::wstring& name
        );

        void set_name (
            const dlib::ustring& name
        );

        const std::string name (
        ) const;

        const std::wstring wname (
        ) const;

        const dlib::ustring uname (
        ) const;

        void wrap_around (
            const rectangle& rect
        );

        void set_main_font (
            const font* f
        );

    protected:

        void draw (
            const canvas& c
        ) const;

    private:

        void make_name_fit_in_rect (
        );

        dlib::ustring name_;
        unsigned long name_width;
        unsigned long name_height;

        // restricted functions
        named_rectangle(named_rectangle&);        // copy constructor
        named_rectangle& operator=(named_rectangle&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class mouse_tracker
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class mouse_tracker : public dragable 
    {

    public:

        mouse_tracker(  
            drawable_window& w
        ); 

        ~mouse_tracker(
        );

        void show (
        );

        void hide (
        );

        void enable (
        );

        void disable (
        );

        void set_pos (
            long x,
            long y
        );

        void set_main_font (
            const font* f
        );

    protected:

        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        );

        void on_drag (
        );

        void draw (
            const canvas& c
        ) const;

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        );


    private:

        const long offset;
        named_rectangle nr;
        label x_label;
        label y_label; 
        std::ostringstream sout;

        long click_x, click_y;

        // restricted functions
        mouse_tracker(mouse_tracker&);        // copy constructor
        mouse_tracker& operator=(mouse_tracker&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // function message_box()  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace message_box_helper
    {
        class box_win : public drawable_window
        {
            void initialize (
            )
            {
                msg.set_pos(20,20);
                msg.set_text(message);
                rectangle msg_rect = msg.get_rect();
                btn_ok.set_name("OK");
                btn_ok.set_size(60,btn_ok.height());
                if (msg_rect.width() >= 60)
                    btn_ok.set_pos(msg_rect.width()/2+msg_rect.left()-btn_ok.width()/2,msg_rect.bottom()+15);
                else
                    btn_ok.set_pos(20,msg_rect.bottom()+15);
                btn_ok.set_click_handler(*this,&box_win::on_click);

                rectangle size = btn_ok.get_rect() + msg_rect;
                set_size(size.right()+20,size.bottom()+20);


                show();
                set_title(title);
            }
        public:
            box_win (
                const std::string& title_,
                const std::string& message_
            ) : 
                drawable_window(false),
                title(convert_mbstring_to_wstring(title_)),
                message(convert_mbstring_to_wstring(message_)),
                msg(*this),
                btn_ok(*this)
            {
                initialize();
            }
            box_win (
                const std::wstring& title_,
                const std::wstring& message_
            ) : 
                drawable_window(false),
                title(title_),
                message(message_),
                msg(*this),
                btn_ok(*this)
            {
                initialize();
            }
            box_win (
                const dlib::ustring& title_,
                const dlib::ustring& message_
            ) : 
                drawable_window(false),
                title(convert_utf32_to_wstring(title_)),
                message(convert_utf32_to_wstring(message_)),
                msg(*this),
                btn_ok(*this)
            {
                initialize();
            }

            ~box_win (
            )
            {
               close_window();
            }

            template <
                typename T
                >
            void set_click_handler (
                T& object,
                void (T::*event_handler_)()
            )
            {
                auto_mutex M(wm);
                event_handler.set(object,event_handler_);
            }

        private:

            static void deleter_thread (
                void* param
            )
            {
                // The point of this extra member function pointer stuff is to allow the user
                // to end the program from within the callback.  So we want to destroy the 
                // window *before* we call their callback.
                box_win& w = *reinterpret_cast<box_win*>(param);
                w.close_window();
                member_function_pointer<>::kernel_1a event_handler(w.event_handler);
                delete &w;
                if (event_handler.is_set())
                    event_handler();
            }

            void on_click (
            )
            {
                hide();
                create_new_thread(&deleter_thread,this);
            }

            on_close_return_code on_window_close (
            )
            {
                // The point of this extra member function pointer stuff is to allow the user
                // to end the program within the callback.  So we want to destroy the 
                // window *before* we call their callback. 
                member_function_pointer<>::kernel_1a event_handler_copy(event_handler);
                delete this;
                if (event_handler_copy.is_set())
                    event_handler_copy();
                return CLOSE_WINDOW;
            }

            const std::wstring title;
            const std::wstring message;
            label msg;
            button btn_ok;

            member_function_pointer<>::kernel_1a event_handler;
        };

        class blocking_box_win : public drawable_window
        {
            void initialize (
            )
            {
                msg.set_pos(20,20);
                msg.set_text(message);
                rectangle msg_rect = msg.get_rect();
                btn_ok.set_name("OK");
                btn_ok.set_size(60,btn_ok.height());
                if (msg_rect.width() >= 60)
                    btn_ok.set_pos(msg_rect.width()/2+msg_rect.left()-btn_ok.width()/2,msg_rect.bottom()+15);
                else
                    btn_ok.set_pos(20,msg_rect.bottom()+15);
                btn_ok.set_click_handler(*this,&blocking_box_win::on_click);

                rectangle size = btn_ok.get_rect() + msg_rect;
                set_size(size.right()+20,size.bottom()+20);


                set_title(title);
                show();
            }
        public:
            blocking_box_win (
                const std::string& title_,
                const std::string& message_
            ) : 
                drawable_window(false),
                title(convert_mbstring_to_wstring(title_)),
                message(convert_mbstring_to_wstring(message_)),
                msg(*this),
                btn_ok(*this)
            {
                initialize();
            }

            blocking_box_win (
                const std::wstring& title_,
                const std::wstring& message_
            ) : 
                drawable_window(false),
                title(title_),
                message(message_),
                msg(*this),
                btn_ok(*this)
            {
                initialize();
            }

            blocking_box_win (
                const dlib::ustring& title_,
                const dlib::ustring& message_
            ) : 
                drawable_window(false),
                title(convert_utf32_to_wstring(title_)),
                message(convert_utf32_to_wstring(message_)),
                msg(*this),
                btn_ok(*this)
            {
                initialize();
            }

            ~blocking_box_win (
            )
            {
                close_window();
            }

        private:

            void on_click (
            )
            {
                close_window();
            }

            const std::wstring title;
            const std::wstring message;
            label msg;
            button btn_ok;
        };
    }

    template <
        typename T
        >
    void message_box (
        const std::string& title,
        const std::string& message,
        T& object,
        void (T::*event_handler)() 
    )
    {
        using namespace message_box_helper;
        box_win* win = new box_win(title,message);
        win->set_click_handler(object,event_handler);
    }

    inline void message_box (
        const std::string& title,
        const std::string& message
    )
    {
        using namespace message_box_helper;
        new box_win(title,message);
    }

    inline void message_box_blocking (
        const std::string& title,
        const std::string& message
    )
    {
        using namespace message_box_helper;
        blocking_box_win w(title,message);
        w.wait_until_closed();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class list_box
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace list_box_helper{
    template <typename S = std::string>
    class list_box : public drawable, 
                     public enumerable<const S>
    {
        /*!
            INITIAL VALUE
                - ms_enabled == false
                - items.size() == 0
                - pos == 0
                - text_start = 0
                - last_selected = 0

            CONVENTION
                - size() == items.size()
                - (*this)[i] == items[i].name
                - is_selected(i) == items[i].is_selected

                - items[i].width == the width of items[i].name as given by font::compute_size() 
                - items[i].height == the height of items[i].name as given by font::compute_size() 

                - items[pos] == the item currently being displayed at the top of the list box
                - sbv == our vertical scroll bar
                - sbh == our horizontal scroll bar
                - text_area == the area that is free to be used for text display (e.g. not occluded 
                  by scroll bars or anything)
                - text_start == the amount of pixels the text should be shifted to the left (but the
                  part outside this widget should be clipped).  This is used by the horizontal 
                  scroll bar.
                - pos == the first line that is shown in the list box
                - last_selected == the last item the user selected
        !*/

    public:

        list_box(  
            drawable_window& w
        );

        ~list_box(
        );

        void set_size (
            unsigned long width_,
            unsigned long height_
        );

        void set_pos (
            long x,
            long y
        );

        bool is_selected (
            unsigned long index
        ) const;

        void select (
            unsigned long index 
        );

        void unselect (
            unsigned long index 
        );

        template <typename T>
        void get_selected (
            T& list
        ) const
        {
            auto_mutex M(m);
            list.clear();
            for (unsigned long i = 0; i < items.size(); ++i)
            {
                if (items[i].is_selected)
                {
                    unsigned long idx = i;
                    list.enqueue(idx);
                }
            }
        }

        template <typename T>
        void load (
            const T& list
        )
        {
            auto_mutex M(m);
            items.clear();
            unsigned long i = 0;
            items.set_max_size(list.size());
            items.set_size(list.size());
            list.reset();
            while (list.move_next())
            {
                items[i].is_selected = false;
                items[i].name = list.element();
                mfont->compute_size(items[i].name,items[i].width, items[i].height);
                ++i;
            }
            pos = 0;
            adjust_sliders();
            parent.invalidate_rectangle(rect);
            last_selected = 0;
        }

        const S& operator[] (
            unsigned long index
        ) const;

        bool multiple_select_enabled (
        ) const;

        void enable_multiple_select (
        ); 

        void disable_multiple_select (
        );

        template <
            typename T
            >
        void set_double_click_handler (
            T& object,
            void (T::*eh)(unsigned long index)
        ) { auto_mutex M(m); event_handler.set(object,eh); }

        template <
            typename T
            >
        void set_click_handler (
            T& object,
            void (T::*eh)(unsigned long index)
        ) { auto_mutex M(m); single_click_event_handler.set(object,eh); }

        bool at_start (
        ) const;

        void reset (
        ) const;

        bool current_element_valid (
        ) const;

        const S& element (
        ) const;

        const S& element (
        );

        bool move_next (
        ) const;

        unsigned long size (
        ) const;

        void show(
        );

        void hide (
        );

        void enable (
        );

        void disable (
        );

        void set_z_order (
            long order
        );

        unsigned long get_selected (
        ) const;

        void set_main_font (
            const font* f
        );

    private:

        void sbv_handler (
        );

        void sbh_handler (
        );

        void adjust_sliders (
        );
        /*!
            requires
                - m is locked
            ensures
                - adjusts the scroll bars so that they are properly displayed
        !*/

        void on_wheel_up (
            unsigned long state
        );

        void on_wheel_down (
            unsigned long state
        );

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        );

        void draw (
            const canvas& c
        ) const;

        template <typename SS>
        struct data
        {
            SS name;
            bool is_selected;
            unsigned long width;
            unsigned long height;
        };

        const static long pad = 2;

        bool ms_enabled;
        typename array<data<S> >::kernel_2a_c items;
        member_function_pointer<unsigned long>::kernel_1a event_handler;
        member_function_pointer<unsigned long>::kernel_1a single_click_event_handler;
        unsigned long pos;
        unsigned long text_start;
        unsigned long last_selected;
        scroll_bar sbv;
        scroll_bar sbh;
        rectangle text_area;


        // restricted functions
        list_box(list_box&);        // copy constructor
        list_box& operator=(list_box&);    // assignment operator
    };
    }
    typedef list_box_helper::list_box<std::string> list_box;
    typedef list_box_helper::list_box<std::wstring> wlist_box;
    typedef list_box_helper::list_box<dlib::ustring> ulist_box;
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // function open_file_box() 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace open_file_box_helper
    {
        class box_win : public drawable_window
        {
        public:
            box_win (
                const std::string& title,
                bool has_text_field = false
            ) : 
                lbl_dirs(*this),
                lbl_files(*this),
                lbl_file_name(*this),
                lb_dirs(*this),
                lb_files(*this),
                btn_ok(*this),
                btn_cancel(*this),
                btn_root(*this),
                tf_file_name(*this)
            {
                if (has_text_field == false)
                {
                    tf_file_name.hide();
                    lbl_file_name.hide();
                }
                else
                {
                    lbl_file_name.set_text("File: ");
                }

                cur_dir = -1;
                set_size(500,300);

                lbl_dirs.set_text("Directories:");
                lbl_files.set_text("Files:");
                btn_ok.set_name("Ok");
                btn_cancel.set_name("Cancel");
                btn_root.set_name("/");

                btn_root.set_click_handler(*this,&box_win::on_root_click);
                btn_cancel.set_click_handler(*this,&box_win::on_cancel_click);
                btn_ok.set_click_handler(*this,&box_win::on_open_click);
                lb_dirs.set_double_click_handler(*this,&box_win::on_dirs_click);
                lb_files.set_click_handler(*this,&box_win::on_files_click);
                lb_files.set_double_click_handler(*this,&box_win::on_files_double_click);


                btn_root.set_pos(5,5);

                set_sizes();
                set_title(title);

                on_root_click();

                // make it so that the file box starts out in our current working
                // directory
                std::string full_name(get_current_dir());

                while (full_name.size() > 0)
                {
                    std::string::size_type pos = full_name.find_first_of("\\/");
                    std::string left(full_name.substr(0,pos));
                    if (pos != std::string::npos)
                        full_name = full_name.substr(pos+1);
                    else
                        full_name.clear();

                    if (left.size() > 0)
                       enter_folder(left); 
                }

       
                show();
            }

            ~box_win (
            )
            {
               close_window();
            }

            template <
                typename T
                >
            void set_click_handler (
                T& object,
                void (T::*event_handler_)(const std::string&)
            )
            {
                auto_mutex M(wm);
                event_handler.set(object,event_handler_);
            }

        private:

            void set_sizes(
            )
            {
                unsigned long width, height;
                get_size(width,height);


                if (lbl_file_name.is_hidden())
                {
                    lbl_dirs.set_pos(0,btn_root.bottom()+5);
                    lb_dirs.set_pos(0,lbl_dirs.bottom());
                    lb_dirs.set_size(width/2,height-lb_dirs.top()-btn_cancel.height()-10);

                    lbl_files.set_pos(lb_dirs.right(),btn_root.bottom()+5);
                    lb_files.set_pos(lb_dirs.right(),lbl_files.bottom());
                    lb_files.set_size(width-lb_files.left(),height-lb_files.top()-btn_cancel.height()-10);

                    btn_ok.set_pos(width - btn_ok.width()-25,lb_files.bottom()+5);
                    btn_cancel.set_pos(btn_ok.left() - btn_cancel.width()-5,lb_files.bottom()+5);
                }
                else
                {

                    lbl_dirs.set_pos(0,btn_root.bottom()+5);
                    lb_dirs.set_pos(0,lbl_dirs.bottom());
                    lb_dirs.set_size(width/2,height-lb_dirs.top()-btn_cancel.height()-10-tf_file_name.height());

                    lbl_files.set_pos(lb_dirs.right(),btn_root.bottom()+5);
                    lb_files.set_pos(lb_dirs.right(),lbl_files.bottom());
                    lb_files.set_size(width-lb_files.left(),height-lb_files.top()-btn_cancel.height()-10-tf_file_name.height());

                    lbl_file_name.set_pos(lb_files.left(), lb_files.bottom()+8);
                    tf_file_name.set_pos(lbl_file_name.right(), lb_files.bottom()+5);
                    tf_file_name.set_width(width-tf_file_name.left()-5);

                    btn_ok.set_pos(width - btn_ok.width()-25,tf_file_name.bottom()+5);
                    btn_cancel.set_pos(btn_ok.left() - btn_cancel.width()-5,tf_file_name.bottom()+5);
                }

            }

            void on_window_resized (
            )
            {
                set_sizes();
            }

            void deleter_thread (
            ) 
            {  
                close_window();
                delete this; 
            }

            void enter_folder (
                const std::string& folder_name
            )
            {
                if (btn_root.is_checked())
                    btn_root.set_unchecked();
                if (cur_dir != -1)
                    sob[cur_dir]->set_unchecked();


                const std::string old_path = path;
                const long old_cur_dir = cur_dir;

                scoped_ptr<toggle_button> new_btn(new toggle_button(*this));
                new_btn->set_name(folder_name);
                new_btn->set_click_handler(*this,&box_win::on_path_button_click);

                // remove any path buttons that won't be part of the path anymore
                if (sob.size())
                {
                    while (sob.size() > (unsigned long)(cur_dir+1))
                    {
                        scoped_ptr<toggle_button> junk;
                        sob.remove(cur_dir+1,junk);
                    }
                }

                if (sob.size())
                    new_btn->set_pos(sob[sob.size()-1]->right()+5,sob[sob.size()-1]->top());
                else
                    new_btn->set_pos(btn_root.right()+5,btn_root.top());

                cur_dir = sob.size();
                sob.add(sob.size(),new_btn);

                path += folder_name + directory::get_separator();
                if (set_dir(prefix + path) == false)
                {
                    sob.remove(sob.size()-1,new_btn);
                    path = old_path;
                    cur_dir = old_cur_dir;
                }
                else
                {

                sob[cur_dir]->set_checked();
                }
            }

            void on_dirs_click (
                unsigned long idx
            )
            {
                enter_folder(lb_dirs[idx]);
            }

            void on_files_click (
                unsigned long idx
            )
            {
                if (tf_file_name.is_hidden() == false)
                {
                    tf_file_name.set_text(lb_files[idx]);
                }
            }

            void on_files_double_click (
                unsigned long 
            )
            {
                on_open_click();
            }

            void on_cancel_click (
            )
            {
                hide();
                create_new_thread<box_win,&box_win::deleter_thread>(*this);
            }

            void on_open_click (
            )
            {
                if (lb_files.get_selected() != lb_files.size() || tf_file_name.text().size() > 0)
                {
                    if (event_handler.is_set())
                    {
                        if (tf_file_name.is_hidden())
                            event_handler(prefix + path + lb_files[lb_files.get_selected()]);
                        else if (tf_file_name.text().size() > 0)
                            event_handler(prefix + path + tf_file_name.text());
                    }
                    hide();
                    create_new_thread<box_win,&box_win::deleter_thread>(*this);
                }
            }

            void on_path_button_click (
                toggle_button& btn
            )
            {
                if (btn_root.is_checked())
                    btn_root.set_unchecked();
                if (cur_dir != -1)
                    sob[cur_dir]->set_unchecked();
                std::string new_path;

                for (unsigned long i = 0; i < sob.size(); ++i)
                {
                    new_path += sob[i]->name() + directory::get_separator();
                    if (sob[i].get() == &btn)
                    {
                        cur_dir = i;
                        sob[i]->set_checked();
                        break;
                    }
                }
                if (path != new_path)
                {
                    path = new_path;
                    set_dir(prefix+path);
                }
            }

            struct case_insensitive_compare
            {
                bool operator() (
                    const std::string& a,
                    const std::string& b
                ) const
                {
                    std::string::size_type i, size;
                    size = std::min(a.size(),b.size());
                    for (i = 0; i < size; ++i)
                    {
                        if (std::tolower(a[i]) < std::tolower(b[i]))
                            return true;
                        else if (std::tolower(a[i]) > std::tolower(b[i]))
                            return false;
                    }
                    if (a.size() < b.size())
                        return true;
                    else
                        return false;
                }
            };

            bool set_dir (
                const std::string& dir
            )
            {
                try
                {
                    directory d(dir);
                    queue<directory>::kernel_1a_c qod;
                    queue<file>::kernel_1a_c qof;
                    queue<std::string>::sort_1a_c qos;
                    d.get_dirs(qod);
                    d.get_files(qof);

                    qod.reset();
                    while (qod.move_next())
                    {
                        std::string temp = qod.element().name();
                        qos.enqueue(temp);
                    }
                    qos.sort(case_insensitive_compare());
                    lb_dirs.load(qos);
                    qos.clear();

                    qof.reset();
                    while (qof.move_next())
                    {
                        std::string temp = qof.element().name();
                        qos.enqueue(temp);
                    }
                    qos.sort(case_insensitive_compare());
                    lb_files.load(qos);
                    return true;
                }
                catch (directory::listing_error& )
                {
                    return false;
                }
                catch (directory::dir_not_found&)
                {
                    return false;
                }
            }

            void on_root_click (
            )
            {
                btn_root.set_checked();
                if (cur_dir != -1)
                    sob[cur_dir]->set_unchecked();

                queue<directory>::kernel_1a_c qod, qod2;
                queue<file>::kernel_1a_c qof;
                queue<std::string>::sort_1a_c qos;
                get_filesystem_roots(qod);
                path.clear();
                cur_dir = -1;
                if (qod.size() == 1)
                {
                    qod.current().get_files(qof);
                    qod.current().get_dirs(qod2);
                    prefix = qod.current().full_name();

                    qod2.reset();
                    while (qod2.move_next())
                    {
                        std::string temp = qod2.element().name();
                        qos.enqueue(temp);
                    }
                    qos.sort(case_insensitive_compare());
                    lb_dirs.load(qos);
                    qos.clear();

                    qof.reset();
                    while (qof.move_next())
                    {
                        std::string temp = qof.element().name();
                        qos.enqueue(temp);
                    }
                    qos.sort(case_insensitive_compare());
                    lb_files.load(qos);
                }
                else
                {
                    prefix.clear();
                    qod.reset();
                    while (qod.move_next())
                    {
                        std::string temp = qod.element().full_name();
                        temp = temp.substr(0,temp.size()-1);
                        qos.enqueue(temp);
                    }
                    qos.sort(case_insensitive_compare());
                    lb_dirs.load(qos);
                    qos.clear();
                    lb_files.load(qos);
                }
            }

            on_close_return_code on_window_close (
            )
            {
                delete this;
                return CLOSE_WINDOW;
            }

            label lbl_dirs;
            label lbl_files;
            label lbl_file_name;
            list_box lb_dirs;
            list_box lb_files;
            button btn_ok;
            button btn_cancel;
            toggle_button btn_root;
            text_field tf_file_name;
            std::string path;
            std::string prefix;
            int cur_dir;

            member_function_pointer<const std::string&>::kernel_1a event_handler;
            sequence<scoped_ptr<toggle_button> >::kernel_2a_c sob;
        };
    }

    template <
        typename T
        >
    void open_file_box (
        T& object,
        void (T::*event_handler)(const std::string&) 
    )
    {
        using namespace open_file_box_helper;
        box_win* win = new box_win("Open File",true);
        win->set_click_handler(object,event_handler);
    }

    template <
        typename T
        >
    void open_existing_file_box (
        T& object,
        void (T::*event_handler)(const std::string&) 
    )
    {
        using namespace open_file_box_helper;
        box_win* win = new box_win("Open File");
        win->set_click_handler(object,event_handler);
    }

    template <
        typename T
        >
    void save_file_box (
        T& object,
        void (T::*event_handler)(const std::string&) 
    )
    {
        using namespace open_file_box_helper;
        box_win* win = new box_win("Save File",true);
        win->set_click_handler(object,event_handler);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // class menu_bar
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class menu_bar : public drawable
    {
        /*!
            INITIAL VALUE
                - menus.size() == 0
                - open_menu == 0 

            CONVENTION 
                - size() == menus.size() 
                - all menu data is stored in menus
                - menus[x].name == the name of the xth menu
                - if (menus[x].underline_pos != std::string::npos) then
                    - menus[x].underline_pos == the position of the character in the
                      menu name that should be underlined
                    - menus[x].underline_p1 != menus[x].underline_p2
                      and these two points define the underline bar
                - else
                    - menus[x].underline_p1 == menus[x].underline_p2
                - menus[x].menu == menu(x)
                - menus[x].rect == the rectangle in which menus[x].name is drawn
                - menus[x].bgrect == the rectangle for the xth menu button

                - if (there is an open menu on the screen) then
                    - open_menu == the index of the open menu from menus 
                - else
                    - open_menu == menus.size() 
        !*/

    public:
        menu_bar(
            drawable_window& w
        ) : 
            drawable(w, 0xFFFF), // listen for all events
            open_menu(0)
        {
            adjust_position();
            enable_events();
        }

        ~menu_bar()
        { disable_events(); parent.invalidate_rectangle(rect); }

        // this function does nothing
        void set_pos(long,long){}

        void set_main_font (
            const font* f
        )
        {
            auto_mutex M(m);
            mfont = f;
            adjust_position();
            compute_menu_geometry();
            parent.invalidate_rectangle(rect);
        }

        void set_number_of_menus (
            unsigned long num
        )
        {
            auto_mutex M(m);
            menus.set_max_size(num);
            menus.set_size(num);
            open_menu = menus.size();
            compute_menu_geometry();

            for (unsigned long i = 0; i < menus.size(); ++i)
            {
                menus[i].menu.set_on_hide_handler(*this,&menu_bar::on_popup_hide);
            }

            parent.invalidate_rectangle(rect);
        }

        unsigned long number_of_menus (
        ) const
        {
            auto_mutex M(m);
            return menus.size();
        }

        void set_menu_name (
            unsigned long idx,
            const std::string name,
            char underline_ch = '\0'
        )
        {
            set_menu_name(idx, convert_mbstring_to_wstring(name), underline_ch);
        }

        void set_menu_name (
            unsigned long idx,
            const std::wstring name,
            char underline_ch = '\0'
        )
        {
            set_menu_name(idx, convert_wstring_to_utf32(name), underline_ch);
        }

        void set_menu_name (
            unsigned long idx,
            const dlib::ustring name,
            char underline_ch = '\0'
        )
        {
            DLIB_ASSERT ( idx < number_of_menus() ,
                    "\tvoid menu_bar::set_menu_name()"
                    << "\n\tidx:               " << idx
                    << "\n\tnumber_of_menus(): " << number_of_menus() 
                    );
            auto_mutex M(m);
            menus[idx].name = name.c_str();
            menus[idx].underline_pos = name.find_first_of(underline_ch);
            compute_menu_geometry();
            parent.invalidate_rectangle(rect);
        }

        const std::string menu_name (
            unsigned long idx
        ) const
        {
            return convert_wstring_to_mbstring(menu_wname(idx));
        }

        const std::wstring menu_wname (
            unsigned long idx
        ) const
        {
            return convert_utf32_to_wstring(menu_uname(idx));
        }

        const dlib::ustring menu_uname (
            unsigned long idx
        ) const
        {
            DLIB_ASSERT ( idx < number_of_menus() ,
                    "\tstd::string menu_bar::menu_name()"
                    << "\n\tidx:               " << idx
                    << "\n\tnumber_of_menus(): " << number_of_menus() 
                    );
            auto_mutex M(m);
            return menus[idx].name.c_str();
        }

        popup_menu& menu (
            unsigned long idx
        )
        {
            DLIB_ASSERT ( idx < number_of_menus() ,
                    "\tpopup_menu& menu_bar::menu()"
                    << "\n\tidx:               " << idx
                    << "\n\tnumber_of_menus(): " << number_of_menus() 
                    );
            auto_mutex M(m);
            return menus[idx].menu;
        }

        const popup_menu& menu (
            unsigned long idx
        ) const
        {
            DLIB_ASSERT ( idx < number_of_menus() ,
                    "\tconst popup_menu& menu_bar::menu()"
                    << "\n\tidx:               " << idx
                    << "\n\tnumber_of_menus(): " << number_of_menus() 
                    );
            auto_mutex M(m);
            return menus[idx].menu;
        }

    protected:

        void on_window_resized (
        )
        {
            adjust_position();
            hide_menu();
        }

        void draw (
            const canvas& c
        ) const
        {
            rectangle area(rect.intersect(c));
            if (area.is_empty())
                return;

            const unsigned char opacity = 40;
            fill_rect_with_vertical_gradient(c, rect,rgb_alpha_pixel(255,255,255,opacity),
                                                    rgb_alpha_pixel(0,0,0,opacity));

            // first draw the border between the menu and the rest of the window
            draw_line(c, point(rect.left(),rect.bottom()-1), 
                      point(rect.right(),rect.bottom()-1), 100);
            draw_line(c, point(rect.left(),rect.bottom()), 
                      point(rect.right(),rect.bottom()), 255);

            // now draw all the menu buttons
            for (unsigned long i = 0; i < menus.size(); ++i)
            {
                mfont->draw_string(c,menus[i].rect, menus[i].name );
                if (menus[i].underline_p1 != menus[i].underline_p2)
                    draw_line(c, menus[i].underline_p1, menus[i].underline_p2);

                if (open_menu == i)
                {
                    fill_rect_with_vertical_gradient(c, menus[i].bgrect,rgb_alpha_pixel(255,255,0,40),  rgb_alpha_pixel(0,0,0,40));
                }
            }
        }

        void on_window_moved (
        )
        {
            hide_menu();
        }

        void on_focus_lost (
        )
        {
            hide_menu();
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long ,
            long x,
            long y,
            bool 
        )
        {

            if (rect.contains(x,y) == false || btn != (unsigned long)base_window::LEFT)
            {
                hide_menu();
                return;
            }

            unsigned long old_menu = menus.size();

            // if a menu is currently open then save its index
            if (open_menu != menus.size())
            {
                old_menu = open_menu;
                hide_menu();
            }

            // figure out which menu should be open if any
            for (unsigned long i = 0; i < menus.size(); ++i)
            {
                if (menus[i].bgrect.contains(x,y))
                {
                    if (old_menu != i)
                        show_menu(i);

                    break;
                }
            }

        }

        void on_mouse_move (
            unsigned long ,
            long x,
            long y
        )
        {
            // if the mouse is over the menu_bar and some menu is currently open
            if (rect.contains(x,y) && open_menu != menus.size())
            {
                // if the mouse is still in the same rectangle then don't do anything
                if (menus[open_menu].bgrect.contains(x,y) == false)
                {
                    // figure out which menu should be instead   
                    for (unsigned long i = 0; i < menus.size(); ++i)
                    {
                        if (menus[i].bgrect.contains(x,y))
                        {
                            show_menu(i);
                            break;
                        }
                    }

                }
            }
        }

        void on_keydown (
            unsigned long key,
            bool is_printable,
            unsigned long state
        )
        {
            if (state&base_window::KBD_MOD_ALT)
            {
                // check if the key matches any of our underlined keys
                for (unsigned long i = 0; i < menus.size(); ++i)
                {
                    // if we have found a matching key
                    if (is_printable && 
                        menus[i].underline_pos != std::string::npos &&
                        std::tolower(menus[i].name[menus[i].underline_pos]) == std::tolower(key))
                    {
                        show_menu(i);
                        menus[open_menu].menu.select_first_item();
                        return;
                    }
                }
            }
            
            if (open_menu != menus.size())
            {
                unsigned long i = open_menu;
                // if the submenu doesn't use this key for something then we will
                if (menus[open_menu].menu.forwarded_on_keydown(key,is_printable,state) == false)
                {
                    if (key == base_window::KEY_LEFT)
                    {
                        i = (i+menus.size()-1)%menus.size();
                        show_menu(i);
                        menus[open_menu].menu.select_first_item();
                    }
                    else if (key == base_window::KEY_RIGHT)
                    {
                        i = (i+1)%menus.size();
                        show_menu(i);
                        menus[open_menu].menu.select_first_item();
                    }
                    else if (key == base_window::KEY_ESC)
                    {
                        hide_menu();
                    }
                }
            }
        }

    private:

        void show_menu (
            unsigned long i
        )
        {
            rectangle temp;

            // menu already open so do nothing
            if (i == open_menu)
                return;

            // if a menu is currently open
            if (open_menu != menus.size())
            {
                menus[open_menu].menu.hide();
                temp = menus[open_menu].bgrect;
            }

            // display the new menu
            open_menu = i;
            long wx, wy;
            parent.get_pos(wx,wy);
            wx += menus[i].bgrect.left();
            wy += menus[i].bgrect.bottom()+1;
            menus[i].menu.set_pos(wx,wy);
            menus[i].menu.show();
            parent.invalidate_rectangle(menus[i].bgrect+temp);
        }

        void hide_menu (
        )
        {
            // if a menu is currently open
            if (open_menu != menus.size())
            {
                menus[open_menu].menu.hide();
                parent.invalidate_rectangle(menus[open_menu].bgrect);
                open_menu = menus.size();
            }
        }

        void on_popup_hide (
        )
        {
            // if a menu is currently open
            if (open_menu != menus.size())
            {
                parent.invalidate_rectangle(menus[open_menu].bgrect);
                open_menu = menus.size();
            }
        }

        struct menu_data
        {
            menu_data():underline_pos(dlib::ustring::npos){}

            dlib::ustring name;
            dlib::ustring::size_type underline_pos;
            popup_menu menu;
            rectangle rect;
            rectangle bgrect;
            point underline_p1;
            point underline_p2;
        };

        array<menu_data>::kernel_2a_c menus;
        unsigned long open_menu;

        // restricted functions
        menu_bar(menu_bar&);        // copy constructor
        menu_bar& operator=(menu_bar&);    // assignment operator

        void compute_menu_geometry (
        )
        {
            long x = 7;
            long bg_x = 0;
            for (unsigned long i = 0; i < menus.size(); ++i)
            {
                // compute the locations of the text rectangles
                menus[i].rect.set_top(5);
                menus[i].rect.set_left(x);
                menus[i].rect.set_bottom(rect.bottom()-2);

                unsigned long width, height;
                mfont->compute_size(menus[i].name,width,height);
                menus[i].rect = resize_rect_width(menus[i].rect, width);
                x = menus[i].rect.right()+10;

                menus[i].bgrect.set_top(0);
                menus[i].bgrect.set_left(bg_x);
                menus[i].bgrect.set_bottom(rect.bottom()-2);
                menus[i].bgrect.set_right(x-5);
                bg_x = menus[i].bgrect.right()+1;

                if (menus[i].underline_pos != std::string::npos)
                {
                    // now compute the location of the underline bar
                    rectangle r1 = mfont->compute_cursor_rect(
                        menus[i].rect, 
                        menus[i].name,
                        menus[i].underline_pos);

                    rectangle r2 = mfont->compute_cursor_rect(
                        menus[i].rect, 
                        menus[i].name,
                        menus[i].underline_pos+1);

                    menus[i].underline_p1.x() = r1.left()+1;
                    menus[i].underline_p2.x() = r2.left()-1;
                    menus[i].underline_p1.y() = r1.bottom()-mfont->height()+mfont->ascender()+2;
                    menus[i].underline_p2.y() = r2.bottom()-mfont->height()+mfont->ascender()+2;
                }
                else
                {
                    // there is no underline in this case
                    menus[i].underline_p1 = menus[i].underline_p2;
                }

            }
        }

        void adjust_position (
        )
        {
            unsigned long width, height;
            rectangle old(rect);
            parent.get_size(width,height);
            rect.set_left(0);
            rect.set_top(0);
            rect = resize_rect(rect,width,mfont->height()+10);
            parent.invalidate_rectangle(old+rect);
        }

    };

// ----------------------------------------------------------------------------------------

    template <typename graph_type>
    class directed_graph_drawer : public zoomable_region 
    {
        /*!
            INITIAL VALUE
                - edge_selected == false
                - mouse_drag == false
                - selected_node == 0
                - graph_.number_of_nodes() == 0
                - external_graph.number_of_nodes() == 0
                - radius == 25
                - last_mouse_click_in_display == false

            CONVENTION
                - radius == the radius of the nodes when they aren't zoomed
                - external_graph and graph_ have the same graph structure
                - external_graph == graph()
                - external_graph.node(i) == graph_node(i)

                - if (one of the nodes is selected) then
                    - selected_node < graph_.number_of_nodes()
                    - graph_.node(selected_node) == the selected node
                - else
                    - selected_node == graph_.number_of_nodes()

                - if (the user is dragging a node with the mouse) then
                    - mouse_drag == true
                    - drag_offset == the vector from the mouse position to the
                      center of the node
                - else
                    - mouse_drag == false

                - if (the user has selected an edge) then
                    - edge_selected == true
                    - the parent node is graph_.node(selected_edge_parent)
                    - the child node is graph_.node(selected_edge_parent)
                - else
                    - edge_selected == false

                - for all valid i:
                    - graph_.node(i).data.p == the center of the node in graph space
                    - graph_.node(i).data.name == node_label(i) 
                    - graph_.node(i).data.color == node_color(i) 
                    - graph_.node(i).data.str_rect == a rectangle sized to contain graph_.node(i).data.name

                - if (the last mouse click in our parent window as in our display_rect_ ) then
                    - last_mouse_click_in_display == true
                - else
                    - last_mouse_click_in_display == false
        !*/

    public:
        directed_graph_drawer (
            drawable_window& w
        ) :
            zoomable_region(w,MOUSE_CLICK | MOUSE_WHEEL | KEYBOARD_EVENTS),
            radius(25),
            edge_selected(false),
            last_mouse_click_in_display(false)
        {
            mouse_drag = false;
            selected_node = 0;

            // Whenever you make your own drawable (or inherit from dragable or button_action)
            // you have to remember to call this function to enable the events.  The idea
            // here is that you can perform whatever setup you need to do to get your 
            // object into a valid state without needing to worry about event handlers 
            // triggering before you are ready.
            enable_events();
        }

        ~directed_graph_drawer (
        )
        {
            // Disable all further events for this drawable object.  We have to do this 
            // because we don't want draw() events coming to this object while or after 
            // it has been destructed.
            disable_events();

            // Tell the parent window to redraw its area that previously contained this
            // drawable object.
            parent.invalidate_rectangle(rect);
        }

        void clear_graph (
        )
        {
            auto_mutex M(m);
            graph_.clear();
            external_graph.clear();
            parent.invalidate_rectangle(display_rect());
        }

        const typename graph_type::node_type& graph_node (
            unsigned long i
        )  const
        {
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\tgraph_type::node_type& directed_graph_drawer::graph_node(i)"
                    << "\n\ti:                 " << i 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            return external_graph.node(i);
        }

        typename graph_type::node_type& graph_node (
            unsigned long i
        ) 
        {
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\tgraph_type::node_type& directed_graph_drawer::graph_node(i)"
                    << "\n\ti:                 " << i 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            return external_graph.node(i);
        }

        const graph_type& graph (
        ) const
        {
            return external_graph;
        }

        void save_graph (
            std::ostream& out
        )
        {
            auto_mutex M(m);
            serialize(external_graph, out);
            serialize(graph_, out);
            parent.invalidate_rectangle(display_rect());
        }

        void load_graph (
            std::istream& in 
        )
        {
            auto_mutex M(m);
            deserialize(external_graph, in);
            deserialize(graph_, in);
            parent.invalidate_rectangle(display_rect());
        }

        unsigned long number_of_nodes (
        ) const
        {
            auto_mutex M(m);
            return graph_.number_of_nodes();
        }

        void set_node_label (
            unsigned long i,
            const std::string& label
        )
        {
            set_node_label(i, convert_mbstring_to_wstring(label));
        }

        void set_node_label (
            unsigned long i,
            const std::wstring& label
        )
        {
            set_node_label(i, convert_wstring_to_utf32(label));
        }

        void set_node_label (
            unsigned long i,
            const dlib::ustring& label
        )
        {
            auto_mutex M(m);
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\tvoid directed_graph_drawer::set_node_label(i,label)"
                    << "\n\ti:                 " << i 
                    << "\n\tlabel:             " << narrow(label) 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            graph_.node(i).data.name = label.c_str();
            unsigned long width, height;
            mfont->compute_size(label,width,height);
            graph_.node(i).data.str_rect = rectangle(width,height);
            parent.invalidate_rectangle(display_rect());
        }

        void set_node_color (
            unsigned long i,
            rgb_pixel color
        )
        {
            auto_mutex M(m);
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\tvoid directed_graph_drawer::set_node_color(i,label)"
                    << "\n\ti:                 " << i 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            graph_.node(i).data.color = color;
            parent.invalidate_rectangle(display_rect());
        }

        rgb_pixel node_color (
            unsigned long i
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\trgb_pixel directed_graph_drawer::node_color(i)"
                    << "\n\ti:                 " << i 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            return graph_.node(i).data.color;
        }

        const std::string node_label (
            unsigned long i
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\tconst std::ustring directed_graph_drawer::node_label(i)"
                    << "\n\ti:                 " << i 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            return narrow(graph_.node(i).data.name);
        }

        const std::wstring node_wlabel (
            unsigned long i
        ) const
        {
            return convert_utf32_to_wstring(node_ulabel(i));
        }

        const dlib::ustring node_ulabel (
            unsigned long i
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( i < number_of_nodes() ,
                    "\tconst std::ustring directed_graph_drawer::node_label(i)"
                    << "\n\ti:                 " << i 
                    << "\n\tnumber_of_nodes(): " << number_of_nodes() 
                    );
            return graph_.node(i).data.name.c_str();
        }

        template <
            typename T
            >
        void set_node_selected_handler (
            T& object,
            void (T::*event_handler_)(unsigned long)
        )
        {
            auto_mutex M(m);
            node_selected_handler.set(object,event_handler_);
        }

        template <
            typename T
            >
        void set_node_deselected_handler (
            T& object,
            void (T::*event_handler_)(unsigned long)
        )
        {
            auto_mutex M(m);
            node_deselected_handler.set(object,event_handler_);
        }

        template <
            typename T
            >
        void set_node_deleted_handler (
            T& object,
            void (T::*event_handler_)()
        )
        {
            auto_mutex M(m);
            node_deleted_handler.set(object,event_handler_);
        }

        template <
            typename T
            >
        void set_graph_modified_handler (
            T& object,
            void (T::*event_handler_)()
        )
        {
            auto_mutex M(m);
            graph_modified_handler.set(object,event_handler_);
        }

    protected:

        void on_keydown (
            unsigned long key,          
            bool is_printable,
            unsigned long state
        )
        {
            // ignore all keyboard input if the last thing the user clicked on 
            // wasn't the display area
            if (last_mouse_click_in_display == false)
                return;

            // if a node is selected
            if (selected_node != graph_.number_of_nodes())
            {
                // deselect the node if the user hits escape
                if (key == base_window::KEY_ESC)
                {
                    parent.invalidate_rectangle(display_rect());
                    if (node_deselected_handler.is_set())
                        node_deselected_handler(selected_node);
                    selected_node = graph_.number_of_nodes();
                }

                // delete the node if the user hits delete 
                if (key == base_window::KEY_DELETE || key == base_window::KEY_BACKSPACE)
                {
                    parent.invalidate_rectangle(display_rect());
                    graph_.remove_node(selected_node);
                    external_graph.remove_node(selected_node);
                    selected_node = graph_.number_of_nodes();
                    mouse_drag = false;
                    if (graph_modified_handler.is_set())
                        graph_modified_handler();
                    if (node_deleted_handler.is_set())
                        node_deleted_handler();
                }
            }

            // if an edge is selected
            if (edge_selected)
            {
                // deselect the node if the user hits escape
                if (key == base_window::KEY_ESC)
                {
                    parent.invalidate_rectangle(display_rect());
                    edge_selected = false;
                }

                // delete the node if the user hits delete 
                if (key == base_window::KEY_DELETE || key == base_window::KEY_BACKSPACE)
                {
                    parent.invalidate_rectangle(display_rect());
                    graph_.remove_edge(selected_edge_parent, selected_edge_child);
                    external_graph.remove_edge(selected_edge_parent, selected_edge_child);
                    edge_selected = false;

                    if (graph_modified_handler.is_set())
                        graph_modified_handler();
                }
            }
        }


        void on_mouse_move (
            unsigned long state,
            long x,
            long y
        )
        {
            if (mouse_drag)
            {
                const point p(nearest_point(display_rect(),point(x,y)));

                point center = drag_offset + p;
                graph_.node(selected_node).data.p = gui_to_graph_space(center);
                parent.invalidate_rectangle(display_rect());
            }
            else
            {
                zoomable_region::on_mouse_move(state,x,y);
            }

            // check if the mouse isn't being dragged anymore
            if ((state & base_window::LEFT) == 0)
            {
                mouse_drag = false;
            }
        }

        void on_mouse_up (
            unsigned long btn,
            unsigned long state,
            long x,
            long y
        )
        {
            mouse_drag = false;
            zoomable_region::on_mouse_up(btn,state,x,y);
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        )
        {
            bool redraw = false;

            if (display_rect().contains(x,y) && 
                (btn == base_window::RIGHT || btn == base_window::LEFT) && 
                (state & base_window::SHIFT) == 0 )
            {
                // start out saying no edge is selected
                if (edge_selected)
                {
                    edge_selected = false;
                    redraw = true;
                }

                bool click_hit_node = false;
                dlib::vector<double> p(gui_to_graph_space(point(x,y)));
                // check if this click is on an existing node
                for (unsigned long i = 0; i < graph_.number_of_nodes(); ++i)
                {
                    dlib::vector<double> n(graph_.node(i).data.p);
                    if ((p-n).length() < radius)
                    {
                        click_hit_node = true;
                        point center = graph_to_gui_space(graph_.node(i).data.p);
                        mouse_drag = true;
                        drag_offset = center - point(x,y);

                        // only do something if the click isn't on the currently
                        // selected node
                        if (selected_node != i)
                        {
                            // send out the deselected event if appropriate
                            if (selected_node != graph_.number_of_nodes() && node_deselected_handler.is_set())
                                node_deselected_handler(selected_node);

                            selected_node = i;
                            redraw = true;
                            if (node_selected_handler.is_set())
                                node_selected_handler(selected_node);
                        }
                        break;
                    }
                }

                // if the click didn't hit any node then make sure nothing is selected
                if (click_hit_node == false && selected_node != graph_.number_of_nodes())
                {
                    if (node_deselected_handler.is_set())
                        node_deselected_handler(selected_node);
                    selected_node = graph_.number_of_nodes();
                    redraw = true;
                }


                // check if this click is on an edge if we didn't click on a node
                if (click_hit_node == false)
                {
                    for (unsigned long n = 0; n < graph_.number_of_nodes() && edge_selected == false; ++n)
                    {
                        const dlib::vector<double> parent_center(graph_to_gui_space(graph_.node(n).data.p));
                        for (unsigned long e = 0; e < graph_.node(n).number_of_children() && edge_selected == false; ++e)
                        {
                            const dlib::vector<double> child_center(graph_to_gui_space(graph_.node(n).child(e).data.p));

                            rectangle area;
                            area += parent_center;
                            area += child_center;
                            // if the point(x,y) is between the two nodes then lets consider it further
                            if (area.contains(point(x,y)))
                            {
                                p = point(x,y);
                                const dlib::vector<double> z(0,0,1);
                                // find the distance from the line between the two nodes
                                const dlib::vector<double> perpendicular(z.cross(parent_center-child_center).normalize());
                                double distance = std::abs((child_center-p).dot(perpendicular));
                                if (distance < 8)
                                {
                                    edge_selected = true;
                                    selected_edge_parent = n;
                                    selected_edge_child = graph_.node(n).child(e).index();
                                    redraw = true;
                                }
                            }
                        }
                    }
                }


                // if the click didn't land on any node then add a new one if this was
                // a right mouse button click
                if (click_hit_node == false && btn == base_window::RIGHT)
                {
                    const unsigned long n = graph_.add_node();
                    external_graph.add_node();

                    graph_.node(n).data.p = gui_to_graph_space(point(x,y));

                    redraw = true;
                    selected_node = n;
                    mouse_drag = false;
                    if (graph_modified_handler.is_set())
                        graph_modified_handler();

                    if (node_selected_handler.is_set())
                        node_selected_handler(selected_node);

                }
                else if (selected_node == graph_.number_of_nodes())
                {
                    // in this case the click landed in the white area between nodes
                    zoomable_region::on_mouse_down( btn, state, x, y, is_double_click);
                }
            }

            // If the user is shift clicking with the mouse then see if we
            // should add a new edge.
            if (display_rect().contains(x,y) && 
                btn == base_window::LEFT && 
                (state & base_window::SHIFT) && 
                selected_node != graph_.number_of_nodes() )
            {
                dlib::vector<double> p(gui_to_graph_space(point(x,y)));
                // check if this click is on an existing node
                for (unsigned long i = 0; i < graph_.number_of_nodes(); ++i)
                {
                    dlib::vector<double> n(graph_.node(i).data.p);
                    if ((p-n).length() < radius)
                    {
                        // add the edge if it doesn't already exist and isn't an edge back to 
                        // the same node
                        if (graph_.has_edge(selected_node,i) == false && selected_node != i &&
                            graph_.has_edge(i, selected_node) == false)
                        {
                            graph_.add_edge(selected_node,i);
                            external_graph.add_edge(selected_node,i);
                            redraw = true;

                            if (graph_modified_handler.is_set())
                                graph_modified_handler();
                        }
                        break;
                    }
                }
            }


            if (redraw)
                parent.invalidate_rectangle(display_rect());


            if (display_rect().contains(x,y) == false)
                last_mouse_click_in_display = false;
            else
                last_mouse_click_in_display = true;
        }

        void draw (
            const canvas& c
        ) const
        {
            zoomable_region::draw(c);

            rectangle area = c.intersect(display_rect());
            if (area.is_empty() == true)
                return;


            if (enabled)
                fill_rect(c,display_rect(),255);
            else
                fill_rect(c,display_rect(),128);


            const unsigned long rad = static_cast<unsigned long>(radius*zoom_scale());
            point center;


            // first draw all the edges
            for (unsigned long i = 0; i < graph_.number_of_nodes(); ++i)
            {
                center = graph_to_gui_space(graph_.node(i).data.p);
                const rectangle circle_area(centered_rect(center,2*(rad+8),2*(rad+8)));

                // draw lines to all this node's parents 
                const dlib::vector<double> z(0,0,1);
                for (unsigned long j = 0; j < graph_.node(i).number_of_parents(); ++j)
                {
                    point p(graph_to_gui_space(graph_.node(i).parent(j).data.p));

                    rgb_pixel color(0,0,0);
                    // if this is the selected edge then draw it with red instead of black
                    if (edge_selected && selected_edge_child == i && selected_edge_parent == graph_.node(i).parent(j).index())
                    {
                        color.red = 255;
                        // we need to be careful when drawing this line to not draw it over the node dots since it 
                        // has a different color from them and would look weird
                        dlib::vector<double> v(p-center);
                        v = v.normalize()*rad;
                        draw_line(c,center+v,p-v ,color, area);
                    }
                    else
                    {
                        draw_line(c,center,p ,color, area);
                    }


                    // draw the triangle pointing to this node
                    if (area.intersect(circle_area).is_empty() == false)
                    {
                        dlib::vector<double> v(p-center);
                        v = v.normalize();

                        dlib::vector<double> cross = z.cross(v).normalize();
                        dlib::vector<double> r(center + v*rad);
                        for (double i = 0; i < 8*zoom_scale(); i += 0.1)
                            draw_line(c,(r+v*i)+cross*i, (r+v*i)-cross*i,color,area);
                    }
                }
            }


            // now draw all the node dots
            for (unsigned long i = 0; i < graph_.number_of_nodes(); ++i)
            {
                center = graph_to_gui_space(graph_.node(i).data.p);
                const rectangle circle_area(centered_rect(center,2*(rad+8),2*(rad+8)));

                // draw the actual dot for this node
                if (area.intersect(circle_area).is_empty()==false)
                {
                    rgb_alpha_pixel color;
                    assign_pixel(color, graph_.node(i).data.color);
                    // this node is in area so lets draw it and all of it's edges as well
                    draw_solid_circle(c,center,rad-3,color,area);
                    color.alpha = 240;
                    draw_circle(c,center,rad-3,color,area);
                    color.alpha = 200;
                    draw_circle(c,center,rad-2.5,color,area);
                    color.alpha = 160;
                    draw_circle(c,center,rad-2.0,color,area);
                    color.alpha = 120;
                    draw_circle(c,center,rad-1.5,color,area);
                    color.alpha = 80;
                    draw_circle(c,center,rad-1.0,color,area);
                    color.alpha = 40;
                    draw_circle(c,center,rad-0.5,color,area);

                }


                if (i == selected_node)
                    draw_circle(c,center,rad+5,rgb_pixel(0,0,255),area);
            }


            // now draw all the strings last
            for (unsigned long i = 0; i < graph_.number_of_nodes(); ++i)
            {
                center = graph_to_gui_space(graph_.node(i).data.p);
                rectangle circle_area(centered_rect(center,2*rad+3,2*rad+3));
                if (area.intersect(circle_area).is_empty()==false)
                {
                    rgb_pixel color = graph_.node(i).data.color;
                    // invert this color
                    color.red = 255-color.red;
                    color.green = 255-color.green;
                    color.blue = 255-color.blue;
                    sout << i;
                    unsigned long width, height;
                    mfont->compute_size(sout.str(),width,height);
                    rectangle str_rect(centered_rect(center, width,height));
                    if (circle_area.contains(str_rect))
                    {
                        mfont->draw_string(c,str_rect,sout.str(),color,0,std::string::npos,area);

                        // draw the label for this node if it isn't empty
                        if(graph_.node(i).data.name.size() > 0)
                        {
                            rectangle str_rect(graph_.node(i).data.str_rect);
                            str_rect = centered_rect(center.x(), center.y()-rad-mfont->height(),  str_rect.width(), str_rect.height());
                            mfont->draw_string(c,str_rect,graph_.node(i).data.name,0,0,std::string::npos,area);
                        }
                    }
                    sout.str("");
                }
            }
        }

    private:

        struct data
        {
            data() : color(0,0,0) {}
            vector<double> p;
            dlib::ustring name;
            rectangle str_rect;
            rgb_pixel color;
        };

        friend void serialize(const data& item, std::ostream& out)
        {
            serialize(item.p, out);
            serialize(item.name, out);
            serialize(item.str_rect, out);
            serialize(item.color, out);
        }

        friend void deserialize(data& item, std::istream& in)
        {
            deserialize(item.p, in);
            deserialize(item.name, in);
            deserialize(item.str_rect, in);
            deserialize(item.color, in);
        }

        mutable std::ostringstream sout;

        const double radius;
        unsigned long selected_node;
        bool mouse_drag; // true if the user is dragging a node
        point drag_offset; 

        bool edge_selected;
        unsigned long selected_edge_parent;
        unsigned long selected_edge_child;

        member_function_pointer<unsigned long>::kernel_1a node_selected_handler;
        member_function_pointer<unsigned long>::kernel_1a node_deselected_handler;
        member_function_pointer<>::kernel_1a node_deleted_handler;
        member_function_pointer<>::kernel_1a graph_modified_handler;

        graph_type external_graph;
        // rebind the graph_ type to make us a graph_ of data structs
        typename graph_type::template rebind<data,char, typename graph_type::mem_manager_type>::other graph_;

        bool last_mouse_click_in_display;
    };

// ----------------------------------------------------------------------------------------

    class text_grid : public scrollable_region 
    {
        /*!
            INITIAL VALUE
                - has_focus == false
                - vertical_scroll_increment() == 10
                - horizontal_scroll_increment() == 10
                - border_color_ == rgb_pixel(128,128,128)

            CONVENTION
                - grid.nr() == row_height.size()
                - grid.nc() == col_width.size()
                - border_color() == border_color_
                - text(r,c) == grid[r][c].text
                - text_color(r,c) == grid[r][c].text_color
                - background_color(r,c) == grid[r][c].bg_color

                - if (the user has clicked on this widget and caused one of the
                    boxes to have input focus) then
                    - has_focus == true
                    - grid[active_row][active_col] == the active text box
                    - cursor_pos == the position of the cursor in the above box
                    - if (the cursor should be displayed) then
                        - show_cursor == true
                    - else
                        - show_cursor == false
                - else
                    - has_focus == false
        !*/

    public:
        text_grid (
            drawable_window& w
        ) :
            scrollable_region(w, KEYBOARD_EVENTS | MOUSE_CLICK | FOCUS_EVENTS ),
            has_focus(false),
            cursor_timer(*this,&text_grid::timer_action),
            border_color_(128,128,128)
        {

            cursor_timer.set_delay_time(500);
            set_vertical_scroll_increment(10);
            set_horizontal_scroll_increment(10);
            enable_events();
        }

        ~text_grid (
        )
        {
            // Disable all further events for this drawable object.  We have to do this 
            // because we don't want draw() events coming to this object while or after 
            // it has been destructed.
            disable_events();

            // wait for the timer to stop doing its thing
            cursor_timer.stop_and_wait();
            // Tell the parent window to redraw its area that previously contained this
            // drawable object.
            parent.invalidate_rectangle(rect);
        }

        void set_grid_size (
            unsigned long rows,
            unsigned long cols
        )
        {
            auto_mutex M(m);
            row_height.set_max_size(rows);
            row_height.set_size(rows);

            col_width.set_max_size(cols);
            col_width.set_size(cols);

            grid.set_size(rows,cols);

            for (unsigned long i = 0; i < row_height.size(); ++i)
                row_height[i] = (mfont->height()*3)/2;
            for (unsigned long i = 0; i < col_width.size(); ++i)
                col_width[i] = mfont->height()*5;

            compute_total_rect();
            compute_bg_rects();
        }

        unsigned long number_of_columns (
        ) const
        {
            auto_mutex M(m);
            return grid.nc();
        }

        unsigned long number_of_rows (
        ) const
        {
            auto_mutex M(m);
            return grid.nr();
        }

        int next_free_user_event_number (
        ) const
        {
            return scrollable_region::next_free_user_event_number()+1;
        }

        rgb_pixel border_color (
        ) const
        {
            auto_mutex M(m);
            return border_color_;
        }

        void set_border_color (
            rgb_pixel color
        )
        {
            auto_mutex M(m);
            border_color_ = color;
            parent.invalidate_rectangle(rect);
        }

        const std::string text (
            unsigned long row,
            unsigned long col
        ) const
        {
            return convert_wstring_to_mbstring(wtext(row, col));
        }

        const std::wstring wtext (
            unsigned long row,
            unsigned long col
        ) const
        {
            return convert_utf32_to_wstring(utext(row, col));
        }

        const dlib::ustring utext (
            unsigned long row,
            unsigned long col
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tconst std::string text_grid::text(row,col)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            return grid[row][col].text.c_str();
        }

        void set_text (
            unsigned long row,
            unsigned long col,
            const std::string& str
        ) 
        {
            set_text(row, col, convert_mbstring_to_wstring(str));
        }

        void set_text (
            unsigned long row,
            unsigned long col,
            const std::wstring& str
        ) 
        {
            set_text(row, col, convert_wstring_to_utf32(str));
        }

        void set_text (
            unsigned long row,
            unsigned long col,
            const dlib::ustring& str
        ) 
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tvoid text_grid::set_text(row,col)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            grid[row][col].text = str.c_str();
            parent.invalidate_rectangle(get_text_rect(row,col));
        }

        const rgb_pixel text_color (
            unsigned long row,
            unsigned long col
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tconst rgb_pixel text_grid::text_color(row,col)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            return grid[row][col].text_color;
        }

        void set_text_color (
            unsigned long row,
            unsigned long col,
            const rgb_pixel color
        ) 
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tvoid text_grid::set_text_color(row,col,color)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            grid[row][col].text_color = color;
            parent.invalidate_rectangle(get_text_rect(row,col));
        }

        const rgb_pixel background_color (
            unsigned long row,
            unsigned long col
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tconst rgb_pixel text_grid::background_color(row,col,color)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            return grid[row][col].bg_color;
        }

        void set_background_color (
            unsigned long row,
            unsigned long col,
            const rgb_pixel color
        ) 
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tvoid text_grid::set_background_color(row,col,color)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            grid[row][col].bg_color = color;
            parent.invalidate_rectangle(get_bg_rect(row,col));
        }

        bool is_editable (
            unsigned long row,
            unsigned long col
        ) const
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tbool text_grid::is_editable(row,col)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\tthis:             " << this
                    );
            return grid[row][col].is_editable;
        }

        void set_editable (
            unsigned long row,
            unsigned long col,
            bool editable
        ) 
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows()  && col < number_of_columns(),
                    "\tvoid text_grid::set_editable(row,col,editable)"
                    << "\n\trow:              " << row 
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\teditable:         " << editable 
                    << "\n\tthis:             " << this
                    );
            grid[row][col].is_editable = editable;
            if (has_focus && active_row == static_cast<long>(row) && active_col == static_cast<long>(col))
            {
                drop_input_focus();
            }
        }

        void set_column_width (
            unsigned long col,
            unsigned long width
        )
        {
            auto_mutex M(m);
            DLIB_ASSERT ( col < number_of_columns(),
                    "\tvoid text_grid::set_column_width(col,width)"
                    << "\n\tcol:              " << col 
                    << "\n\tnumber_of_columns(): " << number_of_columns() 
                    << "\n\twidth:            " << width 
                    << "\n\tthis:             " << this
                    );
            col_width[col] = width;
            compute_total_rect();
            compute_bg_rects();
        }

        void set_row_height (
            unsigned long row,
            unsigned long height 
        )
        {
            auto_mutex M(m);
            DLIB_ASSERT ( row < number_of_rows() ,
                    "\tvoid text_grid::set_row_height(row,height)"
                    << "\n\trow:              " << row 
                    << "\n\tnumber_of_rows(): " << number_of_rows() 
                    << "\n\theight:           " << height 
                    << "\n\tthis:             " << this
                    );
            row_height[row] = height;
            compute_total_rect();
            compute_bg_rects();
        }

        void disable (
        ) 
        {
            auto_mutex M(m);
            scrollable_region::disable();
            drop_input_focus();
        }

        void hide (
        ) 
        {
            auto_mutex M(m);
            scrollable_region::hide();
            drop_input_focus();
        }

        template <
            typename T
            >
        void set_text_modified_handler (
            T& object,
            void (T::*eh)(unsigned long, unsigned long)
        ) { text_modified_handler.set(object,eh); }

    private:

        void on_user_event (
            int num
        )
        {
            // ignore this user event if it isn't for us
            if (num != scrollable_region::next_free_user_event_number())
                return;

            if (has_focus && !recent_cursor_move && enabled && !hidden)
            {
                show_cursor = !show_cursor;
                parent.invalidate_rectangle(get_text_rect(active_row,active_col));
            }
            recent_cursor_move = false;
        }

        void timer_action (
        ) { parent.trigger_user_event(this,scrollable_region::next_free_user_event_number()); }
        /*!
            ensures
                - flips the state of show_cursor 
        !*/

        void compute_bg_rects (
        )
        {
            // loop over each element in the grid and figure out what its rectangle should be
            // with respect to the total_rect()
            point p1, p2;
            p1.y() = total_rect().top();
            for (long row = 0; row < grid.nr(); ++row)
            {
                p1.x() = total_rect().left();
                p2.y() = p1.y() + row_height[row]-1;
                for (long col = 0; col < grid.nc(); ++col)
                {
                    // if this is the last box in this row make it super wide so that it always
                    // goes to the end of the widget
                    if (col+1 == grid.nc())
                        p2.x() = 1000000;
                    else
                        p2.x() = p1.x() + col_width[col]-1;

                    // at this point p1 is the upper left corner of this box and p2 is the 
                    // lower right corner of the box;
                    rectangle bg_rect(p1);
                    bg_rect += p2;

                    grid[row][col].bg_rect = translate_rect(bg_rect, -total_rect().left(), -total_rect().top());


                    p1.x() += 1 + col_width[col];
                }
                p1.y() += 1 + row_height[row];
            }
        }

        void compute_total_rect (
        )
        {
            if (grid.size() == 0)
            {
                set_total_rect_size(0,0);
            }
            else
            {
                unsigned long width = col_width.size()-1;
                unsigned long height = row_height.size()-1;

                for (unsigned long i = 0; i < col_width.size(); ++i)
                    width += col_width[i];
                for (unsigned long i = 0; i < row_height.size(); ++i)
                    height += row_height[i];

                set_total_rect_size(width,height);
            }
        }

        void on_keydown (
            unsigned long key,          
            bool is_printable,
            unsigned long state
        )
        {
            // ignore this event if we are disabled or hidden
            if (!enabled || hidden)
                return;

            if (has_focus)
            {
                if (is_printable)
                {
                    // if the user hit the tab key then jump to the next box
                    if (key == '\t')
                    {
                        if (active_col+1 == grid.nc())
                        {
                            if (active_row+1 == grid.nr())
                                move_cursor(0,0,0);
                            else
                                move_cursor(active_row+1,0,0);
                        }
                        else
                        {
                            move_cursor(active_row,active_col+1,0);
                        }
                    }
                    if (key == '\n')
                    {
                        // ignore the enter key
                    }
                    else if (grid[active_row][active_col].is_editable)
                    {
                        // insert the key the user pressed into the string
                        grid[active_row][active_col].text.insert(cursor_pos,1,static_cast<char>(key));
                        move_cursor(active_row,active_col,cursor_pos+1);

                        if (text_modified_handler.is_set())
                            text_modified_handler(active_row,active_col);
                    }
                }
                else if ((state & base_window::KBD_MOD_CONTROL))
                {
                    if (key == base_window::KEY_LEFT)
                        move_cursor(active_row,active_col-1,0);
                    else if (key == base_window::KEY_RIGHT)
                        move_cursor(active_row,active_col+1,0);
                    else if (key == base_window::KEY_UP)
                        move_cursor(active_row-1,active_col,0);
                    else if (key == base_window::KEY_DOWN)
                        move_cursor(active_row+1,active_col,0);
                    else if (key == base_window::KEY_END)
                        move_cursor(active_row,active_col,grid[active_row][active_col].text.size());
                    else if (key == base_window::KEY_HOME)
                        move_cursor(active_row,active_col,0);
                }
                else
                {
                    if (key == base_window::KEY_LEFT)
                        move_cursor(active_row,active_col,cursor_pos-1);
                    else if (key == base_window::KEY_RIGHT)
                        move_cursor(active_row,active_col,cursor_pos+1);
                    else if (key == base_window::KEY_UP)
                        move_cursor(active_row-1,active_col,0);
                    else if (key == base_window::KEY_DOWN)
                        move_cursor(active_row+1,active_col,0);
                    else if (key == base_window::KEY_END)
                        move_cursor(active_row,active_col,grid[active_row][active_col].text.size());
                    else if (key == base_window::KEY_HOME)
                        move_cursor(active_row,active_col,0);
                    else if (key == base_window::KEY_BACKSPACE)
                    {
                        if (cursor_pos > 0 && grid[active_row][active_col].is_editable)
                        {
                            grid[active_row][active_col].text.erase(
                                grid[active_row][active_col].text.begin()+cursor_pos-1,
                                grid[active_row][active_col].text.begin()+cursor_pos);
                            move_cursor(active_row,active_col,cursor_pos-1);

                            if (text_modified_handler.is_set())
                                text_modified_handler(active_row,active_col);
                        }
                    }
                    else if (key == base_window::KEY_DELETE)
                    {
                        if (cursor_pos < static_cast<long>(grid[active_row][active_col].text.size()) &&
                             grid[active_row][active_col].is_editable)
                        {
                            grid[active_row][active_col].text.erase(
                                grid[active_row][active_col].text.begin()+cursor_pos);
                            move_cursor(active_row,active_col,cursor_pos);

                            if (text_modified_handler.is_set())
                                text_modified_handler(active_row,active_col);
                        }
                    }
                }
            } // if (has_focus)
        }

        void on_mouse_down (
            unsigned long btn,
            unsigned long state,
            long x,
            long y,
            bool is_double_click
        )
        {
            if (display_rect().contains(x,y) && enabled && !hidden)
            {
                // figure out which box this click landed in
                rectangle hit;

                // find which column we hit
                unsigned long col = 0;
                long box_x = total_rect().left();
                for (unsigned long i = 0; i < col_width.size(); ++i)
                {
                    if (box_x <= x && (x < box_x+static_cast<long>(col_width[i]) || (i+1 == col_width.size())))
                    {
                        col = i;
                        hit.set_left(box_x);
                        hit.set_right(box_x+col_width[i]-1);
                        break;
                    }
                    else
                    {
                        box_x += col_width[i]+1;
                    }
                }

                // find which row we hit
                unsigned long row = 0;
                long box_y = total_rect().top();
                for (unsigned long i = 0; i < row_height.size(); ++i)
                {
                    if (box_y <= y && y < box_y+static_cast<long>(row_height[i]))
                    {
                        row = i;
                        hit.set_top(box_y);
                        hit.set_bottom(box_y+row_height[i]-1);
                        break;
                    }
                    else
                    {
                        box_y += row_height[i]+1;
                    }
                }

                // if we hit a box
                if (hit.is_empty() == false)
                {
                    move_cursor(row, 
                                col,
                                mfont->compute_cursor_pos(get_text_rect(row,col), grid[row][col].text, x, y, grid[row][col].first)
                    );
                }
                else
                {
                    drop_input_focus();
                }
            }
            else
            {
                drop_input_focus();
            }
        }

        void on_mouse_up (
            unsigned long btn,
            unsigned long state,
            long x,
            long y
        ) 
        {
        }

        void on_focus_lost (
        )
        {
            drop_input_focus();
        }

        void draw (
            const canvas& c
        ) const
        {
            scrollable_region::draw(c);
            rectangle area = c.intersect(display_rect());
            if (area.is_empty() == true)
                return;

            if (enabled)
                fill_rect(c, area, 255); 

            // don't do anything if the grid is empty
            if (grid.size() == 0)
                return;

            // draw all the vertical lines
            point p1, p2;
            p1.x() = p2.x() = total_rect().left();
            p1.y() = total_rect().top();
            p2.y() = total_rect().bottom();
            for (unsigned long i = 0; i < col_width.size()-1; ++i)
            {
                p1.x() += col_width[i];
                p2.x() += col_width[i];
                if (enabled)
                    draw_line(c,p1,p2,border_color_,area);
                else
                    draw_line(c,p1,p2,128,area);
                p1.x() += 1;
                p2.x() += 1;
            }

            // draw all the horizontal lines
            p1.y() = p2.y() = total_rect().top();
            p1.x() = display_rect().left();
            p2.x() = display_rect().right();
            for (unsigned long i = 0; i < row_height.size(); ++i)
            {
                p1.y() += row_height[i];
                p2.y() += row_height[i];
                if (enabled)
                    draw_line(c,p1,p2,border_color_,area);
                else
                    draw_line(c,p1,p2,128,area);
                p1.y() += 1;
                p2.y() += 1;
            }

            // draw the backgrounds and text for each box
            for (long row = 0; row < grid.nr(); ++row)
            {
                for (long col = 0; col < grid.nc(); ++col)
                {
                    rectangle bg_rect(get_bg_rect(row,col));

                    rectangle text_rect(get_text_rect(row,col));

                    if (enabled)
                    {
                        fill_rect(c,bg_rect.intersect(area),grid[row][col].bg_color);

                        mfont->draw_string(c,
                                           text_rect, 
                                           grid[row][col].text, 
                                           grid[row][col].text_color, 
                                           grid[row][col].first, 
                                           std::string::npos, 
                                           area);
                    }
                    else
                    {
                        mfont->draw_string(c,
                                           text_rect, 
                                           grid[row][col].text, 
                                           128, 
                                           grid[row][col].first, 
                                           std::string::npos, 
                                           area);
                    }

                    // if this box has input focus then draw it with a cursor
                    if (has_focus && active_col == col && active_row == row && show_cursor)
                    {
                        rectangle cursor_rect = mfont->compute_cursor_rect(text_rect,
                                                                           grid[row][col].text,
                                                                           cursor_pos,
                                                                           grid[row][col].first);
                        draw_rectangle(c,cursor_rect,0,area);
                    }

                }
            }


        }

        rectangle get_text_rect (
            unsigned long row,
            unsigned long col
        ) const
        {
            rectangle bg_rect(get_bg_rect(row,col));
            long padding = (bg_rect.height() - mfont->height())/2 + (bg_rect.height() - mfont->height())%2;
            if (padding < 0)
                padding = 0;
            bg_rect.set_left(bg_rect.left()+padding);
            bg_rect.set_top(bg_rect.top()+padding);
            bg_rect.set_right(bg_rect.right()-padding);
            bg_rect.set_bottom(bg_rect.bottom()-padding);
            return bg_rect;
        }

        rectangle get_bg_rect (
            unsigned long row,
            unsigned long col
        ) const
        {
            return translate_rect(grid[row][col].bg_rect, total_rect().left(), total_rect().top());
        }

        struct data_type
        {
            data_type(): text_color(0,0,0), bg_color(255,255,255),
            first(0), is_editable(true) 
            {}

            dlib::ustring text;
            rgb_pixel text_color;
            rgb_pixel bg_color;
            rectangle bg_rect;
            dlib::ustring::size_type first;
            bool is_editable;
        };

        void drop_input_focus (
        )
        {
            if (has_focus)
            {
                parent.invalidate_rectangle(get_text_rect(active_row,active_col));
                has_focus = false;
                show_cursor = false;
                cursor_timer.stop();
            }
        }

        void move_cursor (
            long row,
            long col,
            long new_cursor_pos
        )
        {
            // don't do anything if the grid is empty
            if (grid.size() == 0)
            {
                return;
            }

            if (row < 0)
                row = 0;
            if (row >= grid.nr())
                row = grid.nr()-1;
            if (col < 0)
                col = 0;
            if (col >= grid.nc())
                col = grid.nc()-1;

            if (new_cursor_pos < 0)
            {
                if (col == 0)
                {
                    new_cursor_pos = 0;
                }
                else 
                {
                    --col;
                    new_cursor_pos = grid[row][col].text.size();
                }
            }

            if (new_cursor_pos > static_cast<long>(grid[row][col].text.size()))
            {
                if (col+1 == grid.nc())
                {
                    new_cursor_pos = grid[row][col].text.size();
                }
                else 
                {
                    ++col;
                    new_cursor_pos = 0;
                }
            }

            // if some other box had the input focus then redraw it
            if (has_focus && (active_row != row || active_col != col ))
            {
                parent.invalidate_rectangle(get_text_rect(active_row,active_col));
            }

            if (has_focus == false)
            {
                cursor_timer.start();
            }

            has_focus = true;
            recent_cursor_move = true;
            show_cursor = true;
            active_row = row;
            active_col = col;
            cursor_pos = new_cursor_pos;

            // adjust the first character to draw so that the string is displayed well
            rectangle text_rect(get_text_rect(active_row,active_col));
            rectangle cursor_rect = mfont->compute_cursor_rect(text_rect,
                                                               grid[row][col].text,
                                                               cursor_pos,
                                                               grid[row][col].first);

            // if the cursor rect is too far to the left of the string
            if (cursor_pos < static_cast<long>(grid[row][col].first))
            {
                if (cursor_pos > 5)
                {
                    grid[row][col].first = cursor_pos - 5;
                }
                else
                {
                    grid[row][col].first = 0;
                }
            }
            // if the cursor rect is too far to the right of the string
            else if (cursor_rect.left() > text_rect.right())
            {
                long distance = (cursor_rect.left() - text_rect.right()) + text_rect.width()/3;
                // find the letter that is distance pixels from the start of the string
                long sum = 0;
                for (unsigned long i = grid[row][col].first; i < grid[row][col].text.size(); ++i)
                {
                    sum += (*mfont)[grid[row][col].text[i]].width();
                    if (sum >= distance)
                    {
                        grid[row][col].first = i;
                        break;
                    }
                }
            }

            scroll_to_rect(get_bg_rect(row,col));

            // redraw our box
            parent.invalidate_rectangle(text_rect);

        }

        array2d<data_type>::kernel_1a_c grid;
        array<unsigned long>::kernel_2a_c col_width;
        array<unsigned long>::kernel_2a_c row_height;
        bool has_focus;
        long active_col;
        long active_row;
        long cursor_pos;
        bool show_cursor;
        bool recent_cursor_move;
        timer<text_grid>::kernel_2a cursor_timer;
        rgb_pixel border_color_;
        member_function_pointer<unsigned long, unsigned long>::kernel_1a_c text_modified_handler;
    };

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "widgets.cpp"
#endif

#endif // DLIB_WIDGETs_

