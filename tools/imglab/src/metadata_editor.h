// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_METADATA_EdITOR_H__
#define DLIB_METADATA_EdITOR_H__

#include <dlib/gui_widgets.h>
#include <dlib/data_io.h>
#include <dlib/pixel.h>
#include <map>

// ----------------------------------------------------------------------------------------

class color_mapper
{
public:

    dlib::rgb_alpha_pixel operator() (
        const std::string& str
    ) 
    {
        auto i = colors.find(str);
        if (i != colors.end())
        {
            return i->second;
        }
        else
        {
            using namespace dlib;
            hsi_pixel pix;
            pix.h = reverse(colors.size());
            pix.s = 255;
            pix.i = 150;
            rgb_alpha_pixel result;
            assign_pixel(result, pix);
            colors[str] = result;
            return result;
        }
    }

private:

    // We use a bit reverse here because it causes us to evenly spread the colors as we
    // allocated them. First the colors are maximally different, then become interleaved
    // and progressively more similar as they are allocated.
    unsigned char reverse(unsigned char b)
    {
        // reverse the order of the bits in b.
        b = ((b * 0x0802LU & 0x22110LU) | (b * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16; 
        return b;
    }

    std::map<std::string, dlib::rgb_alpha_pixel> colors;
};

// ----------------------------------------------------------------------------------------

class metadata_editor : public dlib::drawable_window 
{
public:
    metadata_editor(
        const std::string& filename_
    );

    ~metadata_editor();

    void add_labelable_part_name (
        const std::string& name
    );

private:

    void file_save();
    void file_save_as();
    void remove_selected_images();

    virtual void on_window_resized();
    virtual void on_keydown (
        unsigned long key,
        bool is_printable,
        unsigned long state
    );

    void on_lb_images_clicked(unsigned long idx); 
    void select_image(unsigned long idx);
    void save_metadata_to_file (const std::string& file);
    void load_image(unsigned long idx);
    void load_image_and_set_size(unsigned long idx);
    void on_image_clicked(const dlib::point& p, bool is_double_click, unsigned long btn);
    void on_overlay_rects_changed();
    void on_overlay_label_changed();
    void on_overlay_rect_selected(const dlib::image_display::overlay_rect& orect);

    void display_about();

    std::string filename;
    dlib::image_dataset_metadata::dataset metadata;

    dlib::menu_bar mbar;
    dlib::list_box lb_images;
    unsigned long image_pos;

    dlib::image_display display;
    dlib::label overlay_label_name;
    dlib::text_field overlay_label;

    unsigned long keyboard_jump_pos;
    time_t last_keyboard_jump_pos_update;
    bool display_equialized_image = false;
    color_mapper string_to_color;
};

// ----------------------------------------------------------------------------------------


#endif // DLIB_METADATA_EdITOR_H__

