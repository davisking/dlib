// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "metadata_editor.h"
#include <dlib/array.h>
#include <dlib/queue.h>
#include <dlib/static_set.h>
#include <dlib/misc_api.h>
#include <dlib/image_io.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <sstream>
#include <ctime>

using namespace std;
using namespace dlib;

extern const char* VERSION;

// ----------------------------------------------------------------------------------------

metadata_editor::
metadata_editor(
    const std::string& filename_
) : 
    mbar(*this),
    lb_images(*this),
    image_pos(0),
    display(*this),
    overlay_label_name(*this),
    overlay_label(*this),
    keyboard_jump_pos(0),
    last_keyboard_jump_pos_update(0)
{
    file metadata_file(filename_);
    filename = metadata_file.full_name();
    // Make our current directory be the one that contains the metadata file.  We 
    // do this because that file might contain relative paths to the image files
    // we are supposed to be loading.
    set_current_dir(get_parent_directory(metadata_file).full_name());

    load_image_dataset_metadata(metadata, filename);

    dlib::array<std::string>::expand_1a files;
    files.resize(metadata.images.size());
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        files[i] = metadata.images[i].filename;
    }
    lb_images.load(files);
    lb_images.enable_multiple_select();

    lb_images.set_click_handler(*this, &metadata_editor::on_lb_images_clicked);

    overlay_label_name.set_text("Next Label: ");
    overlay_label.set_width(200);

    display.set_image_clicked_handler(*this, &metadata_editor::on_image_clicked);
    display.set_overlay_rects_changed_handler(*this, &metadata_editor::on_overlay_rects_changed);
    display.set_overlay_rect_selected_handler(*this, &metadata_editor::on_overlay_rect_selected);
    overlay_label.set_text_modified_handler(*this, &metadata_editor::on_overlay_label_changed);

    mbar.set_number_of_menus(2);
    mbar.set_menu_name(0,"File",'F');
    mbar.set_menu_name(1,"Help",'H');


    mbar.menu(0).add_menu_item(menu_item_text("Save",*this,&metadata_editor::file_save,'S'));
    mbar.menu(0).add_menu_item(menu_item_text("Save As",*this,&metadata_editor::file_save_as,'A'));
    mbar.menu(0).add_menu_item(menu_item_separator());
    mbar.menu(0).add_menu_item(menu_item_text("Remove Selected Images",*this,&metadata_editor::remove_selected_images,'R'));
    mbar.menu(0).add_menu_item(menu_item_separator());
    mbar.menu(0).add_menu_item(menu_item_text("Exit",static_cast<base_window&>(*this),&drawable_window::close_window,'x'));

    mbar.menu(1).add_menu_item(menu_item_text("About",*this,&metadata_editor::display_about,'A'));

    // set the size of this window.
    on_window_resized();
    load_image_and_set_size(0);
    on_window_resized();
    if (image_pos < lb_images.size() )
        lb_images.select(image_pos);

    // make sure the window is centered on the screen.
    unsigned long width, height;
    get_size(width, height);
    unsigned long screen_width, screen_height;
    get_display_size(screen_width, screen_height);
    set_pos((screen_width-width)/2, (screen_height-height)/2);

    show();
} 

// ----------------------------------------------------------------------------------------

metadata_editor::
~metadata_editor(
)
{
    close_window();
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
add_labelable_part_name (
    const std::string& name
)
{
    display.add_labelable_part_name(name);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
file_save()
{
    save_metadata_to_file(filename);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
save_metadata_to_file (
    const std::string& file
)
{
    try
    {
        save_image_dataset_metadata(metadata, file);
    }
    catch (dlib::error& e)
    {
        message_box("Error saving file", e.what());
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
file_save_as()
{
    save_file_box(*this, &metadata_editor::save_metadata_to_file);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
remove_selected_images()
{
    dlib::queue<unsigned long>::kernel_1a list;
    lb_images.get_selected(list);
    list.reset();
    unsigned long min_idx = lb_images.size();
    while (list.move_next())
    {
        lb_images.unselect(list.element());
        min_idx = std::min(min_idx, list.element());
    }


    // remove all the selected items from metadata.images
    dlib::static_set<unsigned long>::kernel_1a to_remove;
    to_remove.load(list);
    std::vector<dlib::image_dataset_metadata::image> images;
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        if (to_remove.is_member(i) == false)
        {
            images.push_back(metadata.images[i]);
        }
    }
    images.swap(metadata.images);


    // reload metadata into lb_images
    dlib::array<std::string>::expand_1a files;
    files.resize(metadata.images.size());
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        files[i] = metadata.images[i].filename;
    }
    lb_images.load(files);


    if (min_idx != 0)
        min_idx--;
    select_image(min_idx);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_window_resized(
)
{
    drawable_window::on_window_resized();

    unsigned long width, height;
    get_size(width, height);

    lb_images.set_pos(0,mbar.bottom()+1);
    lb_images.set_size(180, height - mbar.height());

    overlay_label_name.set_pos(lb_images.right()+10, mbar.bottom() + (overlay_label.height()-overlay_label_name.height())/2+1);
    overlay_label.set_pos(overlay_label_name.right(), mbar.bottom()+1);
    display.set_pos(lb_images.right(), overlay_label.bottom()+3);

    display.set_size(width - display.left(), height - display.top());
}

// ----------------------------------------------------------------------------------------

void propagate_boxes(
    dlib::image_dataset_metadata::dataset& data,
    unsigned long prev,
    unsigned long next
)
{
    if (prev == next || next >= data.images.size())
        return;

    array2d<rgb_pixel> img1, img2;
    dlib::load_image(img1, data.images[prev].filename);
    dlib::load_image(img2, data.images[next].filename);
    for (unsigned long i = 0; i < data.images[prev].boxes.size(); ++i)
    {
        correlation_tracker tracker;
        tracker.start_track(img1, data.images[prev].boxes[i].rect);
        tracker.update(img2);
        dlib::image_dataset_metadata::box box = data.images[prev].boxes[i];
        box.rect = tracker.get_position();
        data.images[next].boxes.push_back(box);
    }
}

// ----------------------------------------------------------------------------------------

void propagate_labels(
    const std::string& label,
    dlib::image_dataset_metadata::dataset& data,
    unsigned long prev,
    unsigned long next
)
{
    if (prev == next || next >= data.images.size())
        return;


    for (unsigned long i = 0; i < data.images[prev].boxes.size(); ++i)
    {
        if (data.images[prev].boxes[i].label != label)
            continue;

        // figure out which box in the next image matches the current one the best
        const rectangle cur = data.images[prev].boxes[i].rect;
        double best_overlap = 0;
        unsigned long best_idx = 0;
        for (unsigned long j = 0; j < data.images[next].boxes.size(); ++j)
        {
            const rectangle next_box = data.images[next].boxes[j].rect;
            const double overlap = cur.intersect(next_box).area()/(double)(cur+next_box).area();
            if (overlap > best_overlap)
            {
                best_overlap = overlap;
                best_idx = j;
            }
        }

        // If we found a matching rectangle in the next image and the best match doesn't
        // already have a label.
        if (best_overlap > 0.5 && data.images[next].boxes[best_idx].label == "")
        {
            data.images[next].boxes[best_idx].label = label;
        }
    }

}

// ----------------------------------------------------------------------------------------

bool has_label_or_all_boxes_labeled (
    const std::string& label,
    const dlib::image_dataset_metadata::image& img 
)
{
    if (label.size() == 0)
        return true;

    bool all_boxes_labeled = true;
    for (unsigned long i = 0; i < img.boxes.size(); ++i)
    {
        if (img.boxes[i].label == label)
            return true;
        if (img.boxes[i].label.size() == 0)
            all_boxes_labeled = false;
    }

    return all_boxes_labeled;
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_keydown (
    unsigned long key,
    bool is_printable,
    unsigned long state
)
{
    drawable_window::on_keydown(key, is_printable, state);

    if (is_printable)
    {
        if (key == '\t')
        {
            overlay_label.give_input_focus();
            overlay_label.select_all_text();
        }

        // If the user types a number then jump to that image.
        if ('0' <= key && key <= '9' && metadata.images.size() != 0 && !overlay_label.has_input_focus())
        {
            time_t curtime = time(0);
            // If it's been a while since the user typed numbers then forget the last jump
            // position and start accumulating numbers over again.
            if (curtime-last_keyboard_jump_pos_update >= 2)
                keyboard_jump_pos = 0;
            last_keyboard_jump_pos_update = curtime;

            keyboard_jump_pos *= 10;
            keyboard_jump_pos += key-'0';
            if (keyboard_jump_pos >= metadata.images.size())
                keyboard_jump_pos = metadata.images.size()-1;

            image_pos = keyboard_jump_pos;
            select_image(image_pos);
        }
        else
        {
            last_keyboard_jump_pos_update = 0;
        }

        if (key == '=')
        {
            display.zoom_in();
        }

        if (key == '-')
        {
            display.zoom_out();
        }

        if (key == 'd' && (state&base_window::KBD_MOD_ALT))
        {
            remove_selected_images();
        }

        if (key == 'e' && !overlay_label.has_input_focus())
        {
            display_equialized_image = !display_equialized_image;
            select_image(image_pos);
        }

        // Make 'w' and 's' act like KEY_UP and KEY_DOWN
        if ((key == 'w' || key == 'W') && !overlay_label.has_input_focus())
        {
            key = base_window::KEY_UP;
        }
        else if ((key == 's' || key == 'S') && !overlay_label.has_input_focus())
        {
            key = base_window::KEY_DOWN;
        }
        else
        {
            return;
        }
    }

    if (key == base_window::KEY_UP)
    {
        if ((state&KBD_MOD_CONTROL) && (state&KBD_MOD_SHIFT))
        {
            // Don't do anything if there are no boxes in the current image.
            if (metadata.images[image_pos].boxes.size() == 0)
                return;
            // Also don't do anything if there *are* boxes in the next image.
            if (image_pos > 1 && metadata.images[image_pos-1].boxes.size() != 0)
                return;

            propagate_boxes(metadata, image_pos, image_pos-1);
        }
        else if (state&base_window::KBD_MOD_CONTROL)
        {
            // If the label we are supposed to propagate doesn't exist in the current image
            // then don't advance.
            if (!has_label_or_all_boxes_labeled(display.get_default_overlay_rect_label(),metadata.images[image_pos]))
                return;

            // if the next image is going to be empty then fast forward to the next one
            while (image_pos > 1 && metadata.images[image_pos-1].boxes.size() == 0)
                --image_pos;

            propagate_labels(display.get_default_overlay_rect_label(), metadata, image_pos, image_pos-1);
        }
        select_image(image_pos-1);
    }
    else if (key == base_window::KEY_DOWN)
    {
        if ((state&KBD_MOD_CONTROL) && (state&KBD_MOD_SHIFT))
        {
            // Don't do anything if there are no boxes in the current image.
            if (metadata.images[image_pos].boxes.size() == 0)
                return;
            // Also don't do anything if there *are* boxes in the next image.
            if (image_pos+1 < metadata.images.size() && metadata.images[image_pos+1].boxes.size() != 0)
                return;

            propagate_boxes(metadata, image_pos, image_pos+1);
        }
        else if (state&base_window::KBD_MOD_CONTROL)
        {
            // If the label we are supposed to propagate doesn't exist in the current image
            // then don't advance.
            if (!has_label_or_all_boxes_labeled(display.get_default_overlay_rect_label(),metadata.images[image_pos]))
                return;

            // if the next image is going to be empty then fast forward to the next one
            while (image_pos+1 < metadata.images.size() && metadata.images[image_pos+1].boxes.size() == 0)
                ++image_pos;

            propagate_labels(display.get_default_overlay_rect_label(), metadata, image_pos, image_pos+1);
        }
        select_image(image_pos+1);
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
select_image(
    unsigned long idx
)
{
    if (idx < lb_images.size())
    {
        // unselect all currently selected images
        dlib::queue<unsigned long>::kernel_1a list;
        lb_images.get_selected(list);
        list.reset();
        while (list.move_next())
        {
            lb_images.unselect(list.element());
        }


        lb_images.select(idx);
        load_image(idx);
    }
    else if (lb_images.size() == 0)
    {
        display.clear_overlay();
        array2d<unsigned char> empty_img;
        display.set_image(empty_img);
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_lb_images_clicked(
    unsigned long idx
) 
{ 
    load_image(idx);
}

// ----------------------------------------------------------------------------------------

std::vector<dlib::image_display::overlay_rect> get_overlays (
    const dlib::image_dataset_metadata::image& data,
    color_mapper& string_to_color 
)
{
    std::vector<dlib::image_display::overlay_rect> temp(data.boxes.size());
    for (unsigned long i = 0; i < temp.size(); ++i)
    {
        temp[i].rect = data.boxes[i].rect;
        temp[i].label = data.boxes[i].label;
        temp[i].parts = data.boxes[i].parts;
        temp[i].crossed_out = data.boxes[i].ignore;
        temp[i].color = string_to_color(data.boxes[i].label);
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
load_image(
    unsigned long idx
)
{
    if (idx >= metadata.images.size())
        return;

    image_pos = idx; 

    array2d<rgb_pixel> img;
    display.clear_overlay();
    try
    {
        dlib::load_image(img, metadata.images[idx].filename);
        set_title(metadata.name + " #"+cast_to_string(idx)+": " +metadata.images[idx].filename);
    }
    catch (exception& e)
    {
        message_box("Error loading image", e.what());
    }

    if (display_equialized_image)
        equalize_histogram(img);
    display.set_image(img);
    display.add_overlay(get_overlays(metadata.images[idx], string_to_color));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
load_image_and_set_size(
    unsigned long idx
)
{
    if (idx >= metadata.images.size())
        return;

    image_pos = idx; 

    array2d<rgb_pixel> img;
    display.clear_overlay();
    try
    {
        dlib::load_image(img, metadata.images[idx].filename);
        set_title(metadata.name + " #"+cast_to_string(idx)+": " +metadata.images[idx].filename);
    }
    catch (exception& e)
    {
        message_box("Error loading image", e.what());
    }


    unsigned long screen_width, screen_height;
    get_display_size(screen_width, screen_height);


    unsigned long needed_width = display.left() + img.nc() + 4;
    unsigned long needed_height = display.top() + img.nr() + 4;
	if (needed_width < 300) needed_width = 300;
	if (needed_height < 300) needed_height = 300;

    if (needed_width > 100 + screen_width)
        needed_width = screen_width - 100;
    if (needed_height > 100 + screen_height)
        needed_height = screen_height - 100;

    set_size(needed_width, needed_height);


    if (display_equialized_image)
        equalize_histogram(img);
    display.set_image(img);
    display.add_overlay(get_overlays(metadata.images[idx], string_to_color));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_overlay_rects_changed(
)
{
    using namespace dlib::image_dataset_metadata;
    if (image_pos < metadata.images.size())
    {
        const std::vector<image_display::overlay_rect>& rects = display.get_overlay_rects();

        std::vector<box>& boxes = metadata.images[image_pos].boxes;

        boxes.clear();
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            box temp;
            temp.label = rects[i].label;
            temp.rect = rects[i].rect;
            temp.parts = rects[i].parts;
            temp.ignore = rects[i].crossed_out;
            boxes.push_back(temp);
        }
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_image_clicked(
    const point& /*p*/, bool /*is_double_click*/, unsigned long /*btn*/
)
{
    display.set_default_overlay_rect_color(string_to_color(trim(overlay_label.text())));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_overlay_label_changed(
)
{
    display.set_default_overlay_rect_label(trim(overlay_label.text()));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_overlay_rect_selected(
    const image_display::overlay_rect& orect
)
{
    overlay_label.set_text(orect.label);
    display.set_default_overlay_rect_label(orect.label);
    display.set_default_overlay_rect_color(string_to_color(orect.label));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
display_about(
)
{
    std::ostringstream sout;
    sout << wrap_string("Image Labeler v" + string(VERSION) + "." ,0,0) << endl << endl;
    sout << wrap_string("This program is a tool for labeling images with rectangles. " ,0,0) << endl << endl;

    sout << wrap_string("You can add a new rectangle by holding the shift key, left clicking "
                        "the mouse, and dragging it.  New rectangles are given the label from the \"Next Label\" "
                        "field at the top of the application.  You can quickly edit the contents of the Next Label field "
                        "by hitting the tab key. Double clicking "
                        "a rectangle selects it and the delete key removes it.  You can also mark "
                        "a rectangle as ignored by hitting the i or END keys when it is selected.  Ignored "
                        "rectangles are visually displayed with an X through them.  You can remove an image "
                        "entirely by selecting it in the list on the left and pressing alt+d."
                        ,0,0) << endl << endl;

    sout << wrap_string("It is also possible to label object parts by selecting a rectangle and "
                        "then right clicking.  A popup menu will appear and you can select a part label. "
                        "Note that you must define the allowable part labels by giving --parts on the "
                        "command line.  An example would be '--parts \"leye reye nose mouth\"'. "
                        "Alternatively, if you don't give --parts you can simply select a rectangle and shift+left "
                        "click to add parts. Parts added this way will be labeled with integer labels starting from 0. "
                        "You can only use this simpler part adding mode if all the parts in a rectangle are already "
                        "labeled with integer labels or the rectangle has no parts at all."
                        ,0,0) << endl << endl;

    sout << wrap_string("Press the down or s key to select the next image in the list and the up or w "
                        "key to select the previous one.",0,0) << endl << endl;

    sout << wrap_string("Additionally, you can hold ctrl and then scroll the mouse wheel to zoom.  A normal left click "
                        "and drag allows you to navigate around the image.  Holding ctrl and "
                        "left clicking a rectangle will give it the label from the Next Label field. "
                        "Holding shift + right click and then dragging allows you to move things around. "
                        "Holding ctrl and pressing the up or down keyboard keys will propagate "
                        "rectangle labels from one image to the next and also skip empty images. " 
                        "Similarly, holding ctrl+shift will propagate entire boxes via a visual tracking " 
                        "algorithm from one image to the next. "
                        "Finally, typing a number on the keyboard will jump you to a specific image.",0,0) << endl << endl;

    sout << wrap_string("You can also toggle image histogram equalization by pressing the e key."
                        ,0,0) << endl;


    message_box("About Image Labeler",sout.str());
}

// ----------------------------------------------------------------------------------------

