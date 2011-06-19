// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "metadata_editor.h"
#include <dlib/array.h>
#include <dlib/queue.h>
#include <dlib/static_set.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

metadata_editor::
metadata_editor(
    const std::string& filename_
) : 
    filename(filename_),
    mbar(*this),
    lb_images(*this),
    image_pos(0)
{
    load_image_dataset_metadata(metadata, filename);

    dlib::array<std::string>::expand_1a files;
    files.resize(metadata.images.size());
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        files[i] = metadata.images[i].filename;
    }
    lb_images.load(files);
    lb_images.enable_multiple_select();

    select_image(0);
    lb_images.set_click_handler(*this, &metadata_editor::on_lb_images_clicked);


    mbar.set_number_of_menus(1);
    mbar.set_menu_name(0,"File",'F');


    mbar.menu(0).add_menu_item(menu_item_text("Save",*this,&metadata_editor::file_save,'S'));
    mbar.menu(0).add_menu_item(menu_item_text("Save As",*this,&metadata_editor::file_save_as,'A'));
    mbar.menu(0).add_menu_item(menu_item_separator());
    mbar.menu(0).add_menu_item(menu_item_text("Remove Selected Images",*this,&metadata_editor::remove_selected_images,'R'));
    mbar.menu(0).add_menu_item(menu_item_separator());
    mbar.menu(0).add_menu_item(menu_item_text("Quit",static_cast<base_window&>(*this),&drawable_window::close_window,'Q'));


    // set the size of this window
    set_size(430,380);

    on_window_resized();

    set_title("Image Dataset Metadata Editor");
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
    while (list.move_next())
    {
        lb_images.unselect(list.element());
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


    select_image(0);
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
        return;

    if (key == base_window::KEY_UP)
    {
        select_image(image_pos-1);
    }
    else if (key == base_window::KEY_DOWN)
    {
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
        image_pos = idx;
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_lb_images_clicked(
    unsigned long idx
) 
{ 
    image_pos = idx; 
}

// ----------------------------------------------------------------------------------------

