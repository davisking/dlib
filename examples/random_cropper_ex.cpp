// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    When you are training a convolutional neural network using the loss_mmod loss
    layer, you need to generate a bunch of identically sized training images.  The
    random_cropper is a convenient tool to help you crop out a bunch of
    identically sized images from a training dataset.

    This example shows you what it does exactly and talks about some of its options.
*/


#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "Give an image dataset XML file to run this program." << endl;
        cout << "For example, if you are running from the examples folder then run this program by typing" << endl;
        cout << "   ./random_cropper_ex faces/training.xml" << endl;
        cout << endl;
        return 0;
    }

    // First lets load a dataset
    std::vector<matrix<rgb_pixel>> images;
    std::vector<std::vector<mmod_rect>> boxes;
    load_image_dataset(images, boxes, argv[1]);

    // Here we make our random_cropper.  It has a number of options. 
    random_cropper cropper;
    // We can tell it how big we want the cropped images to be.
    cropper.set_chip_dims(400,400);
    // Also, when doing cropping, it will map the object annotations from the
    // dataset to the cropped image as well as perform random scale jittering.
    // You can tell it how much scale jittering you would like by saying "please
    // make the objects in the crops have a min and max size of such and such".
    // You do that by calling these two functions.  Here we are saying we want the
    // objects in our crops to be no more than 0.8*400 pixels in height and width.
    cropper.set_max_object_size(0.8);
    // And also that they shouldn't be too small. Specifically, each object's smallest
    // dimension (i.e. height or width) should be at least 60 pixels and at least one of
    // the dimensions must be at least 80 pixels.  So the smallest objects the cropper will
    // output will be either 80x60 or 60x80.
    cropper.set_min_object_size(80,60);
    // The cropper can also randomly mirror and rotate crops, which we ask it to
    // perform as well.
    cropper.set_randomly_flip(true);
    cropper.set_max_rotation_degrees(50);
    // This fraction of crops are from random parts of images, rather than being centered
    // on some object.
    cropper.set_background_crops_fraction(0.2);

    // Now ask the cropper to generate a bunch of crops.  The output is stored in
    // crops and crop_boxes.
    std::vector<matrix<rgb_pixel>> crops;
    std::vector<std::vector<mmod_rect>> crop_boxes;
    // Make 1000 crops.
    cropper(1000, images, boxes, crops, crop_boxes);

    // Finally, lets look at the results
    image_window win;
    for (size_t i = 0; i < crops.size(); ++i)
    {
        win.clear_overlay();
        win.set_image(crops[i]);
        for (auto b : crop_boxes[i])
        {
            // Note that mmod_rect has an ignore field.  If an object was labeled
            // ignore in boxes then it will still be labeled as ignore in
            // crop_boxes.  Moreover, objects that are not well contained within
            // the crop are also set to ignore.
            if (b.ignore)
                win.add_overlay(b.rect, rgb_pixel(255,255,0)); // draw ignored boxes as orange 
            else
                win.add_overlay(b.rect, rgb_pixel(255,0,0));   // draw other boxes as red
        }
        cout << "Hit enter to view the next random crop.";
        cin.get();
    }

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}





