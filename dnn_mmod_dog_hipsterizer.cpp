// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to run a CNN based dog face detector using dlib.  The
    example loads a pretrained model and uses it to find dog faces in images.
    We also use the dlib::shape_predictor to find the location of the eyes and
    nose and then draw glasses and a mustache onto each dog found :)

    Users who are just learning about dlib's deep learning API should read the
    dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples to learn how
    the API works.  For an introduction to the object detection method you
    should read dnn_mmod_ex.cpp

    TRAINING THE MODEL
        Finally, users interested in how the dog face detector was trained should
        read the dnn_mmod_ex.cpp example program.  It should be noted that the
        dog face detector used in this example uses a bigger training dataset and
        larger CNN architecture than what is shown in dnn_mmod_ex.cpp, but
        otherwise training is the same.  If you compare the net_type statements
        in this file and dnn_mmod_ex.cpp you will see that they are very similar
        except that the number of parameters has been increased.
        Additionally, the following training parameters were different during
        training: The following lines in dnn_mmod_ex.cpp were changed from
            mmod_options options(face_boxes_train, 40,40);
            trainer.set_iterations_without_progress_threshold(300);
        to the following when training the model used in this example:
            mmod_options options(face_boxes_train, 80,80);
            trainer.set_iterations_without_progress_threshold(8000);
        Also, the random_cropper was left at its default settings,  So we didn't
        call these functions:
            cropper.set_chip_dims(200, 200);
            cropper.set_min_object_size(40,40);
        The training data used to create the model is also available at
        http://dlib.net/files/data/CU_dogs_fully_labeled.tar.gz
        Lastly, the shape_predictor was trained with default settings except we
        used the following non-default settings: cascade depth=20, tree
        depth=5, padding=0.2
*/

#include <string>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <new>

using namespace std;
using namespace dlib;



// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

const unsigned MAX_SIZE = 2000*1500; //Based on bearface 
const unsigned CHIP_SIZE = 112;


// ----------------------------------------------------------------------------------------
// upscale image if it will remain under MAX_SIZE, return if scaled
// ----------------------------------------------------------------------------------------
bool upscale_image (matrix<rgb_pixel>& img) { //float& pxRatio
	//long origImgSize = img.size();
    if (img.size() < (MAX_SIZE/4)) //If expanding will remain under MAX_SIZE (*2 to both dims)
	{
	  cout << "upscaling..." << endl;
	  pyramid_up(img);
	  //pxRatio = sqrt (img.size () / origImgSize);
	  cout << "Upscaled image size: " << img.size() << endl;
	  return true;
	}
	return false;
}



int main(int argc, char** argv) try
{
    if (argc < 3)
    {
        cout << "Call this program like this:" << endl;
        cout << "./seal.exe seal.dat PHOTO_DIR photo1 photo2 ..." << endl;
        return 0;
    }

    // load the models
    net_type net;
    deserialize(argv[1]) >> net;

    int num_chips = 0;
    string curFolder = argv[2];
    string chipFolder = curFolder + "Chips";

    //go through and chip each image
    for (int i = 3; i < argc; ++i)
    {
        string curImg = argv[i];
        string curImageFile = curFolder + "/" + curImg;
        matrix<rgb_pixel> img;
        load_image(img, curImageFile);

        std::vector<dlib::mmod_rect> dets;
        try {
            cout << "try pyramid up " << curImageFile << endl;
            pyramid_up(img);
            dets = net(img);
        }
        catch(std::bad_alloc& ba) { //Catches bad alllocation (too big)
            cout << "bad alloc pyramid up" << endl;
            try {
                cout << "try pyramid down" << endl;
                load_image(img, curImageFile); //Reload image, smaller
                dets = net(img);
            }
            catch(std::bad_alloc& ba) {
                cout << "bad alloc pyramid down, skipping" << endl;
            }
        }

        for (auto&& d : dets)
        {
            cout << "num_chips " << num_chips << endl;
            // extract the face chip
            matrix<rgb_pixel> face_chip;
            cout << "d.rect... " << endl;
            chip_details face_chip_details = chip_details(d.rect, chip_dims(CHIP_SIZE, CHIP_SIZE)); //Optionally add angle
            int left = face_chip_details.rect.left(), top = face_chip_details.rect.top();
            int right = face_chip_details.rect.right(), bottom = face_chip_details.rect.bottom();
            extract_image_chip(img, face_chip_details, face_chip); //Img, rectangle for each chip, chip destination

            //remove the .jpg part of the curImg
            int dotIdx = curImg.find(".");
            if (dotIdx != std::string::npos) curImg = curImg.substr(0, dotIdx);

            // save the face chip
            // the name of the chipped photo will be in the format:
            // <count>_<original_photo>_ChippedAt_<top_left_coordinate_of_the_chipped_photo_within_the_original_photo>.jpeg
            // located in the folder <original_folder>Chips
            string location = "(" + to_string(left) + "," + to_string(top) + "),(" + to_string(right) + "," + to_string(bottom) + ")";
            string filename = to_string(num_chips) + "_" + curImg + "_ChippedAt_" + location + ".jpeg";
            string filedir = chipFolder + "/" + filename;
            num_chips++;
            save_jpeg(face_chip, filedir, 100); 
            cout << filename << " saved to " << chipFolder << endl;
        }
    }
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}
