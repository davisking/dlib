// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to train a CNN based object detector using dlib's 
    loss_mmod loss layer.  This loss layer implements the Max-Margin Object
    Detection loss as described in the paper:
        Max-Margin Object Detection by Davis E. King (http://arxiv.org/abs/1502.00046).
    This is the same loss used by the popular SVM+HOG object detector in dlib
    (see fhog_object_detector_ex.cpp) except here we replace the HOG features
    with a CNN and train the entire detector end-to-end.  This allows us to make
    much more powerful detectors.

    It would be a good idea to become familiar with dlib's DNN tooling before
    reading this example.  So you should read dnn_introduction_ex.cpp and
    dnn_introduction2_ex.cpp before reading this example program.
    
    Just like in the fhog_object_detector_ex.cpp example, we are going to train
    a simple face detector based on the very small training dataset in the
    examples/faces folder.  As we will see, even with this small dataset the
    MMOD method is able to make a working face detector.  However, for real
    applications you should train with more data for an even better result.
*/


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

// The first thing we do is define our CNN.  The CNN is going to be evaluated
// convolutionally over an entire image pyramid.  Think of it like a normal
// sliding window classifier.  This means you need to define a CNN that can look
// at some part of an image and decide if it is an object of interest.  In this
// example I've defined a CNN with a receptive field of a little over 50x50
// pixels.  This is reasonable for face detection since you can clearly tell if
// a 50x50 image contains a face.  Other applications may benefit from CNNs with
// different architectures.  
// 
// In this example our CNN begins with 3 downsampling layers.  These layers will
// reduce the size of the image by 8x and output a feature map with
// 32 dimensions.  Then we will pass that through 4 more convolutional layers to
// get the final output of the network.  The last layer has only 1 channel and
// the values in that last channel are large when the network thinks it has
// found an object at a particular location.


// Let's begin the network definition by creating some network blocks.

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    // In this example we are going to train a face detector based on the
    // small faces dataset in the examples/faces directory.  So the first
    // thing we do is load that dataset.  This means you need to supply the
    // path to this faces folder as a command line argument so we will know
    // where it is.
    if (argc != 2)
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./dnn_mmod_ex faces" << endl;
        cout << endl;
        return 0;
    }
    const std::string faces_directory = argv[1];
    // The faces directory contains a training dataset and a separate
    // testing dataset.  The training data consists of 4 images, each
    // annotated with rectangles that bound each human face.  The idea is 
    // to use this training data to learn to identify human faces in new
    // images.  
    // 
    // Once you have trained an object detector it is always important to
    // test it on data it wasn't trained on.  Therefore, we will also load
    // a separate testing set of 5 images.  Once we have a face detector
    // created from the training data we will see how well it works by
    // running it on the testing images. 
    // 
    // So here we create the variables that will hold our dataset.
    // images_train will hold the 4 training images and face_boxes_train
    // holds the locations of the faces in the training images.  So for
    // example, the image images_train[0] has the faces given by the
    // rectangles in face_boxes_train[0].
    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> face_boxes_train, face_boxes_test;

    // Now we load the data.  These XML files list the images in each dataset
    // and also contain the positions of the face boxes.  Obviously you can use
    // any kind of input format you like so long as you store the data into
    // images_train and face_boxes_train.  But for convenience dlib comes with
    // tools for creating and loading XML image datasets.  Here you see how to
    // load the data.  To create the XML files you can use the imglab tool which
    // can be found in the tools/imglab folder.  It is a simple graphical tool
    // for labeling objects in images with boxes.  To see how to use it read the
    // tools/imglab/README.txt file.
    load_image_dataset(images_train, face_boxes_train, faces_directory+"/training.xml");
    load_image_dataset(images_test, face_boxes_test, faces_directory+"/testing.xml");


    cout << "num training images: " << images_train.size() << endl;
    cout << "num testing images:  " << images_test.size() << endl;


    // The MMOD algorithm has some options you can set to control its behavior.  However,
    // you can also call the constructor with your training annotations and a "target
    // object size" and it will automatically configure itself in a reasonable way for your
    // problem.  Here we are saying that faces are still recognizably faces when they are
    // 40x40 pixels in size.  You should generally pick the smallest size where this is
    // true.  Based on this information the mmod_options constructor will automatically
    // pick a good sliding window width and height.  It will also automatically set the
    // non-max-suppression parameters to something reasonable.  For further details see the
    // mmod_options documentation.
    mmod_options options(face_boxes_train, 40*40);
    cout << "detection window width,height:      " << options.detector_width << "," << options.detector_height << endl;
    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;

    // Now we are ready to create our network and trainer.  
    net_type net(options);
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(5));
    trainer.set_iterations_without_progress_threshold(300);


    // Now let's train the network.  We are going to use mini-batches of 150
    // images.   The images are random crops from our training set (see
    // random_cropper_ex.cpp for a discussion of the random_cropper). 
    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels; 
    random_cropper cropper;
    cropper.set_chip_dims(200, 200);
    cropper.set_min_object_height(0.2);
    dlib::rand rnd;
    // Run the trainer until the learning rate gets small.  This will probably take several
    // hours.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        cropper(150, images_train, face_boxes_train, mini_batch_samples, mini_batch_labels);
        // We can also randomly jitter the colors and that often helps a detector
        // generalize better to new images.
        for (auto&& img : mini_batch_samples)
            disturb_colors(img, rnd);

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }
    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("mmod_network.dat") << net;


    // Now that we have a face detector we can test it.  The first statement tests it
    // on the training data.  It will print the precision, recall, and then average precision.
    // This statement should indicate that the network works perfectly on the
    // training data.
    cout << "training results: " << test_object_detection_function(net, images_train, face_boxes_train) << endl;
    // However, to get an idea if it really worked without overfitting we need to run
    // it on images it wasn't trained on.  The next line does this.   Happily,
    // this statement indicates that the detector finds most of the faces in the
    // testing data.
    cout << "testing results:  " << test_object_detection_function(net, images_test, face_boxes_test) << endl;

    // Now lets run the detector on the testing images and look at the outputs.  
    image_window win;
    for (auto&& img : images_test)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
    return 0;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




