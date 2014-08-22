// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*



    The pose estimator was created by using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014

*/


#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        // In this example we are going to train a shape_predictor based on the
        // small faces dataset in the examples/faces directory.  So the first
        // thing we do is load that dataset.  This means you need to supply the
        // path to this faces folder as a command line argument so we will know
        // where it is.
        if (argc != 2)
        {
            cout << "Give the path to the examples/faces directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./train_shape_predictor_ex faces" << endl;
            cout << endl;
            return 0;
        }
        const std::string faces_directory = argv[1];
        // The faces directory contains a training dataset and a separate
        // testing dataset.  The training data consists of 4 images, each
        // annotated with rectangles that bound each human face along with 68
        // face landmarks on each face.  The idea is to use this training data
        // to learn to identify the position of landmarks on human faces in new
        // images. 
        // 
        // Once you have trained a shape_predictor it is always important to
        // test it on data it wasn't trained on.  Therefore, we will also load
        // a separate testing set of 5 images.  Once we have a shape_predictor 
        // created from the training data we will see how well it works by
        // running it on the testing images. 
        // 
        // So here we create the variables that will hold our dataset.
        // images_train will hold the 4 training images and face_boxes_train
        // holds the locations of the faces in the training images.  So for
        // example, the image images_train[0] has the faces given by the
        // full_object_detections in face_boxes_train[0].
        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > faces_train, faces_test;

        // Now we load the data.  These XML files list the images in each
        // dataset and also contain the positions of the face boxes and landmark
        // (called parts in the XML file).  Obviously you can use any kind of
        // input format you like so long as you store the data into images_train
        // and faces_train.  
        load_image_dataset(images_train, faces_train, faces_directory+"/training_with_face_landmarks.xml");
        load_image_dataset(images_test, faces_test, faces_directory+"/testing_with_face_landmarks.xml");

        shape_predictor_trainer trainer;
        shape_predictor sp = trainer.train(images_train, faces_train);


        cout << "mean training error: "<< test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;
        cout << "mean testing error:  "<< test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;

        serialize("sp.dat") << sp;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

double interocular_distance (
    const full_object_detection& det
)
{
    dlib::vector<double,2> l, r;
    double cnt = 0;
    // Find the center of the left eye by averaging the points around 
    // the eye.
    for (unsigned long i = 36; i <= 41; ++i) 
    {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    // Find the center of the right eye by averaging the points around 
    // the eye.
    cnt = 0;
    for (unsigned long i = 42; i <= 47; ++i) 
    {
        r += det.part(i);
        ++cnt;
    }
    r /= cnt;

    // Now return the distance between the centers of the eyes
    return length(l-r);
}

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

