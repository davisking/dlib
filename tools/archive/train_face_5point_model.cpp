
/*

    This is the program that created the http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 model file.

*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/data_io.h>
#include <dlib/statistics.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);
/*!
    ensures
        - returns an object D such that:    
            - D[i][j] == the distance, in pixels, between the eyes for the face represented
              by objects[i][j].
!*/

// ----------------------------------------------------------------------------------------

template <
    typename image_array_type,
    typename T
    >
void add_image_left_right_flips_5points (
    image_array_type& images,
    std::vector<std::vector<T> >& objects
)
{
    // make sure requires clause is not broken
    DLIB_ASSERT( images.size() == objects.size(),
        "\t void add_image_left_right_flips()"
        << "\n\t Invalid inputs were given to this function."
        << "\n\t images.size():  " << images.size() 
        << "\n\t objects.size(): " << objects.size() 
        );

    typename image_array_type::value_type temp;
    std::vector<T> rects;

    const unsigned long num = images.size();
    for (unsigned long j = 0; j < num; ++j)
    {
        const point_transform_affine tran = flip_image_left_right(images[j], temp);

        rects.clear();
        for (unsigned long i = 0; i < objects[j].size(); ++i)
        {
            rects.push_back(impl::tform_object(tran, objects[j][i]));

            DLIB_CASSERT(rects.back().num_parts() == 5);
            swap(rects.back().part(0), rects.back().part(2));
            swap(rects.back().part(1), rects.back().part(3));
        }

        images.push_back(temp);
        objects.push_back(rects);
    }
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            cout << "give the path to the training data folder" << endl;
            return 0;
        }
        const std::string faces_directory = argv[1];
        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > faces_train, faces_test;

        std::vector<std::string> parts_list;
        load_image_dataset(images_train, faces_train, faces_directory+"/train_cleaned.xml", parts_list);
        load_image_dataset(images_test, faces_test, faces_directory+"/test_cleaned.xml");

        add_image_left_right_flips_5points(images_train, faces_train);
        add_image_left_right_flips_5points(images_test, faces_test);
        add_image_rotations(linspace(-20,20,3)*pi/180.0,images_train, faces_train);

        cout << "num training images: "<< images_train.size() << endl;

        for (auto& part : parts_list)
            cout << part << endl;

        shape_predictor_trainer trainer;
        trainer.set_oversampling_amount(40);
        trainer.set_num_test_splits(150);
        trainer.set_feature_pool_size(800);
        trainer.set_num_threads(4);
        trainer.set_cascade_depth(15);
        trainer.be_verbose();

        // Now finally generate the shape model
        shape_predictor sp = trainer.train(images_train, faces_train);

        serialize("shape_predictor_5_face_landmarks.dat") << sp;

        cout << "mean training error: "<< 
            test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;

        cout << "mean testing error:  "<< 
            test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;

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
    // left eye
    l = (det.part(0) + det.part(1))/2;
    // right eye
    r = (det.part(2) + det.part(3))/2;

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

