// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

*/


#include <iostream>
#include <dlib/svm_threaded.h>
#include <dlib/rand.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

class detection
{

public:
    typedef class track track_type;

    matrix<double,0,1> measurements;
};

class track
{

public:
    // This type should be a dlib::matrix capable of storing column vectors or an
    // unsorted sparse vector type as defined in dlib/svm/sparse_vector_abstract.h.
    typedef matrix<double,0,1> feature_vector_type;

    track()
    {
        time_since_last_association = 0;
    }

    void get_similarity_features (
        const detection& det,
        feature_vector_type& feats
    ) const
    {
        feats = abs(last_measurements - det.measurements);
    }

    void update_track (
        const detection& det
    )
    {
        last_measurements = det.measurements;
        time_since_last_association = 0;
    }

    void propagate_track (
    )
    {
        ++time_since_last_association;
    }

    matrix<double,0,1> last_measurements;
    unsigned long time_since_last_association;
};

// ----------------------------------------------------------------------------------------

typedef std::vector<labeled_detection<detection> > detections_at_single_time_step;
typedef std::vector<detections_at_single_time_step> track_history;

dlib::rand rnd;
const long num_objects = 4;
const long num_properties = 6;
std::vector<matrix<double,0,1> > object_properties(num_objects);

void initialize_object_properties()
{
    for (unsigned long i = 0; i < object_properties.size(); ++i)
        object_properties[i] = randm(num_properties,1,rnd);
}

// ----------------------------------------------------------------------------------------

track_history make_random_tracking_data_for_training()
{
    track_history data;


    const int num_time_steps = 100;
    for (int i = 0; i < num_time_steps; ++i)
    {
        detections_at_single_time_step dets(3);
        dets[0].det.measurements = object_properties[0] + randm(num_properties,1,rnd)*0.1;
        dets[0].label = 0;

        dets[1].det.measurements = object_properties[1] + randm(num_properties,1,rnd)*0.1;
        dets[1].label = 1;

        dets[2].det.measurements = object_properties[2] + randm(num_properties,1,rnd)*0.1;
        dets[2].label = 2;

        data.push_back(dets);
    }

    for (int i = 0; i < num_time_steps; ++i)
    {
        detections_at_single_time_step dets(2);
        dets[0].det.measurements = object_properties[0] + randm(num_properties,1,rnd)*0.1;
        dets[0].label = 0;

        dets[1].det.measurements = object_properties[3] + randm(num_properties,1,rnd)*0.1;
        dets[1].label = 3;

        data.push_back(dets);
    }

    return data;
}

// ----------------------------------------------------------------------------------------

std::vector<detection> make_random_detections(unsigned long num_dets)
{
    std::vector<detection> dets(num_dets);
    for (unsigned long i = 0; i < dets.size(); ++i)
    {
        dets[i].measurements = object_properties[i] + randm(num_properties,1,rnd)*0.1;
    }
    return dets;
}

// ----------------------------------------------------------------------------------------

int main()
{
    initialize_object_properties();


    std::vector<track_history> data;
    data.push_back(make_random_tracking_data_for_training());
    data.push_back(make_random_tracking_data_for_training());
    data.push_back(make_random_tracking_data_for_training());
    data.push_back(make_random_tracking_data_for_training());
    data.push_back(make_random_tracking_data_for_training());


    structural_track_association_trainer trainer;
    trainer.be_verbose();
    trainer.set_c(100);

    track_association_function<detection> assoc = trainer.train(data);

    cout << "accuracy on training data: "<< test_track_association_function(assoc, data) << endl;
    cout << "cross validation: "<< cross_validate_track_association_trainer(trainer, data, 5) << endl;

    std::vector<detection> dets;
    std::vector<track> tracks;

    cout << "number of tracks: "<< tracks.size() << endl;
    dets = make_random_detections(3);
    assoc(tracks, dets);
    cout << "number of tracks: "<< tracks.size() << endl;

    dets = make_random_detections(3);
    assoc(tracks, dets);
    cout << "number of tracks: "<< tracks.size() << endl;

    dets = make_random_detections(4);
    assoc(tracks, dets);
    cout << "number of tracks: "<< tracks.size() << endl;

    dets = make_random_detections(3);
    assoc(tracks, dets);
    cout << "number of tracks: "<< tracks.size() << endl;
    for (unsigned long i = 0; i < tracks.size(); ++i)
        cout << "   time since last association: "<< tracks[i].time_since_last_association << endl;

    dets = make_random_detections(3);
    assoc(tracks, dets);
    cout << "number of tracks: "<< tracks.size() << endl;
    for (unsigned long i = 0; i < tracks.size(); ++i)
        cout << "   time since last association: "<< tracks[i].time_since_last_association << endl;


    ofstream fout("track_assoc.svm", ios::binary);
    serialize(assoc, fout);
    fout.close();

    ifstream fin("track_assoc.svm", ios::binary);
    deserialize(assoc, fin);
}

// ----------------------------------------------------------------------------------------

