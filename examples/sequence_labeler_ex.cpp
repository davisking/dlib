// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the support vector machine
    utilities from the dlib C++ Library.  

    This example creates a simple set of data to train on and then shows
    you how to use the cross validation and svm training functions
    to find a good decision function that can classify examples in our
    data set.


    The data used in this example will be 2 dimensional data and will
    come from a distribution where points with a distance less than 10
    from the origin are labeled +1 and all other points are labeled
    as -1.
        
*/


#include <iostream>
#include "dlib/svm_threaded.h"
#include "dlib/rand.h"

using namespace std;
using namespace dlib;


const unsigned long num_label_states = 3; // the "hidden" states
const unsigned long num_sample_states = 3;

// ----------------------------------------------------------------------------------------

class feature_extractor
{
public:
    typedef unsigned long sample_type; 

    unsigned long num_features() const
    {
        return num_label_states*num_label_states + num_label_states*num_sample_states;
    }

    unsigned long order() const 
    { 
        return 1; 
    }

    unsigned long num_labels() const 
    { 
        return num_label_states; 
    }

    template <typename feature_setter, typename EXP>
    void get_features (
        feature_setter& set_feature,
        const std::vector<sample_type>& x,
        const matrix_exp<EXP>& y,
        unsigned long position
    ) const
    {
        if (y.size() > 1)
            set_feature(y(1)*num_label_states + y(0));

        set_feature(num_label_states*num_label_states +
                    y(0)*num_sample_states + x[position]);
    }
};

void serialize(const feature_extractor&, std::ostream&) {}
void deserialize(feature_extractor&, std::istream&) {}

// ----------------------------------------------------------------------------------------

void make_dataset (
    const matrix<double>& emission_probabilities,
    const matrix<double>& transition_probabilities,
    std::vector<std::vector<unsigned long> >& samples,
    std::vector<std::vector<unsigned long> >& labels,
    unsigned long dataset_size
);
/*!
    requires
        - transition_probabilities.nr() == transition_probabilities.nc()
        - transition_probabilities.nr() == emission_probabilities.nr()
        - The rows of transition_probabilities and emission_probabilities must sum to 1.
          (i.e. sum_cols(transition_probabilities) and sum_cols(emission_probabilities)
          must evaluate to vectors of all 1s.)
    ensures
        - This function randomly samples a bunch of sequences from the HMM defined by 
          transition_probabilities and emission_probabilities. 
        - The HMM is defined by:
            - P(next_label |previous_label) == transition_probabilities(previous_label, next_label)
            - P(next_sample|next_label)     == emission_probabilities  (next_label,     next_sample)
        - #samples.size() == labels.size() == dataset_size
        - for all valid i:
            - #labels[i] is a randomly sampled sequence of hidden states from the
              given HMM.  #samples[i] is its corresponding randomly sampled sequence
              of observed states.
!*/

// ----------------------------------------------------------------------------------------

int main()
{

    // set this up so emission_probabilities(L,X) == The probability of a state with label L 
    // emitting an X.
    matrix<double> emission_probabilities(num_label_states,num_sample_states);
    emission_probabilities = 0.5, 0.5, 0.0,
                             0.0, 0.5, 0.5,
                             0.5, 0.0, 0.5;

    matrix<double> transition_probabilities(num_label_states, num_label_states);

    transition_probabilities = 0.05, 0.90, 0.05,
                               0.05, 0.05, 0.90,
                               0.90, 0.05, 0.05;
                    


    std::vector<std::vector<unsigned long> > samples;
    std::vector<std::vector<unsigned long> > labels;
    make_dataset(emission_probabilities, transition_probabilities,
                 samples, labels, 1000);

    cout << "samples.size(): "<< samples.size() << endl;

    // print out some of the randomly sampled sequences
    for (int i = 0; i < 10; ++i)
    {
        cout << "hidden states:   " << trans(vector_to_matrix(labels[i]));
        cout << "observed states: " << trans(vector_to_matrix(samples[i]));
        cout << "******************************" << endl;
    }

    structural_sequence_labeling_trainer<feature_extractor> trainer;
    trainer.set_c(4);
    trainer.set_num_threads(4);


    matrix<double> confusion_matrix;

    // Learn to do sequence labeling from the dataset
    sequence_labeler<feature_extractor> labeler = trainer.train(samples, labels);
    confusion_matrix = test_sequence_labeler(labeler, samples, labels);
    cout << "trained sequence labeler: " << endl;
    cout << confusion_matrix;
    cout << "label accuracy: "<< sum(diag(confusion_matrix))/sum(confusion_matrix) << endl;


    // We can also do cross-validation 
    confusion_matrix = cross_validate_sequence_labeler(trainer, samples, labels, 4);
    cout << "\ncross-validation: " << endl;
    cout << confusion_matrix;
    cout << "label accuracy: "<< sum(diag(confusion_matrix))/sum(confusion_matrix) << endl;



    matrix<double,0,1> true_hmm_model_weights = log(join_cols(reshape_to_column_vector(transition_probabilities),
                                                              reshape_to_column_vector(emission_probabilities)));

    sequence_labeler<feature_extractor> labeler_true(feature_extractor(), true_hmm_model_weights); 

    confusion_matrix = test_sequence_labeler(labeler_true, samples, labels);
    cout << "\nTrue HMM model: " << endl;
    cout << confusion_matrix;
    cout << "label accuracy: "<< sum(diag(confusion_matrix))/sum(confusion_matrix) << endl;







    // Finally, the labeler can be serialized to disk just like most dlib objects.
    ofstream fout("labeler.dat", ios::binary);
    serialize(labeler, fout);
    fout.close();

    // recall from disk
    ifstream fin("labeler.dat", ios::binary);
    deserialize(labeler, fin);
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//              Code for creating a bunch of random samples from our HMM.
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

void sample_hmm (
    dlib::rand& rnd,
    const matrix<double>& transition_probabilities,
    const matrix<double>& emission_probabilities,
    unsigned long previous_label,
    unsigned long& next_label,
    unsigned long& next_sample
)
/*!
    requires
        - previous_label < transition_probabilities.nr()
        - transition_probabilities.nr() == transition_probabilities.nc()
        - transition_probabilities.nr() == emission_probabilities.nr()
        - The rows of transition_probabilities and emission_probabilities must sum to 1.
          (i.e. sum_cols(transition_probabilities) and sum_cols(emission_probabilities)
          must evaluate to vectors of all 1s.)
    ensures
        - This function randomly samples the HMM defined by transition_probabilities
          and emission_probabilities assuming that the previous hidden state
          was previous_label. 
        - The HMM is defined by:
            - P(next_label |previous_label) == transition_probabilities(previous_label, next_label)
            - P(next_sample|next_label)     == emission_probabilities  (next_label,     next_sample)
        - #next_label == the sampled value of the hidden state
        - #next_sample == the sampled value of the observed state
!*/
{
    // sample next_label
    double p = rnd.get_random_double();
    for (long c = 0; p >= 0 && c < transition_probabilities.nc(); ++c)
    {
        next_label = c;
        p -= transition_probabilities(previous_label, c);
    }

    // now sample next_sample
    p = rnd.get_random_double();
    for (long c = 0; p >= 0 && c < emission_probabilities.nc(); ++c)
    {
        next_sample = c;
        p -= emission_probabilities(next_label, c);
    }
}

// ----------------------------------------------------------------------------------------

void make_dataset (
    const matrix<double>& emission_probabilities,
    const matrix<double>& transition_probabilities,
    std::vector<std::vector<unsigned long> >& samples,
    std::vector<std::vector<unsigned long> >& labels,
    unsigned long dataset_size
)
{
    samples.clear();
    labels.clear();

    dlib::rand rnd;

    // now randomly sample some labeled sequences from our Hidden Markov Model
    for (unsigned long iter = 0; iter < dataset_size; ++iter)
    {
        const unsigned long sequence_size = rnd.get_random_32bit_number()%20+3;
        std::vector<unsigned long> sample(sequence_size);
        std::vector<unsigned long> label(sequence_size);

        unsigned long previous_label = rnd.get_random_32bit_number()%num_label_states;
        for (unsigned long i = 0; i < sample.size(); ++i)
        {
            unsigned long next_label, next_sample;
            sample_hmm(rnd, transition_probabilities, emission_probabilities, 
                       previous_label, next_label, next_sample);

            label[i] = next_label;
            sample[i] = next_sample;

            previous_label = next_label;
        }

        samples.push_back(sample);
        labels.push_back(label);
    }
}

// ----------------------------------------------------------------------------------------

