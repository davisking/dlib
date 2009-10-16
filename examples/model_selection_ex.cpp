// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example that shows you some reasonable ways you can perform
    model selection with the dlib C++ Library.  

    This example creates a simple set of data and then shows you how to use 
    the cross validation and optimization routines to determine good model 
    parameters for the purpose of training an svm to classify the sample data.

    The data used in this example will be 2 dimensional data and will
    come from a distribution where points with a distance less than 10
    from the origin are labeled +1 and all other points are labeled
    as -1.
        
*/


#include <iostream>
#include "dlib/svm.h"

using namespace std;
using namespace dlib;

// The svm functions use column vectors to contain a lot of the data on which they they 
// operate. So the first thing we do here is declare a convenient typedef.  

// This typedef declares a matrix with 2 rows and 1 column.  It will be the
// object that contains each of our 2 dimensional samples.   (Note that if you wanted 
// more than 2 features in this vector you can simply change the 2 to something else.
// Or if you don't know how many features you want until runtime then you can put a 0
// here and use the matrix.set_size() member function)
typedef matrix<double, 2, 1> sample_type;

// This is a typedef for the type of kernel we are going to use in this example.
// In this case I have selected the radial basis kernel that can operate on our
// 2D sample_type objects
typedef radial_basis_kernel<sample_type> kernel_type;



class cross_validation_objective
{
public:

    cross_validation_objective (
        const std::vector<sample_type>& samples_,
        const std::vector<double>& labels_
    ) : samples(samples_), labels(labels_) {}

    double operator() (
        const matrix<double>& params
    ) const
    {
        const double gamma = exp(params(0));
        const double nu    = exp(params(1));

        svm_nu_trainer<kernel_type> trainer;
        trainer.set_kernel(kernel_type(gamma));
        trainer.set_nu(nu);

        matrix<double> result = cross_validate_trainer(trainer, samples, labels, 10);
        cout << "gamma: " << setw(11) << gamma << "  nu: " << setw(11) << nu <<  "  cross validation accuracy: " << result;
        return sum(result);
    }

    const std::vector<sample_type>& samples;
    const std::vector<double>& labels;

};

int main()
{

    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Now lets put some data into our samples and labels objects.  We do this
    // by looping over a bunch of points and labeling them according to their
    // distance from the origin.
    for (int r = -20; r <= 20; ++r)
    {
        for (int c = -20; c <= 20; ++c)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            // if this point is less than 10 from the origin
            if (sqrt((double)r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);

        }
    }


    // Here we normalize all the samples by subtracting their mean and dividing by their standard deviation.
    // This is generally a good idea since it often heads off numerical stability problems and also 
    // prevents one large feature from smothering others.  Doing this doesn't matter much in this example
    // so I'm just doing this here so you can see an easy way to accomplish this with 
    // the library.  
    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 


    // Now that we have some data we want to train on it.  However, there are two parameters to the 
    // training.  These are the nu and gamma parameters.  Our choice for these parameters will 
    // influence how good the resulting decision function is.  To test how good a particular choice 
    // of these parameters are we can use the cross_validate_trainer() function to perform n-fold cross
    // validation on our training data.  However, there is a problem with the way we have sampled 
    // our distribution above.  The problem is that there is a definite ordering to the samples.  
    // That is, the first half of the samples look like they are from a different distribution 
    // than the second half do.  This would screw up the cross validation process but we can 
    // fix it by randomizing the order of the samples with the following function call.
    randomize_samples(samples, labels);


    // The nu parameter has a maximum value that is dependent on the ratio of the +1 to -1 
    // labels in the training data.  This function finds that value.
    const double max_nu = maximum_nu(labels);

    // here we make an instance of the svm_nu_trainer object that uses our kernel type.
    svm_nu_trainer<kernel_type> trainer;


    // Lets do a simple grid search

    matrix<double> params = cartesian_product(logspace(log10(20.0), log10(1e-5), 4),  // gamma parameter
                                              logspace(log10(max_nu), log10(1e-5), 4) // nu parameter
                                              );

    cout << "Doing a grid search" << endl;
    matrix<double> best_result(2,1);
    best_result = 0;
    double best_gamma, best_nu;
    set_all_elements(best_result, 0);
    for (long col = 0; col < params.nc(); ++col)
    {
        const double gamma = params(0, col);
        const double nu    = params(1, col);

        trainer.set_kernel(kernel_type(gamma));
        trainer.set_nu(nu);

        matrix<double> result = cross_validate_trainer(trainer, samples, labels, 10);
        cout << "gamma: " << setw(11) << gamma << "  nu: " << setw(11) << nu <<  "  cross validation accuracy: " << result;
        if (sum(result) > sum(best_result))
        {
            best_result = result;
            best_gamma = gamma;
            best_nu = nu;
        }
    }

    cout << "\n best result of grid search: " << sum(best_result) << endl;
    cout << "    best gamma: " << best_gamma << "   best nu: " << best_nu << endl;

    // now lets try out the BOBYQA algorithm

    cout << "\n\n Try the BOBYQA algorithm" << endl;

    params.set_size(2,1);
    params = best_gamma,      // initial gamma
             best_nu; // initial nu

    matrix<double> lower_bound(2,1), upper_bound(2,1);
    lower_bound = 1e-7, // smallest allowed gamma
                  1e-7; // smallest allowed nu
    upper_bound = 100,    // largest allowed gamma
                  max_nu; // largest allowed nu

    params = log(params);
    lower_bound = log(lower_bound);
    upper_bound = log(upper_bound);

    double best_score = find_max_bobyqa(cross_validation_objective(samples, labels),
                                        params,
                                        params.size()*2 + 1,
                                        lower_bound,
                                        upper_bound,
                                        min(upper_bound-lower_bound)/10,
                                        0.01,
                                        100
                                        );

    params = exp(params);
    cout << " best result of BOBYQA: " << best_score << endl;
    cout << "    best gamma: " << params(0) << "   best nu: " << params(1) << endl;

}

