// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example that shows how you can perform model selection with the
    dlib C++ Library.  

    It will create a simple dataset and show you how to use cross validation and
    global optimization to determine good parameters for the purpose of training
    an svm to classify the data.

    The data used in this example will be 2 dimensional data and will come from a
    distribution where points with a distance less than 10 from the origin are
    labeled +1 and all other points are labeled as -1.
        

    As an side, you should probably read the svm_ex.cpp and matrix_ex.cpp example
    programs before you read this one.
*/


#include <iostream>
#include <dlib/svm.h>
#include <dlib/global_optimization.h>

using namespace std;
using namespace dlib;


int main() try
{
    // The svm functions use column vectors to contain a lot of the data on which they 
    // operate. So the first thing we do here is declare a convenient typedef.  

    // This typedef declares a matrix with 2 rows and 1 column.  It will be the
    // object that contains each of our 2 dimensional samples.   
    typedef matrix<double, 2, 1> sample_type;



    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Now let's put some data into our samples and labels objects.  We do this
    // by looping over a bunch of points and labeling them according to their
    // distance from the origin.
    for (double r = -20; r <= 20; r += 0.8)
    {
        for (double c = -20; c <= 20; c += 0.8)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            // if this point is less than 10 from the origin
            if (sqrt(r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);
        }
    }

    cout << "Generated " << samples.size() << " points" << endl;


    // Here we normalize all the samples by subtracting their mean and dividing by their
    // standard deviation.  This is generally a good idea since it often heads off
    // numerical stability problems and also prevents one large feature from smothering
    // others.  Doing this doesn't matter much in this example so I'm just doing this here
    // so you can see an easy way to accomplish this with the library.  
    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 


    // Now that we have some data we want to train on it.  We are going to train a
    // binary SVM with the RBF kernel to classify the data.  However, there are
    // three parameters to the training.  These are the SVM C parameters for each
    // class and the RBF kernel's gamma parameter.  Our choice for these
    // parameters will influence how good the resulting decision function is.  To
    // test how good a particular choice of these parameters is we can use the
    // cross_validate_trainer() function to perform n-fold cross validation on our
    // training data.  However, there is a problem with the way we have sampled
    // our distribution above.  The problem is that there is a definite ordering
    // to the samples.  That is, the first half of the samples look like they are
    // from a different distribution than the second half.  This would screw up
    // the cross validation process, but we can fix it by randomizing the order of
    // the samples with the following function call.
    randomize_samples(samples, labels);


    // And now we get to the important bit.  Here we define a function,
    // cross_validation_score(), that will do the cross-validation we
    // mentioned and return a number indicating how good a particular setting
    // of gamma, c1, and c2 is.
    auto cross_validation_score = [&](const double gamma, const double c1, const double c2) 
    {
        // Make a RBF SVM trainer and tell it what the parameters are supposed to be.
        typedef radial_basis_kernel<sample_type> kernel_type;
        svm_c_trainer<kernel_type> trainer;
        trainer.set_kernel(kernel_type(gamma));
        trainer.set_c_class1(c1);
        trainer.set_c_class2(c2);

        // Finally, perform 10-fold cross validation and then print and return the results.
        matrix<double> result = cross_validate_trainer(trainer, samples, labels, 10);
        cout << "gamma: " << setw(11) << gamma << "  c1: " << setw(11) << c1 <<  "  c2: " << setw(11) << c2 <<  "  cross validation accuracy: " << result;

        // Now return a number indicating how good the parameters are.  Bigger is
        // better in this example.  Here I'm returning the harmonic mean between the
        // accuracies of each class.  However, you could do something else.  For
        // example, you might care a lot more about correctly predicting the +1 class,
        // so you could penalize results that didn't obtain a high accuracy on that
        // class.  You might do this by using something like a weighted version of the
        // F1-score (see http://en.wikipedia.org/wiki/F1_score).     
        return 2*prod(result)/sum(result);
    };


    // And finally, we call this global optimizer that will search for the best parameters.
    // It will call cross_validation_score() 50 times with different settings and return
    // the best parameter setting it finds.  find_max_global() uses a global optimization
    // method based on a combination of non-parametric global function modeling and
    // quadratic trust region modeling to efficiently find a global maximizer.  It usually
    // does a good job with a relatively small number of calls to cross_validation_score().
    // In this example, you should observe that it finds settings that give perfect binary
    // classification of the data.
    auto result = find_max_global(cross_validation_score, 
                                  {1e-5, 1e-5, 1e-5},  // lower bound constraints on gamma, c1, and c2, respectively
                                  {100,  1e6,  1e6},   // upper bound constraints on gamma, c1, and c2, respectively
                                  max_function_calls(50));

    double best_gamma = result.x(0);
    double best_c1    = result.x(1);
    double best_c2    = result.x(2);

    cout << " best cross-validation score: " << result.y << endl;
    cout << " best gamma: " << best_gamma << "   best c1: " << best_c1 << "    best c2: "<< best_c2  << endl;
}
catch (exception& e)
{
    cout << e.what() << endl;
}

