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
#include "dlib/svm.h"

using namespace std;
using namespace dlib;


int main()
{
    // The svm functions use column vectors to contain a lot of the data they operate on
    // So the first thing we do here is declare some convenient typedefs for matrix objects
    // we will be using.

    // This first typedef declares a matrix with 2 rows and 1 column.  It will be the
    // object that contains each of our 2 dimensional samples.   (Note that if you wanted 
    // more than 2 features in this vector you can simply change the 2 to something else)
    typedef matrix<double, 2, 1> sample_type;

    // This is a typedef for a column vector of unknown length that contains our
    // sample_type objects.  Instances of this object will contain our sample data.
    typedef matrix<sample_type,0,1> samples_type;

    // This is a typedef for the type of kernel we are going to use in this example.
    // In this case I have selected the radial basis kernel that can operate on our
    // 2D sample_type objects
    typedef radial_basis_kernel<sample_type> kernel_type;


    // Now we make a samples_type object as well as a column vector to 
    // store the label for each sample in samples.
    samples_type samples;
    matrix<double, 0,1> labels;


    // Now lets put some data into our samples and labels objects.  We do this
    // by looping over 41*41 points and labeling them according to their
    // distance from the origin.
    samples.set_size(41*41);
    labels.set_size(41*41);
    int count = 0;
    for (int r = -20; r <= 20; ++r)
    {
        for (int c = -20; c <= 20; ++c)
        {
            samples(count)(0) = r;
            samples(count)(1) = c;

            // if this point is less than 10 from the origin
            if (sqrt((double)r*r + c*c) <= 10)
                labels(count) = +1;
            else
                labels(count) = -1;

            ++count;
        }
    }


    // Now that we have some data we want to train on it.  However, there are two parameters to the 
    // training.  These are the nu and gamma parameters.  Our choice for these parameters will 
    // influence how good the resulting decision function is.  To test how good a particular choice 
    // of these parameters are we can use the svm_nu_cross_validate() function to perform n-fold cross
    // validation on our training data.  However, there is a problem with the way we have sampled 
    // our distribution above.  The problem is that there is a definite ordering to the samples.  
    // That is, the first half of the samples look like they are from a different distribution 
    // than the second half do.  This would screw up the cross validation process but we can 
    // fix it by randomizing the order of the samples with the following function call.
    randomize_samples(samples, labels);


    // The nu parameter has a maximum value that is dependent on the ratio of the +1 to -1 
    // labels in the training data.  This function finds that value.
    const double max_nu = maximum_nu(labels);

    // Now we loop over some different nu and gamma values to see how good they are.  Note
    // that this is just a simple brute force way to try out a few possible parameter 
    // choices.  You may want to investigate more sophisticated strategies for determining 
    // good parameter choices.
    cout << "doing cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma += 0.1)
    {
        for (double nu = 0.00001; nu < max_nu; nu += 0.1)
        {
            cout << "gamma: " << gamma << "    nu: " << nu;
            // Print out the cross validation accuracy for 3-fold cross validation using the current gamma and nu.  
            // svm_nu_cross_validate() returns a column vector.  The first element of the vector is the fraction
            // of +1 training examples correctly classified and the second number is the fraction of -1 training 
            // examples correctly classified.
            cout << "     cross validation accuracy: " << svm_nu_cross_validate(samples, labels, kernel_type(gamma), nu, 3);
        }
    }


    // From looking at the output of the above loop it turns out that a good value for 
    // nu and gamma for this problem is 0.1 for both.  So that is what we will use.

    // Now we train on the full set of data and obtain the resulting decision function.  We use the
    // value of 0.1 for nu and gamma.  The decision function will return values >= 0 for samples it predicts
    // are in the +1 class and numbers < 0 for samples it predicts to be in the -1 class.
    decision_function<kernel_type> learned_decision_function = svm_nu_train(samples, labels, kernel_type(0.1), 0.1);

    // print out the number of support vectors in the resulting decision function
    cout << "\nnumber of support vectors in our learned_decision_function is " << learned_decision_function.support_vectors.nr() << endl;

    // now lets try this decision_function on some samples we haven't seen before 
    sample_type sample;

    sample(0) = 3.123;
    sample(1) = 2;
    cout << "This sample should be >= 0 and it is classified as a " << learned_decision_function(sample) << endl;

    sample(0) = 3.123;
    sample(1) = 9.3545;
    cout << "This sample should be >= 0 and it is classified as a " << learned_decision_function(sample) << endl;

    sample(0) = 13.123;
    sample(1) = 9.3545;
    cout << "This sample should be < 0 and it is classified as a " << learned_decision_function(sample) << endl;

    sample(0) = 13.123;
    sample(1) = 0;
    cout << "This sample should be < 0 and it is classified as a " << learned_decision_function(sample) << endl;


    // We can also train a decision function that reports a well conditioned probability instead of just a number
    // > 0 for the +1 class and < 0 for the -1 class.  An example of doing that follows:
    probabilistic_decision_function<kernel_type> learned_probabilistic_decision_function = svm_nu_train_prob(samples, labels, kernel_type(0.1), 0.1, 3);

    // print out the number of support vectors in the resulting decision function.  (it should be the same as in the one above)
    cout << "\nnumber of support vectors in our learned_probabilistic_decision_function is " 
         << learned_probabilistic_decision_function.decision_funct.support_vectors.nr() << endl;

    sample(0) = 3.123;
    sample(1) = 2;
    cout << "This +1 example should have high probability.  It's probability is: " << learned_probabilistic_decision_function(sample) << endl;

    sample(0) = 3.123;
    sample(1) = 9.3545;
    cout << "This +1 example should have high probability.  It's probability is: " << learned_probabilistic_decision_function(sample) << endl;

    sample(0) = 13.123;
    sample(1) = 9.3545;
    cout << "This -1 example should have low probability.  It's probability is: " << learned_probabilistic_decision_function(sample) << endl;

    sample(0) = 13.123;
    sample(1) = 0;
    cout << "This -1 example should have low probability.  It's probability is: " << learned_probabilistic_decision_function(sample) << endl;


}

