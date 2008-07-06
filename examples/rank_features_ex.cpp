/*

    This is an example illustrating the use of the rank_features() function 
    from the dlib C++ Library.  

    This example creates a simple set of data and then shows
    you how to use the rank_features() function to find a good 
    set of features (where "good" means the feature set will probably
    work well with a classification algorithm).

    The data used in this example will be 4 dimensional data and will
    come from a distribution where points with a distance less than 10
    from the origin are labeled +1 and all other points are labeled
    as -1.  Note that this data is conceptually 2 dimensional but we
    will add two extra features for the purpose of showing what
    the rank_features() function does.
*/


#include <iostream>
#include "dlib/svm.h"
#include "dlib/rand.h"
#include <vector>

using namespace std;
using namespace dlib;


int main()
{

    // This first typedef declares a matrix with 4 rows and 1 column.  It will be the
    // object that contains each of our 4 dimensional samples.  
    typedef matrix<double, 4, 1> sample_type;



    // Now lets make some vector objects that can hold our samples 
    std::vector<sample_type> samples;
    std::vector<double> labels;

    dlib::rand::float_1a rnd;

    for (int x = -20; x <= 20; ++x)
    {
        for (int y = -20; y <= 20; ++y)
        {
            sample_type samp;

            // the first two features are just the (x,y) position of our points and so
            // we expect them to be good features since our two classes here are points
            // close to the origin and points far away from the origin.
            samp(0) = x;
            samp(1) = y;

            // This is a worthless feature since it is just random noise.  It should
            // be indicated as worthless by the rank_features() function below.
            samp(2) = rnd.get_random_double();

            // This is a version of the y feature that is corrupted by random noise.  It
            // should be ranked as less useful than features 0, and 1, but more useful
            // than the above feature.
            samp(3) = y - rnd.get_random_double()*10;

            // add this sample into our vector of samples.
            samples.push_back(samp);

            // if this point is less than 10 from the origin then label it as a +1 class point.  
            // otherwise it is a -1 class point
            if (sqrt((double)x*x + y*y) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);
        }
    }


    // Here we normalize all the samples by subtracting their mean and dividing by their standard deviation.
    // This is generally a good idea since it often heads off numerical stability problems and also 
    // prevents one large feature from smothering others.
    const sample_type m(mean(vector_to_matrix(samples)));  // compute a mean vector
    const sample_type sd(reciprocal(sqrt(variance(vector_to_matrix(samples))))); // compute a standard deviation vector
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = pointwise_multiply(samples[i] - m, sd); 

    // This is another thing that is often good to do from a numerical stability point of view.  
    // In our case it doesn't really matter.  
    randomize_samples(samples,labels);



    // This is a typedef for the type of kernel we are going to use in this example.
    // In this case I have selected the radial basis kernel that can operate on our
    // 4D sample_type objects.  In general, I would suggest using the same kernel for
    // classification and feature ranking. 
    typedef radial_basis_kernel<sample_type> kernel_type;
    
    // This line here declares the kcentroid object we want to use for feature ranking.  Note that there
    // are two numbers in it.  The first is the argument to the kernel.  The second is a tolerance argument
    // for the kcentroid object.  This tolerance is basically a control on the number of support vectors it
    // will use, with a smaller tolerance giving better accuracy but longer running times.  Generally
    // something in the range 0.01 to 0.001 is a good choice.
    kcentroid<kernel_type> kc(kernel_type(0.05), 0.001);

    // And finally we get to the feature ranking. Here we call rank_features() with the kcentroid we just made,
    // the samples and labels we made above, and the number of features we want it to rank.  
    cout << rank_features(kc, samples, labels, 4) << endl;

    // The output is:
    /*
        0   0.452251 
        1   0.259739 
        3    0.28801 
        2 -0.0347664 
    */

    // The first column is a list of the features in order of decreasing goodness.  So the rank_features() function
    // is telling us that the samples[i](0) and samples[i](1) (i.e. the x and y) features are the best two.  Then
    // after that the next best feature is the samples[i](3) (i.e. the y corrupted by noise) and finally the worst
    // feature is the one that is just random noise.  So in this case rank_features did exactly what we would
    // intuitively expect.


    // The second column of the matrix is a number that indicates how much that feature contributes to the
    // separation of the two classes.  So a bigger number is better and smaller is worse.  What we see above is that
    // the first 3 features all help separate the data and the last one actually hurts us in terms of this metric.

    // So to break it down a little more.
    //    0   0.452251   <-- class separation of feature 0 all by itself
    //    1   0.259739   <-- Additional separation gained from feature 1 if classification is done with features 1 and 0
    //    3    0.28801   <-- Additional separation gained from feature 3 if classification is done with features 3, 0, and 1
    //    2 -0.0347664   <-- Additional separation gained from feature 2 if classification is done with features 2, 3, 0, and 1

}

