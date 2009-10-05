// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the feature ranking 
    tools from the dlib C++ Library.  

    This example creates a simple set of data and then shows
    you how to use the feature ranking function to find a good 
    set of features (where "good" means the feature set will probably
    work well with a classification algorithm).

    The data used in this example will be 4 dimensional data and will
    come from a distribution where points with a distance less than 10
    from the origin are labeled +1 and all other points are labeled
    as -1.  Note that this data is conceptually 2 dimensional but we
    will add two extra features for the purpose of showing what
    the feature ranking function does.
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

    for (int x = -30; x <= 30; ++x)
    {
        for (int y = -30; y <= 30; ++y)
        {
            sample_type samp;

            // the first two features are just the (x,y) position of our points and so
            // we expect them to be good features since our two classes here are points
            // close to the origin and points far away from the origin.
            samp(0) = x;
            samp(1) = y;

            // This is a worthless feature since it is just random noise.  It should
            // be indicated as worthless by the feature ranking below.
            samp(2) = rnd.get_random_double();

            // This is a version of the y feature that is corrupted by random noise.  It
            // should be ranked as less useful than features 0, and 1, but more useful
            // than the above feature.
            samp(3) = y*0.2 + (rnd.get_random_double()-0.5)*10;

            // add this sample into our vector of samples.
            samples.push_back(samp);

            // if this point is less than 15 from the origin then label it as a +1 class point.  
            // otherwise it is a -1 class point
            if (sqrt((double)x*x + y*y) <= 15)
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
    // However, in our case it doesn't really matter.  
    randomize_samples(samples,labels);



    // Finally we get to the feature ranking. Here we call verbose_rank_features_rbf() with
    // the samples and labels we made above.  The 20 is a measure of how much memory and CPU
    // resources the algorithm should use.  Generally bigger values give better results but 
    // take longer to run.
    cout << verbose_rank_features_rbf(samples, labels, 20) << endl;

    // The output is:
    /*
        0 0.810087 
        1        1 
        3 0.873991 
        2 0.668913 
    */

    // The first column is a list of the features in order of decreasing goodness.  So the feature ranking function
    // is telling us that the samples[i](0) and samples[i](1) (i.e. the x and y) features are the best two.  Then
    // after that the next best feature is the samples[i](3) (i.e. the y corrupted by noise) and finally the worst
    // feature is the one that is just random noise.  So in this case the feature ranking did exactly what we would
    // intuitively expect.


    // The second column of the matrix is a number that indicates how much the features up to that point
    // contribute to the separation of the two classes.  So bigger numbers are better since they
    // indicate a larger separation.

    // So to break it down a little more.
    //    1 0.810087   <-- class separation of feature 1 all by itself
    //    0        1   <-- class separation of feature 1 and 0
    //    3 0.873991   <-- class separation of feature 1, 0, and 3
    //    2 0.668913   <-- class separation of feature 1, 0, 3, and 2
        

}

