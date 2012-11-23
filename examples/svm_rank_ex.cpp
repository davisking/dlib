// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the SVM-Rank tool from the dlib
    C++ Library.  This is a tool useful for learning to rank objects.  For
    example, you might use it to learn to rank web pages in response to a
    user's query.  The idea being to rank the most relevant pages higher than
    non-relevant pages.


    In this example, we will create a simple test dataset and show how to learn
    a ranking function on it.  The purpose of the function will be to give
    "relevant" objects higher scores than "non-relevant" objects.  The idea is
    that you use this score to order the objects so that the most relevant
    objects come to the top of the ranked list.
    


    Note that we use dense vectors (i.e. dlib::matrix objects) in this example,
    however, the ranking tools can also use sparse vectors as well.  See
    svm_sparse_ex.cpp for an example.
*/

#include "dlib/svm.h"
#include <iostream>


using namespace std;
using namespace dlib;


int main()
{
    try
    {
        // Make a typedef for the kind of object we will be ranking.  In this
        // example, we are ranking 2-dimensional vectors.  
        typedef matrix<double,2,1> sample_type;


        // Now lets make some testing data.  To make it really simple, lets
        // suppose that vectors with positive values in the first dimension
        // should rank higher than other vectors.  So what we do is make
        // examples of relevant (i.e. high ranking) and non-relevant (i.e. low
        // ranking) vectors and store them into a ranking_pair object like so:
        ranking_pair<sample_type> query;
        sample_type samp;

        // Make one relevant example.
        samp = 1, 0; 
        query.relevant.push_back(samp);

        // Now make a non-relevant example.
        samp = 0, 1; 
        query.nonrelevant.push_back(samp);

        // Now that we have some data, we can use a machine learning method to
        // learn a function that will give high scores to the relevant vectors
        // and low scores to the non-relevant vectors.

        // The first thing we do is select the kernel we want to use.  For the
        // svm_rank_trainer there are only two options.  The linear_kernel and
        // sparse_linear_kernel.  The later is used if you want to use sparse
        // vectors to represent your objects.  Since we are using dense vectors
        // (i.e. dlib::matrix objects to represent the vectors) we use the
        // linear_kernel.
        typedef linear_kernel<sample_type> kernel_type;

        svm_rank_trainer<kernel_type> trainer;
        decision_function<kernel_type> rank = trainer.train(query);

        cout << "ranking score for a relevant vector:     " << rank(query.relevant[0]) << endl;
        cout << "ranking score for a non-relevant vector: " << rank(query.nonrelevant[0]) << endl;

        // If we want an overall measure of ranking accuracy, we can find out
        // how often a non-relevant vector was ranked ahead of a relevant
        // vector like so.  This is a number between 0 and 1.  A value of 1
        // means everything was ranked perfectly.
        cout << "accuracy: " << test_ranking_function(rank, query) << endl;

        // We can also see the ranking weights:
        cout << "learned ranking weights: \n" << rank.basis_vectors(0) << endl;
        // In this case they are:
        //  0.5 
        // -0.5 




        std::vector<ranking_pair<sample_type> > queries;
        queries.push_back(query);
        queries.push_back(query);
        queries.push_back(query);
        queries.push_back(query);

        cout << "cv-accuracy: "<< cross_validate_ranking_trainer(trainer, queries, 4) << endl;

    }
    catch (std::exception& e)
    {
        cout << e.what() << endl;
    }
} 

