// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_LDA_ABSTRACT_Hh_
#ifdef DLIB_LDA_ABSTRACT_Hh_

#include <map>
#include "../matrix.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void compute_lda_transform (
        matrix<T>& X,
        matrix<T,0,1>& M,
        const std::vector<unsigned long>& row_labels,
        unsigned long lda_dims = 500,
        unsigned long extra_pca_dims = 200
    );
    /*!
        requires
            - X.size() != 0
            - row_labels.size() == X.nr()
            - The number of distinct values in row_labels > 1
            - lda_dims != 0
        ensures
            - We interpret X as a collection X.nr() of input vectors, where each row of X
              is one of the vectors.
            - We interpret row_labels[i] as the label of the vector rowm(X,i).
            - This function performs the dimensionality reducing version of linear
              discriminant analysis.  That is, you give it a set of labeled vectors and it
              returns a linear transform that maps the input vectors into a new space that
              is good for distinguishing between the different classes.  In particular,
              this function finds matrices Z and M such that:
                - Given an input vector x, Z*x-M, is the transformed version of x.  That is,
                  Z*x-M maps x into a space where x vectors that share the same class label
                  are near each other. 
                - Z*x-M results in the transformed vectors having zero expected mean.
                - Z.nr() <= lda_dims
                  (it might be less than lda_dims if there are not enough distinct class
                  labels to support lda_dims dimensions).
                - Z.nc() == X.nc()
                - We overwrite the input matrix X and store Z in it.  Therefore, the
                  outputs of this function are in X and M.
            - In order to deal with very high dimensional inputs, we perform PCA internally
              to map the input vectors into a space of at most lda_dims+extra_pca_dims
              prior to performing LDA.
    !*/

// ----------------------------------------------------------------------------------------

    std::pair<double,double> equal_error_rate (
        const std::vector<double>& low_vals,
        const std::vector<double>& high_vals 
    );
    /*!
        ensures
            - This function finds a threshold T that best separates the elements of
              low_vals from high_vals by selecting the threshold with equal error rate.  In
              particular, we try to pick a threshold T such that:
                - for all valid i:
                    - high_vals[i] >= T
                - for all valid i:
                    - low_vals[i] < T
              Where the best T is determined such that the fraction of low_vals >= T is the
              same as the fraction of high_vals < T.
            - Let ERR == the equal error rate.  I.e. the fraction of times low_vals >= T
              and high_vals < T.  Note that 0 <= ERR <= 1.
            - returns make_pair(ERR,T) 
    !*/

// ----------------------------------------------------------------------------------------

    struct roc_point
    {
        double true_positive_rate;
        double false_positive_rate;
        double detection_threshold;
    };

    std::vector<roc_point> compute_roc_curve (
        const std::vector<double>& true_detections,
        const std::vector<double>& false_detections 
    );
    /*!
        requires
            - true_detections.size() != 0
            - false_detections.size() != 0
        ensures
            - This function computes the ROC curve (receiver operating characteristic)
              curve of the given data.  Therefore, we interpret true_detections as
              containing detection scores for a bunch of true detections and
              false_detections as detection scores from a bunch of false detections.  A
              perfect detector would always give higher scores to true detections than to
              false detections, resulting in a true positive rate of 1 and a false positive
              rate of 0, for some appropriate detection threshold.
            - Returns an array, ROC, such that:
                - ROC.size() == true_detections.size()+false_detections.size()
                - for all valid i:
                    - If you were to accept all detections with a score >= ROC[i].detection_threshold 
                      then you would obtain a true positive rate of ROC[i].true_positive_rate and a 
                      false positive rate of ROC[i].false_positive_rate.
                - ROC is ordered such that low detection rates come first.  That is, the
                  curve is swept from a high detection threshold to a low threshold.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LDA_ABSTRACT_Hh_


