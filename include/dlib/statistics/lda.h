// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LDA_Hh_
#define DLIB_LDA_Hh_

#include "lda_abstract.h"
#include "../algs.h"
#include <map>
#include "../matrix.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {

        inline std::map<unsigned long,unsigned long> make_class_labels(
            const std::vector<unsigned long>& row_labels
        )
        {
            std::map<unsigned long,unsigned long> class_labels;
            for (unsigned long i = 0; i < row_labels.size(); ++i)
            {
                const unsigned long next = class_labels.size();
                if (class_labels.count(row_labels[i]) == 0)
                    class_labels[row_labels[i]] = next;
            }
            return class_labels;
        }

    // ------------------------------------------------------------------------------------

        template <
            typename T
            >
        matrix<T,0,1> center_matrix (
            matrix<T>& X
        )
        {
            matrix<T,1> mean;
            for (long r = 0; r < X.nr(); ++r)
                mean += rowm(X,r);
            mean /= X.nr();

            for (long r = 0; r < X.nr(); ++r)
                set_rowm(X,r) -= mean;

            return trans(mean);
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void compute_lda_transform (
        matrix<T>& X,
        matrix<T,0,1>& mean,
        const std::vector<unsigned long>& row_labels,
        unsigned long lda_dims = 500,
        unsigned long extra_pca_dims = 200
    )
    {
        std::map<unsigned long,unsigned long> class_labels = impl::make_class_labels(row_labels);
        // LDA can only give out at most class_labels.size()-1 dimensions so don't try to
        // compute more than that.
        lda_dims = std::min<unsigned long>(lda_dims, class_labels.size()-1);

        // make sure requires clause is not broken
        DLIB_CASSERT(class_labels.size() > 1,
            "\t void compute_lda_transform()"
            << "\n\t You can't call this function if the number of distinct class labels is less than 2."
            );
        DLIB_CASSERT(X.size() != 0 && (long)row_labels.size() == X.nr() && lda_dims != 0,
            "\t void compute_lda_transform()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t X.size():          " << X.size()
            << "\n\t row_labels.size(): " << row_labels.size()
            << "\n\t lda_dims:          " << lda_dims
            );


        mean = impl::center_matrix(X);
        // Do PCA to reduce dims
        matrix<T> pu,pw,pv;
        svd_fast(X, pu, pw, pv, lda_dims+extra_pca_dims, 4);
        pu.set_size(0,0); // free RAM, we don't need pu.
        X = X*pv;


        matrix<T> class_means(class_labels.size(), X.nc());
        class_means = 0;
        matrix<T,0,1> class_counts(class_labels.size());
        class_counts = 0;

        // First compute the means of each class
        for (unsigned long i = 0; i < row_labels.size(); ++i)
        {
            const unsigned long class_idx = class_labels[row_labels[i]];
            set_rowm(class_means,class_idx) += rowm(X,i);
            class_counts(class_idx)++;
        }
        class_means = inv(diagm(class_counts))*class_means;
        // subtract means from the data
        for (unsigned long i = 0; i < row_labels.size(); ++i)
        {
            const unsigned long class_idx = class_labels[row_labels[i]];
            set_rowm(X,i) -= rowm(class_means,class_idx);
        }

        // Note that we are using the formulas from the paper Using Discriminant
        // Eigenfeatures for Image Retrieval by Swets and Weng.
        matrix<T> Sw = trans(X)*X;
        matrix<T> Sb = trans(class_means)*class_means;
        matrix<T> A, H;
        matrix<T,0,1> W;
        svd3(Sw, A, W, H);
        W = sqrt(W);
        W = reciprocal(lowerbound(W,max(W)*1e-5));
        A = trans(H*diagm(W))*Sb*H*diagm(W);
        matrix<T> v,s,u;
        svd3(A, v, s, u);
        matrix<T> tform = H*diagm(W)*u;
        // pick out only the number of dimensions we are supposed to for the output, unless
        // we should just keep them all, then don't do anything. 
        if ((long)lda_dims <= tform.nc())
        {
            rsort_columns(tform, s);
            tform = colm(tform, range(0, lda_dims-1));
        }

        X = trans(pv*tform);
        mean = X*mean;
    }

// ----------------------------------------------------------------------------------------

    struct roc_point
    {
        double true_positive_rate;
        double false_positive_rate;
        double detection_threshold;
    };

    inline std::vector<roc_point> compute_roc_curve (
        const std::vector<double>& true_detections,
        const std::vector<double>& false_detections 
    )
    {
        DLIB_CASSERT(true_detections.size() != 0);
        DLIB_CASSERT(false_detections.size() != 0);

        std::vector<std::pair<double,int> > temp;
        temp.reserve(true_detections.size()+false_detections.size());
        // We use -1 for true labels and +1 for false so when we call std::sort() below it will sort
        // runs with equal detection scores so false come first.  This will avoid it seeming like we
        // can separate true from false when scores are equal in the loop below.
        const int true_label = -1;
        const int false_label = +1;
        for (unsigned long i = 0; i < true_detections.size(); ++i)
            temp.push_back(std::make_pair(true_detections[i], true_label));
        for (unsigned long i = 0; i < false_detections.size(); ++i)
            temp.push_back(std::make_pair(false_detections[i], false_label));

        std::sort(temp.rbegin(), temp.rend());


        std::vector<roc_point> roc_curve;
        roc_curve.reserve(temp.size());

        double num_false_included = 0;
        double num_true_included = 0;
        for (unsigned long i = 0; i < temp.size(); ++i)
        {
            if (temp[i].second == true_label)
                num_true_included++;
            else
                num_false_included++;

            roc_point p;
            p.true_positive_rate = num_true_included/true_detections.size();
            p.false_positive_rate = num_false_included/false_detections.size();
            p.detection_threshold = temp[i].first;
            roc_curve.push_back(p);
        }

        return roc_curve;
    }

// ----------------------------------------------------------------------------------------

    inline std::pair<double,double> equal_error_rate (
        const std::vector<double>& low_vals,
        const std::vector<double>& high_vals 
    )
    {
        if (low_vals.size() == 0 && high_vals.size() == 0)
            return std::make_pair(0,0);
        else if (low_vals.size() == 0) 
            return std::make_pair(0, min(mat(high_vals)));
        else if (high_vals.size() == 0) 
            return std::make_pair(0, max(mat(low_vals))+1);

        // Find the point of equal error rates
        double best_thresh = 0;
        double best_error = 0;
        double best_delta = std::numeric_limits<double>::infinity();
        for (const auto& pt : compute_roc_curve(high_vals, low_vals)) 
        {
            const double false_negative_rate = 1-pt.true_positive_rate;
            const double delta = std::abs(false_negative_rate - pt.false_positive_rate);
            if (delta < best_delta) 
            {
                best_delta = delta;
                best_error = std::max(false_negative_rate, pt.false_positive_rate);
                best_thresh = pt.detection_threshold;
            }
        }

        return std::make_pair(best_error, best_thresh);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LDA_Hh_

