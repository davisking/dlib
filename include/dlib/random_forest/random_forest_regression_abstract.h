// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RANdOM_FOREST_REGRESION_ABSTRACT_H_
#ifdef DLIB_RANdOM_FOREST_REGRESION_ABSTRACT_H_

#include <vector>
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class dense_feature_extractor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for extracting features from objects.  In particular,
                it is designed to be used with the random forest regression tools discussed
                below.

                This particular feature extract does almost nothing since it works on
                vectors in R^n and simply selects elements from each vector.  However, the
                tools below are templated and allow you to design your own feature extractors
                that operate on whatever object types you create.  So for example, maybe
                you want to perform regression on images rather than vectors.  Moreover,
                your feature extraction could be more complex.  Maybe you are selecting
                differences between pairs of pixels in an image or doing something
                involving geometric transforms during feature extraction.  Any of these
                kinds of more complex feature extraction patterns can be realized with the
                random forest tools by implementing your own feature extractor object and
                using it with the random forest objects.

                Therefore, you should consider this dense_feature_extractor as an example
                that documents the interface as well as the simple default extractor for
                use with dense vectors.


            THREAD SAFETY
                It is safe to call const members of this object from multiple threads.  ANY
                USER DEFINED FEATURE EXTRACTORS MUST ALSO MEET THIS GUARONTEE AS WELL SINCE
                IT IS ASSUMED BY THE RANDOM FOREST TRAINING ROUTINES.
        !*/

    public:
        typedef uint32_t feature;
        typedef matrix<double,0,1> sample_type;

        dense_feature_extractor(
        );
        /*!
            ensures
                - #max_num_feats() == 0
        !*/

        void setup (
            const std::vector<sample_type>& x,
            const std::vector<double>& y 
        );
        /*!
            requires
                - x.size() == y.size()
                - x.size() > 0
                - x[0].size() > 0
                - all the vectors in x have the same dimensionality.
            ensures
                - Configures this feature extractor to work on the given training data.
                  For dense feature extractors all we do is record the dimensionality of
                  the training vectors.
                - #max_num_feats() == x[0].size()
                  (In general, setup() sets max_num_feats() to some non-zero value so that
                  the other methods of this object can then be called.  The point of setup() 
                  is to allow a feature extractor to gather whatever statistics it needs from 
                  training data.  That is, more complex feature extraction strategies my
                  themselves be trained from data.)
        !*/

        void get_random_features (
            dlib::rand& rnd,
            size_t num,
            std::vector<feature>& feats
        ) const;
        /*!
            requires
                - max_num_feats() != 0
            ensures
                - #feats.size() == min(num, max_num_feats())
                - This function randomly identifies num features and stores them into feats.  
                  These feature objects can then be used with extract_feature_value() to
                  obtain a value from any particular sample_type object.  This value is the
                  "feature value" used by a decision tree algorithm to deice how to split
                  and traverse trees.   
                - The above two conditions define the behavior of get_random_features() in
                  general. For this specific implementation of the feature extraction interface 
                  this function selects num integer values from the range [0, max_num_feats()), 
                  without replacement.  These values are stored into feats.
        !*/

        double extract_feature_value (
            const sample_type& item,
            const feature& f
        ) const;
        /*!
            requires
                - #max_num_feats() != 0
                - f was produced from a call to get_random_features().
            ensures
                - Extracts the feature value corresponding to f. For this simple dense
                  feature extractor this simply means returning item(f).  But in general
                  you can design feature extractors that do something more complex.
        !*/

        size_t max_num_feats (
        ) const;
        /*!
            ensures
                - returns the number of distinct features this object might extract.  That is,
                  a feature extractor essentially defines a mapping from sample_type objects to
                  vectors in R^max_num_feats().
        !*/
    };

    void serialize(const dense_feature_extractor& item, std::ostream& out);
    void deserialize(dense_feature_extractor& item, std::istream& in);
    /*!
        provides serialization support
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    struct internal_tree_node
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is an internal node in a regression tree.  See the code of
                random_forest_regression_function to see how it is used to create a tree.
        !*/

        uint32_t left;
        uint32_t right;
        float split_threshold;
        typename feature_extractor::feature split_feature;
    };

    template <typename feature_extractor>
    void serialize(const internal_tree_node<feature_extractor>& item, std::ostream& out);
    template <typename feature_extractor>
    void deserialize(internal_tree_node<feature_extractor>& item, std::istream& in);
    /*!
        provides serialization support
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor = dense_feature_extractor
        >
    class random_forest_regression_function
    {
        /*!
            REQUIREMENTS ON feature_extractor
                feature_extractor must be dense_feature_extractor or a type with a
                compatible interface.

            WHAT THIS OBJECT REPRESENTS
                This object represents a regression forest.  This is a collection of
                decision trees that take an object as input and each vote on a real value
                to associate with the object.  The final real value output is the average
                of all the votes from each of the trees.
        !*/

    public:

        typedef feature_extractor feature_extractor_type;
        typedef typename feature_extractor::sample_type sample_type;

        random_forest_regression_function(
        );
        /*!
            ensures
                - #num_trees() == 0
        !*/

        random_forest_regression_function (
            feature_extractor_type&& fe_,
            std::vector<std::vector<internal_tree_node<feature_extractor>>>&& trees_,
            std::vector<std::vector<float>>&& leaves_
        );
        /*!
            requires
                - trees.size() > 0
                - trees.size() = leaves.size()
                - for all valid i:
                    - leaves[i].size() > 0 
                    - trees[i].size()+leaves[i].size() > the maximal left or right index values in trees[i].
                      (i.e. each left or right value must index to some existing internal tree node or leaf node).
            ensures
                - #get_internal_tree_nodes() == trees_
                - #get_tree_leaves() == leaves_
                - #get_feature_extractor() == fe_
        !*/

        size_t get_num_trees(
        ) const;
        /*!
            ensures
                - returns the number of trees in this regression forest.
        !*/

        const std::vector<std::vector<internal_tree_node<feature_extractor>>>& get_internal_tree_nodes (
        ) const; 
        /*!
            ensures
                - returns the internal tree nodes that define the regression trees.
                - get_internal_tree_nodes().size() == get_num_trees()
        !*/

        const std::vector<std::vector<float>>& get_tree_leaves (
        ) const; 
        /*!
            ensures
                - returns the tree leaves that define the regression trees.
                - get_tree_leaves().size() == get_num_trees()
        !*/

        const feature_extractor_type& get_feature_extractor (
        ) const;
        /*!
            ensures
                - returns the feature extractor used by the trees. 
        !*/

        double operator() (
            const sample_type& x
        ) const;
        /*!
            requires
                - get_num_trees() > 0
            ensures
                - Maps x to a real value and returns the value.  To do this, we find the
                  get_num_trees() leaf values associated with x and then return the average
                  of these leaf values.   
        !*/
    };

    void serialize(const random_forest_regression_function& item, std::ostream& out);
    void deserialize(random_forest_regression_function& item, std::istream& in);
    /*!
        provides serialization support
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor = dense_feature_extractor
        >
    class random_forest_regression_trainer
    {
        /*!
            REQUIREMENTS ON feature_extractor
                feature_extractor must be dense_feature_extractor or a type with a
                compatible interface.

            WHAT THIS OBJECT REPRESENTS
                This object implements Breiman's classic random forest regression
                algorithm.  The algorithm learns to map objects, nominally vectors in R^n,
                into the reals.  It essentially optimizes the mean squared error by fitting
                a bunch of decision trees, each of which vote on the output value of the
                regressor. The final prediction is obtained by averaging all the
                predictions. 

                For more information on the algorithm see:
                    Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
        !*/

    public:
        typedef feature_extractor feature_extractor_type;
        typedef random_forest_regression_function<feature_extractor> trained_function_type;
        typedef typename feature_extractor::sample_type sample_type;


        random_forest_regression_trainer (
        );
        /*!
            ensures
                - #get_min_samples_per_leaf() == 5
                - #get_num_trees() == 1000
                - #get_feature_subsampling_frac() == 1.0/3.0
                - #get_feature_extractor() == a default initialized feature extractor.
                - #get_random_seed() == ""
                - this object is not verbose.
        !*/

        const feature_extractor_type& get_feature_extractor (
        ) const;
        /*!
            ensures
                - returns the feature extractor used when train() is invoked.
        !*/

        void set_feature_extractor (
            const feature_extractor_type& feat_extractor
        );
        /*!
            ensures
                - #get_feature_extractor() == feat_extractor
        !*/

        void set_seed (
            const std::string& seed
        );
        /*!
            ensures
                - #get_random_seed() == seed
        !*/

        const std::string& get_random_seed (
        ) const;
        /*!
            ensures
                - A central part of this algorithm is random selection of both training
                  samples and features. This function returns the seed used to initialized
                  the random number generator used for these random selections.
        !*/

        size_t get_num_trees (
        ) const;
        /*!
            ensures
                - Random forests built by this object will contain get_num_trees() trees.
        !*/

        void set_num_trees (
            size_t num
        );
        /*!
            requires
                - num > 0
            ensures
                - #get_num_trees() == num
        !*/

        void set_feature_subsampling_fraction (
            double frac
        );
        /*!
            requires
                - 0 < frac <= 1
            ensures
                - #get_feature_subsampling_frac() == frac
        !*/

        double get_feature_subsampling_frac(
        ) const;
        /*!
            ensures
                - When we build trees, at each node we don't look at all the available
                  features.  We consider only get_feature_subsampling_frac() fraction of
                  them, selected at random.
        !*/

        void set_min_samples_per_leaf (
            size_t num
        );
        /*!
            requires
                - num > 0
            ensures
                - #get_min_samples_per_leaf() == num
        !*/

        size_t get_min_samples_per_leaf(
        ) const;
        /*!
            ensures
                - When building trees, each leaf node in a tree will contain at least
                  get_min_samples_per_leaf() samples.  This means that the output votes of
                  each tree are averages of at least get_min_samples_per_leaf() y values.
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - This object will print status messages to standard out so that the
                  progress of training can be tracked..
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - this object will not print anything to standard out
        !*/

        random_forest_regression_function<feature_extractor> train (
            const std::vector<sample_type>& x,
            const std::vector<double>& y,
            std::vector<double>& oob_values 
        ) const;
        /*!
            requires
                - x.size() == y.size()
                - x.size() > 0
                - Running following code:
                    auto fe = get_feature_extractor()
                    fe.setup(x,y);
                  Must be valid and result in fe.max_num_feats() != 0
            ensures
                - This function fits a regression forest to the given training data.  The
                  goal being to regress x to y in the mean squared sense.  It therefore
                  fits regression trees and returns the resulting random_forest_regression_function 
                  RF, which will have the following properties:
                    - RF.get_num_trees() == get_num_trees()
                    - for all valid i:
                        - RF(x[i]) should output a value close to y[i]
                    - RF.get_feature_extractor() will be a copy of this->get_feature_extractor() 
                      that has been configured by a call the feature extractor's setup() routine.
                  To run the algorithm we need to use a feature extractor.  We obtain a
                  valid feature extractor by making a copy of get_feature_extractor(), then
                  invoking setup(x,y) on it.  This feature extractor is what is used to fit
                  the trees and is also the feature extractor stored in the returned random
                  forest.
                - #oob_values.size() == y.size()
                - for all valid i:  
                    - #oob_values[i] == the "out of bag" prediction for y[i].  It is
                      calculated by computing the average output from trees not trained on
                      y[i].  This is similar to a leave-one-out cross-validation prediction
                      of y[i] and can be used to estimate the generalization error of the
                      regression forest.  
                - Training uses all the available CPU cores.
        !*/

        random_forest_regression_function<feature_extractor> train (
            const std::vector<sample_type>& x,
            const std::vector<double>& y 
        ) const;
        /*!
            requires
                - x.size() == y.size()
                - x.size() > 0
                - Running following code:
                    auto fe = get_feature_extractor()
                    fe.setup(x,y);
                  Must be valid and result in fe.max_num_feats() != 0
            ensures
                - This function is identical to train(x,y,oob_values) except that the
                  oob_values are not calculated.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RANdOM_FOREST_REGRESION_ABSTRACT_H_

