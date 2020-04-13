// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_LOSS_ABSTRACT_H_
#ifdef DLIB_DNn_LOSS_ABSTRACT_H_

#include "core_abstract.h"
#include "../image_processing/full_object_detection_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class EXAMPLE_LOSS_LAYER_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A loss layer is the final layer in a deep neural network.  It computes the
                task loss.  That is, it computes a number that tells us how well the
                network is performing on some task, such as predicting a binary label.  

                You can use one of the loss layers that comes with dlib (defined below).
                But importantly, you are able to define your own loss layers to suit your
                needs.  You do this by creating a class that defines an interface matching
                the one described by this EXAMPLE_LOSS_LAYER_ class.  Note that there is no
                dlib::EXAMPLE_LOSS_LAYER_ type.  It is shown here purely to document the
                interface that a loss layer must implement.

                A loss layer can optionally provide a to_label() method that converts the
                output of a network into a user defined type.  If to_label() is not
                provided then the operator() methods of add_loss_layer will not be
                available, but otherwise everything will function as normal.

                Finally, note that there are two broad flavors of loss layer, supervised
                and unsupervised.  The EXAMPLE_LOSS_LAYER_ as shown here is a supervised
                layer.  To make an unsupervised loss you simply leave out the
                training_label_type typedef and the truth iterator argument to
                compute_loss_value_and_gradient().
        !*/

    public:

        // In most cases training_label_type and output_label_type will be the same type.
        typedef whatever_type_you_use_for_training_labels training_label_type;
        typedef whatever_type_you_use_for_outout_labels   output_label_type;

        EXAMPLE_LOSS_LAYER_ (
        );
        /*!
            ensures
                - EXAMPLE_LOSS_LAYER_ objects are default constructable.
        !*/

        EXAMPLE_LOSS_LAYER_ (
            const EXAMPLE_LOSS_LAYER_& item
        );
        /*!
            ensures
                - EXAMPLE_LOSS_LAYER_ objects are copy constructable.
        !*/

        // Implementing to_label() is optional.
        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of
                  layers_abstract.h.
                - input_tensor was given as input to the network sub and the outputs are
                  now visible in layer<i>(sub).get_output(), for all valid i.
                - input_tensor.num_samples() > 0
                - input_tensor.num_samples()%sub.sample_expansion_factor() == 0.
                - iter == an iterator pointing to the beginning of a range of
                  input_tensor.num_samples()/sub.sample_expansion_factor() elements.  Moreover,
                  they must be output_label_type elements.
            ensures
                - Converts the output of the provided network to output_label_type objects and
                  stores the results into the range indicated by iter.  In particular, for
                  all valid i, it will be the case that:
                    *(iter+i/sub.sample_expansion_factor()) is populated based on the output of
                    sub and corresponds to the ith sample in input_tensor.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of
                  layers_abstract.h.
                - input_tensor was given as input to the network sub and the outputs are
                  now visible in layer<i>(sub).get_output(), for all valid i.
                - input_tensor.num_samples() > 0
                - input_tensor.num_samples()%sub.sample_expansion_factor() == 0.
                - for all valid i:
                    - layer<i>(sub).get_gradient_input() has the same dimensions as
                      layer<i>(sub).get_output().
                    - layer<i>(sub).get_gradient_input() contains all zeros (i.e.
                      initially, all input gradients are 0).
                - truth == an iterator pointing to the beginning of a range of
                  input_tensor.num_samples()/sub.sample_expansion_factor() elements.  Moreover,
                  they must be training_label_type elements.
                - for all valid i:
                    - *(truth+i/sub.sample_expansion_factor()) is the label of the ith sample in
                      input_tensor.
            ensures
                - This function computes a loss function that describes how well the output
                  of sub matches the expected labels given by truth.  Let's write the loss
                  function as L(input_tensor, truth, sub).  
                - Then compute_loss_value_and_gradient() computes the gradient of L() with
                  respect to the outputs in sub.  Specifically, compute_loss_value_and_gradient() 
                  assigns the gradients into sub by performing the following tensor
                  assignments, for all valid i: 
                    - layer<i>(sub).get_gradient_input() = the gradient of
                      L(input_tensor,truth,sub) with respect to layer<i>(sub).get_output().
                      Note that, since get_gradient_input() is zero initialized, you don't
                      have to write gradient information to layers that have a zero
                      loss gradient.
                - returns L(input_tensor,truth,sub)
        !*/
    };

    std::ostream& operator<<(std::ostream& out, const EXAMPLE_LOSS_LAYER_& item);
    /*!
        print a string describing this layer.
    !*/

    void to_xml(const EXAMPLE_LOSS_LAYER_& item, std::ostream& out);
    /*!
        This function is optional, but required if you want to print your networks with
        net_to_xml().  Therefore, to_xml() prints a layer as XML.
    !*/

    void serialize(const EXAMPLE_LOSS_LAYER_& item, std::ostream& out);
    void deserialize(EXAMPLE_LOSS_LAYER_& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    // For each loss layer you define, always define an add_loss_layer template so that
    // layers can be easily composed.  Moreover, the convention is that the layer class
    // ends with an _ while the add_loss_layer template has the same name but without the
    // trailing _.
    template <typename SUBNET>
    using EXAMPLE_LOSS_LAYER = add_loss_layer<EXAMPLE_LOSS_LAYER_, SUBNET>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class loss_binary_hinge_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the hinge loss, which is
                appropriate for binary classification problems.  Therefore, the possible
                labels when using this loss are +1 and -1.  Moreover, it will cause the
                network to produce outputs > 0 when predicting a member of the +1 class and
                values < 0 otherwise.
        !*/
    public:

        typedef float training_label_type;
        typedef float output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the raw score for each classified object.  If the score
            is > 0 then the classifier is predicting the +1 class, otherwise it is
            predicting the -1 class.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all values pointed to by truth are +1 or -1.
        !*/

    };

    template <typename SUBNET>
    using loss_binary_hinge = add_loss_layer<loss_binary_hinge_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_binary_log_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the log loss, which is
                appropriate for binary classification problems.  Therefore, there are two possible
                classes of labels: positive (> 0) and negative (< 0) when using this loss.
                The absolute value of the label represents its weight.  Putting a larger weight
                on a sample increases the importance of getting its prediction correct during 
                training.  A good rule of thumb is to use weights with absolute value 1 unless 
                you have a very unbalanced training dataset, in that case, give larger weight
                to the class with less training examples.
                
                This loss will cause the network to produce outputs > 0 when predicting a
                member of the positive class and values < 0 otherwise.

                To be more specific, this object contains a sigmoid layer followed by a 
                cross-entropy layer.  
        !*/
    public:

        typedef float training_label_type;
        typedef float output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the raw score for each classified object.  If the score
            is > 0 then the classifier is predicting the +1 class, otherwise it is
            predicting the -1 class.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all values pointed to by truth are non-zero.  Nominally they should be +1 or -1,
                  each indicating the desired class label.
        !*/

    };

    template <typename SUBNET>
    using loss_binary_log = add_loss_layer<loss_binary_log_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the multiclass logistic
                regression loss (e.g. negative log-likelihood loss), which is appropriate
                for multiclass classification problems.  This means that the possible
                labels when using this loss are integers >= 0.  
                
                Moreover, if after training you were to replace the loss layer of the
                network with a softmax layer, the network outputs would give the
                probabilities of each class assignment.  That is, if you have K classes
                then the network should output tensors with the tensor::k()'th dimension
                equal to K.  Applying softmax to these K values gives the probabilities of
                each class.  The index into that K dimensional vector with the highest
                probability is the predicted class label.
        !*/

    public:

        typedef unsigned long training_label_type;
        typedef unsigned long output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted class for each classified object.  The number
            of possible output classes is sub.get_output().k().
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all values pointed to by truth are < sub.get_output().k()
        !*/

    };

    template <typename SUBNET>
    using loss_multiclass_log = add_loss_layer<loss_multiclass_log_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <typename label_type>
    struct weighted_label
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents the truth label of a single sample, together with
                an associated weight (the higher the weight, the more emphasis the
                corresponding sample is given during the training).
                This object is used in the following loss layers:
                    - loss_multiclass_log_weighted_ with unsigned long as label_type
                    - loss_multiclass_log_per_pixel_weighted_ with uint16_t as label_type,
                      since, in semantic segmentation, 65536 classes ought to be enough for
                      anybody. 
        !*/
        weighted_label()
        {}

        weighted_label(label_type label, float weight = 1.f)
            : label(label), weight(weight)
        {}

        // The ground truth label
        label_type label{};

        // The weight of the corresponding sample
        float weight = 1.f;
    };

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_weighted_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the multiclass logistic
                regression loss (e.g. negative log-likelihood loss), which is appropriate
                for multiclass classification problems.  It is basically just like the
                loss_multiclass_log except that it lets you define per-sample weights,
                which might be useful e.g. if you want to emphasize rare classes while
                training.  If the classification problem is difficult, a flat weight
                structure may lead the network to always predict the most common label,
                in particular if the degree of imbalance is high.  To emphasize a certain
                class or classes, simply increase the weights of the corresponding samples,
                relative to the weights of other pixels.

                Note that if you set all the weights equals to 1, then you get
                loss_multiclass_log_ as a special case.
        !*/

    public:

        typedef dlib::weighted_label<unsigned long> weighted_label;
        typedef weighted_label training_label_type;
        typedef unsigned long output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted class for each classified object.  The number
            of possible output classes is sub.get_output().k().
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all values pointed to by truth are < sub.get_output().k()
        !*/

    };

    template <typename SUBNET>
    using loss_multiclass_log_weighted = add_loss_layer<loss_multiclass_log_weighted_, SUBNET>;// ----------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------

    class loss_multimulticlass_log_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements a collection of
                multiclass classifiers.  An example will make its use clear.  So suppose,
                for example, that you want to make something that takes a picture of a
                vehicle and answers the following questions:
                    - What type of vehicle is it? A sedan or a truck?
                    - What color is it? red, green, blue, gray, or black?
                You need two separate multi-class classifiers to do this.  One to decide
                the type of vehicle, and another to decide the color.  The
                loss_multimulticlass_log_ allows you to pack these two classifiers into one
                neural network.  This means that when you use the network to process an
                image it will output 2 labels for each image, the type label and the color
                label.  

                To create a loss_multimulticlass_log_ for the above case you would
                construct it as follows:
                    std::map<std::string,std::vector<std::string>> labels;
                    labels["type"] = {"sedan", "truck"};
                    labels["color"] = {"red", "green", "blue", "gray", "black"};
                    loss_multimulticlass_log_ myloss(labels);
                Then you could use myloss with a network object and train it to do this
                task.  More generally, you can use any number of classifiers and labels
                when using this object.  Finally, each of the classifiers uses a standard
                multi-class logistic regression loss.
        !*/

    public:

        loss_multimulticlass_log_(
        ); 
        /*!
            ensures
                - #number_of_labels() == 0
                - #get_labels().size() == 0
        !*/

        loss_multimulticlass_log_ (
            const std::map<std::string,std::vector<std::string>>& labels
        );
        /*!
            requires
                - Each vector in labels must contain at least 2 strings.  I.e. each
                  classifier must have at least two possible labels.
            ensures
                - #number_of_labels() == the total number of strings in all the
                  std::vectors in labels.
                - #number_of_classifiers() == labels.size()
                - #get_labels() == labels
        !*/

        unsigned long number_of_labels(
        ) const; 
        /*!
            ensures
                - returns the total number of labels known to this loss.  This is the count of 
                  all the labels in each classifier.
        !*/

        unsigned long number_of_classifiers(
        ) const; 
        /*!
            ensures
                - returns the number of classifiers defined by this loss.
        !*/

        std::map<std::string,std::vector<std::string>> get_labels ( 
        ) const;
        /*!
            ensures
                - returns the names of the classifiers and labels used by this loss.  In
                  particular, if the returned object is L then:
                    - L[CLASS] == the set of labels used by the classifier CLASS.
                    - L.size() == number_of_classifiers()
                    - The count of strings in the vectors in L == number_of_labels()
        !*/

        class classifier_output
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object stores the predictions from one of the classifiers in
                    loss_multimulticlass_log_.  It allows you to find out the most likely
                    string label predicted by that classifier, as well as get the class
                    conditional probability of any of the classes in the classifier.
            !*/

        public:

            classifier_output(
            );
            /*!
                ensures
                    - #num_classes() == 0
            !*/

            size_t num_classes(
            ) const; 
            /*!
                ensures
                    - returns the number of possible classes output by this classifier.
            !*/

            double probability_of_class (
                size_t i
            ) const;
            /*!
                requires
                    - i < num_classes()
                ensures
                    - returns the probability that the true class has a label of label(i).
                    - The sum of probability_of_class(j) for j in the range [0, num_classes()) is always 1.
            !*/

            const std::string& label(
                size_t i
            ) const;
            /*!
                requires
                    - i < num_classes()
                ensures
                    - returns the string label for the ith class.
            !*/

            operator std::string(
            ) const;
            /*!
                requires
                    - num_classes() != 0
                ensures
                    - returns the string label for the most probable class.
            !*/

            friend std::ostream& operator<< (std::ostream& out, const classifier_output& item);
            /*!
                requires
                    - num_classes() != 0
                ensures
                    - prints the most probable class label to out.
            !*/

        };

        // Both training_label_type and output_label_type should always have sizes equal to
        // number_of_classifiers().  That is, the std::map should have an entry for every
        // classifier known to this loss.
        typedef std::map<std::string,std::string> training_label_type;
        typedef std::map<std::string,classifier_output> output_label_type;


        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - number_of_labels() != 0
                - sub.get_output().k() == number_of_labels()
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - number_of_labels() != 0
                - sub.get_output().k() == number_of_labels()
                    It should be noted that the last layer in your network should usually
                    be an fc layer.  If so, you can satisfy this requirement of k() being
                    number_of_labels() by calling set_num_outputs() prior to training your
                    network like so:
                    your_network.subnet().layer_details().set_num_outputs(your_network.loss_details().number_of_labels());
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - All the std::maps pointed to by truth contain entries for all the
                  classifiers known to this loss.  That is, it must be valid to call
                  truth[i][classifier] for any of the classifiers known to this loss.  To
                  say this another way, all the training samples must contain labels for
                  each of the classifiers defined by this loss.

                  To really belabor this, this also means that truth[i].size() ==
                  get_labels().size() and that both truth[i] and get_labels() have the same
                  set of key strings.  It also means that the value strings in truth[i]
                  must be strings known to the loss, i.e. they are valid labels according
                  to get_labels().
        !*/
    };

    template <typename SUBNET>
    using loss_multimulticlass_log = add_loss_layer<loss_multimulticlass_log_, SUBNET>;

    // Allow comparison between classifier_outputs and std::string to check if the
    // predicted class is a particular string.
    inline bool operator== (const std::string& lhs, const loss_multimulticlass_log_::classifier_output& rhs)
    { return lhs == static_cast<const std::string&>(rhs); }
    inline bool operator== (const loss_multimulticlass_log_::classifier_output& lhs, const std::string& rhs)
    { return rhs == static_cast<const std::string&>(lhs); }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    enum class use_image_pyramid : uint8_t
    {
        no,
        yes
    };

    struct mmod_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object contains all the parameters that control the behavior of loss_mmod_.
        !*/

    public:

        struct detector_window_details
        {
            detector_window_details() = default; 
            detector_window_details(unsigned long w, unsigned long h) : width(w), height(h) {}
            detector_window_details(unsigned long w, unsigned long h, const std::string& l) : width(w), height(h), label(l) {}

            unsigned long width = 0;
            unsigned long height = 0;
            std::string label;

            friend inline void serialize(const detector_window_details& item, std::ostream& out);
            friend inline void deserialize(detector_window_details& item, std::istream& in);
        };

        mmod_options() = default;

        // This kind of object detector is a sliding window detector.  The detector_windows
        // field determines how many sliding windows we will use and what the shape of each
        // window is.  It also determines the output label applied to each detection
        // identified by each window.  Since you will usually use the MMOD loss with an
        // image pyramid, the detector sizes also determine the size of the smallest object
        // you can detect.
        std::vector<detector_window_details> detector_windows;

        // These parameters control how we penalize different kinds of mistakes.  See 
        // Max-Margin Object Detection by Davis E. King (http://arxiv.org/abs/1502.00046)
        // for further details.
        double loss_per_false_alarm = 1;
        double loss_per_missed_target = 1;

        // A detection must have an intersection-over-union value greater than this for us
        // to consider it a match against a ground truth box.
        double truth_match_iou_threshold = 0.5;

        // When doing non-max suppression, we use overlaps_nms to decide if a box overlaps
        // an already output detection and should therefore be thrown out.
        test_box_overlap overlaps_nms = test_box_overlap(0.4);

        // Any mmod_rect in the training data that has its ignore field set to true defines
        // an "ignore zone" in an image.  Any detection from that area is totally ignored
        // by the optimizer.  Therefore, this overlaps_ignore field defines how we decide
        // if a box falls into an ignore zone.  You use these ignore zones if there are
        // objects in your dataset that you are unsure if you want to detect or otherwise
        // don't care if the detector gets them or not.  
        test_box_overlap overlaps_ignore;

        // Usually the detector would be scale-invariant, and used with an image pyramid.
        // However, sometimes scale-invariance may not be desired.
        use_image_pyramid assume_image_pyramid = use_image_pyramid::yes;

        // By default, the mmod loss doesn't train any bounding box regression model.  But
        // if you set use_bounding_box_regression == true then it expects the network to
        // output a tensor with detector_windows.size()*5 channels rather than just
        // detector_windows.size() channels.  The 4 extra channels per window are trained
        // to give a bounding box regression output that improves the positioning of the
        // output detection box.
        bool use_bounding_box_regression = false; 
        // When using bounding box regression, bbr_lambda determines how much you care
        // about getting the bounding box shape correct vs just getting the detector to
        // find objects.  That is, the objective function being optimized is
        // basic_mmod_loss + bbr_lambda*bounding_box_regression_loss.  So setting
        // bbr_lambda to a larger value will cause the overall loss to care more about
        // getting the bounding box shape correct.
        double bbr_lambda = 100; 

        mmod_options (
            const std::vector<std::vector<mmod_rect>>& boxes,
            const unsigned long target_size,      
            const unsigned long min_target_size,   
            const double min_detector_window_overlap_iou = 0.75
        );
        /*!
            requires
                - 0 < min_target_size <= target_size
                - 0.5 < min_detector_window_overlap_iou < 1
            ensures
                - use_image_pyramid_ == use_image_pyramid::yes
                - This function should be used when scale-invariance is desired, and
                  input_rgb_image_pyramid is therefore used as the input layer.
                - This function tries to automatically set the MMOD options to reasonable
                  values, assuming you have a training dataset of boxes.size() images, where
                  the ith image contains objects boxes[i] you want to detect.
                - The most important thing this function does is decide what detector
                  windows should be used.  This is done by finding a set of detector
                  windows that are sized such that:
                    - When slid over an image pyramid, each box in boxes will have an
                      intersection-over-union with one of the detector windows of at least
                      min_detector_window_overlap_iou.  That is, we will make sure that
                      each box in boxes could potentially be detected by one of the
                      detector windows.  This essentially comes down to picking detector
                      windows with aspect ratios similar to the aspect ratios in boxes.
                      Note that we also make sure that each box can be detected by a window
                      with the same label.  For example, if all the boxes had the same
                      aspect ratio but there were 4 different labels used in boxes then
                      there would be 4 resulting detector windows, one for each label.
                    - The longest edge of each detector window is target_size pixels in
                      length, unless the window's shortest side would be less than
                      min_target_size pixels in length.  In this case the shortest side
                      will be set to min_target_size length, and the other side sized to
                      preserve the aspect ratio of the window.  
                  This means that target_size and min_target_size control the size of the
                  detector windows, while the aspect ratios of the detector windows are
                  automatically determined by the contents of boxes.  It should also be
                  emphasized that the detector isn't going to be able to detect objects
                  smaller than any of the detector windows.  So consider that when setting
                  these sizes.
                - This function will also set the overlaps_nms tester to the most
                  restrictive tester that doesn't reject anything in boxes.
        !*/

        mmod_options (
            use_image_pyramid use_image_pyramid,
            const std::vector<std::vector<mmod_rect>>& boxes,
            const double min_detector_window_overlap_iou = 0.75
        );
        /*!
            requires
                - use_image_pyramid == use_image_pyramid::no
                - 0.5 < min_detector_window_overlap_iou < 1
            ensures
                - This function should be used when scale-invariance is not desired, and
                  there is no intention to apply an image pyramid.
                - This function tries to automatically set the MMOD options to reasonable
                  values, assuming you have a training dataset of boxes.size() images, where
                  the ith image contains objects boxes[i] you want to detect.
                - The most important thing this function does is decide what detector
                  windows should be used.  This is done by finding a set of detector
                  windows that are sized such that:
                    - When slid over an image, each box in boxes will have an
                      intersection-over-union with one of the detector windows of at least
                      min_detector_window_overlap_iou.  That is, we will make sure that
                      each box in boxes could potentially be detected by one of the
                      detector windows.
                - This function will also set the overlaps_nms tester to the most
                  restrictive tester that doesn't reject anything in boxes.
        !*/
    };

    void serialize(const mmod_options& item, std::ostream& out);
    void deserialize(mmod_options& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    class loss_mmod_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the Max Margin Object
                Detection loss defined in the paper:
                    Max-Margin Object Detection by Davis E. King (http://arxiv.org/abs/1502.00046).
               
                This means you use this loss if you want to detect the locations of objects
                in images.

                It should also be noted that this loss layer requires an input layer that
                defines the following functions:
                    - image_contained_point()
                    - tensor_space_to_image_space()
                    - image_space_to_tensor_space()
                A reference implementation of them and their definitions can be found in
                the input_rgb_image_pyramid object, which is the recommended input layer to
                be used with loss_mmod_.
        !*/

    public:

        typedef std::vector<mmod_rect> training_label_type;
        typedef std::vector<mmod_rect> output_label_type;

        loss_mmod_(
        );
        /*!
            ensures
                - #get_options() == mmod_options()
        !*/

        loss_mmod_(
            mmod_options options_
        );
        /*!
            ensures
                - #get_options() == options_
        !*/

        const mmod_options& get_options (
        ) const;
        /*!
            ensures
                - returns the options object that defines the general behavior of this loss layer.
        !*/

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter,
            double adjust_threshold = 0
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            Also, the output labels are std::vectors of mmod_rects where, for each mmod_rect R,
            we have the following interpretations:
                - R.rect == the location of an object in the image.
                - R.detection_confidence the score for the object, the bigger the score the
                  more confident the detector is that an object is really there.  Only
                  objects with a detection_confidence > adjust_threshold are output.  So if
                  you want to output more objects (that are also of less confidence) you
                  can call to_label() with a smaller value of adjust_threshold.
                - R.ignore == false (this value is unused by to_label()).
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            Also, the loss value returned is roughly equal to the average number of
            mistakes made per image.  This is the sum of false alarms and missed
            detections, weighted by the loss weights for these types of mistakes specified
            in the mmod_options.
        !*/
    };

    template <typename SUBNET>
    using loss_mmod = add_loss_layer<loss_mmod_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_metric_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it allows you to learn to map objects
                into a vector space where objects sharing the same class label are close to
                each other, while objects with different labels are far apart.   

                To be specific, it optimizes the following loss function which considers
                all pairs of objects in a mini-batch and computes a different loss depending 
                on their respective class labels.  So if objects A1 and A2 in a mini-batch
                share the same class label then their contribution to the loss is:
                    max(0, length(A1-A2)-get_distance_threshold() + get_margin())

                While if A1 and B1 have different class labels then their contribution to
                the loss function is:
                    max(0, get_distance_threshold()-length(A1-B1) + get_margin())

                Therefore, this loss layer optimizes a version of the hinge loss.
                Moreover, the loss is trying to make sure that all objects with the same
                label are within get_distance_threshold() distance of each other.
                Conversely, if two objects have different labels then they should be more
                than get_distance_threshold() distance from each other in the learned
                embedding.  So this loss function gives you a natural decision boundary for
                deciding if two objects are from the same class.

                Finally, the loss balances the number of negative pairs relative to the
                number of positive pairs.  Therefore, if there are N pairs that share the
                same identity in a mini-batch then the algorithm will only include the N
                worst non-matching pairs in the loss.  That is, the algorithm performs hard
                negative mining on the non-matching pairs.  This is important since there
                are in general way more non-matching pairs than matching pairs.  So to
                avoid imbalance in the loss this kind of hard negative mining is useful.
        !*/
    public:

        typedef unsigned long training_label_type;
        typedef matrix<float,0,1> output_label_type;

        loss_metric_(
        );
        /*!
            ensures
                - #get_margin() == 0.04
                - #get_distance_threshold() == 0.6
        !*/

        loss_metric_(
            float margin,
            float dist_thresh
        );
        /*!
            requires
                - margin > 0
                - dist_thresh > 0
            ensures
                - #get_margin() == margin
                - #get_distance_threshold() == dist_thresh
        !*/

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            This loss expects the network to produce a single vector (per sample) as
            output.  This vector is the learned embedding.  Therefore, to_label() just
            copies these output vectors from the network into the output label_iterators
            given to this function, one for each sample in the input_tensor.
        !*/

        float get_margin() const; 
        /*!
            ensures
                - returns the margin value used by the loss function.  See the discussion
                  in WHAT THIS OBJECT REPRESENTS for details.
        !*/

        float get_distance_threshold() const; 
        /*!
            ensures
                - returns the distance threshold value used by the loss function.  See the discussion
                  in WHAT THIS OBJECT REPRESENTS for details.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient() 
            except it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
        !*/

    };

    template <typename SUBNET>
    using loss_metric = add_loss_layer<loss_metric_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_ranking_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the pairwise ranking
                loss described in the paper:
                    Optimizing Search Engines using Clickthrough Data by Thorsten Joachims

                This is the same loss function used by the dlib::svm_rank_trainer object.
                Therefore, it is generally appropriate when you have a two class problem
                and you want to learn a function that ranks one class before the other.  

                So for example, suppose you have two classes of data.  Objects of type A
                and objects of type B.  Moreover, suppose that you want to sort the objects
                so that A objects always come before B objects.  This loss will help you
                learn a function that assigns a real number to each object such that A
                objects get a larger number assigned to them than B objects.  This lets you
                then sort the objects according to the output of the neural network and
                obtain the desired result of having A objects come before B objects.

                The training labels should be positive values for objects you want to get
                high scores and negative for objects that should get small scores.  So
                relative to our A/B example, you would give A objects labels of +1 and B
                objects labels of -1.  This should cause the learned network to give A
                objects large positive values and B objects negative values.


                Finally, the specific loss function is:
                    For all pairs of positive vs negative training examples A_i and B_j respectively:
                      sum_ij:  max(0, B_i - A_j + margin_ij)
                where margin_ij = the label for A_j minus the label for B_i.  If you
                always use +1 and -1 labels then the margin is always 2.  However, this
                formulation allows you to give certain training samples different weight by
                adjusting the training labels appropriately.  
        !*/

    public:

        typedef float training_label_type;
        typedef float output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted ranking score.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
        !*/

    };

    template <typename SUBNET>
    using loss_ranking = add_loss_layer<loss_ranking_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_epsilon_insensitive_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the epsilon insensitive
                loss, which is appropriate for regression problems.  In particular, this
                loss function is;
                    loss(y1,y2) = abs(y1-y2)<epsilon ? 0 : abs(y1-y2)-epsilon

                Therefore, the loss is basically just the abs() loss except there is a dead
                zone around zero, causing the loss to not care about mistakes of magnitude
                smaller than epsilon.
        !*/
    public:

        typedef float training_label_type;
        typedef float output_label_type;

        loss_epsilon_insensitive_(
        ) = default;
        /*!
            ensures
                - #get_epsilon() == 1
        !*/

        loss_epsilon_insensitive_(
            double eps
        );
        /*!
            requires
                - eps >= 0
            ensures
                - #get_epsilon() == eps
        !*/

        double get_epsilon (
        ) const;
        /*!
            ensures
                - returns the epsilon value used in the loss function.  Mistakes in the
                  regressor smaller than get_epsilon() are ignored by the loss function.
        !*/

        void set_epsilon(
            double eps
        );
        /*!
            requires
                - eps >= 0
            ensures
                - #get_epsilon() == eps
        !*/

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted continuous variable.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
        !*/

    };

    template <typename SUBNET>
    using loss_epsilon_insensitive = add_loss_layer<loss_epsilon_insensitive_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_mean_squared_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the mean squared loss, which is
                appropriate for regression problems.
        !*/
    public:

        typedef float training_label_type;
        typedef float output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted continuous variable.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
        !*/

    };

    template <typename SUBNET>
    using loss_mean_squared = add_loss_layer<loss_mean_squared_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_mean_squared_multioutput_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the mean squared loss,
                which is appropriate for regression problems.  It is basically just like
                loss_mean_squared_ except that it lets you define multiple outputs instead
                of just 1.
        !*/
    public:

        typedef matrix<float> training_label_type;
        typedef matrix<float> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted continuous variable.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - (*(truth + idx)).nc() == 1 for all idx such that 0 <= idx < sub.get_output().num_samples()
                - (*(truth + idx)).nr() == sub.get_output().k() for all idx such that 0 <= idx < sub.get_output().num_samples()
        !*/

    };

    template <typename SUBNET>
    using loss_mean_squared_multioutput = add_loss_layer<loss_mean_squared_multioutput_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_binary_log_per_pixel_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the log loss, which is
                appropriate for binary classification problems.  It is basically just like
                loss_binary_log_ except that it lets you define matrix outputs instead
                of scalar outputs.  It should be useful, for example, in segmentation
                where we want to classify each pixel of an image, and also get at least
                some sort of confidence estimate for each pixel.
        !*/
    public:

        typedef matrix<float> training_label_type;
        typedef matrix<float> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the raw score for each classified object.  If the score
            is > 0 then the classifier is predicting the +1 class, otherwise it is
            predicting the -1 class.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all pixel values pointed to by truth correspond to the desired target values.
                  Nominally they should be +1 or -1, each indicating the desired class label,
                  or 0 to indicate that the corresponding pixel is to be ignored.
        !*/

    };

    template <typename SUBNET>
    using loss_binary_log_per_pixel = add_loss_layer<loss_binary_log_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_per_pixel_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the multiclass logistic
                regression loss (e.g. negative log-likelihood loss), which is appropriate
                for multiclass classification problems.  It is basically just like
                loss_multiclass_log_ except that it lets you define matrix outputs instead
                of scalar outputs.  It should be useful, for example, in semantic
                segmentation where we want to classify each pixel of an image.
        !*/
    public:

        // In semantic segmentation, if you don't know the ground-truth of some pixel,
        // set the label of that pixel to this value. When you do so, the pixel will be
        // ignored when computing gradients.
        static const uint16_t label_to_ignore = std::numeric_limits<uint16_t>::max();

        // In semantic segmentation, 65535 classes ought to be enough for anybody.
        typedef matrix<uint16_t> training_label_type;
        typedef matrix<uint16_t> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted class for each classified element.  The number
            of possible output classes is sub.get_output().k().
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all values pointed to by truth are < sub.get_output().k() or are equal to label_to_ignore.
        !*/

    };

    template <typename SUBNET>
    using loss_multiclass_log_per_pixel = add_loss_layer<loss_multiclass_log_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_per_pixel_weighted_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the multiclass logistic
                regression loss (e.g. negative log-likelihood loss), which is appropriate
                for multiclass classification problems.  It is basically just like
                loss_multiclass_log_per_pixel_ except that it lets you define per-pixel
                weights, which may be useful e.g. if you want to emphasize rare classes
                while training.  (If the classification problem is difficult, a flat weight
                structure may lead the network to always predict the most common label, in
                particular if the degree of imbalance is high.  To emphasize a certain
                class or classes, simply increase the weights of the corresponding pixels,
                relative to the weights of the other pixels.)

                Note that if you set the weight to 0 whenever a pixel's label is equal to
                loss_multiclass_log_per_pixel_::label_to_ignore, and to 1 otherwise, then
                you essentially get loss_multiclass_log_per_pixel_ as a special case.
        !*/
    public:

        typedef dlib::weighted_label<uint16_t> weighted_label;
        typedef matrix<weighted_label> training_label_type;
        typedef matrix<uint16_t> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output label is the predicted class for each classified element.  The number
            of possible output classes is sub.get_output().k().
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - all labels pointed to by truth are < sub.get_output().k(), or the corresponding weight
                  is zero.
        !*/

    };

    template <typename SUBNET>
    using loss_multiclass_log_per_pixel_weighted = add_loss_layer<loss_multiclass_log_per_pixel_weighted_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_mean_squared_per_pixel_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the mean squared loss,
                which is appropriate for regression problems.  It is basically just like
                loss_mean_squared_multioutput_ except that it lets you define matrix or
                image outputs, instead of vector.
        !*/
    public:

        typedef matrix<float> training_label_type;
        typedef matrix<float> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output labels are the predicted continuous variables.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().k() == 1
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - for all idx such that 0 <= idx < sub.get_output().num_samples():
                    - sub.get_output().nr() == (*(truth + idx)).nr()
                    - sub.get_output().nc() == (*(truth + idx)).nc()
        !*/
    };

    template <typename SUBNET>
    using loss_mean_squared_per_pixel = add_loss_layer<loss_mean_squared_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template<long _num_channels>
    class loss_mean_squared_per_channel_and_pixel_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, it implements the mean squared loss,
                which is appropriate for regression problems.  It is basically just like
                loss_mean_squared_per_pixel_ except that it computes the loss over all
                channels, not just the first one.
        !*/
    public:

        typedef std::array<matrix<float>, _num_channels> training_label_type;
        typedef std::array<matrix<float>, _num_channels> output_label_type;


        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.get_output().k() == _num_channels
                - sub.sample_expansion_factor() == 1
            and the output labels are the predicted continuous variables.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().k() == _num_channels
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - for all idx such that 0 <= idx < sub.get_output().num_samples():
                    - sub.get_output().nr() == (*(truth + idx)).nr()
                    - sub.get_output().nc() == (*(truth + idx)).nc()
        !*/
    };

    template <long num_channels, typename SUBNET>
    using loss_mean_squared_per_channel_and_pixel = add_loss_layer<loss_mean_squared_per_channel_and_pixel_<num_channels>, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_dot_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, selecting this loss means you want
                maximize the dot product between the output of a network and a set of
                training vectors.  The loss is therefore the negative dot product.  To be
                very specific, if X is the output vector of a network and Y is a training
                label (also a vector), then the loss for this training sample is: -dot(X,Y)
        !*/

    public:

        typedef matrix<float,0,1> training_label_type;
        typedef matrix<float,0,1> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
            and the output labels are simply the final network outputs stuffed into a
            vector.  To be very specific, the output is the following for all valid i:
                *(iter+i) == trans(rowm(mat(sub.get_output()),i))
        !*/


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::compute_loss_value_and_gradient()
            except it has the additional calling requirements that:
                - sub.get_output().num_samples() == input_tensor.num_samples()
                - sub.sample_expansion_factor() == 1
                - Let NETWORK_OUTPUT_DIMS == sub.get_output().size()/sub.get_output().num_samples()
                - for all idx such that 0 <= idx < sub.get_output().num_samples():
                    - NETWORK_OUTPUT_DIMS == (*(truth + idx)).size()
        !*/
    };

    template <typename SUBNET>
    using loss_dot = add_loss_layer<loss_dot_, SUBNET>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_LOSS_ABSTRACT_H_

