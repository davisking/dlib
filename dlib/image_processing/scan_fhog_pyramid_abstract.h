// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SCAN_fHOG_PYRAMID_ABSTRACT_H__
#ifdef DLIB_SCAN_fHOG_PYRAMID_ABSTRACT_H__

#include <vector>
#include "../image_transforms/fhog_abstract.h"
#include "object_detector_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type
        >
    matrix<unsigned char> draw_fhog (
        const object_detector<scan_fhog_pyramid<Pyramid_type> >& detector,
        const unsigned long weight_index = 0,
        const long cell_draw_size = 15
    );
    /*!
        requires
            - cell_draw_size > 0
            - weight_index < detector.num_detectors()
            - detector.get_w().size() >= detector.get_scanner().get_num_dimensions()
              (i.e. the detector must have been populated with a HOG filter)
        ensures
            - Converts the HOG filters in the given detector (specifically, the filters in
              detector.get_w(weight_index)) into an image suitable for display on the
              screen.  In particular, we draw all the HOG cells into a grayscale image in a
              way that shows the magnitude and orientation of the gradient energy in each
              cell.  The resulting image is then returned.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type
        >
    unsigned long num_separable_filters (
        const object_detector<scan_fhog_pyramid<Pyramid_type> >& detector
    );
    /*!
        requires
            - detector.get_w().size() >= detector.get_scanner().get_num_dimensions()
              (i.e. the detector must have been populated with a HOG filter)
        ensures
            - Returns the number of separable filters necessary to represent the HOG
              filters in the given detector.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type
        >
    class scan_fhog_pyramid : noncopyable
    {
        /*!
            REQUIREMENTS ON Pyramid_type
                - Must be one of the pyramid_down objects defined in
                  dlib/image_transforms/image_pyramid_abstract.h or an object with a
                  compatible interface

            INITIAL VALUE
                - get_padding()   == 1
                - get_cell_size() == 8
                - get_detection_window_width()   == 64
                - get_detection_window_height()  == 64
                - get_max_pyramid_levels()       == 1000
                - get_min_pyramid_layer_width()  == 64
                - get_min_pyramid_layer_height() == 64
                - get_nuclear_norm_regularization_strength() == 0

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for running a fixed sized sliding window classifier
                over an image pyramid.  In particular,  it slides a linear classifier over
                a HOG pyramid as discussed in the paper:  
                    Histograms of Oriented Gradients for Human Detection by Navneet Dalal
                    and Bill Triggs, CVPR 2005
                However, we augment the method slightly to use the version of HOG features 
                from: 
                    Object Detection with Discriminatively Trained Part Based Models by
                    P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
                    IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010
                Since these HOG features have been shown to give superior performance. 

            THREAD SAFETY
                Concurrent access to an instance of this object is not safe and should be
                protected by a mutex lock except for the case where you are copying the
                configuration (via copy_configuration()) of a scan_fhog_pyramid object to
                many other threads.  In this case, it is safe to copy the configuration of
                a shared object so long as no other operations are performed on it.
        !*/

    public:
        typedef matrix<double,0,1> feature_vector_type;
        typedef Pyramid_type pyramid_type;

        scan_fhog_pyramid (
        );  
        /*!
            ensures
                - this object is properly initialized
        !*/

        template <
            typename image_type
            >
        void load (
            const image_type& img
        );
        /*!
            requires
                - image_type == is an implementation of array2d/array2d_kernel_abstract.h
                - img contains some kind of pixel type. 
                  (i.e. pixel_traits<typename image_type::type> is defined)
            ensures
                - #is_loaded_with_image() == true
                - This object is ready to run a classifier over img to detect object
                  locations.  Call detect() to do this.
        !*/

        bool is_loaded_with_image (
        ) const;
        /*!
            ensures
                - returns true if this object has been loaded with an image to process and
                  false otherwise.
        !*/

        void copy_configuration (
            const scan_fhog_pyramid& item
        );
        /*!
            ensures
                - Copies all the state information of item into *this, except for state 
                  information populated by load().  More precisely, given two scan_fhog_pyramid
                  objects S1 and S2, the following sequence of instructions should always 
                  result in both of them having the exact same state:
                    S2.copy_configuration(S1);
                    S1.load(img);
                    S2.load(img);
        !*/

        void set_detection_window_size (
            unsigned long window_width,
            unsigned long window_height
        );
        /*!
            requires
                - window_width > 0
                - window_height > 0
            ensures
                - When detect() is called, this object scans a window that is of the given
                  width and height (in pixels) over each layer in an image pyramid.  This
                  means that the rectangle detections which come out of detect() will have
                  a width to height ratio approximately equal to window_width/window_height
                  and will be approximately window_width*window_height pixels in area or
                  larger.  Therefore, the smallest object that can be detected is roughly
                  window_width by window_height pixels in size.
                - #get_detection_window_width() == window_width
                - #get_detection_window_height() == window_height
                - Since we use a HOG feature representation, the detection procedure works
                  as follows:
                    Step 1. Make an image pyramid.
                    Step 2. Convert each layer of the image pyramid into a 31 band HOG "image".
                    Step 3. Scan a linear classifier over each HOG image in the pyramid. 
                  Moreover, the HOG features quantize the input image into a grid of cells,
                  each cell being get_cell_size() by get_cell_size() pixels in size.  So
                  when we scan the object detector over the pyramid we are scanning an
                  appropriately sized window over these smaller quantized HOG features.  In
                  particular, the size of the window we scan over the HOG feature pyramid
                  is #get_fhog_window_width() by #get_fhog_window_height() HOG cells in
                  size.    
        !*/

        unsigned long get_detection_window_width (
        ) const;
        /*!
            ensures
                - returns the width, in pixels, of the detection window that is scanned
                  over the image when detect() is called.    
        !*/

        inline unsigned long get_detection_window_height (
        ) const; 
        /*!
            ensures
                - returns the height, in pixels, of the detection window that is scanned
                  over the image when detect() is called.  
        !*/

        unsigned long get_fhog_window_width (
        ) const; 
        /*!
            ensures
                - Returns the width of the HOG scanning window in terms of HOG cell blocks.
                  Note that this is a function of get_detection_window_width(), get_cell_size(), 
                  and get_padding() and is therefore not something you set directly. 
                - #get_fhog_window_width() is approximately equal to the number of HOG cells 
                  that fit into get_detection_window_width() pixels plus 2*get_padding()
                  since we include additional padding around each window to add context.
        !*/

        unsigned long get_fhog_window_height (
        ) const;
        /*!
            ensures
                - Returns the height of the HOG scanning window in terms of HOG cell blocks.  
                  Note that this is a function of get_detection_window_height(), get_cell_size(), 
                  and get_padding() and is therefore not something you set directly. 
                - #get_fhog_window_height() is approximately equal to the number of HOG cells 
                  that fit into get_detection_window_height() pixels plus 2*get_padding()
                  since we include additional padding around each window to add context.
        !*/

        void set_padding (
            unsigned long new_padding
        );
        /*!
            ensures
                - #get_padding() == new_padding
        !*/

        unsigned long get_padding (
        ) const;
        /*!
            ensures
                - The HOG windows scanned over the HOG pyramid can include additional HOG
                  cells outside the detection window.  This can help add context and
                  improve detection accuracy.  This function returns the number of extra
                  HOG cells added onto the border of the HOG windows which are scanned by
                  detect().
        !*/

        unsigned long get_cell_size (
        ) const;
        /*!
            ensures
                - Returns the size of the HOG cells.  Each HOG cell is square and contains
                  get_cell_size()*get_cell_size() pixels.
        !*/

        void set_cell_size (
            unsigned long new_cell_size
        );
        /*!
            requires
                - new_cell_size > 0
            ensures
                - #get_cell_size() == new_cell_size
        !*/

        inline long get_num_dimensions (
        ) const;
        /*!
            ensures
                - get_fhog_window_width()*get_fhog_window_height()*31
                  (i.e. The number of features is equal to the size of the HOG window
                  times 31 since there are 31 channels in the HOG feature representation.)
        !*/

        inline unsigned long get_num_detection_templates (
        ) const { return 1; }
        /*!
            ensures
                - returns 1.  Note that this function is here only for compatibility with 
                  the scan_image_pyramid object.  Notionally, its return value indicates 
                  that a scan_fhog_pyramid object is always ready to detect objects once
                  an image has been loaded.
        !*/

        inline unsigned long get_num_movable_components_per_detection_template (
        ) const { return 0; }
        /*!
            ensures
                - returns 0.  Note that this function is here only for compatibility with
                  the scan_image_pyramid object.  Its return value means that this object
                  does not support using movable part models.
        !*/

        unsigned long get_max_pyramid_levels (
        ) const;
        /*!
            ensures
                - returns the maximum number of image pyramid levels this object will use.
                  Note that #get_max_pyramid_levels() == 1 indicates that no image pyramid
                  will be used at all.  That is, only the original image will be processed
                  and no lower scale versions will be created.  
        !*/

        void set_max_pyramid_levels (
            unsigned long max_levels
        );
        /*!
            requires
                - max_levels > 0
            ensures
                - #get_max_pyramid_levels() == max_levels
        !*/

        void set_min_pyramid_layer_size (
            unsigned long width,
            unsigned long height 
        );
        /*!
            requires
                - width > 0
                - height > 0
            ensures
                - #get_min_pyramid_layer_width() == width
                - #get_min_pyramid_layer_height() == height
        !*/

        inline unsigned long get_min_pyramid_layer_width (
        ) const;
        /*!
            ensures
                - returns the smallest allowable width of an image in the image pyramid.
                  All pyramids will always include the original input image, however, no
                  pyramid levels will be created which have a width smaller than the
                  value returned by this function.
        !*/

        inline unsigned long get_min_pyramid_layer_height (
        ) const;
        /*!
            ensures
                - returns the smallest allowable height of an image in the image pyramid.
                  All pyramids will always include the original input image, however, no
                  pyramid levels will be created which have a height smaller than the
                  value returned by this function.
        !*/

        fhog_filterbank build_fhog_filterbank (
            const feature_vector_type& weights 
        ) const;
        /*!
            requires
                - weights.size() >= get_num_dimensions()
            ensures
                - Creates and then returns a fhog_filterbank object FB such that:
                    - FB.get_num_dimensions() == get_num_dimensions()
                    - FB.get_filters() == the values in weights unpacked into 31 filters.
                    - FB.num_separable_filters() == the number of separable filters necessary to
                      represent all the filters in FB.get_filters().
        !*/

        class fhog_filterbank 
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object represents a HOG filter bank.  That is, the classifier that
                    is slid over a HOG pyramid is a set of 31 linear filters, each
                    get_fhog_window_width() rows by get_fhog_window_height() columns in
                    size.  This object contains that set of 31 filters.  
            !*/

        public:
            long get_num_dimensions(
            ) const;
            /*!
                ensures
                    - Returns the total number of values in the filters.  
            !*/

            const std::vector<matrix<float> >& get_filters(
            ) const; 
            /*!
                ensures
                    - returns the set of 31 HOG filters in this object.
            !*/

            unsigned long num_separable_filters(
            ) const;
            /*!
                ensures
                    - returns the number of separable filters necessary to represent all
                      the filters in get_filters().
            !*/
        };

        void detect (
            const fhog_filterbank& w,
            std::vector<std::pair<double, rectangle> >& dets,
            const double thresh
        ) const;
        /*!
            requires
                - w.get_num_dimensions() == get_num_dimensions()
                - is_loaded_with_image() == true
            ensures
                - Scans the HOG filter defined by w over the HOG pyramid that was populated
                  by the last call to load() and stores all object detections into #dets.  
                - for all valid i:
                    - #dets[i].second == The object box which produced this detection.  This rectangle gives
                      the location of the detection.  Note that the rectangle will have been converted back into
                      the original image input space.  That is, if this detection was made at a low level in the
                      image pyramid then the object box will have been automatically mapped up the pyramid layers
                      to the original image space.  Or in other words, if you plot #dets[i].second on top of the 
                      image given to load() it will show up in the right place.
                    - #dets[i].first == The score for this detection.  This value is equal to dot(w, feature vector
                      for this sliding window location).
                    - #dets[i].first >= thresh
                - #dets will be sorted in descending order. (i.e.  #dets[i].first >= #dets[j].first for all i, and j>i)
                - Elements of w beyond index get_num_dimensions()-1 are ignored.  I.e. only the first
                  get_num_dimensions() are used.
                - Note that no form of non-max suppression is performed.  If a window has a score >= thresh
                  then it is reported in #dets.
        !*/

        void detect (
            const feature_vector_type& w,
            std::vector<std::pair<double, rectangle> >& dets,
            const double thresh
        ) const;
        /*!
            requires
                - w.size() >= get_num_dimensions()
                - is_loaded_with_image() == true
            ensures
                - performs: detect(build_fhog_filterbank(w), dets, thresh)
        !*/

        void get_feature_vector (
            const full_object_detection& obj,
            feature_vector_type& psi
        ) const;
        /*!
            requires
                - obj.num_parts() == 0 
                - is_loaded_with_image() == true
                - psi.size() >= get_num_dimensions()
                  (i.e. psi must have preallocated its memory before this function is called)
            ensures
                - This function allows you to determine the feature vector used for an
                  object detection output from detect().  Note that this vector is
                  added to psi.  Note also that you must use get_full_object_detection() to
                  convert a rectangle from detect() into the needed full_object_detection.
                - The dimensionality of the vector added to psi is get_num_dimensions().  This
                  means that elements of psi after psi(get_num_dimensions()-1) are not modified.
                - Since scan_fhog_pyramid only searches a limited set of object locations,
                  not all possible rectangles can be output by detect().  So in the case
                  where obj.get_rect() could not arise from a call to detect(), this
                  function will map obj.get_rect() to the nearest possible rectangle and
                  then add the feature vector for the mapped rectangle into #psi.
                - get_best_matching_rect(obj.get_rect()) == the rectangle obj.get_rect()
                  gets mapped to for feature extraction.
        !*/

        full_object_detection get_full_object_detection (
            const rectangle& rect,
            const feature_vector_type& w
        ) const;
        /*!
            ensures
                - returns full_object_detection(rect)
                  (This function is here only for compatibility with the scan_image_pyramid
                  object)
        !*/

        const rectangle get_best_matching_rect (
            const rectangle& rect
        ) const;
        /*!
            ensures
                - Since scan_fhog_pyramid only searches a limited set of object locations,
                  not all possible rectangles can be represented.  Therefore, this function
                  allows you to supply a rectangle and obtain the nearest possible
                  candidate object location rectangle.
        !*/

        double get_nuclear_norm_regularization_strength (
        ) const;
        /*!
            ensures
                - If the number of separable filters in a fhog_filterbank is small then the
                  filter bank can be scanned over an image much faster than a normal set of
                  31 filters.  Therefore, this object provides the option to encourage
                  machine learning methods that learn a HOG filter bank (i.e.
                  structural_object_detection_trainer) to select filter banks that have
                  this beneficial property.  In particular, the value returned by
                  get_nuclear_norm_regularization_strength() is a multiplier on a nuclear
                  norm regularizer which will encourage the selection of filters that use a
                  small number of separable components.  Larger values encourage tend to
                  give a smaller number of separable filters. 
                - if (get_nuclear_norm_regularization_strength() == 0) then
                    - This feature is disabled
                - else
                    - A nuclear norm regularizer will be added when
                      structural_object_detection_trainer is used to learn a HOG filter
                      bank.  Note that this can make the training process take
                      significantly longer (but can result in faster object detectors).
        !*/

        void set_nuclear_norm_regularization_strength (
            double strength
        );
        /*!
            requires
                - strength >= 0
            ensures
                - #get_nuclear_norm_regularization_strength() == strength
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    void serialize (
        const scan_fhog_pyramid<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support 
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T>
    void deserialize (
        scan_fhog_pyramid<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SCAN_fHOG_PYRAMID_ABSTRACT_H__


