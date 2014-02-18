// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_TRACK_ASSOCiATION_FUNCTION_ABSTRACT_H__
#ifdef DLIB_TRACK_ASSOCiATION_FUNCTION_ABSTRACT_H__

#include <vector>
#include "assignment_function_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class example_detection
    {
        /*!
            WHAT THIS OBJECT REPRESENTS

        !*/

    public:
        // Each detection object should be designed to work with a specific track object.
        // This typedef lets you determine which track type is meant for use with this
        // detection object.
        typedef struct example_track track_type;

    };

// ----------------------------------------------------------------------------------------

    class example_track
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:
        // This type should be a dlib::matrix capable of storing column vectors
        // or an unsorted sparse vector type as defined in dlib/svm/sparse_vector_abstract.h.
        typedef matrix_or_sparse_vector_type feature_vector_type;

        example_track(
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        void get_similarity_features (
            const example_detection& det,
            feature_vector_type& feats
        ) const;
        /*!
            ensures
                - #feats == A feature vector that contains information describing how
                  likely it is that det is a detection from the object corresponding to
                  this track.  That is, the feature vector should contain information that
                  lets someone decide if det should be associated to this track.
        !*/

        void update_track (
            const example_detection& det
        );
        /*!
            ensures
                - Updates this track with the given detection assuming that det is the most
                  current observation of the object under track. 
        !*/

        void propagate_track (
        );
        /*!
            ensures
                - propagates this track forward in time one time step.
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        typename detection_type
        > 
    class feature_extractor_track_association
    {
        /*!
            WHAT THIS OBJECT REPRESENTS 
                This object is an adapter that converts from the detection/track style
                interface defined above to the feature extraction interface required by the
                association rule learning tools in dlib.  Specifically, it converts the
                detection/track interface into a form usable by the assignment_function and
                its trainer structural_assignment_trainer.
        !*/

    public:
        typedef typename detection_type::track_type track_type;
        typedef typename track_type::feature_vector_type feature_vector_type;
        typedef detection_type lhs_element;
        typedef track_type rhs_element;

        unsigned long num_features(
        ) const; 
        /*!
            ensures
                - returns the dimensionality of the feature vectors produced by get_features().
        !*/

        void get_features (
            const detection_type& det,
            const track_type& track,
            feature_vector_type& feats
        ) const;
        /*!
            ensures
                - performs: track.get_similarity_features(det, feats);
        !*/
    };

    void serialize (const feature_extractor_track_association& item, std::ostream& out);
    void deserialize (feature_extractor_track_association& item, std::istream& in);
    /*!
        Provides serialization and deserialization support.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename detection_type_
        >
    class track_association_function
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/
    public:

        typedef detection_type_ detection_type;
        typedef typename detection_type::track_type track_type;
        typedef assignment_function<feature_extractor_track_association<detection_type> > association_function_type;

        track_association_function(
        );
        /*!
            ensures
                - #get_assignment_function() will be default initialized.
        !*/

        track_association_function (
            const association_function_type& assoc_
        ); 
        /*!
            ensures
                - #get_assignment_function() == assoc
        !*/

        const association_function_type& get_assignment_function (
        ) const;
        /*!
            ensures
                - returns the assignment_function used by this object to assign detections
                  to tracks.
        !*/

        void operator() (
            std::vector<track_type>& tracks,
            const std::vector<detection_type>& dets
        ) const;
        /*!
            ensures
                - This function uses get_assignment_function() to assign all the detections
                  in dets to their appropriate track in tracks.  Then each track which
                  associates to a detection is updated by calling update_track() with the
                  associated detection.  
                - Detections that don't associate with any of the elements of tracks will
                  spawn new tracks.  For each unassociated detection, this is done by
                  creating a new track_type object, calling update_track() on it with the
                  new detection, and then adding the new track into tracks.
                - Tracks that don't have a detection associate to them are propagated
                  forward in time by calling propagate_track() on them.  That is, we call
                  propagate_track() only on tracks that do not get associated with a
                  detection.
        !*/
    };

    void serialize (const track_association_function& item, std::ostream& out);
    void deserialize (track_association_function& item, std::istream& in);
    /*!
        Provides serialization and deserialization support.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TRACK_ASSOCiATION_FUNCTION_ABSTRACT_H__


