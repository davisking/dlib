// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_HASHED_IMAGE_FEATUrES_ABSTRACT_H__
#ifdef DLIB_HASHED_IMAGE_FEATUrES_ABSTRACT_H__

#include <vector>
#include "../matrix.h"
#include "../statistics.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class hashed_feature_image : noncopyable
    {
        /*!
            REQUIREMENTS ON feature_extractor 
                - must be an object with an interface compatible with dlib::hog_image

            INITIAL VALUE
                 - size() == 0
                 - get_num_dimensions() == 1000
                 - has_image_statistics() == false
                 - get_scales() == logspace(-1,1,3)

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for performing image feature extraction.  In
                particular, it wraps another image feature extractor and converts
                the wrapped image feature vectors into indicator vectors.  It does
                this by hashing each feature vector into the range [0, get_num_dimensions()-1]
                and then returns a new vector which is zero everywhere except for
                the position determined by the hash.  Additionally, this object can be
                configured to hash each feature vector into multiple bins, thereby 
                creating an indicator vector with multiple non-zero indicator features.


            NOTATION 
                let BASE_FE denote the base feature_extractor object contained inside 
                the hashed_feature_image.
        !*/

    public:

        typedef std::vector<std::pair<unsigned int,double> > descriptor_type;

        hashed_feature_image (
        ); 
        /*!
            ensures
                - this object is properly initialized
        !*/

        void clear (
        );
        /*!
            ensures
                - this object will have its initial value
        !*/

        void set_scales (
            const matrix<double,1,0>& new_scales
        );
        /*!
            requires
                - new_scales.size() > 0
            ensures
                - #get_scales() == new_scales
        !*/

        const matrix<double,1,0>& get_scales (
        ) const;
        /*!
            ensures
                - When a feature vector from BASE_FE is hashed, it is hashed into exactly 
                  get_scales().size() hash bins.  Each hash is computed as follows:
                    - first normalize the feature vector
                    - then multiply it by an element of get_scales()
                    - then convert the resulting vector to a vector of dlib::int32
                    - finally, hash the integer vector into a hash bin.
                - The size of the numbers in get_scales() determines how "big" the hash bins are.  
                  A very small scale value would result in all input vectors being hashed into the 
                  same bin, while larger scale values would result in only similar vectors 
                  falling into the same bins.  However, a scale value too large would result in
                  all vectors going into different bins.  In this case, the bins are too fine 
                  grained.
        !*/

        template <
            typename image_type
            >
        void accumulate_image_statistics (
            const image_type& img
        );
        /*!
            requires
                - image_type == is an implementation of array2d/array2d_kernel_abstract.h
                - pixel_traits<typename image_type::type>::has_alpha == false
            ensures
                - if (img is large enough to have at least two local features in it) then
                    - #has_image_statistics() == true
                - This function will accumulate image statistics across multiple calls. 
                  Therefore, it can be beneficial to pass in many images.
        !*/

        bool has_image_statistics (
        ) const;
        /*!
            ensures
                - Part of the hashing step is to normalize the features produced by
                  BASE_FE.  This function returns true if we have accumulated the necessary
                  information to perform this normalization and false otherwise.
        !*/

        void copy_configuration (
            const feature_extractor& item
        );
        /*!
            ensures
                - performs BASE_FE.copy_configuration(item)
        !*/

        void copy_configuration (
            const hashed_feature_image& item
        );
        /*!
            ensures
                - copies all the state information of item into *this, except for state 
                  information populated by load().  More precisely, given two hashed_feature_image 
                  objects H1 and H2, the following sequence of instructions should always 
                  result in both of them having the exact same state.
                    H2.copy_configuration(H1);
                    H1.load(img);
                    H2.load(img);
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
                - pixel_traits<typename image_type::type>::has_alpha == false
            ensures
                - performs BASE_FE.load(img)
                  i.e. does feature extraction.  The features can be accessed using
                  operator() as defined below.
        !*/

        unsigned long size (
        ) const;
        /*!
            ensures
                - returns BASE_FE.size() 
        !*/

        long nr (
        ) const;
        /*!
            ensures
                - returns BASE_FE.nr() 
        !*/

        long nc (
        ) const;
        /*!
            ensures
                - returns BASE_FE.nc() 
        !*/

        long get_num_dimensions (
        ) const;
        /*!
            ensures
                - returns the dimensionality of the feature vectors returned by operator().  
                  In this case, this is the number of hash bins.
        !*/

        void set_num_dimensions (
            long new_num_dims
        );
        /*!
            requires
                - new_num_dims > 0
            ensures
                - #get_num_dimensions() == new_num_dims
        !*/

        const descriptor_type& operator() (
            long row,
            long col
        ) const;
        /*!
            requires
                - has_image_statistics() == true
                - 0 <= row < nr()
                - 0 <= col < nc()
            ensures
                - hashes BASE_FE(row,col) and returns resulting indicator vector. 
                - Returns a vector V such that:
                    - V.size() == get_scales().size()
                    - for all valid i: 0 <= V[i].first < get_num_dimensions()
                    - if (BASE_FE(row,col) hashes into bin B) then
                        - V contains an element with .first == B and .second == 1
                - Note that the returned vector is represented as a sparse vector but 
                  the indices are not in sorted order.   Moreover, there might even be 
                  duplicate entires for a particular dimension.  This means you can't use 
                  many of the sparse vector operations defined in dlib::sparse_vector.    
        !*/

        const rectangle get_block_rect (
            long row,
            long col
        ) const;
        /*!
            requires
                - 0 <= row < nr()
                - 0 <= col < nc()
            ensures
                - returns BASE_FE.get_block_rect(row,col)
                  I.e. returns a rectangle that tells you what part of the original image is associated
                  with a particular feature vector.
        !*/

        const point image_to_feat_space (
            const point& p
        ) const;
        /*!
            ensures
                - returns BASE_FE.image_to_feat_space(p)
                  I.e. Each local feature is extracted from a certain point in the input image.
                  This function returns the identity of the local feature corresponding
                  to the image location p.  Or in other words, let P == image_to_feat_space(p), 
                  then (*this)(P.y(),P.x()) == the local feature closest to, or centered at, 
                  the point p in the input image.  Note that some image points might not have 
                  corresponding feature locations.  E.g. border points or points outside the 
                  image.  In these cases the returned point will be outside get_rect(*this).
        !*/

        const rectangle image_to_feat_space (
            const rectangle& rect
        ) const;
        /*!
            ensures
                - returns BASE_FE.image_to_feat_space(rect)
                  I.e. returns rectangle(image_to_feat_space(rect.tl_corner()), image_to_feat_space(rect.br_corner()));
                  (i.e. maps a rectangle from image space to feature space)
        !*/

        const point feat_to_image_space (
            const point& p
        ) const;
        /*!
            ensures
                - returns BASE_FE.feat_to_image_space(p)
                  I.e. returns the location in the input image space corresponding to the center
                  of the local feature at point p.  In other words, this function computes
                  the inverse of image_to_feat_space().  Note that it may only do so approximately, 
                  since more than one image location might correspond to the same local feature.  
                  That is, image_to_feat_space() might not be invertible so this function gives 
                  the closest possible result.
        !*/

        const rectangle feat_to_image_space (
            const rectangle& rect
        ) const;
        /*!
            ensures
                - returns BASE_FE.feat_to_image_space(rect)
                  I.e. return rectangle(feat_to_image_space(rect.tl_corner()), feat_to_image_space(rect.br_corner()));
                  (i.e. maps a rectangle from feature space to image space)
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void serialize (
        const hashed_feature_image<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support 
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void deserialize (
        hashed_feature_image<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_HASHED_IMAGE_FEATUrES_ABSTRACT_H__



