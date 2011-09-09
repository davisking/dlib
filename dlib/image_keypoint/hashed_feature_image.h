// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_HASHED_IMAGE_FEATUrES_H__
#define DLIB_HASHED_IMAGE_FEATUrES_H__

#include "hashed_feature_image_abstract.h"
#include <vector>
#include "../algs.h"
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
            INITIAL VALUE
                - scales == logspace(-1, 1, 3)
                - num_dims == 1000

            CONVENTION
                - scales.size() > 0
                - num_dims == get_num_dimensions()
                - if (has_image_statistics()) then
                    - rs[i] == the statistics of feature element i.  I.e. the stats of fe(r,c)(i)
                      over a set of images supplied to accumulate_image_statistics().
                    - inv_stddev[i] == 1.0/(rs[i].stddev() + 1e-10)

        !*/

    public:

        typedef std::vector<std::pair<unsigned int,double> > descriptor_type;

        hashed_feature_image (
        ); 

        void clear (
        );

        void set_scales (
            const matrix<double,1,0>& new_scales
        );

        const matrix<double,1,0>& get_scales (
        ) const;

        template <
            typename image_type
            >
        inline void accumulate_image_statistics (
            const image_type& img
        );


        void copy_configuration (
            const feature_extractor& item
        );

        void copy_configuration (
            const hashed_feature_image& item
        );

        template <
            typename image_type
            >
        inline void load (
            const image_type& img
        );

        inline unsigned long size (
        ) const;

        inline long nr (
        ) const;

        inline long nc (
        ) const;

        inline long get_num_dimensions (
        ) const;

        void set_num_dimensions (
            long new_num_dims
        );

        inline const descriptor_type& operator() (
            long row,
            long col
        ) const;

        inline const rectangle get_block_rect (
            long row,
            long col
        ) const;

        inline const point image_to_feat_space (
            const point& p
        ) const;

        inline const rectangle image_to_feat_space (
            const rectangle& rect
        ) const;

        inline const point feat_to_image_space (
            const point& p
        ) const;

        inline const rectangle feat_to_image_space (
            const rectangle& rect
        ) const;

        template <typename T>
        friend void serialize (
            const hashed_feature_image<T>& item,
            std::ostream& out
        );

        template <typename T>
        friend void deserialize (
            hashed_feature_image<T>& item,
            std::istream& in 
        );

    private:

        inline bool has_image_statistics (
        ) const;

        feature_extractor fe;
        typename feature_extractor::descriptor_type inv_stddev;
        std::vector<running_stats<double> > rs;
        matrix<double,1,0> scales;
        long num_dims;

        // Transient variables.  These are here just so they don't have to get constructed over
        // and over inside operator().  I.e. they don't logically contribute to the state of 
        // this object.
        mutable typename feature_extractor::descriptor_type scaled_feats;
        mutable matrix<int32,0,1> quantized_feats;
        mutable descriptor_type hash_feats;

    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    void serialize (
        const hashed_feature_image<T>& item,
        std::ostream& out
    )
    {
        serialize(item.fe, out);
        serialize(item.inv_stddev, out);
        serialize(item.rs, out);
        serialize(item.scales, out);
        serialize(item.num_dims, out);
    }

    template <typename T>
    void deserialize (
        hashed_feature_image<T>& item,
        std::istream& in 
    )
    {
        deserialize(item.fe, in);
        deserialize(item.inv_stddev, in);
        deserialize(item.rs, in);
        deserialize(item.scales, in);
        deserialize(item.num_dims, in);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                        hashed_feature_image member functions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    hashed_feature_image<feature_extractor>::
    hashed_feature_image (
    ) : 
        num_dims(1000) 
    {
        scales = logspace(-1,1,3);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void hashed_feature_image<feature_extractor>::
    clear (
    )
    {
        fe.clear();
        inv_stddev = 0;
        scales = logspace(-1,1,3);
        rs.clear();
        num_dims = 1000;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void hashed_feature_image<feature_extractor>::
    set_scales (
        const matrix<double,1,0>& new_scales
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(new_scales.size() > 0,
            "\t void hashed_feature_image::set_scales()"
            << "\n\t size of new_scales should not be zero"
            << "\n\t this: " << this
            );

        scales = new_scales;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const matrix<double,1,0>& hashed_feature_image<feature_extractor>::
    get_scales (
    ) const
    {
        return scales;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    template <
        typename image_type
        >
    void hashed_feature_image<feature_extractor>::
    accumulate_image_statistics (
        const image_type& img
    )
    {
        feature_extractor temp;
        temp.load(img);

        if (temp.size() == 0)
            return;

        rs.resize(temp(0,0).size());


        typename feature_extractor::descriptor_type des;

        for (long r = 0; r < temp.nr(); ++r)
        {
            for (long c = 0; c < temp.nc(); ++c)
            {
                des = temp(r,c);
                for (long i = 0; i < des.size(); ++i)
                {
                    rs[i].add(des(i));
                }
            }
        }

        if (rs[0].current_n() <= 1)
            return;

        // keep inv_stddev up to date with rs.
        inv_stddev.set_size(des.nr(), des.nc());
        for (long i = 0; i < des.size(); ++i)
        {
            inv_stddev(i) = 1.0/(rs[i].stddev() + 1e-10);
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void hashed_feature_image<feature_extractor>::
    copy_configuration (
        const feature_extractor& item
    )
    {
        fe.copy_configuration(item);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void hashed_feature_image<feature_extractor>::
    copy_configuration (
        const hashed_feature_image& item
    )
    {
        rs = item.rs;
        inv_stddev = item.inv_stddev;
        scales = item.scales;
        fe.copy_configuration(item.fe);
        num_dims = item.num_dims;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    bool hashed_feature_image<feature_extractor>::
    has_image_statistics (
    ) const
    {
        // if we have enough data to compute standard deviations of the features
        if (rs.size() > 0 && rs[0].current_n() > 1)
            return true;
        else
            return false;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    template <
        typename image_type
        >
    void hashed_feature_image<feature_extractor>::
    load (
        const image_type& img
    )
    {
        fe.load(img);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    unsigned long hashed_feature_image<feature_extractor>::
    size (
    ) const
    {
        return fe.size();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    long hashed_feature_image<feature_extractor>::
    nr (
    ) const
    {
        return fe.nr();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    long hashed_feature_image<feature_extractor>::
    nc (
    ) const
    {
        return fe.nc();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    long hashed_feature_image<feature_extractor>::
    get_num_dimensions (
    ) const
    {
        return num_dims;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void hashed_feature_image<feature_extractor>::
    set_num_dimensions (
        long new_num_dims
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(new_num_dims > 0,
            "\t void hashed_feature_image::set_num_dimensions()"
            << "\n\t You can't have zero dimensions"
            << "\n\t this: " << this
            );

        num_dims = new_num_dims;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const typename hashed_feature_image<feature_extractor>::descriptor_type& hashed_feature_image<feature_extractor>::
    operator() (
        long row,
        long col
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(0 <= row && row < nr() &&
                    0 <= col && col < nc(),
            "\t descriptor_type hashed_feature_image::operator(row,col)"
            << "\n\t Invalid inputs were given to this function"
            << "\n\t row:  " << row
            << "\n\t col:  " << col 
            << "\n\t nr(): " << nr()
            << "\n\t nc(): " << nc()
            << "\n\t this: " << this
            );

        hash_feats.resize(scales.size());
        if (has_image_statistics())
            scaled_feats = pointwise_multiply(fe(row,col), inv_stddev);
        else
            scaled_feats = fe(row,col);

        for (long i = 0; i < scales.size(); ++i)
        {
            quantized_feats = matrix_cast<int32>(scales(i)*scaled_feats);
            hash_feats[i] = std::make_pair(hash(quantized_feats)%num_dims,1);
        }
        return hash_feats;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const rectangle hashed_feature_image<feature_extractor>::
    get_block_rect (
        long row,
        long col
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(0 <= row && row < nr() &&
                    0 <= col && col < nc(),
            "\t rectangle hashed_feature_image::get_block_rect(row,col)"
            << "\n\t Invalid inputs were given to this function"
            << "\n\t row:  " << row
            << "\n\t col:  " << col 
            << "\n\t nr(): " << nr()
            << "\n\t nc(): " << nc()
            << "\n\t this: " << this
            );

        return fe.get_block_rect(row,col);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const point hashed_feature_image<feature_extractor>::
    image_to_feat_space (
        const point& p
    ) const
    {
        return fe.image_to_feat_space(p);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const rectangle hashed_feature_image<feature_extractor>::
    image_to_feat_space (
        const rectangle& rect
    ) const
    {
        return fe.image_to_feat_space(rect);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const point hashed_feature_image<feature_extractor>::
    feat_to_image_space (
        const point& p
    ) const
    {
        return fe.feat_to_image_space(p);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    const rectangle hashed_feature_image<feature_extractor>::
    feat_to_image_space (
        const rectangle& rect
    ) const 
    {
        return fe.feat_to_image_space(rect);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_HASHED_IMAGE_FEATUrES_H__


