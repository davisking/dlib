// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RANKING_ToOLS_ABSTRACT_H__
#ifdef DLIB_RANKING_ToOLS_ABSTRACT_H__


#include "../algs.h"
#include "../matrix.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct ranking_pair
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

        ranking_pair() {}

        ranking_pair(
            const std::vector<T>& r, 
            const std::vector<T>& nr
        ) :
            relevant(r), nonrelevant(nr) 
        {}

        std::vector<T> relevant;
        std::vector<T> nonrelevant;
    };

    template <
        typename T
        >
    void serialize (
        const ranking_pair<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    template <
        typename T
        >
    void deserialize (
        ranking_pair<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool is_ranking_problem (
        const std::vector<ranking_pair<T> >& samples
    )
    {
        if (samples.size() == 0)
            return false;


        for (unsigned long i = 0; i < samples.size(); ++i)
        {
            if (samples[i].relevant.size() == 0)
                return false;
            if (samples[i].nonrelevant.size() == 0)
                return false;
        }

        // If these are dense vectors then they must all have the same dimensionality.
        if (is_matrix<T>::value)
        {
            const long dims = max_index_plus_one(samples[0].relevant);
            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                for (unsigned long j = 0; j < samples[i].relevant.size(); ++j)
                {
                    if (samples[i].relevant[j].size() != dims)
                        return false;
                }
                for (unsigned long j = 0; j < samples[i].nonrelevant.size(); ++j)
                {
                    if (samples[i].nonrelevant[j].size() != dims)
                        return false;
                }
            }
        }

        return true;
    }

    template <
        typename T
        >
    unsigned long max_index_plus_one (
        const ranking_pair<T>& item
    )
    {
        return std::max(max_index_plus_one(item.relevant), max_index_plus_one(item.nonrelevant));
    }

    template <
        typename T
        >
    unsigned long max_index_plus_one (
        const std::vector<ranking_pair<T> >& samples
    )
    {
        unsigned long dims = 0;
        for (unsigned long i = 0; i < samples.size(); ++i)
        {
            dims = std::max(dims, max_index_plus_one(samples[i]));
        }
        return dims;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    void count_ranking_inversions (
        const std::vector<T>& x,
        const std::vector<T>& y,
        std::vector<unsigned long>& x_count,
        std::vector<unsigned long>& y_count
    );
    /*!
        ensures
            - This function counts how many times we see a y value greater than or equal to
              x value.  This is done efficiently in O(n*log(n)) time via the use of quick
              sort.
            - #x_count.size() == x.size()
            - #y_count.size() == y.size()
            - for all valid i:
                - #x_count[i] == how many times a value in y was >= x[i].
            - for all valid j:
                - #y_count[j] == how many times a value in x was <= y[j].
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename ranking_function,
        typename T
        >
    double test_ranking_function (
        const ranking_function& funct,
        const std::vector<ranking_pair<T> >& samples
    );
    /*!
        ensures
            - returns the fraction of ranking pairs predicted correctly.
            - TODO
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type,
        typename T
        >
    double cross_validate_ranking_trainer (
        const trainer_type& trainer,
        const std::vector<ranking_pair<T> >& samples,
        const long folds
    );
    /*!
        requires
            - is_ranking_problem(samples) == true
            - 1 < folds <= samples.size()
        ensures
            - TODO
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RANKING_ToOLS_ABSTRACT_H__


