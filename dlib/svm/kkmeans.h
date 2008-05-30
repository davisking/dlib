// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KKMEANs_
#define DLIB_KKMEANs_

#include <cmath>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include "kernel_abstract.h"
#include "../array.h"
#include "kcentroid.h"
#include "kkmeans_abstract.h"
#include "../noncopyable.h"
#include "../smart_pointers.h"

namespace dlib
{

    template <
        typename kernel_type
        >
    class kkmeans : public noncopyable
    {
    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;

        kkmeans (
            const kcentroid<kernel_type>& kc_ 
        ):
            kc(kc_)
        {
            set_number_of_centers(1);
        }

        ~kkmeans()
        {
        }

        void set_kcentroid (
            const kcentroid<kernel_type>& kc_
        )
        {
            kc = kc_;
            set_number_of_centers(number_of_centers());
        }

        const kcentroid<kernel_type>& get_kcentroid (
            unsigned long i
        ) const
        {
            return *centers[i];
        }

        void set_number_of_centers (
            unsigned long num
        )
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(num > 0,
                "\tvoid kkmeans::set_number_of_centers"
                << "\n\tYou can't set the number of centers to zero"
                << "\n\tthis: " << this
                );

            centers.set_max_size(num);
            centers.set_size(num);

            for (unsigned long i = 0; i < centers.size(); ++i)
            {
                centers[i].reset(new kcentroid<kernel_type>(kc));
            }
        }

        unsigned long number_of_centers (
        ) const
        {
            return centers.size();
        }

        template <typename matrix_type>
        void train (
            const matrix_type& samples,
            const matrix_type& initial_centers 
        )
        {
            // clear out the old data and initialize the centers
            for (unsigned long i = 0; i < centers.size(); ++i)
            {
                centers[i]->clear_dictionary();
                centers[i]->train(initial_centers(i));
            }

            assignments.expand(samples.size());

            bool assignment_changed = true;

            // loop until the centers stabilize 
            while (assignment_changed)
            {
                assignment_changed = false;

                // loop over all the samples and assign them to their closest centers
                for (long i = 0; i < samples.size(); ++i)
                {
                    // find the best center
                    unsigned long best_center = 0;
                    scalar_type best_score = (*centers[0])(samples(i));
                    for (unsigned long c = 1; c < centers.size(); ++c)
                    {
                        scalar_type temp = (*centers[c])(samples(i));
                        if (temp < best_score)
                        {
                            best_score = temp;
                            best_center = c;
                        }
                    }

                    // if the current sample changed centers then make note of that
                    if (assignments[i] != best_center)
                    {
                        assignments[i] = best_center;
                        assignment_changed = true;
                    }
                }

                if (assignment_changed)
                {
                    // now clear out the old data 
                    for (unsigned long i = 0; i < centers.size(); ++i)
                        centers[i]->clear_dictionary();

                    // recalculate the cluster centers 
                    for (unsigned long i = 0; i < assignments.size(); ++i)
                        centers[assignments[i]]->train(samples(i));
                }

            }


        }

        unsigned long operator() (
            const sample_type& sample
        ) const
        {
            unsigned long label = 0;
            scalar_type best_score = (*centers[0])(sample);

            // figure out which center the given sample is closest too
            for (unsigned long i = 1; i < centers.size(); ++i)
            {
                scalar_type temp = (*centers[i])(sample);
                if (temp < best_score)
                {
                    label = i;
                    best_score = temp;
                }
            }

            return label;
        }

        void swap (
            kkmeans& item
        )
        {
            centers.swap(item.centers);
            kc.swap(item.kc);
            assignments.swap(item.assignments);
        }

        friend void serialize(const kkmeans& item, std::ostream& out)
        {
            serialize(item.centers, out);
            serialize(item.kc, out);
            serialize(item.assignments, out);
        }

        friend void deserialize(kkmeans& item, std::istream& in)
        {
            deserialize(item.centers, in);
            deserialize(item.kc, in);
            deserialize(item.assignments, in);
        }

    private:

        typename array<scoped_ptr<kcentroid<kernel_type> > >::expand_1b_c centers;
        kcentroid<kernel_type> kc;

        // temp variables
        array<unsigned long>::expand_1b_c assignments;
    };

// ----------------------------------------------------------------------------------------

    template <typename kernel_type>
    void swap(kkmeans<kernel_type>& a, kkmeans<kernel_type>& b)
    { a.swap(b); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KKMEANs_


