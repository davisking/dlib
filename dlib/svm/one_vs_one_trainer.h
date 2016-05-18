// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ONE_VS_ONE_TRAiNER_Hh_
#define DLIB_ONE_VS_ONE_TRAiNER_Hh_

#include "one_vs_one_trainer_abstract.h"

#include "one_vs_one_decision_function.h"
#include <vector>

#include "../unordered_pair.h"
#include "multiclass_tools.h"

#include <sstream>
#include <iostream>

#include "../any.h"
#include <map>
#include <set>
#include "../threads.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename any_trainer,
        typename label_type_ = double
        >
    class one_vs_one_trainer
    {
    public:
        typedef label_type_ label_type;

        typedef typename any_trainer::sample_type sample_type;
        typedef typename any_trainer::scalar_type scalar_type;
        typedef typename any_trainer::mem_manager_type mem_manager_type;

        typedef one_vs_one_decision_function<one_vs_one_trainer> trained_function_type;

        one_vs_one_trainer (
        ) :
            verbose(false),
            num_threads(4)
        {}

        void set_trainer (
            const any_trainer& trainer
        )
        {
            default_trainer = trainer;
            trainers.clear();
        }

        void set_trainer (
            const any_trainer& trainer,
            const label_type& l1,
            const label_type& l2
        )
        {
            trainers[make_unordered_pair(l1,l2)] = trainer;
        }

        void be_verbose (
        )
        {
            verbose = true;
        }

        void be_quiet (
        )
        {
            verbose = false;
        }

        void set_num_threads (
            unsigned long num
        )
        {
            num_threads = num;
        }

        unsigned long get_num_threads (
        ) const
        {
            return num_threads;
        }

        struct invalid_label : public dlib::error
        {
            invalid_label(const std::string& msg, const label_type& l1_, const label_type& l2_
                ) : dlib::error(msg), l1(l1_), l2(l2_) {};

            virtual ~invalid_label(
            ) throw() {}

            label_type l1, l2;
        };

        trained_function_type train (
            const std::vector<sample_type>& all_samples,
            const std::vector<label_type>& all_labels
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_learning_problem(all_samples,all_labels),
                "\t trained_function_type one_vs_one_trainer::train(all_samples,all_labels)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t all_samples.size():     " << all_samples.size()
                << "\n\t all_labels.size():      " << all_labels.size()
                );

            const std::vector<label_type> distinct_labels = select_all_distinct_labels(all_labels);


            // fill pairs with all the pairs of labels.
            std::vector<unordered_pair<label_type> > pairs;
            for (unsigned long i = 0; i < distinct_labels.size(); ++i)
            {
                for (unsigned long j = i+1; j < distinct_labels.size(); ++j)
                {
                    pairs.push_back(unordered_pair<label_type>(distinct_labels[i], distinct_labels[j]));

                    // make sure we have a trainer for this pair
                    const typename binary_function_table::const_iterator itr = trainers.find(pairs.back());
                    if (itr == trainers.end() && default_trainer.is_empty())
                    {
                        std::ostringstream sout;
                        sout << "In one_vs_one_trainer, no trainer registered for the ("
                             << pairs.back().first << ", " << pairs.back().second << ") label pair.";
                        throw invalid_label(sout.str(), pairs.back().first, pairs.back().second);
                    }
                }
            }



            // Now train on all the label pairs.
            parallel_for_helper helper(all_samples,all_labels,default_trainer,trainers,verbose,pairs);
            parallel_for(num_threads, 0, pairs.size(), helper, 500);

            if (helper.error_message.size() != 0)
            {
                throw dlib::error("binary trainer threw while training one vs. one classifier.  Error was: " + helper.error_message);
            }
            return trained_function_type(helper.dfs);
        }

    private:

        typedef std::map<unordered_pair<label_type>, any_trainer> binary_function_table;

        struct parallel_for_helper
        {
            parallel_for_helper(
                const std::vector<sample_type>& all_samples_,
                const std::vector<label_type>& all_labels_,
                const any_trainer& default_trainer_,
                const binary_function_table& trainers_,
                const bool verbose_,
                const std::vector<unordered_pair<label_type> >& pairs_
            ) :
                all_samples(all_samples_),
                all_labels(all_labels_),
                default_trainer(default_trainer_),
                trainers(trainers_),
                verbose(verbose_),
                pairs(pairs_)
            {}

            void operator()(long i) const
            {
                try
                {
                    std::vector<sample_type> samples;
                    std::vector<scalar_type> labels;

                    const unordered_pair<label_type> p = pairs[i];

                    // pick out the samples corresponding to these two classes
                    for (unsigned long k = 0; k < all_samples.size(); ++k)
                    {
                        if (all_labels[k] == p.first)
                        {
                            samples.push_back(all_samples[k]);
                            labels.push_back(+1);
                        }
                        else if (all_labels[k] == p.second)
                        {
                            samples.push_back(all_samples[k]);
                            labels.push_back(-1);
                        }
                    }

                    if (verbose)
                    {
                        auto_mutex lock(class_mutex);
                        std::cout << "Training classifier for " << p.first << " vs. " << p.second << std::endl;
                    }

                    any_trainer trainer;
                    // now train a binary classifier using the samples we selected
                    { auto_mutex lock(class_mutex);
                    const typename binary_function_table::const_iterator itr = trainers.find(p);
                    if (itr != trainers.end())
                        trainer = itr->second;
                    else
                        trainer = default_trainer;
                    }

                    any_decision_function<sample_type,scalar_type> binary_df = trainer.train(samples, labels);

                    auto_mutex lock(class_mutex);
                    dfs[p] = binary_df;
                }
                catch (std::exception& e)
                {
                    auto_mutex lock(class_mutex);
                    error_message = e.what();
                }
            }

            mutable typename trained_function_type::binary_function_table dfs;
            mutex class_mutex;
            mutable std::string error_message;

            const std::vector<sample_type>& all_samples;
            const std::vector<label_type>& all_labels;
            const any_trainer& default_trainer;
            const binary_function_table& trainers;
            const bool verbose;
            const std::vector<unordered_pair<label_type> >& pairs;
        };

        
        any_trainer default_trainer;
        binary_function_table trainers;
        bool verbose;
        unsigned long num_threads;

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ONE_VS_ONE_TRAiNER_Hh_

