// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_VALIDATION_H_
#define DLIB_DNn_VALIDATION_H_

#include "../svm/cross_validate_object_detection_trainer_abstract.h"
#include "../svm/cross_validate_object_detection_trainer.h"
#include "layers.h"
#include <set>

namespace dlib
{
    namespace impl
    {
        inline std::set<std::string> get_labels (
            const std::vector<mmod_rect>& rects1,
            const std::vector<mmod_rect>& rects2
        )
        {
            std::set<std::string> labels;
            for (auto& rr : rects1)
                labels.insert(rr.label);
            for (auto& rr : rects2)
                labels.insert(rr.label);
            return labels;
        }
    }

    template <
        typename SUBNET,
        typename image_array_type
        >
    const matrix<double,1,3> test_object_detection_function (
        loss_mmod<SUBNET>& detector,
        const image_array_type& images,
        const std::vector<std::vector<mmod_rect>>& truth_dets,
        const test_box_overlap& overlap_tester = test_box_overlap(),
        const double adjust_threshold = 0,
        const test_box_overlap& overlaps_ignore_tester = test_box_overlap()
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT( is_learning_problem(images,truth_dets) == true , 
                    "\t matrix test_object_detection_function()"
                    << "\n\t invalid inputs were given to this function"
                    << "\n\t is_learning_problem(images,truth_dets): " << is_learning_problem(images,truth_dets)
                    << "\n\t images.size(): " << images.size() 
                    );



        double correct_hits = 0;
        double total_true_targets = 0;

        std::vector<std::pair<double,bool> > all_dets;
        unsigned long missing_detections = 0;

        resizable_tensor temp;

        for (unsigned long i = 0; i < images.size(); ++i)
        {
            std::vector<mmod_rect> hits; 
            detector.to_tensor(&images[i], &images[i]+1, temp);
            detector.subnet().forward(temp);
            detector.loss_details().to_label(temp, detector.subnet(), &hits, adjust_threshold);


            for (auto& label : impl::get_labels(truth_dets[i], hits))
            {
                std::vector<full_object_detection> truth_boxes;
                std::vector<rectangle> ignore;
                std::vector<std::pair<double,rectangle>> boxes;
                // copy hits and truth_dets into the above three objects
                for (auto&& b : truth_dets[i])
                {
                    if (b.ignore)
                    {
                        ignore.push_back(b);
                    }
                    else if (b.label == label)
                    {
                        truth_boxes.push_back(full_object_detection(b.rect));
                        ++total_true_targets;
                    }
                }
                for (auto&& b : hits)
                {
                    if (b.label == label)
                        boxes.push_back(std::make_pair(b.detection_confidence, b.rect));
                }

                correct_hits += impl::number_of_truth_hits(truth_boxes, ignore, boxes, overlap_tester, all_dets, missing_detections, overlaps_ignore_tester);
            }
        }

        std::sort(all_dets.rbegin(), all_dets.rend());

        double precision, recall;

        double total_hits = all_dets.size();

        if (total_hits == 0)
            precision = 1;
        else
            precision = correct_hits / total_hits;

        if (total_true_targets == 0)
            recall = 1;
        else
            recall = correct_hits / total_true_targets;

        matrix<double, 1, 3> res;
        res = precision, recall, average_precision(all_dets, missing_detections);
        return res;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_VALIDATION_H_

