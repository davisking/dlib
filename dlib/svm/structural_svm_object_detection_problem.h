// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_H__
#define DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_H__

#include "../matrix.h"
#include "structural_svm_problem_threaded.h"
#include <sstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type,
        typename image_array_type 
        >
    class structural_svm_object_detection_problem : public structural_svm_problem_threaded<matrix<double,0,1> >,
                                                    noncopyable
    {
    public:

        structural_svm_object_detection_problem(
            const image_scanner_type& scanner,
            const overlap_tester_type& overlap_tester,
            const image_array_type& images_,
            const std::vector<std::vector<rectangle> >& truth_rects,
            unsigned long num_threads = 2
        ) :
            structural_svm_problem_threaded<matrix<double,0,1> >(num_threads),
            boxes_overlap(overlap_tester),
            images(images_),
            rects(truth_rects),
            overlap_eps(0.5)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_learning_problem(images_, truth_rects) && 
                         scanner.get_num_detection_templates() > 0,
                "\t structural_svm_object_detection_problem::structural_svm_object_detection_problem()"
                << "\n\t Invalid inputs were given to this function "
                << "\n\t scanner.get_num_detection_templates(): " << scanner.get_num_detection_templates()
                << "\n\t is_learning_problem(images_,truth_rects): " << is_learning_problem(images_,truth_rects)
                << "\n\t this: " << this
                );

            scanner_config.copy_configuration(scanner);

            max_num_dets = 0;
            for (unsigned long i = 0; i < rects.size(); ++i)
            {
                if (rects.size() > max_num_dets)
                    max_num_dets = rects.size();
            }
            max_num_dets = max_num_dets*3 + 10;
        }

        void set_overlap_eps (
            double eps
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(0 < eps && eps < 1, 
                "\t structural_svm_object_detection_problem::set_overlap_eps(eps)"
                << "\n\t Invalid inputs were given to this function "
                << "\n\t eps:  " << eps 
                << "\n\t this: " << this
                );

            overlap_eps = eps;
        }

        double get_overlap_eps (
        ) const
        {
            return overlap_eps;
        }

    private:
        virtual long get_num_dimensions (
        ) const 
        {
            return scanner_config.get_num_dimensions() + 
                1;// for threshold
        }

        virtual long get_num_samples (
        ) const 
        {
            return images.size();
        }

        virtual void get_truth_joint_feature_vector (
            long idx,
            feature_vector_type& psi 
        ) const 
        {
            image_scanner_type scanner;
            scanner.copy_configuration(scanner_config);

            scanner.load(images[idx]);
            psi.set_size(get_num_dimensions());
            std::vector<rectangle> mapped_rects;
            scanner.get_feature_vector(rects[idx], psi, mapped_rects);
            psi(scanner.get_num_dimensions()) = -1.0*rects[idx].size();
            std::cout << "truth psi length: "<< length(psi) << std::endl;
            std::cout << "max truth psi: "<< max(abs(psi)) << std::endl;
            std::cout << "index of max truth psi: "<< index_of_max(abs(psi)) << std::endl;

            // check if any of the boxes overlap.  If they do then it is impossible for
            // us to learn to correctly classify this sample
            for (unsigned long i = 0; i < mapped_rects.size(); ++i)
            {
                for (unsigned long j = i+1; j < mapped_rects.size(); ++j)
                {
                    if (boxes_overlap(mapped_rects[i], mapped_rects[j]))
                    {
                        using namespace std;
                        ostringstream sout;
                        sout << "impossible labeling found!" << endl;
                        sout << "rect1: "<< mapped_rects[i] << endl;
                        sout << "rect2: "<< mapped_rects[j] << endl;
                        sout << "in image " << idx << endl;
                        throw dlib::error(sout.str());
                    }
                }
            }

            // make sure the mapped rectangles are within overlap_eps of the
            // truth rectangles.
            for (unsigned long i = 0; i < mapped_rects.size(); ++i)
            {
                const double area = (rects[idx][i].intersect(mapped_rects[i])).area();
                const double total_area = (rects[idx][i] + mapped_rects[i]).area();
                if (area/total_area <= overlap_eps)
                {
                    using namespace std;
                    ostringstream sout;
                    sout << "overlap_eps is too tight!" << endl;
                    sout << "overlap_eps:    "<< overlap_eps << endl;
                    sout << "mapped overlap: "<< area/total_area << endl;
                    sout << "width/height:   "<< rects[idx][i].width()/(double)rects[idx][i].height() << endl;
                    sout << "area:           "<< rects[idx][i].area() << endl;
                    sout << "true rect:      "<< rects[idx][i] << endl;
                    sout << "mapped rect:    "<< mapped_rects[i] << endl;
                    sout << "in image        " << idx << endl;
                    throw dlib::error(sout.str());
                }

            }

            //std::cout << "TRUTH PSI: "<< trans(psi) << std::endl;
            //abort();
        }

        virtual void separation_oracle (
            const long idx,
            const matrix_type& current_solution,
            scalar_type& loss,
            feature_vector_type& psi
        ) const 
        {
            image_scanner_type scanner;
            scanner.copy_configuration(scanner_config);

            std::vector<std::pair<double, rectangle> > dets;
            const double thresh = current_solution(scanner.get_num_dimensions());

            const double loss_per_error = 1;
            scanner.load(images[idx]);

            scanner.detect(current_solution, dets, thresh-loss_per_error);


            // The loss will measure the number of incorrect detections.  A detection is
            // incorrect if it doesn't hit a truth rectangle or if it is a duplicate detection
            // on a truth rectangle.
            loss = rects[idx].size()*loss_per_error;

            // Measure the risk loss augmented score for the detections which hit a truth rect.
            std::vector<double> truth_score_hits(rects[idx].size(), 0);

            std::vector<rectangle> final_dets;
            // The point of this loop is to fill out the truth_score_hits array. 
            for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
            {
                if (overlaps_any_box(final_dets, dets[i].second))
                    continue;

                const std::pair<double,unsigned int> truth = find_max_overlap(rects[idx], dets[i].second);

                final_dets.push_back(dets[i].second);

                const double truth_overlap = truth.first;
                // if hit truth rect
                if (truth_overlap > overlap_eps)
                {
                    // if this is the first time we have seen a detect which hit rects[truth.second]
                    const double score = dets[i].first - thresh;
                    if (truth_score_hits[truth.second] == 0)
                        truth_score_hits[truth.second] += score - loss_per_error;
                    else
                        truth_score_hits[truth.second] += score + loss_per_error;
                }
            }


            double expected_psi_dot_w = 0;
            // keep track of which truth boxes we have hit so far.
            std::vector<bool> hit_truth_table(rects[idx].size(), false);
            final_dets.clear();
            // Now figure out which detections jointly maximize the loss and detection score sum.  We
            // need to take into account the fact that allowing a true detection in the output, while 
            // initially reducing the loss, may allow us to increase the loss later with many duplicate
            // detections.
            for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
            {
                if (overlaps_any_box(final_dets, dets[i].second))
                    continue;

                const std::pair<double,unsigned int> truth = find_max_overlap(rects[idx], dets[i].second);

                const double truth_overlap = truth.first;
                if (truth_overlap > overlap_eps)
                {
                    if (truth_score_hits[truth.second] >= 0)
                    {
                        if (!hit_truth_table[truth.second])
                        {
                            hit_truth_table[truth.second] = true;
                            final_dets.push_back(dets[i].second);
                            expected_psi_dot_w += dets[i].first;
                            loss -= loss_per_error;
                        }
                        else
                        {
                            final_dets.push_back(dets[i].second);
                            expected_psi_dot_w += dets[i].first;
                            loss += loss_per_error;
                        }
                    }
                }
                else
                {
                    // didn't hit anything
                    final_dets.push_back(dets[i].second);
                    expected_psi_dot_w += dets[i].first;
                    loss += loss_per_error;
                }
            }

            psi.set_size(get_num_dimensions());
            psi = 0;
            //feature_vector_type other_psi;
            //other_psi.set_size(get_num_dimensions());
            //other_psi = 0;
            //scanner.get_feature_vector(final_dets, other_psi);
            std::vector<rectangle> mapped_rects;
            scanner.get_feature_vector(final_dets, psi, mapped_rects);
            DLIB_CASSERT(mapped_rects.size() == final_dets.size(),"");
            if (std::abs(expected_psi_dot_w  - dot(psi,current_solution)) > 1e-8)
            {
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                std::cout << "   psi dot w:          "<< dot(psi,current_solution) << std::endl;
                std::cout << "   expected psi dot w: "<< expected_psi_dot_w << std::endl;
                std::cout << "   final_dets.size(): "<< final_dets.size() << std::endl;
                for (unsigned long i = 0; i < final_dets.size(); ++i)
                {
                    DLIB_CASSERT(final_dets[i] == mapped_rects[i], final_dets[i] << "   " << mapped_rects[i] << "   i: " << i);
                }
                abort();
            }
            for (unsigned long i = 0; i < final_dets.size(); ++i)
            {
                DLIB_CASSERT(final_dets[i] == mapped_rects[i], final_dets[i] << "   " << mapped_rects[i] << "   i: " << i);
            }
            psi(scanner.get_num_dimensions()) = -1.0*final_dets.size();

            std::cout << "LOSS: "<< loss << std::endl;
            std::cout << "   final_dets.size(): "<< final_dets.size() << std::endl;
            //std::cout << "PSI: "<< trans(psi) << std::endl;
        }


        bool overlaps_any_box (
            const std::vector<rectangle>& rects,
            const dlib::rectangle& rect
        ) const
        {
            for (unsigned long i = 0; i < rects.size(); ++i)
            {
                if (boxes_overlap(rects[i], rect))
                    return true;
            }
            return false;
        }

        std::pair<double,unsigned int> find_max_overlap(
            const std::vector<rectangle>& boxes,
            const rectangle rect
        ) const
        /*!
            ensures
                - determines which rectangle in boxes overlaps rect the most and
                  returns the amount of this overlap.  Specifically, the overlap is
                  a number O with the following properties:
                    - 0 <= O <= 1
                    - Let R be the maximum overlap rectangle in boxes, then
                      O == (R.intersect(rect)).area() / (R + rect).area
                    - O == 0 if there is no overlap with any rectangle.
        !*/
        {
            double overlap = 0;
            unsigned int best_idx = 0;
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {

                const unsigned long area = rect.intersect(boxes[i]).area();
                if (area != 0)
                {
                    const double new_overlap = area / static_cast<double>((rect + boxes[i]).area());
                    if (new_overlap > overlap)
                    {
                        overlap = new_overlap;
                        best_idx = i;
                    }
                }
            }

            return std::make_pair(overlap,best_idx);
        }



        overlap_tester_type boxes_overlap;

        image_scanner_type scanner_config;

        const image_array_type& images;
        const std::vector<std::vector<rectangle> >& rects;

        unsigned long max_num_dets;
        double overlap_eps;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_H__


