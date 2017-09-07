// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_LOSS_H_
#define DLIB_DNn_LOSS_H_

#include "loss_abstract.h"
#include "core.h"
#include "../matrix.h"
#include "tensor_tools.h"
#include "../geometry.h"
#include "../image_processing/box_overlap_testing.h"
#include "../image_processing/full_object_detection.h"
#include <sstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class loss_binary_hinge_ 
    {
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
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 && 
                         output_tensor.k() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                *iter++ = out_data[i];
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 && 
                         output_tensor.k() == 1);

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            const float* out_data = output_tensor.host();
            float* g = grad.host_write_only();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const float y = *truth++;
                DLIB_CASSERT(y == +1 || y == -1, "y: " << y);
                const float temp = 1-y*out_data[i];
                if (temp > 0)
                {
                    loss += scale*temp;
                    g[i] = -scale*y;
                }
                else
                {
                    g[i] = 0;
                }
            }
            return loss;
        }

        friend void serialize(const loss_binary_hinge_& , std::ostream& out)
        {
            serialize("loss_binary_hinge_", out);
        }

        friend void deserialize(loss_binary_hinge_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_binary_hinge_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_binary_hinge_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_binary_hinge_& )
        {
            out << "loss_binary_hinge";
            return out;
        }

        friend void to_xml(const loss_binary_hinge_& /*item*/, std::ostream& out)
        {
            out << "<loss_binary_hinge/>";
        }

    };

    template <typename SUBNET>
    using loss_binary_hinge = add_loss_layer<loss_binary_hinge_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_binary_log_ 
    {
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
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 && 
                         output_tensor.k() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                *iter++ = out_data[i];
            }
        }


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 && 
                         output_tensor.k() == 1);
            DLIB_CASSERT(grad.nr() == 1 && 
                         grad.nc() == 1 && 
                         grad.k() == 1);

            tt::sigmoid(grad, output_tensor);

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            float* g = grad.host();
            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const float y = *truth++;
                DLIB_CASSERT(y == +1 || y == -1, "y: " << y);
                float temp;
                if (y > 0)
                {
                    temp = log1pexp(-out_data[i]);
                    loss += scale*temp;
                    g[i] = scale*(g[i]-1);
                }
                else
                {
                    temp = -(-out_data[i]-log1pexp(-out_data[i]));
                    loss += scale*temp;
                    g[i] = scale*g[i];
                }
            }
            return loss;
        }

        friend void serialize(const loss_binary_log_& , std::ostream& out)
        {
            serialize("loss_binary_log_", out);
        }

        friend void deserialize(loss_binary_log_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_binary_log_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_binary_log_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_binary_log_& )
        {
            out << "loss_binary_log";
            return out;
        }

        friend void to_xml(const loss_binary_log_& /*item*/, std::ostream& out)
        {
            out << "<loss_binary_log/>";
        }

    };

    template <typename SUBNET>
    using loss_binary_log = add_loss_layer<loss_binary_log_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_ 
    {
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
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 );
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());


            // Note that output_tensor.k() should match the number of labels.

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                // The index of the largest output for this sample is the label.
                *iter++ = index_of_max(rowm(mat(output_tensor),i));
            }
        }


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 && 
                         grad.nc() == 1);

            tt::softmax(grad, output_tensor);

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            float* g = grad.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const long y = (long)*truth++;
                // The network must produce a number of outputs that is equal to the number
                // of labels when using this type of loss.
                DLIB_CASSERT(y < output_tensor.k(), "y: " << y << ", output_tensor.k(): " << output_tensor.k());
                for (long k = 0; k < output_tensor.k(); ++k)
                {
                    const unsigned long idx = i*output_tensor.k()+k;
                    if (k == y)
                    {
                        loss += scale*-std::log(g[idx]);
                        g[idx] = scale*(g[idx]-1);
                    }
                    else
                    {
                        g[idx] = scale*g[idx];
                    }
                }
            }
            return loss;
        }

        friend void serialize(const loss_multiclass_log_& , std::ostream& out)
        {
            serialize("loss_multiclass_log_", out);
        }

        friend void deserialize(loss_multiclass_log_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_multiclass_log_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_multiclass_log_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_multiclass_log_& )
        {
            out << "loss_multiclass_log";
            return out;
        }

        friend void to_xml(const loss_multiclass_log_& /*item*/, std::ostream& out)
        {
            out << "<loss_multiclass_log/>";
        }

    };

    template <typename SUBNET>
    using loss_multiclass_log = add_loss_layer<loss_multiclass_log_, SUBNET>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    struct mmod_options
    {
    public:

        struct detector_window_details
        {
            detector_window_details() = default; 
            detector_window_details(unsigned long w, unsigned long h) : width(w), height(h) {}
            detector_window_details(unsigned long w, unsigned long h, const std::string& l) : width(w), height(h), label(l) {}

            unsigned long width = 0;
            unsigned long height = 0;
            std::string label;

            friend inline void serialize(const detector_window_details& item, std::ostream& out)
            {
                int version = 2;
                serialize(version, out);
                serialize(item.width, out);
                serialize(item.height, out);
                serialize(item.label, out);
            }

            friend inline void deserialize(detector_window_details& item, std::istream& in)
            {
                int version = 0;
                deserialize(version, in);
                if (version != 1 && version != 2)
                    throw serialization_error("Unexpected version found while deserializing dlib::mmod_options::detector_window_details");
                deserialize(item.width, in);
                deserialize(item.height, in);
                if (version == 2)
                    deserialize(item.label, in);
            }

        };

        mmod_options() = default;

        std::vector<detector_window_details> detector_windows;
        double loss_per_false_alarm = 1;
        double loss_per_missed_target = 1;
        double truth_match_iou_threshold = 0.5;
        test_box_overlap overlaps_nms = test_box_overlap(0.4);
        test_box_overlap overlaps_ignore;

        mmod_options (
            const std::vector<std::vector<mmod_rect>>& boxes,
            const unsigned long target_size,       // We want the length of the longest dimension of the detector window to be this.
            const unsigned long min_target_size,   // But we require that the smallest dimension of the detector window be at least this big.
            const double min_detector_window_overlap_iou = 0.75
        )
        {
            DLIB_CASSERT(0 < min_target_size && min_target_size <= target_size);
            DLIB_CASSERT(0.5 < min_detector_window_overlap_iou && min_detector_window_overlap_iou < 1);

            // Figure out what detector windows we will need.
            for (auto& label : get_labels(boxes))
            {
                for (auto ratio : find_covering_aspect_ratios(boxes, test_box_overlap(min_detector_window_overlap_iou), label))
                {
                    double detector_width;
                    double detector_height;
                    if (ratio < 1)
                    {
                        detector_height = target_size;
                        detector_width = ratio*target_size;
                        if (detector_width < min_target_size)
                        {
                            detector_height = min_target_size;
                            detector_width = min_target_size/ratio;
                        }
                    }
                    else
                    {
                        detector_width = target_size;
                        detector_height = target_size/ratio;
                        if (detector_height < min_target_size)
                        {
                            detector_width = min_target_size*ratio;
                            detector_height = min_target_size;
                        }
                    }

                    detector_window_details p((unsigned long)std::round(detector_width), (unsigned long)std::round(detector_height), label);
                    detector_windows.push_back(p);
                }
            }

            DLIB_CASSERT(detector_windows.size() != 0, "You can't call mmod_options's constructor with a set of boxes that is empty (or only contains ignored boxes).");



            // Convert from mmod_rect to rectangle so we can call
            // find_tight_overlap_tester().
            std::vector<std::vector<rectangle>> temp;
            for (auto&& bi : boxes)
            {
                std::vector<rectangle> rtemp;
                for (auto&& b : bi)
                {
                    if (b.ignore)
                        continue;
                    rtemp.push_back(b.rect);
                }
                temp.push_back(std::move(rtemp));
            }
            overlaps_nms = find_tight_overlap_tester(temp);
            // Relax the non-max-suppression a little so that it doesn't accidentally make
            // it impossible for the detector to output boxes matching the training data.
            // This could be a problem with the tightest possible nms test since there is
            // some small variability in how boxes get positioned between the training data
            // and the coordinate system used by the detector when it runs.  So relaxing it
            // here takes care of that.
            auto iou_thresh             = advance_toward_1(overlaps_nms.get_iou_thresh());
            auto percent_covered_thresh = advance_toward_1(overlaps_nms.get_percent_covered_thresh());
            overlaps_nms = test_box_overlap(iou_thresh, percent_covered_thresh);
        }


    private:

        static double advance_toward_1 (
            double val
        )
        {
            if (val < 1)
                val += (1-val)*0.1;
            return val;
        }

        static size_t count_overlaps (
            const std::vector<rectangle>& rects,
            const test_box_overlap& overlaps,
            const rectangle& ref_box
        )
        {
            size_t cnt = 0;
            for (auto& b : rects)
            {
                if (overlaps(b, ref_box))
                    ++cnt;
            }
            return cnt;
        }

        static std::vector<rectangle> find_rectangles_overlapping_all_others (
            std::vector<rectangle> rects,
            const test_box_overlap& overlaps
        )
        {
            std::vector<rectangle> exemplars;
            dlib::rand rnd;

            while(rects.size() > 0)
            {
                // Pick boxes at random and see if they overlap a lot of other boxes.  We will try
                // 500 different boxes each iteration and select whichever hits the most others to
                // add to our exemplar set.
                rectangle best_ref_box;
                size_t best_cnt = 0;
                for (int iter = 0; iter < 500; ++iter)
                {
                    rectangle ref_box = rects[rnd.get_random_64bit_number()%rects.size()];
                    size_t cnt = count_overlaps(rects, overlaps, ref_box);
                    if (cnt >= best_cnt)
                    {
                        best_cnt = cnt;
                        best_ref_box = ref_box;
                    }
                }

                // Now mark all the boxes the new ref box hit as hit.
                for (size_t i = 0; i < rects.size(); ++i)
                {
                    if (overlaps(rects[i], best_ref_box))
                    {
                        // remove box from rects so we don't hit it again later
                        swap(rects[i], rects.back());
                        rects.pop_back();
                        --i;
                    }
                }

                exemplars.push_back(best_ref_box);
            }

            return exemplars;
        }

        static std::set<std::string> get_labels (
            const std::vector<std::vector<mmod_rect>>& rects
        )
        {
            std::set<std::string> labels;
            for (auto& rr : rects)
            {
                for (auto& r : rr)
                    labels.insert(r.label);
            }
            return labels;
        }

        static std::vector<double> find_covering_aspect_ratios (
            const std::vector<std::vector<mmod_rect>>& rects,
            const test_box_overlap& overlaps,
            const std::string& label
        )
        {
            std::vector<rectangle> boxes;
            // Make sure all the boxes have the same size and position, so that the only thing our
            // checks for overlap will care about is aspect ratio (i.e. scale and x,y position are
            // ignored).
            for (auto& bb : rects)
            {
                for (auto&& b : bb)
                {
                    if (!b.ignore && b.label == label)
                        boxes.push_back(move_rect(set_rect_area(b.rect,400*400), point(0,0)));
                }
            }

            std::vector<double> ratios;
            for (auto r : find_rectangles_overlapping_all_others(boxes, overlaps))
                ratios.push_back(r.width()/(double)r.height());
            return ratios;
        }

    };

    inline void serialize(const mmod_options& item, std::ostream& out)
    {
        int version = 2;

        serialize(version, out);
        serialize(item.detector_windows, out);
        serialize(item.loss_per_false_alarm, out);
        serialize(item.loss_per_missed_target, out);
        serialize(item.truth_match_iou_threshold, out);
        serialize(item.overlaps_nms, out);
        serialize(item.overlaps_ignore, out);
    }

    inline void deserialize(mmod_options& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 2 && version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::mmod_options");
        if (version == 1)
        {
            unsigned long width;
            unsigned long height;
            deserialize(width, in);
            deserialize(height, in);
            item.detector_windows = {mmod_options::detector_window_details(width, height)};
        }
        else
        {
            deserialize(item.detector_windows, in);
        }
        deserialize(item.loss_per_false_alarm, in);
        deserialize(item.loss_per_missed_target, in);
        deserialize(item.truth_match_iou_threshold, in);
        deserialize(item.overlaps_nms, in);
        deserialize(item.overlaps_ignore, in);
    }

// ----------------------------------------------------------------------------------------

    class loss_mmod_ 
    {
        struct intermediate_detection
        {
            intermediate_detection() = default; 

            intermediate_detection(
                rectangle rect_
            ) : rect(rect_) {}

            intermediate_detection(
                rectangle rect_,
                double detection_confidence_,
                size_t tensor_offset_,
                long channel
            ) : rect(rect_), detection_confidence(detection_confidence_), tensor_offset(tensor_offset_), tensor_channel(channel) {}

            rectangle rect;
            double detection_confidence = 0;
            size_t tensor_offset = 0;
            long tensor_channel = 0;

            bool operator<(const intermediate_detection& item) const { return detection_confidence < item.detection_confidence; }
        };

    public:

        typedef std::vector<mmod_rect> training_label_type;
        typedef std::vector<mmod_rect> output_label_type;

        loss_mmod_() {}

        loss_mmod_(mmod_options options_) : options(options_) {}

        const mmod_options& get_options (
        ) const { return options; }

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter,
            double adjust_threshold = 0
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(sub.sample_expansion_factor() == 1,  sub.sample_expansion_factor());

            std::vector<intermediate_detection> dets_accum;
            output_label_type final_dets;
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                tensor_to_dets(input_tensor, output_tensor, i, dets_accum, adjust_threshold, sub);

                // Do non-max suppression
                final_dets.clear();
                for (unsigned long i = 0; i < dets_accum.size(); ++i)
                {
                    if (overlaps_any_box_nms(final_dets, dets_accum[i].rect))
                        continue;

                    final_dets.push_back(mmod_rect(dets_accum[i].rect,
                                                   dets_accum[i].detection_confidence,
                                                   options.detector_windows[dets_accum[i].tensor_channel].label));
                }

                *iter++ = std::move(final_dets);
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size());



            // we will scale the loss so that it doesn't get really huge
            const double scale = 1.0/output_tensor.size();
            double loss = 0;

            float* g = grad.host_write_only();
            for (size_t i = 0; i < grad.size(); ++i)
                g[i] = 0;

            const float* out_data = output_tensor.host();

            std::vector<size_t> truth_idxs;  truth_idxs.reserve(truth->size());
            std::vector<intermediate_detection> dets;
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                tensor_to_dets(input_tensor, output_tensor, i, dets, -options.loss_per_false_alarm, sub);

                const unsigned long max_num_dets = 50 + truth->size()*5;

                // The loss will measure the number of incorrect detections.  A detection is
                // incorrect if it doesn't hit a truth rectangle or if it is a duplicate detection
                // on a truth rectangle.
                loss += truth->size()*options.loss_per_missed_target;
                for (auto&& x : *truth)
                {
                    if (!x.ignore)
                    {
                        size_t k;
                        point p;
                        if(image_rect_to_feat_coord(p, input_tensor, x, x.label, sub, k))
                        {
                            // Ignore boxes that can't be detected by the CNN.
                            loss -= 1;
                            continue;
                        }
                        const size_t idx = (k*output_tensor.nr() + p.y())*output_tensor.nc() + p.x();
                        loss -= out_data[idx];
                        // compute gradient
                        g[idx] = -scale;
                        truth_idxs.push_back(idx);
                    }
                    else
                    {
                        // This box was ignored so shouldn't have been counted in the loss.
                        loss -= 1;
                        truth_idxs.push_back(0);
                    }
                }

                // Measure the loss augmented score for the detections which hit a truth rect.
                std::vector<double> truth_score_hits(truth->size(), 0);

                // keep track of which truth boxes we have hit so far.
                std::vector<bool> hit_truth_table(truth->size(), false);

                std::vector<intermediate_detection> final_dets;
                // The point of this loop is to fill out the truth_score_hits array. 
                for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
                {
                    if (overlaps_any_box_nms(final_dets, dets[i].rect))
                        continue;

                    const auto& det_label = options.detector_windows[dets[i].tensor_channel].label;

                    const std::pair<double,unsigned int> hittruth = find_best_match(*truth, dets[i].rect, det_label);

                    final_dets.push_back(dets[i].rect);

                    const double truth_match = hittruth.first;
                    // if hit truth rect
                    if (truth_match > options.truth_match_iou_threshold)
                    {
                        // if this is the first time we have seen a detect which hit (*truth)[hittruth.second]
                        const double score = dets[i].detection_confidence;
                        if (hit_truth_table[hittruth.second] == false)
                        {
                            hit_truth_table[hittruth.second] = true;
                            truth_score_hits[hittruth.second] += score;
                        }
                        else
                        {
                            truth_score_hits[hittruth.second] += score + options.loss_per_false_alarm;
                        }
                    }
                }

                // Check if any of the truth boxes are unobtainable because the NMS is
                // killing them.  If so, automatically set those unobtainable boxes to
                // ignore and print a warning message to the user.
                for (size_t i = 0; i < hit_truth_table.size(); ++i)
                {
                    if (!hit_truth_table[i] && !(*truth)[i].ignore) 
                    {
                        // So we didn't hit this truth box.  Is that because there is
                        // another, different truth box, that overlaps it according to NMS?
                        const std::pair<double,unsigned int> hittruth = find_best_match(*truth, (*truth)[i], i);
                        if (hittruth.second == i || (*truth)[hittruth.second].ignore)
                            continue;
                        rectangle best_matching_truth_box = (*truth)[hittruth.second];
                        if (options.overlaps_nms(best_matching_truth_box, (*truth)[i]))
                        {
                            const size_t idx = truth_idxs[i];
                            // We are ignoring this box so we shouldn't have counted it in the
                            // loss in the first place.  So we subtract out the loss values we
                            // added for it in the code above.
                            loss -= 1-out_data[idx];
                            g[idx] = 0;
                            std::cout << "Warning, ignoring object.  We encountered a truth rectangle located at " << (*truth)[i].rect;
                            std::cout << " that is suppressed by non-max-suppression ";
                            std::cout << "because it is overlapped by another truth rectangle located at " << best_matching_truth_box 
                                      << " (IoU:"<< box_intersection_over_union(best_matching_truth_box,(*truth)[i]) <<", Percent covered:" 
                                      << box_percent_covered(best_matching_truth_box,(*truth)[i]) << ")." << std::endl;
                        }
                    }
                }

                hit_truth_table.assign(hit_truth_table.size(), false);
                final_dets.clear();


                // Now figure out which detections jointly maximize the loss and detection score sum.  We
                // need to take into account the fact that allowing a true detection in the output, while 
                // initially reducing the loss, may allow us to increase the loss later with many duplicate
                // detections.
                for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
                {
                    if (overlaps_any_box_nms(final_dets, dets[i].rect))
                        continue;

                    const auto& det_label = options.detector_windows[dets[i].tensor_channel].label;

                    const std::pair<double,unsigned int> hittruth = find_best_match(*truth, dets[i].rect, det_label);

                    const double truth_match = hittruth.first;
                    if (truth_match > options.truth_match_iou_threshold)
                    {
                        if (truth_score_hits[hittruth.second] > options.loss_per_missed_target)
                        {
                            if (!hit_truth_table[hittruth.second])
                            {
                                hit_truth_table[hittruth.second] = true;
                                final_dets.push_back(dets[i]);
                                loss -= options.loss_per_missed_target;
                            }
                            else
                            {
                                final_dets.push_back(dets[i]);
                                loss += options.loss_per_false_alarm;
                            }
                        }
                    }
                    else if (!overlaps_ignore_box(*truth, dets[i].rect))
                    {
                        // didn't hit anything
                        final_dets.push_back(dets[i]);
                        loss += options.loss_per_false_alarm;
                    }
                }

                for (auto&& x : final_dets)
                {
                    loss += out_data[x.tensor_offset];
                    g[x.tensor_offset] += scale;
                }

                ++truth;
                g        += output_tensor.k()*output_tensor.nr()*output_tensor.nc();
                out_data += output_tensor.k()*output_tensor.nr()*output_tensor.nc();
            } // END for (long i = 0; i < output_tensor.num_samples(); ++i)


            // Here we scale the loss so that it's roughly equal to the number of mistakes
            // in an image.  Note that this scaling is different than the scaling we
            // applied to the gradient but it doesn't matter since the loss value isn't
            // used to update parameters.  It's used only for display and to check if we
            // have converged.  So it doesn't matter that they are scaled differently and
            // this way the loss that is displayed is readily interpretable to the user.
            return loss/output_tensor.num_samples();
        }


        friend void serialize(const loss_mmod_& item, std::ostream& out)
        {
            serialize("loss_mmod_", out);
            serialize(item.options, out);
        }

        friend void deserialize(loss_mmod_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_mmod_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_mmod_.");
            deserialize(item.options, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_mmod_& item)
        {
            out << "loss_mmod\t (";

            out << "detector_windows:(";
            auto& opts = item.options;
            for (size_t i = 0; i < opts.detector_windows.size(); ++i)
            {
                out << opts.detector_windows[i].width << "x" << opts.detector_windows[i].height;
                if (i+1 < opts.detector_windows.size())
                    out << ",";
            }
            out << ")";
            out << ", loss per FA:" << opts.loss_per_false_alarm;
            out << ", loss per miss:" << opts.loss_per_missed_target;
            out << ", truth match IOU thresh:" << opts.truth_match_iou_threshold;
            out << ", overlaps_nms:("<<opts.overlaps_nms.get_iou_thresh()<<","<<opts.overlaps_nms.get_percent_covered_thresh()<<")";
            out << ", overlaps_ignore:("<<opts.overlaps_ignore.get_iou_thresh()<<","<<opts.overlaps_ignore.get_percent_covered_thresh()<<")";

            out << ")";
            return out;
        }

        friend void to_xml(const loss_mmod_& /*item*/, std::ostream& out)
        {
            // TODO, add options fields
            out << "<loss_mmod/>";
        }

    private:

        template <typename net_type>
        void tensor_to_dets (
            const tensor& input_tensor,
            const tensor& output_tensor,
            long i,
            std::vector<intermediate_detection>& dets_accum,
            double adjust_threshold,
            const net_type& net 
        ) const
        {
            DLIB_CASSERT(net.sample_expansion_factor() == 1,net.sample_expansion_factor());
            DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size());
            const float* out_data = output_tensor.host() + output_tensor.k()*output_tensor.nr()*output_tensor.nc()*i;
            // scan the final layer and output the positive scoring locations
            dets_accum.clear();
            for (long k = 0; k < output_tensor.k(); ++k)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        double score = out_data[(k*output_tensor.nr() + r)*output_tensor.nc() + c];
                        if (score > adjust_threshold)
                        {
                            dpoint p = output_tensor_to_input_tensor(net, point(c,r));
                            drectangle rect = centered_drect(p, options.detector_windows[k].width, options.detector_windows[k].height);
                            rect = input_layer(net).tensor_space_to_image_space(input_tensor,rect);

                            dets_accum.push_back(intermediate_detection(rect, score, (k*output_tensor.nr() + r)*output_tensor.nc() + c, k));
                        }
                    }
                }
            }
            std::sort(dets_accum.rbegin(), dets_accum.rend());
        }

        size_t find_best_detection_window (
            rectangle rect,
            const std::string& label
        ) const
        {
            rect = move_rect(set_rect_area(rect, 400*400), point(0,0));

            // Figure out which detection window in options.detector_windows has the most
            // similar aspect ratio to rect.
            size_t best_i = 0;
            double best_ratio_diff = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < options.detector_windows.size(); ++i)
            {
                if (options.detector_windows[i].label != label)
                    continue;
                rectangle det_window = centered_rect(point(0,0), options.detector_windows[i].width, options.detector_windows[i].height);
                det_window = move_rect(set_rect_area(det_window, 400*400), point(0,0));

                double iou = box_intersection_over_union(rect, det_window);
                if (iou > best_ratio_diff)
                {
                    best_ratio_diff = iou;
                    best_i = i;
                }
            }
            return best_i;
        }

        template <typename net_type>
        bool image_rect_to_feat_coord (
            point& tensor_p,
            const tensor& input_tensor,
            const rectangle& rect,
            const std::string& label,
            const net_type& net,
            size_t& det_idx
        ) const 
        {
            using namespace std;
            if (!input_layer(net).image_contained_point(input_tensor,center(rect)))
            {
                std::ostringstream sout;
                sout << "Encountered a truth rectangle located at " << rect << " that is outside the image." << endl;
                sout << "The center of each truth rectangle must be within the image." << endl;
                throw impossible_labeling_error(sout.str());
            }

            det_idx = find_best_detection_window(rect,label); 

            // Compute the scale we need to be at to get from rect to our detection window.
            // Note that we compute the scale as the max of two numbers.  It doesn't
            // actually matter which one we pick, because if they are very different then
            // it means the box can't be matched by the sliding window.  But picking the
            // max causes the right error message to be selected in the logic below.
            const double scale = std::max(options.detector_windows[det_idx].width/(double)rect.width(), options.detector_windows[det_idx].height/(double)rect.height());
            const rectangle mapped_rect = input_layer(net).image_space_to_tensor_space(input_tensor, std::min(1.0,scale), rect);

            // compute the detection window that we would use at this position.
            tensor_p = center(mapped_rect);
            rectangle det_window = centered_rect(tensor_p, options.detector_windows[det_idx].width,options.detector_windows[det_idx].height);
            det_window = input_layer(net).tensor_space_to_image_space(input_tensor, det_window);

            // make sure the rect can actually be represented by the image pyramid we are
            // using.
            if (box_intersection_over_union(rect, det_window) <= options.truth_match_iou_threshold)
            {
                std::cout << "Warning, ignoring object.  We encountered a truth rectangle with a width and height of " << rect.width() << " and " << rect.height() << ".  ";
                std::cout << "The image pyramid and sliding windows can't output a rectangle of this shape.  ";
                const double detector_area = options.detector_windows[det_idx].width*options.detector_windows[det_idx].height;
                if (mapped_rect.area()/detector_area <= options.truth_match_iou_threshold)
                {
                    std::cout << "This is because the rectangle is smaller than the best matching detection window, which has a width ";
                    std::cout << "and height of " << options.detector_windows[det_idx].width << " and " << options.detector_windows[det_idx].height << "." << std::endl;
                }
                else
                {
                    std::cout << "This is because the rectangle's aspect ratio is too different from the best matching detection window, ";
                    std::cout << "which has a width and height of " << options.detector_windows[det_idx].width << " and " << options.detector_windows[det_idx].height << "." << std::endl;
                }
                return true;
            }

            // now map through the CNN to the output layer.
            tensor_p = input_tensor_to_output_tensor(net,tensor_p);

            const tensor& output_tensor = net.get_output();
            if (!get_rect(output_tensor).contains(tensor_p))
            {
                std::cout << "Warning, ignoring object.  We encountered a truth rectangle located at " << rect << " that is too close to the edge ";
                std::cout << "of the image to be captured by the CNN features." << std::endl;
                return true;
            }

            return false;
        }


        bool overlaps_ignore_box (
            const std::vector<mmod_rect>& boxes,
            const rectangle& rect
        ) const
        {
            for (auto&& b : boxes)
            {
                if (b.ignore && options.overlaps_ignore(b, rect))
                    return true;
            }
            return false;
        }

        std::pair<double,unsigned int> find_best_match(
            const std::vector<mmod_rect>& boxes,
            const rectangle& rect,
            const std::string& label
        ) const
        {
            double match = 0;
            unsigned int best_idx = 0;
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {
                if (boxes[i].ignore || boxes[i].label != label)
                    continue;

                const double new_match = box_intersection_over_union(rect, boxes[i]);
                if (new_match > match)
                {
                    match = new_match;
                    best_idx = i;
                }
            }

            return std::make_pair(match,best_idx);
        }

        std::pair<double,unsigned int> find_best_match(
            const std::vector<mmod_rect>& boxes,
            const rectangle& rect,
            const size_t excluded_idx
        ) const
        {
            double match = 0;
            unsigned int best_idx = 0;
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {
                if (boxes[i].ignore || excluded_idx == i)
                    continue;

                const double new_match = box_intersection_over_union(rect, boxes[i]);
                if (new_match > match)
                {
                    match = new_match;
                    best_idx = i;
                }
            }

            return std::make_pair(match,best_idx);
        }

        template <typename T>
        inline bool overlaps_any_box_nms (
            const std::vector<T>& rects,
            const rectangle& rect
        ) const
        {
            for (auto&& r : rects)
            {
                if (options.overlaps_nms(r.rect, rect))
                    return true;
            }
            return false;
        }


        mmod_options options;

    };

    template <typename SUBNET>
    using loss_mmod = add_loss_layer<loss_mmod_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_metric_ 
    {
    public:

        typedef unsigned long training_label_type;
        typedef matrix<float,0,1> output_label_type;

        loss_metric_() = default;

        loss_metric_(
            float margin_,
            float dist_thresh_
        ) : margin(margin_), dist_thresh(dist_thresh_) 
        {
            DLIB_CASSERT(margin_ > 0);
            DLIB_CASSERT(dist_thresh_ > 0);
        }

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1);

            const float* p = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                *iter = mat(p,output_tensor.k(),1);

                ++iter;
                p += output_tensor.k();
            }
        }


        float get_margin() const { return margin; }
        float get_distance_threshold() const { return dist_thresh; }

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 && 
                         grad.nc() == 1);



            temp.set_size(output_tensor.num_samples(), output_tensor.num_samples());
            grad_mul.copy_size(temp);

            tt::gemm(0, temp, 1, output_tensor, false, output_tensor, true);


            std::vector<double> temp_threshs;
            const float* d = temp.host();
            double loss = 0;
            double num_pos_samps = 0.0001;
            double num_neg_samps = 0.0001;
            for (long r = 0; r < temp.num_samples(); ++r)
            {
                auto xx = d[r*temp.num_samples() + r];
                const auto x_label = *(truth + r);
                for (long c = r+1; c < temp.num_samples(); ++c)
                {
                    const auto y_label = *(truth + c);
                    if (x_label == y_label)
                    {
                        ++num_pos_samps;
                    }
                    else
                    {
                        ++num_neg_samps;

                        // Figure out what distance threshold, when applied to the negative pairs,
                        // causes there to be an equal number of positive and negative pairs.
                        auto yy = d[c*temp.num_samples() + c];
                        auto xy = d[r*temp.num_samples() + c];
                        // compute the distance between x and y samples.
                        auto d2 = xx + yy - 2*xy;
                        if (d2 < 0)
                            d2 = 0;
                        temp_threshs.push_back(d2);
                    }
                }
            }
            // The whole objective function is multiplied by this to scale the loss
            // relative to the number of things in the mini-batch.
            const double scale = 0.5/num_pos_samps;
            DLIB_CASSERT(num_pos_samps>=1, "Make sure each mini-batch contains both positive pairs and negative pairs");
            DLIB_CASSERT(num_neg_samps>=1, "Make sure each mini-batch contains both positive pairs and negative pairs");

            std::sort(temp_threshs.begin(), temp_threshs.end());
            const float neg_thresh = std::sqrt(temp_threshs[std::min(num_pos_samps,num_neg_samps)-1]);

            // loop over all the pairs of training samples and compute the loss and
            // gradients.  Note that we only use the hardest negative pairs and that in
            // particular we pick the number of negative pairs equal to the number of
            // positive pairs so everything is balanced.
            float* gm = grad_mul.host();
            for (long r = 0; r < temp.num_samples(); ++r)
            {
                gm[r*temp.num_samples() + r] = 0;
                const auto x_label = *(truth + r);
                auto xx = d[r*temp.num_samples() + r];
                for (long c = 0; c < temp.num_samples(); ++c)
                {
                    if (r==c)
                        continue;
                    const auto y_label = *(truth + c);
                    auto yy = d[c*temp.num_samples() + c];
                    auto xy = d[r*temp.num_samples() + c];

                    // compute the distance between x and y samples.
                    auto d2 = xx + yy - 2*xy;
                    if (d2 <= 0)
                        d2 = 0;
                    else 
                        d2 = std::sqrt(d2);

                    // It should be noted that the derivative of length(x-y) with respect
                    // to the x vector is the unit vector (x-y)/length(x-y).  If you stare
                    // at the code below long enough you will see that it's just an
                    // application of this formula.

                    if (x_label == y_label)
                    {
                        // Things with the same label should have distances < dist_thresh between
                        // them.  If not then we experience non-zero loss.
                        if (d2 < dist_thresh-margin)
                        {
                            gm[r*temp.num_samples() + c] = 0;
                        }
                        else
                        {
                            loss += scale*(d2 - (dist_thresh-margin));
                            gm[r*temp.num_samples() + r] += scale/d2;
                            gm[r*temp.num_samples() + c] = -scale/d2;
                        }
                    }
                    else
                    {
                        // Things with different labels should have distances > dist_thresh between
                        // them.  If not then we experience non-zero loss.
                        if (d2 > dist_thresh+margin || d2 > neg_thresh)
                        {
                            gm[r*temp.num_samples() + c] = 0;
                        }
                        else
                        {
                            loss += scale*((dist_thresh+margin) - d2);
                            // don't divide by zero (or a really small number)
                            d2 = std::max(d2, 0.001f);
                            gm[r*temp.num_samples() + r] -= scale/d2;
                            gm[r*temp.num_samples() + c] = scale/d2;
                        }
                    }
                }
            }


            tt::gemm(0, grad, 1, grad_mul, false, output_tensor, false); 

            return loss;
        }

        friend void serialize(const loss_metric_& item, std::ostream& out)
        {
            serialize("loss_metric_2", out);
            serialize(item.margin, out);
            serialize(item.dist_thresh, out);
        }

        friend void deserialize(loss_metric_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "loss_metric_")
            {
                // These values used to be hard coded, so for this version of the metric
                // learning loss we just use these values.
                item.margin = 0.1;
                item.dist_thresh = 0.75;
                return;
            }
            else if (version == "loss_metric_2")
            {
                deserialize(item.margin, in);
                deserialize(item.dist_thresh, in);
            }
            else
            {
                throw serialization_error("Unexpected version found while deserializing dlib::loss_metric_.  Instead found " + version);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_metric_& item )
        {
            out << "loss_metric (margin="<<item.margin<<", distance_threshold="<<item.dist_thresh<<")";
            return out;
        }

        friend void to_xml(const loss_metric_& item, std::ostream& out)
        {
            out << "<loss_metric margin='"<<item.margin<<"' distance_threshold='"<<item.dist_thresh<<"'/>";
        }

    private:
        float margin = 0.04;
        float dist_thresh = 0.6;


        // These variables are only here to avoid being reallocated over and over in
        // compute_loss_value_and_gradient()
        mutable resizable_tensor temp, grad_mul;

    };

    template <typename SUBNET>
    using loss_metric = add_loss_layer<loss_metric_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_mean_squared_
    {
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
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const tensor& output_tensor = sub.get_output();

            DLIB_CASSERT(output_tensor.nr() == 1 &&
                         output_tensor.nc() == 1 &&
                         output_tensor.k() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                *iter++ = out_data[i];
            }
        }


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 &&
                         output_tensor.nc() == 1 &&
                         output_tensor.k() == 1);
            DLIB_CASSERT(grad.nr() == 1 &&
                         grad.nc() == 1 &&
                         grad.k() == 1);

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            float* g = grad.host_write_only();
            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const float y = *truth++;
                const float temp1 = y - out_data[i];
                const float temp2 = scale*temp1;
                loss += 0.5*temp2*temp1;
                g[i] = -temp2;

            }
            return loss;
        }

        friend void serialize(const loss_mean_squared_& , std::ostream& out)
        {
            serialize("loss_mean_squared_", out);
        }

        friend void deserialize(loss_mean_squared_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_mean_squared_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_mean_squared_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_mean_squared_& )
        {
            out << "loss_mean_squared";
            return out;
        }

        friend void to_xml(const loss_mean_squared_& /*item*/, std::ostream& out)
        {
            out << "<loss_mean_squared/>";
        }

    };

    template <typename SUBNET>
    using loss_mean_squared = add_loss_layer<loss_mean_squared_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_mean_squared_multioutput_
    {
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
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const tensor& output_tensor = sub.get_output();

            DLIB_CASSERT(output_tensor.nr() == 1 &&
                         output_tensor.nc() == 1)
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                *iter++ = mat(out_data, output_tensor.k(), 1);
                out_data += output_tensor.k();
            }
        }


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 &&
                         output_tensor.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 &&
                         grad.nc() == 1);
            DLIB_CASSERT(grad.k() == output_tensor.k());
            const long k = output_tensor.k();
            for (long idx = 0; idx < output_tensor.num_samples(); ++idx)
            {
                const_label_iterator truth_matrix_ptr = (truth + idx);
                DLIB_CASSERT((*truth_matrix_ptr).nr() == k &&
                             (*truth_matrix_ptr).nc() == 1);
            }

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            float* g = grad.host_write_only();
            const float* out_data = output_tensor.host();
            matrix<float> ytrue;
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                ytrue = *truth++;
                for (long j = 0; j < output_tensor.k(); ++j)
                {
                    const float y = ytrue(j, 0);
                    const float temp1 = y - *out_data++;
                    const float temp2 = scale*temp1;
                    loss += 0.5*temp2*temp1;
                    *g = -temp2;
                    ++g;
                }

            }
            return loss;
        }

        friend void serialize(const loss_mean_squared_multioutput_& , std::ostream& out)
        {
            serialize("loss_mean_squared_multioutput_", out);
        }

        friend void deserialize(loss_mean_squared_multioutput_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_mean_squared_multioutput_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_mean_squared_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_mean_squared_multioutput_& )
        {
            out << "loss_mean_squared_multioutput";
            return out;
        }

        friend void to_xml(const loss_mean_squared_multioutput_& /*item*/, std::ostream& out)
        {
            out << "<loss_mean_squared_multioutput/>";
        }

    };

    template <typename SUBNET>
    using loss_mean_squared_multioutput = add_loss_layer<loss_mean_squared_multioutput_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_per_pixel_
    {
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
        static void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        )
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const tensor& output_tensor = sub.get_output();

            DLIB_CASSERT(output_tensor.k() >= 1); // Note that output_tensor.k() should match the number of labels.
            DLIB_CASSERT(output_tensor.k() < std::numeric_limits<uint16_t>::max());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* const out_data = output_tensor.host();

            // The index of the largest output for each element is the label.
            const auto find_label = [&](long sample, long r, long c) 
            {
                uint16_t label = 0;
                float max_value = out_data[tensor_index(output_tensor, sample, 0, r, c)];
                for (long k = 1; k < output_tensor.k(); ++k) 
                {
                    const float value = out_data[tensor_index(output_tensor, sample, k, r, c)];
                    if (value > max_value) 
                    {
                        label = static_cast<uint16_t>(k);
                        max_value = value;
                    }
                }
                return label;
            };

            for (long i = 0; i < output_tensor.num_samples(); ++i, ++iter) 
            {
                iter->set_size(output_tensor.nr(), output_tensor.nc());
                for (long r = 0; r < output_tensor.nr(); ++r) 
                {
                    for (long c = 0; c < output_tensor.nc(); ++c) 
                    {
                        // The index of the largest output for this element is the label.
                        iter->operator()(r, c) = find_label(i, r, c);
                    }
                }
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.k() >= 1);
            DLIB_CASSERT(output_tensor.k() < std::numeric_limits<uint16_t>::max());
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                         output_tensor.nc() == grad.nc() &&
                         output_tensor.k() == grad.k());
            for (long idx = 0; idx < output_tensor.num_samples(); ++idx)
            {
                const_label_iterator truth_matrix_ptr = (truth + idx);
                DLIB_CASSERT(truth_matrix_ptr->nr() == output_tensor.nr() &&
                             truth_matrix_ptr->nc() == output_tensor.nc(),
                             "truth size = " << truth_matrix_ptr->nr() << " x " << truth_matrix_ptr->nc() << ", "
                             "output size = " << output_tensor.nr() << " x " << output_tensor.nc());
            }

            tt::softmax(grad, output_tensor);

            // The loss we output is the average loss over the mini-batch, and also over each element of the matrix output.
            const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc());
            double loss = 0;
            float* const g = grad.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i, ++truth)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const uint16_t y = truth->operator()(r, c);
                        // The network must produce a number of outputs that is equal to the number
                        // of labels when using this type of loss.
                        DLIB_CASSERT(static_cast<long>(y) < output_tensor.k() || y == label_to_ignore,
                                        "y: " << y << ", output_tensor.k(): " << output_tensor.k());
                        for (long k = 0; k < output_tensor.k(); ++k)
                        {
                            const size_t idx = tensor_index(output_tensor, i, k, r, c);
                            if (k == y)
                            {
                                loss += scale*-std::log(g[idx]);
                                g[idx] = scale*(g[idx] - 1);
                            }
                            else if (y == label_to_ignore)
                            {
                                g[idx] = 0.f;
                            }
                            else
                            {
                                g[idx] = scale*g[idx];
                            }
                        }
                    }
                }
            }
            return loss;
        }

        friend void serialize(const loss_multiclass_log_per_pixel_& , std::ostream& out)
        {
            serialize("loss_multiclass_log_per_pixel_", out);
        }

        friend void deserialize(loss_multiclass_log_per_pixel_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_multiclass_log_per_pixel_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_multiclass_log_per_pixel_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_multiclass_log_per_pixel_& )
        {
            out << "loss_multiclass_log_per_pixel";
            return out;
        }

        friend void to_xml(const loss_multiclass_log_per_pixel_& /*item*/, std::ostream& out)
        {
            out << "<loss_multiclass_log_per_pixel/>";
        }

    private:
        static size_t tensor_index(const tensor& t, long sample, long k, long row, long column)
        {
            // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
            return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
        }

    };

    template <typename SUBNET>
    using loss_multiclass_log_per_pixel = add_loss_layer<loss_multiclass_log_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_per_pixel_weighted_
    {
    public:

        struct weighted_label
        {
            weighted_label()
            {}

            weighted_label(uint16_t label, float weight = 1.f)
                : label(label), weight(weight)
            {}

            // In semantic segmentation, 65536 classes ought to be enough for anybody.
            uint16_t label = 0;
            float weight = 1.f;
        };

        typedef matrix<weighted_label> training_label_type;
        typedef matrix<uint16_t> output_label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        static void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        )
        {
            loss_multiclass_log_per_pixel_::to_label(input_tensor, sub, iter);
        }

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.k() >= 1);
            DLIB_CASSERT(output_tensor.k() < std::numeric_limits<uint16_t>::max());
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                         output_tensor.nc() == grad.nc() &&
                         output_tensor.k() == grad.k());
            for (long idx = 0; idx < output_tensor.num_samples(); ++idx)
            {
                const_label_iterator truth_matrix_ptr = (truth + idx);
                DLIB_CASSERT(truth_matrix_ptr->nr() == output_tensor.nr() &&
                             truth_matrix_ptr->nc() == output_tensor.nc(),
                             "truth size = " << truth_matrix_ptr->nr() << " x " << truth_matrix_ptr->nc() << ", "
                             "output size = " << output_tensor.nr() << " x " << output_tensor.nc());
            }

            tt::softmax(grad, output_tensor);

            // The loss we output is the weighted average loss over the mini-batch, and also over each element of the matrix output.
            const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc());
            double loss = 0;
            float* const g = grad.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i, ++truth)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const weighted_label& weighted_label = truth->operator()(r, c);
                        const uint16_t y = weighted_label.label;
                        const float weight = weighted_label.weight;
                        // The network must produce a number of outputs that is equal to the number
                        // of labels when using this type of loss.
                        DLIB_CASSERT(static_cast<long>(y) < output_tensor.k() || weight == 0.f,
                                        "y: " << y << ", output_tensor.k(): " << output_tensor.k());
                        for (long k = 0; k < output_tensor.k(); ++k)
                        {
                            const size_t idx = tensor_index(output_tensor, i, k, r, c);
                            if (k == y)
                            {
                                loss += weight*scale*-std::log(g[idx]);
                                g[idx] = weight*scale*(g[idx] - 1);
                            }
                            else
                            {
                                g[idx] = weight*scale*g[idx];
                            }
                        }
                    }
                }
            }
            return loss;
        }

        friend void serialize(const loss_multiclass_log_per_pixel_weighted_& , std::ostream& out)
        {
            serialize("loss_multiclass_log_per_pixel_weighted_", out);
        }

        friend void deserialize(loss_multiclass_log_per_pixel_weighted_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_multiclass_log_per_pixel_weighted_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_multiclass_log_per_pixel_weighted_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_multiclass_log_per_pixel_weighted_& )
        {
            out << "loss_multiclass_log_per_pixel_weighted";
            return out;
        }

        friend void to_xml(const loss_multiclass_log_per_pixel_weighted_& /*item*/, std::ostream& out)
        {
            out << "<loss_multiclass_log_per_pixel_weighted/>";
        }

    private:
        static size_t tensor_index(const tensor& t, long sample, long k, long row, long column)
        {
            // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
            return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
        }

    };

    template <typename SUBNET>
    using loss_multiclass_log_per_pixel_weighted = add_loss_layer<loss_multiclass_log_per_pixel_weighted_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_mean_squared_per_pixel_
    {
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
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const tensor& output_tensor = sub.get_output();

            DLIB_CASSERT(output_tensor.k() == 1, "output k = " << output_tensor.k());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i, ++iter)
            {
                iter->set_size(output_tensor.nr(), output_tensor.nc());
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        iter->operator()(r, c) = out_data[tensor_index(output_tensor, i, 0, r, c)];
                    }
                }
            }
        }


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.k() >= 1);
            DLIB_CASSERT(output_tensor.k() < std::numeric_limits<uint16_t>::max());
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                output_tensor.nc() == grad.nc() &&
                output_tensor.k() == grad.k());
            for (long idx = 0; idx < output_tensor.num_samples(); ++idx)
            {
                const_label_iterator truth_matrix_ptr = (truth + idx);
                DLIB_CASSERT(truth_matrix_ptr->nr() == output_tensor.nr() &&
                    truth_matrix_ptr->nc() == output_tensor.nc(),
                    "truth size = " << truth_matrix_ptr->nr() << " x " << truth_matrix_ptr->nc() << ", "
                    "output size = " << output_tensor.nr() << " x " << output_tensor.nc());
            }

            // The loss we output is the average loss over the mini-batch, and also over each element of the matrix output.
            const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc());
            double loss = 0;
            float* const g = grad.host();
            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i, ++truth)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const float y = truth->operator()(r, c);
                        const size_t idx = tensor_index(output_tensor, i, 0, r, c);
                        const float temp1 = y - out_data[idx];
                        const float temp2 = scale*temp1;
                        loss += 0.5*temp2*temp1;
                        g[idx] = -temp2;
                    }
                }
            }
            return loss;
        }

        friend void serialize(const loss_mean_squared_per_pixel_& , std::ostream& out)
        {
            serialize("loss_mean_squared_per_pixel_", out);
        }

        friend void deserialize(loss_mean_squared_per_pixel_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_mean_squared_per_pixel_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_mean_squared_per_pixel_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_mean_squared_per_pixel_& )
        {
            out << "loss_mean_squared_per_pixel";
            return out;
        }

        friend void to_xml(const loss_mean_squared_per_pixel_& /*item*/, std::ostream& out)
        {
            out << "<loss_mean_squared_per_pixel/>";
        }

    private:
        static size_t tensor_index(const tensor& t, long sample, long k, long row, long column)
        {
            // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
            return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
        }
    };

    template <typename SUBNET>
    using loss_mean_squared_per_pixel = add_loss_layer<loss_mean_squared_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------


}

#endif // DLIB_DNn_LOSS_H_

