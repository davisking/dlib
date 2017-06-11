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

        mmod_options() = default;

        unsigned long detector_width = 80;
        unsigned long detector_height = 80;
        double loss_per_false_alarm = 1;
        double loss_per_missed_target = 1;
        double truth_match_iou_threshold = 0.5;
        test_box_overlap overlaps_nms = test_box_overlap(0.4);
        test_box_overlap overlaps_ignore;

        mmod_options (
            const std::vector<std::vector<mmod_rect>>& boxes,
            const unsigned long target_size = 6400
        )
        {
            std::vector<std::vector<rectangle>> temp;

            // find the average width and height.  Then we will set the detector width and
            // height to match the average aspect ratio of the boxes given the target_size.
            running_stats<double> avg_width, avg_height;
            for (auto&& bi : boxes)
            {
                std::vector<rectangle> rtemp;
                for (auto&& b : bi)
                {
                    if (b.ignore)
                        continue;
                    avg_width.add(b.rect.width());
                    avg_height.add(b.rect.height());
                    rtemp.push_back(b.rect);
                }
                temp.push_back(std::move(rtemp));
            }

            // now adjust the box size so that it is about target_pixels pixels in size
            double size = avg_width.mean()*avg_height.mean();
            double scale = std::sqrt(target_size/size);

            detector_width = (unsigned long)(avg_width.mean()*scale+0.5);
            detector_height = (unsigned long)(avg_height.mean()*scale+0.5);
            // make sure the width and height never round to zero.
            if (detector_width == 0)
                detector_width = 1;
            if (detector_height == 0)
                detector_height = 1;


            overlaps_nms = find_tight_overlap_tester(temp);
            // Relax the non-max-suppression a little so that it doesn't accidentally make
            // it impossible for the detector to output boxes matching the training data.
            // This could be a problem with the tightest possible nms test since there is
            // some small variability in how boxes get positioned between the training data
            // and the coordinate system used by the detector when it runs.  So relaxing it
            // here takes care of that.
            double relax_amount = 0.10;
            auto iou_thresh = std::min(1.0, overlaps_nms.get_iou_thresh()+relax_amount);
            auto percent_covered_thresh = std::min(1.0, overlaps_nms.get_percent_covered_thresh()+relax_amount);
            overlaps_nms = test_box_overlap(iou_thresh, percent_covered_thresh);
        }
    };

    inline void serialize(const mmod_options& item, std::ostream& out)
    {
        int version = 1;

        serialize(version, out);
        serialize(item.detector_width, out);
        serialize(item.detector_height, out);
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
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::mmod_options");
        deserialize(item.detector_width, in);
        deserialize(item.detector_height, in);
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
            intermediate_detection() : detection_confidence(0), tensor_offset(0) {}

            intermediate_detection(
                rectangle rect_
            ) : rect(rect_), detection_confidence(0), tensor_offset(0) {}

            intermediate_detection(
                rectangle rect_,
                double detection_confidence_,
                size_t tensor_offset_
            ) : rect(rect_), detection_confidence(detection_confidence_), tensor_offset(tensor_offset_) {}

            rectangle rect;
            double detection_confidence;
            size_t tensor_offset;

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
            DLIB_CASSERT(output_tensor.k() == 1);
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

                    final_dets.push_back(mmod_rect(dets_accum[i].rect, dets_accum[i].detection_confidence));
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
            DLIB_CASSERT(output_tensor.k() == 1);



            // we will scale the loss so that it doesn't get really huge
            const double scale = 1.0/output_tensor.size();
            double loss = 0;

            float* g = grad.host_write_only();
            // zero initialize grad.
            for (auto&& x : grad)
                x = 0;

            const float* out_data = output_tensor.host();

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
                        point p = image_rect_to_feat_coord(input_tensor, x, sub);
                        loss -= out_data[p.y()*output_tensor.nc() + p.x()];
                        // compute gradient
                        g[p.y()*output_tensor.nc() + p.x()] = -scale;
                    }
                    else
                    {
                        // This box was ignored so shouldn't have been counted in the loss.
                        loss -= 1;
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

                    const std::pair<double,unsigned int> hittruth = find_best_match(*truth, dets[i].rect);

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

                    const std::pair<double,unsigned int> hittruth = find_best_match(*truth, dets[i].rect);

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
                g        += output_tensor.nr()*output_tensor.nc();
                out_data += output_tensor.nr()*output_tensor.nc();
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

        friend std::ostream& operator<<(std::ostream& out, const loss_mmod_& )
        {
            // TODO, add options fields
            out << "loss_mmod";
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
            DLIB_CASSERT(output_tensor.k() == 1);
            const float* out_data = output_tensor.host() + output_tensor.nr()*output_tensor.nc()*i;
            // scan the final layer and output the positive scoring locations
            dets_accum.clear();
            for (long r = 0; r < output_tensor.nr(); ++r)
            {
                for (long c = 0; c < output_tensor.nc(); ++c)
                {
                    double score = out_data[r*output_tensor.nc() + c];
                    if (score > adjust_threshold)
                    {
                        dpoint p = output_tensor_to_input_tensor(net, point(c,r));
                        drectangle rect = centered_drect(p, options.detector_width, options.detector_height);
                        rect = input_layer(net).tensor_space_to_image_space(input_tensor,rect);

                        dets_accum.push_back(intermediate_detection(rect, score, r*output_tensor.nc() + c));
                    }
                }
            }
            std::sort(dets_accum.rbegin(), dets_accum.rend());
        }


        template <typename net_type>
        point image_rect_to_feat_coord (
            const tensor& input_tensor,
            const rectangle& rect,
            const net_type& net
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

            // Compute the scale we need to be at to get from rect to our detection window.
            // Note that we compute the scale as the max of two numbers.  It doesn't
            // actually matter which one we pick, because if they are very different then
            // it means the box can't be matched by the sliding window.  But picking the
            // max causes the right error message to be selected in the logic below.
            const double scale = std::max(options.detector_width/(double)rect.width(), options.detector_height/(double)rect.height());
            const rectangle mapped_rect = input_layer(net).image_space_to_tensor_space(input_tensor, std::min(1.0,scale), rect);

            // compute the detection window that we would use at this position.
            point tensor_p = center(mapped_rect);
            rectangle det_window = centered_rect(tensor_p, options.detector_width,options.detector_height);
            det_window = input_layer(net).tensor_space_to_image_space(input_tensor, det_window);

            // make sure the rect can actually be represented by the image pyramid we are
            // using.
            if (box_intersection_over_union(rect, det_window) <= options.truth_match_iou_threshold)
            {
                std::ostringstream sout;
                sout << "Encountered a truth rectangle with a width and height of " << rect.width() << " and " << rect.height() << "." << endl;
                sout << "The image pyramid and sliding window can't output a rectangle of this shape. " << endl;
                const double detector_area = options.detector_width*options.detector_height;
                if (mapped_rect.area()/detector_area <= options.truth_match_iou_threshold)
                {
                    sout << "This is because the rectangle is smaller than the detection window which has a width" << endl;
                    sout << "and height of " << options.detector_width << " and " << options.detector_height << "." << endl;
                }
                else
                {
                    sout << "This is because the rectangle's aspect ratio is too different from the detection window," << endl;
                    sout << "which has a width and height of " << options.detector_width << " and " << options.detector_height << "." << endl;
                }
                throw impossible_labeling_error(sout.str());
            }

            // now map through the CNN to the output layer.
            tensor_p = input_tensor_to_output_tensor(net,tensor_p);

            const tensor& output_tensor = net.get_output();
            if (!get_rect(output_tensor).contains(tensor_p))
            {
                std::ostringstream sout;
                sout << "Encountered a truth rectangle located at " << rect << " that is too close to the edge" << endl;
                sout << "of the image to be captured by the CNN features." << endl;
                throw impossible_labeling_error(sout.str());
            }

            return tensor_p;
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
            const rectangle& rect
        ) const
        {
            double match = 0;
            unsigned int best_idx = 0;
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {
                if (boxes[i].ignore)
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

}

#endif // DLIB_DNn_LOSS_H_

