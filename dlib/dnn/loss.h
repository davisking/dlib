// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_LOSS_H_
#define DLIB_DNn_LOSS_H_

#include "loss_abstract.h"
#include "core.h"
#include "utilities.h"
#include "../matrix.h"
#include "../cuda/tensor_tools.h"
#include "../geometry.h"
#include "../image_processing/box_overlap_testing.h"
#include "../image_processing/full_object_detection.h"
#include "../svm/ranking_tools.h"
#include <sstream>
#include <map>
#include <unordered_map>

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
            out << "<loss_binary_hinge/>\n";
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
                DLIB_CASSERT(y != 0, "y: " << y);
                float temp;
                if (y > 0)
                {
                    temp = log1pexp(-out_data[i]);
                    loss += y*scale*temp;
                    g[i] = y*scale*(g[i]-1);
                }
                else
                {
                    temp = -(-out_data[i]-log1pexp(-out_data[i]));
                    loss += -y*scale*temp;
                    g[i] = -y*scale*g[i];
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
            out << "<loss_binary_log/>\n";
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
                        loss += scale*-safe_log(g[idx]);
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
            out << "<loss_multiclass_log/>\n";
        }

    };

    template <typename SUBNET>
    using loss_multiclass_log = add_loss_layer<loss_multiclass_log_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_weighted_
    {
    public:

        typedef dlib::weighted_label<unsigned long> weighted_label;
        typedef weighted_label training_label_type;
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
                const auto wl = *truth++;
                const long y = wl.label;
                const float weight = wl.weight;
                // The network must produce a number of outputs that is equal to the number
                // of labels when using this type of loss.
                DLIB_CASSERT(y < output_tensor.k(), "y: " << y << ", output_tensor.k(): " << output_tensor.k());
                for (long k = 0; k < output_tensor.k(); ++k)
                {
                    const unsigned long idx = i*output_tensor.k()+k;
                    if (k == y)
                    {
                        loss += weight*scale*-safe_log(g[idx]);
                        g[idx] =weight*scale*(g[idx]-1);
                    }
                    else
                    {
                        g[idx] = weight*scale*g[idx];
                    }
                }
            }
            return loss;
        }

        friend void serialize(const loss_multiclass_log_weighted_& , std::ostream& out)
        {
            serialize("loss_multiclass_log_weighted_", out);
        }

        friend void deserialize(loss_multiclass_log_weighted_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_multiclass_log_weighted_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_multiclass_log_weighted_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_multiclass_log_weighted_& )
        {
            out << "loss_multiclass_log_weighted";
            return out;
        }

        friend void to_xml(const loss_multiclass_log_weighted_& /*item*/, std::ostream& out)
        {
            out << "<loss_multiclass_log_weighted/>\n";
        }

    };

    template <typename SUBNET>
    using loss_multiclass_log_weighted = add_loss_layer<loss_multiclass_log_weighted_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multimulticlass_log_ 
    {

    public:

        loss_multimulticlass_log_ () = default;

        loss_multimulticlass_log_ (
            const std::map<std::string,std::vector<std::string>>& labels
        )
        {
            for (auto& l : labels)
            {
                possible_labels[l.first] = std::make_shared<decltype(l.second)>(l.second);
                DLIB_CASSERT(l.second.size() >= 2, "Each classifier must have at least two possible labels.");

                for (size_t i = 0; i < l.second.size(); ++i)
                {
                    label_idx_lookup[l.first][l.second[i]] = i;
                    ++total_num_labels;
                }
            }
        }

        unsigned long number_of_labels() const { return total_num_labels; }

        unsigned long number_of_classifiers() const { return possible_labels.size(); }

        std::map<std::string,std::vector<std::string>> get_labels ( 
        ) const 
        {
            std::map<std::string,std::vector<std::string>> info; 
            for (auto& i : possible_labels)
            {
                for (auto& label : *i.second)
                    info[i.first].emplace_back(label);
            }
            return info;
        }

        class classifier_output
        {

        public:
            classifier_output() = default;

            size_t num_classes() const { return class_probs.size(); }

            double probability_of_class (
                size_t i
            ) const 
            { 
                DLIB_CASSERT(i < num_classes());
                return class_probs(i); 
            }

            const std::string& label(
                size_t i
            ) const 
            { 
                DLIB_CASSERT(i < num_classes()); 
                return (*_labels)[i]; 
            }

            operator std::string(
            ) const
            {
                DLIB_CASSERT(num_classes() != 0); 
                return (*_labels)[index_of_max(class_probs)];
            }

            friend std::ostream& operator<< (std::ostream& out, const classifier_output& item)
            {
                DLIB_ASSERT(item.num_classes() != 0); 
                out << static_cast<std::string>(item);
                return out;
            }

        private:

            friend class loss_multimulticlass_log_;

            template <typename EXP>
            classifier_output(
                const matrix_exp<EXP>& class_probs,
                const std::shared_ptr<std::vector<std::string>>& _labels
            ) : 
                class_probs(class_probs), 
                _labels(_labels)
            {
            }

            matrix<float,1,0> class_probs;
            std::shared_ptr<std::vector<std::string>> _labels;
        };

        typedef std::map<std::string,std::string> training_label_type;
        typedef std::map<std::string,classifier_output> output_label_type;


        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter_begin
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 );
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            DLIB_CASSERT(number_of_labels() != 0, "You must give the loss_multimulticlass_log_'s constructor label data before you can use it!");
            DLIB_CASSERT(output_tensor.k() == (long)number_of_labels(), "The output tensor must have " << number_of_labels() << " channels.");


            long k_offset = 0;
            for (auto& l : possible_labels)
            {
                auto iter = iter_begin;
                const std::string& classifier_name = l.first;
                const auto& labels = (*l.second); 
                scratch.set_size(output_tensor.num_samples(), labels.size());
                tt::copy_tensor(false, scratch, 0, output_tensor, k_offset, labels.size());

                tt::softmax(scratch, scratch);

                for (long i = 0; i < scratch.num_samples(); ++i)
                    (*iter++)[classifier_name] = classifier_output(rowm(mat(scratch),i), l.second);

                k_offset += labels.size();
            }
        }


        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth_begin, 
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
            DLIB_CASSERT(number_of_labels() != 0, "You must give the loss_multimulticlass_log_'s constructor label data before you can use it!");
            DLIB_CASSERT(output_tensor.k() == (long)number_of_labels(), "The output tensor must have " << number_of_labels() << " channels.");

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            long k_offset = 0;
            for (auto& l : label_idx_lookup)
            {
                const std::string& classifier_name = l.first;
                const auto& int_labels = l.second; 
                scratch.set_size(output_tensor.num_samples(), int_labels.size());
                tt::copy_tensor(false, scratch, 0, output_tensor, k_offset, int_labels.size());

                tt::softmax(scratch, scratch);


                auto truth = truth_begin;
                float* g = scratch.host();
                for (long i = 0; i < scratch.num_samples(); ++i)
                {
                    const long y = int_labels.at(truth->at(classifier_name));
                    ++truth;

                    for (long k = 0; k < scratch.k(); ++k)
                    {
                        const unsigned long idx = i*scratch.k()+k;
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

                tt::copy_tensor(false, grad, k_offset, scratch, 0, int_labels.size());

                k_offset += int_labels.size();
            }
            return loss;
        }


        friend void serialize(const loss_multimulticlass_log_& item, std::ostream& out)
        {
            serialize("loss_multimulticlass_log_", out);
            serialize(item.get_labels(), out);
        }

        friend void deserialize(loss_multimulticlass_log_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_multimulticlass_log_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_multimulticlass_log_.");

            std::map<std::string,std::vector<std::string>> info; 
            deserialize(info, in);
            item = loss_multimulticlass_log_(info);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_multimulticlass_log_& item)
        {
            out << "loss_multimulticlass_log, labels={";
            for (auto i = item.possible_labels.begin(); i != item.possible_labels.end(); )
            {
                auto& category = i->first;
                auto& labels = *(i->second);
                out << category << ":(";
                for (size_t j = 0; j < labels.size(); ++j)
                {
                    out << labels[j];
                    if (j+1 < labels.size())
                        out << ",";
                }

                out << ")";
                if (++i != item.possible_labels.end())
                    out << ", ";
            }
            out << "}";
            return out;
        }

        friend void to_xml(const loss_multimulticlass_log_& item, std::ostream& out)
        {
            out << "<loss_multimulticlass_log>\n";
            out << item;
            out << "\n</loss_multimulticlass_log>\n";
        }

    private:

        std::map<std::string,std::shared_ptr<std::vector<std::string>>> possible_labels;
        unsigned long total_num_labels = 0;

        // We make it true that: possible_labels[classifier][label_idx_lookup[classifier][label]] == label
        std::map<std::string, std::map<std::string, size_t>> label_idx_lookup;


        // Scratch doesn't logically contribute to the state of this object.  It's just
        // temporary scratch space used by this class.  
        mutable resizable_tensor scratch;


    };

    template <typename SUBNET>
    using loss_multimulticlass_log = add_loss_layer<loss_multimulticlass_log_, SUBNET>;

    inline bool operator== (const std::string& lhs, const loss_multimulticlass_log_::classifier_output& rhs)
    { return lhs == static_cast<const std::string&>(rhs); }
    inline bool operator== (const loss_multimulticlass_log_::classifier_output& lhs, const std::string& rhs)
    { return rhs == static_cast<const std::string&>(lhs); }

// ----------------------------------------------------------------------------------------

    class loss_multibinary_log_
    {
    public:
        typedef std::vector<float> training_label_type;
        typedef std::vector<float> output_label_type;

        loss_multibinary_log_() = default;

        loss_multibinary_log_(double gamma) : gamma(gamma)
        {
            DLIB_CASSERT(gamma >= 0);
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
            DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            // Note that output_tensor.k() should match the number of labels.

            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                output_label_type predictions(output_tensor.k(), 0);
                for (long k = 0; k < output_tensor.k(); ++k)
                {
                    predictions[k] = out_data[i * output_tensor.k() + k];
                }
                *iter++ = std::move(predictions);
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
            DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);

            tt::sigmoid(grad, output_tensor);

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0 / output_tensor.num_samples();
            double loss = 0;
            float* g = grad.host();
            const float* out_data  = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i, ++truth)
            {
                const long long num_label_categories = truth->size();
                DLIB_CASSERT(output_tensor.k() == num_label_categories,
                    "Number of label types should match the number of output channels. "
                    "output_tensor.k(): " << output_tensor.k() 
                    << ", num_label_categories: "<< num_label_categories);
                for (long k = 0; k < output_tensor.k(); ++k)
                {
                    const float y = (*truth)[k];
                    DLIB_CASSERT(y != 0, "y: " << y);
                    const size_t idx = i * output_tensor.k() + k;
                    if (y > 0)
                    {
                        const float temp = log1pexp(-out_data[idx]);
                        const float focus = std::pow(1 - g[idx], gamma);
                        loss += y * scale * temp * focus;
                        g[idx] = y * scale * focus * (g[idx] * (gamma * temp + 1) - 1);
                    }
                    else
                    {
                        const float temp = -(-out_data[idx] - log1pexp(-out_data[idx]));
                        const float focus = std::pow(g[idx], gamma);
                        loss += -y * scale * temp * focus;
                        g[idx] = -y * scale * focus * g[idx] * (gamma * temp + 1);
                    }
                }
            }
            return loss;
        }

        double get_gamma () const { return gamma; }

        friend void serialize(const loss_multibinary_log_& item, std::ostream& out)
        {
            serialize("loss_multibinary_log_2", out);
            serialize(item.gamma, out);
        }

        friend void deserialize(loss_multibinary_log_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "loss_multibinary_log_")
            {
                item.gamma = 0;
                return;
            }
            else if (version == "loss_multibinary_log_2")
            {
                deserialize(item.gamma, in);
            }
            else
            {
                throw serialization_error("Unexpected version found while deserializing dlib::loss_multibinary_log_.");
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_multibinary_log_& item)
        {
            out << "loss_multibinary_log (gamma=" << item.gamma << ")";
            return out;
        }

        friend void to_xml(const loss_multibinary_log_& item, std::ostream& out)
        {
            out << "<loss_multibinary_log gamma='" << item.gamma << "'/>\n";
        }

    private:
        double gamma = 0;
    };

    template <typename SUBNET>
    using loss_multibinary_log = add_loss_layer<loss_multibinary_log_, SUBNET>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    enum class use_image_pyramid : uint8_t
    {
        no,
        yes
    };

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
        bool use_bounding_box_regression = false; 
        double bbr_lambda = 100; 
        // This field is intentionally not serialized because I want people to really think hard
        // about ignoring the warnings that this suppresses.
        bool be_quiet = false;

        use_image_pyramid assume_image_pyramid = use_image_pyramid::yes;

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
                            detector_height = min_target_size/ratio;
                            detector_width = min_target_size;
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

            set_overlap_nms(boxes);
        }

        mmod_options(
            use_image_pyramid assume_image_pyramid,
            const std::vector<std::vector<mmod_rect>>& boxes,
            const double min_detector_window_overlap_iou = 0.75
        )
            : assume_image_pyramid(assume_image_pyramid)
        {
            DLIB_CASSERT(assume_image_pyramid == use_image_pyramid::no);
            DLIB_CASSERT(0.5 < min_detector_window_overlap_iou && min_detector_window_overlap_iou < 1);

            // Figure out what detector windows we will need.
            for (auto& label : get_labels(boxes))
            {
                for (auto rectangle : find_covering_rectangles(boxes, test_box_overlap(min_detector_window_overlap_iou), label))
                {
                    detector_windows.push_back(detector_window_details(rectangle.width(), rectangle.height(), label));
                }
            }

            DLIB_CASSERT(detector_windows.size() != 0, "You can't call mmod_options's constructor with a set of boxes that is empty (or only contains ignored boxes).");

            set_overlap_nms(boxes);
        }

    private:

        void set_overlap_nms(const std::vector<std::vector<mmod_rect>>& boxes)
        {
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

        static std::vector<dlib::rectangle> find_covering_rectangles (
            const std::vector<std::vector<mmod_rect>>& rects,
            const test_box_overlap& overlaps,
            const std::string& label
        )
        {
            std::vector<rectangle> boxes;
            // Make sure all the boxes have the same position, so that the we only check for
            // width and height.
            for (auto& bb : rects)
            {
                for (auto&& b : bb)
                {
                    if (!b.ignore && b.label == label)
                        boxes.push_back(rectangle(b.rect.width(), b.rect.height()));
                }
            }

            return find_rectangles_overlapping_all_others(boxes, overlaps);
        }
    };

    inline void serialize(const mmod_options& item, std::ostream& out)
    {
        int version = 4;

        serialize(version, out);
        serialize(item.detector_windows, out);
        serialize(item.loss_per_false_alarm, out);
        serialize(item.loss_per_missed_target, out);
        serialize(item.truth_match_iou_threshold, out);
        serialize(item.overlaps_nms, out);
        serialize(item.overlaps_ignore, out);
        serialize(static_cast<uint8_t>(item.assume_image_pyramid), out);
        serialize(item.use_bounding_box_regression, out);
        serialize(item.bbr_lambda, out);
    }

    inline void deserialize(mmod_options& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (!(1 <= version && version <= 4))
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
        item.assume_image_pyramid = use_image_pyramid::yes;
        if (version >= 3)
        {
            uint8_t assume_image_pyramid = 0;
            deserialize(assume_image_pyramid, in);
            item.assume_image_pyramid = static_cast<use_image_pyramid>(assume_image_pyramid);
        }
        item.use_bounding_box_regression = mmod_options().use_bounding_box_regression; // use default value since this wasn't provided
        item.bbr_lambda = mmod_options().bbr_lambda; // use default value since this wasn't provided
        if (version >= 4)
        {
            deserialize(item.use_bounding_box_regression, in);
            deserialize(item.bbr_lambda, in);
        }
    }

    inline std::ostream& operator<<(std::ostream& out, const std::vector<mmod_options::detector_window_details>& detector_windows)
    {
        // write detector windows grouped by label
        // example output: aeroplane:74x30,131x30,70x45,54x70,198x30;bicycle:70x57,32x70,70x32,51x70,128x30,30x121;car:70x36,70x60,99x30,52x70,30x83,30x114,30x200

        std::map<std::string, std::deque<mmod_options::detector_window_details>> detector_windows_by_label;
        for (const auto& detector_window : detector_windows)
            detector_windows_by_label[detector_window.label].push_back(detector_window);

        size_t label_count = 0;
        for (const auto& i : detector_windows_by_label)
        {
            const auto& label = i.first;
            const auto& detector_windows = i.second;

            if (label_count++ > 0)
                out << ";";
            out << label << ":";

            for (size_t j = 0; j < detector_windows.size(); ++j)
            {
                out << detector_windows[j].width << "x" << detector_windows[j].height;
                if (j + 1 < detector_windows.size())
                    out << ",";
            }
        }

        return out;
    }

// ----------------------------------------------------------------------------------------

    class loss_mmod_ 
    {
        struct intermediate_detection
        {
            intermediate_detection() = default; 

            intermediate_detection(
                rectangle rect_
            ) : rect(rect_), rect_bbr(rect_) {}

            intermediate_detection(
                rectangle rect_,
                double detection_confidence_,
                size_t tensor_offset_,
                long channel
            ) : rect(rect_), detection_confidence(detection_confidence_), tensor_offset(tensor_offset_), tensor_channel(channel), rect_bbr(rect_) {}

            // rect is the rectangle you get without any bounding box regression.  So it's
            // the basic sliding window box (aka, the "anchor box").
            rectangle rect;
            double detection_confidence = 0;
            size_t tensor_offset = 0;
            long tensor_channel = 0;

            // rect_bbr = rect + bounding box regression.  So more accurate.  Or if bbr is off then
            // this is just rect.  The important thing about rect_bbr is that its the
            // rectangle we use for doing NMS.
            drectangle rect_bbr; 
            size_t tensor_offset_dx = 0;
            size_t tensor_offset_dy = 0;
            size_t tensor_offset_dw = 0;
            size_t tensor_offset_dh = 0;

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
            if (options.use_bounding_box_regression)
            {
                DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size()*5);
            }
            else
            {
                DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size());
            }
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
                    if (overlaps_any_box_nms(final_dets, dets_accum[i].rect_bbr))
                        continue;

                    final_dets.push_back(mmod_rect(dets_accum[i].rect_bbr,
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
            if (options.use_bounding_box_regression)
            {
                DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size()*5);
            }
            else
            {
                DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size());
            }

            double det_thresh_speed_adjust = 0;

            // we will scale the loss so that it doesn't get really huge
            const double scale = 1.0/(output_tensor.nr()*output_tensor.nc()*output_tensor.num_samples()*options.detector_windows.size());
            double loss = 0;

            float* g = grad.host_write_only();
            for (size_t i = 0; i < grad.size(); ++i)
                g[i] = 0;

            const float* out_data = output_tensor.host();

            std::vector<intermediate_detection> dets;
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                tensor_to_dets(input_tensor, output_tensor, i, dets, -options.loss_per_false_alarm + det_thresh_speed_adjust, sub);

                const unsigned long max_num_dets = 50 + truth->size()*5;
                // Prevent calls to tensor_to_dets() from running for a really long time
                // due to the production of an obscene number of detections.
                const unsigned long max_num_initial_dets = max_num_dets*100;
                if (dets.size() > max_num_initial_dets)
                {
                    det_thresh_speed_adjust = std::max(det_thresh_speed_adjust,dets[max_num_initial_dets].detection_confidence + options.loss_per_false_alarm);
                }

                std::vector<int> truth_idxs;
                truth_idxs.reserve(truth->size());

                std::unordered_map<size_t, rectangle> idx_to_truth_rect;

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
                        if(image_rect_to_feat_coord(p, input_tensor, x, x.label, sub, k, options.assume_image_pyramid))
                        {
                            // Ignore boxes that can't be detected by the CNN.
                            loss -= options.loss_per_missed_target;
                            truth_idxs.push_back(-1);
                            continue;
                        }
                        const size_t idx = (k*output_tensor.nr() + p.y())*output_tensor.nc() + p.x();
                        const auto i = idx_to_truth_rect.find(idx);
                        if (i != idx_to_truth_rect.end())
                        {
                            if (!options.be_quiet) 
                            {
                                // Ignore duplicate truth box in feature coordinates.
                                std::cout << "Warning, ignoring object.  We encountered a truth rectangle located at " << x.rect;
                                std::cout << ", and we are ignoring it because it maps to the exact same feature coordinates ";
                                std::cout << "as another truth rectangle located at " << i->second << "." << std::endl;
                            }

                            loss -= options.loss_per_missed_target;
                            truth_idxs.push_back(-1);
                            continue;
                        }
                        loss -= out_data[idx];
                        // compute gradient
                        g[idx] = -scale;
                        truth_idxs.push_back(idx);
                        idx_to_truth_rect[idx] = x.rect;
                    }
                    else
                    {
                        // This box was ignored so shouldn't have been counted in the loss.
                        loss -= options.loss_per_missed_target;
                        truth_idxs.push_back(-1);
                    }
                }

                // Measure the loss augmented score for the detections which hit a truth rect.
                std::vector<double> truth_score_hits(truth->size(), 0);

                // keep track of which truth boxes we have hit so far.
                std::vector<bool> hit_truth_table(truth->size(), false);

                std::vector<intermediate_detection> final_dets;
                // The point of this loop is to fill out the truth_score_hits array. 
                for (size_t i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
                {
                    if (overlaps_any_box_nms(final_dets, dets[i].rect_bbr))
                        continue;

                    const auto& det_label = options.detector_windows[dets[i].tensor_channel].label;

                    const std::pair<double,unsigned int> hittruth = find_best_match(*truth, hit_truth_table, dets[i].rect, det_label);

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
                            const int idx = truth_idxs[i];
                            if (idx != -1)
                            {
                                // We are ignoring this box so we shouldn't have counted it in the
                                // loss in the first place.  So we subtract out the loss values we
                                // added for it in the code above.
                                loss -= options.loss_per_missed_target-out_data[idx];
                                g[idx] = 0;
                                if (!options.be_quiet) 
                                {
                                    std::cout << "Warning, ignoring object.  We encountered a truth rectangle located at " << (*truth)[i].rect;
                                    std::cout << " that is suppressed by non-max-suppression ";
                                    std::cout << "because it is overlapped by another truth rectangle located at " << best_matching_truth_box 
                                        << " (IoU:"<< box_intersection_over_union(best_matching_truth_box,(*truth)[i]) <<", Percent covered:" 
                                        << box_percent_covered(best_matching_truth_box,(*truth)[i]) << ")." << std::endl;
                                }
                            }
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
                    if (overlaps_any_box_nms(final_dets, dets[i].rect_bbr))
                        continue;

                    const auto& det_label = options.detector_windows[dets[i].tensor_channel].label;

                    const std::pair<double,unsigned int> hittruth = find_best_match(*truth, hit_truth_table, dets[i].rect, det_label);

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

                                // Now account for BBR loss and gradient if appropriate.
                                if (options.use_bounding_box_regression)
                                {
                                    double dx = out_data[dets[i].tensor_offset_dx];
                                    double dy = out_data[dets[i].tensor_offset_dy];
                                    double dw = out_data[dets[i].tensor_offset_dw];
                                    double dh = out_data[dets[i].tensor_offset_dh];

                                    dpoint p = dcenter(dets[i].rect);
                                    double w = dets[i].rect.width()-1;
                                    double h = dets[i].rect.height()-1;
                                    drectangle truth_box = (*truth)[hittruth.second].rect;
                                    dpoint p_truth = dcenter(truth_box); 

                                    DLIB_CASSERT(w > 0);
                                    DLIB_CASSERT(h > 0);

                                    double target_dx = (p_truth.x() - p.x())/w;
                                    double target_dy = (p_truth.y() - p.y())/h;
                                    double target_dw = std::log((truth_box.width()-1)/w);
                                    double target_dh = std::log((truth_box.height()-1)/h);


                                    // compute smoothed L1 loss on BBR outputs.  This loss
                                    // is just the MSE loss when the loss is small and L1
                                    // when large.
                                    dx = dx-target_dx;
                                    dy = dy-target_dy;
                                    dw = dw-target_dw;
                                    dh = dh-target_dh;

                                    // use smoothed L1 
                                    double ldx = std::abs(dx)<1 ? 0.5*dx*dx : std::abs(dx)-0.5;
                                    double ldy = std::abs(dy)<1 ? 0.5*dy*dy : std::abs(dy)-0.5;
                                    double ldw = std::abs(dw)<1 ? 0.5*dw*dw : std::abs(dw)-0.5;
                                    double ldh = std::abs(dh)<1 ? 0.5*dh*dh : std::abs(dh)-0.5;

                                    loss += options.bbr_lambda*(ldx + ldy + ldw + ldh);
      
                                    // now compute the derivatives of the smoothed L1 loss
                                    ldx = put_in_range(-1,1, dx);
                                    ldy = put_in_range(-1,1, dy);
                                    ldw = put_in_range(-1,1, dw);
                                    ldh = put_in_range(-1,1, dh);


                                    // also smoothed L1 gradient goes to gradient output
                                    g[dets[i].tensor_offset_dx] += scale*options.bbr_lambda*ldx;
                                    g[dets[i].tensor_offset_dy] += scale*options.bbr_lambda*ldy;
                                    g[dets[i].tensor_offset_dw] += scale*options.bbr_lambda*ldw;
                                    g[dets[i].tensor_offset_dh] += scale*options.bbr_lambda*ldh;
                                }
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

            auto& opts = item.options;

            out << "detector_windows:(" << opts.detector_windows << ")";

            out << ", loss per FA:" << opts.loss_per_false_alarm;
            out << ", loss per miss:" << opts.loss_per_missed_target;
            out << ", truth match IOU thresh:" << opts.truth_match_iou_threshold;
            out << ", use_bounding_box_regression:" << opts.use_bounding_box_regression;
            if (opts.use_bounding_box_regression)
                out << ", bbr_lambda:" << opts.bbr_lambda;
            out << ", overlaps_nms:("<<opts.overlaps_nms.get_iou_thresh()<<","<<opts.overlaps_nms.get_percent_covered_thresh()<<")";
            out << ", overlaps_ignore:("<<opts.overlaps_ignore.get_iou_thresh()<<","<<opts.overlaps_ignore.get_percent_covered_thresh()<<")";

            out << ")";
            return out;
        }

        friend void to_xml(const loss_mmod_& /*item*/, std::ostream& out)
        {
            // TODO, add options fields
            out << "<loss_mmod/>\n";
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
            if (options.use_bounding_box_regression)
            {
                DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size()*5);
            }
            else
            {
                DLIB_CASSERT(output_tensor.k() == (long)options.detector_windows.size());
            }

            const float* out_data = output_tensor.host() + output_tensor.k()*output_tensor.nr()*output_tensor.nc()*i;
            // scan the final layer and output the positive scoring locations
            dets_accum.clear();
            for (long k = 0; k < (long)options.detector_windows.size(); ++k)
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

                            if (options.use_bounding_box_regression)
                            {
                                const auto offset = options.detector_windows.size() + k*4;
                                dets_accum.back().tensor_offset_dx = ((offset+0)*output_tensor.nr() + r)*output_tensor.nc() + c;
                                dets_accum.back().tensor_offset_dy = ((offset+1)*output_tensor.nr() + r)*output_tensor.nc() + c;
                                dets_accum.back().tensor_offset_dw = ((offset+2)*output_tensor.nr() + r)*output_tensor.nc() + c;
                                dets_accum.back().tensor_offset_dh = ((offset+3)*output_tensor.nr() + r)*output_tensor.nc() + c;

                                // apply BBR to dets_accum.back()
                                double dx = out_data[dets_accum.back().tensor_offset_dx];
                                double dy = out_data[dets_accum.back().tensor_offset_dy];
                                double dw = out_data[dets_accum.back().tensor_offset_dw];
                                double dh = out_data[dets_accum.back().tensor_offset_dh];
                                dw = std::exp(dw);
                                dh = std::exp(dh);
                                double w = rect.width()-1;
                                double h = rect.height()-1;
                                rect = translate_rect(rect, dpoint(dx*w,dy*h));
                                rect = centered_drect(rect, w*dw+1, h*dh+1);
                                dets_accum.back().rect_bbr = rect;
                            }
                        }
                    }
                }
            }
            std::sort(dets_accum.rbegin(), dets_accum.rend());
        }

        size_t find_best_detection_window (
            rectangle rect,
            const std::string& label,
            use_image_pyramid assume_image_pyramid
        ) const
        {
            if (assume_image_pyramid == use_image_pyramid::yes)
            {
                rect = move_rect(set_rect_area(rect, 400*400), point(0,0));
            }
            else
            {
                rect = rectangle(rect.width(), rect.height());
            }

            // Figure out which detection window in options.detector_windows is most similar to rect
            // (in terms of aspect ratio, if assume_image_pyramid == use_image_pyramid::yes).
            size_t best_i = 0;
            double best_ratio_diff = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < options.detector_windows.size(); ++i)
            {
                if (options.detector_windows[i].label != label)
                    continue;

                rectangle det_window;
                
                if (options.assume_image_pyramid == use_image_pyramid::yes)
                {
                    det_window = centered_rect(point(0,0), options.detector_windows[i].width, options.detector_windows[i].height);
                    det_window = move_rect(set_rect_area(det_window, 400*400), point(0,0));
                }
                else
                {
                    det_window = rectangle(options.detector_windows[i].width, options.detector_windows[i].height);
                }

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
            size_t& det_idx,
            use_image_pyramid assume_image_pyramid
        ) const 
        {
            if (!input_layer(net).image_contained_point(input_tensor,center(rect)))
            {
                std::ostringstream sout;
                sout << "Encountered a truth rectangle located at " << rect << " that is outside the image." << std::endl;
                sout << "The center of each truth rectangle must be within the image." << std::endl;
                throw impossible_labeling_error(sout.str());
            }

            det_idx = find_best_detection_window(rect,label,assume_image_pyramid);

            double scale = 1.0;
            if (options.assume_image_pyramid == use_image_pyramid::yes)
            {
                // Compute the scale we need to be at to get from rect to our detection window.
                // Note that we compute the scale as the max of two numbers.  It doesn't
                // actually matter which one we pick, because if they are very different then
                // it means the box can't be matched by the sliding window.  But picking the
                // max causes the right error message to be selected in the logic below.
                scale = std::max(options.detector_windows[det_idx].width/(double)rect.width(), options.detector_windows[det_idx].height/(double)rect.height());
            }
            else
            {
                // We don't want invariance to scale.
                scale = 1.0;
            }

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
                    std::cout << "This is either because (1) the final layer's features have too large of a stride across the image, limiting the possible locations the sliding window can search ";
                    std::cout << "or (2) because the rectangle's aspect ratio is too different from the best matching detection window, ";
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
            const std::vector<bool>& hit_truth_table,
            const rectangle& rect,
            const std::string& label
        ) const
        {
            double match = 0;
            unsigned int best_idx = 0;

            for (int allow_duplicate_hit = 0; allow_duplicate_hit <= 1 && match == 0; ++allow_duplicate_hit)
            {
                for (unsigned long i = 0; i < boxes.size(); ++i)
                {
                    if (boxes[i].ignore || boxes[i].label != label)
                        continue;
                    if (!allow_duplicate_hit && hit_truth_table[i])
                        continue;

                    const double new_match = box_intersection_over_union(rect, boxes[i]);
                    if (new_match > match)
                    {
                        match = new_match;
                        best_idx = i;
                    }
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
                item.margin = 0.1f;
                item.dist_thresh = 0.75f;
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
            out << "<loss_metric margin='"<<item.margin<<"' distance_threshold='"<<item.dist_thresh<<"'/>\n";
        }

    private:
        float margin = 0.04f;
        float dist_thresh = 0.6f;


        // These variables are only here to avoid being reallocated over and over in
        // compute_loss_value_and_gradient()
        mutable resizable_tensor temp, grad_mul;

    };

    template <typename SUBNET>
    using loss_metric = add_loss_layer<loss_metric_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_ranking_
    {
    public:

        typedef float training_label_type; // nominally +1/-1
        typedef float output_label_type; // ranking score

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


            std::vector<double> rel_scores;
            std::vector<double> nonrel_scores;
            std::vector<long> rel_idx, nonrel_idx;

            const float* out_data = output_tensor.host();
            float* g = grad.host_write_only();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const float y = *truth++;
                if (y > 0)
                {
                    rel_scores.push_back(out_data[i]-y);
                    rel_idx.push_back(i);
                }
                else if (y < 0)
                {
                    nonrel_scores.push_back(out_data[i]-y);
                    nonrel_idx.push_back(i);
                }
                else
                {
                    g[i] = 0;
                }
            }


            std::vector<unsigned long> rel_counts;
            std::vector<unsigned long> nonrel_counts;
            count_ranking_inversions(rel_scores, nonrel_scores, rel_counts, nonrel_counts);
            const unsigned long total_pairs = rel_scores.size()*nonrel_scores.size();
            DLIB_CASSERT(total_pairs > 0, "You can't give a ranking mini-batch that contains only one class.  Both classes must be represented.");
            const double scale = 1.0/total_pairs;


            double loss = 0;
            for (unsigned long k = 0; k < rel_counts.size(); ++k)
            {
                loss -= rel_counts[k]*rel_scores[k];
                g[rel_idx[k]] = -1.0*rel_counts[k]*scale;
            }

            for (unsigned long k = 0; k < nonrel_counts.size(); ++k)
            {
                loss += nonrel_counts[k]*nonrel_scores[k];
                g[nonrel_idx[k]] = nonrel_counts[k]*scale;
            }

            return loss*scale;
        }

        friend void serialize(const loss_ranking_& , std::ostream& out)
        {
            serialize("loss_ranking_", out);
        }

        friend void deserialize(loss_ranking_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_ranking_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_ranking_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_ranking_& )
        {
            out << "loss_ranking";
            return out;
        }

        friend void to_xml(const loss_ranking_& /*item*/, std::ostream& out)
        {
            out << "<loss_ranking/>\n";
        }

    };

    template <typename SUBNET>
    using loss_ranking = add_loss_layer<loss_ranking_, SUBNET>;

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
                loss += temp2*temp1;
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
            out << "<loss_mean_squared/>\n";
        }

    };

    template <typename SUBNET>
    using loss_mean_squared = add_loss_layer<loss_mean_squared_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_epsilon_insensitive_
    {
    public:

        typedef float training_label_type;
        typedef float output_label_type;

        loss_epsilon_insensitive_() = default;
        loss_epsilon_insensitive_(double eps) : eps(eps) 
        {
            DLIB_CASSERT(eps >= 0, "You can't set a negative error epsilon.");
        }

        double get_epsilon () const { return eps; }
        void set_epsilon(double e)
        {
            DLIB_CASSERT(e >= 0, "You can't set a negative error epsilon.");
            eps = e;
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
                const float err = out_data[i]-y;
                if (err > eps)
                {
                    loss += scale*(err-eps);
                    g[i] = scale;
                }
                else if (err < -eps)
                {
                    loss += scale*(eps-err);
                    g[i] = -scale;
                }
            }
            return loss;
        }

        friend void serialize(const loss_epsilon_insensitive_& item, std::ostream& out)
        {
            serialize("loss_epsilon_insensitive_", out);
            serialize(item.eps, out);
        }

        friend void deserialize(loss_epsilon_insensitive_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_epsilon_insensitive_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_epsilon_insensitive_.");
            deserialize(item.eps, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_epsilon_insensitive_& item)
        {
            out << "loss_epsilon_insensitive epsilon: " << item.eps;
            return out;
        }

        friend void to_xml(const loss_epsilon_insensitive_& item, std::ostream& out)
        {
            out << "<loss_epsilon_insensitive_ epsilon='" << item.eps << "'/>\n";
        }

    private:
        double eps = 1;

    };

    template <typename SUBNET>
    using loss_epsilon_insensitive = add_loss_layer<loss_epsilon_insensitive_, SUBNET>;

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
                    loss += temp2*temp1;
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
            out << "<loss_mean_squared_multioutput/>\n";
        }

    };

    template <typename SUBNET>
    using loss_mean_squared_multioutput = add_loss_layer<loss_mean_squared_multioutput_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_binary_log_per_pixel_
    {
    public:

        typedef matrix<float> training_label_type;
        typedef matrix<float> output_label_type;

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

            DLIB_CASSERT(output_tensor.k() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* const out_data = output_tensor.host();

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
            DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.k() == 1);
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

            double loss;
#ifdef DLIB_USE_CUDA
            cuda_compute(truth, output_tensor, grad, loss);
#else
            cpu_compute(truth, output_tensor, grad, loss);
#endif
            return loss;
        }

        friend void serialize(const loss_binary_log_per_pixel_& , std::ostream& out)
        {
            serialize("loss_binary_log_per_pixel_", out);
        }

        friend void deserialize(loss_binary_log_per_pixel_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_binary_log_per_pixel_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_binary_log_per_pixel_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_binary_log_per_pixel_& )
        {
            out << "loss_binary_log_per_pixel";
            return out;
        }

        friend void to_xml(const loss_binary_log_per_pixel_& /*item*/, std::ostream& out)
        {
            out << "<loss_binary_log_per_pixel/>\n";
        }

    private:

#ifdef DLIB_USE_CUDA
        cuda::compute_loss_binary_log_per_pixel cuda_compute;
#else
        cpu::compute_loss_binary_log_per_pixel cpu_compute;
#endif
    };

    template <typename SUBNET>
    using loss_binary_log_per_pixel = add_loss_layer<loss_binary_log_per_pixel_, SUBNET>;

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


            double loss;
#ifdef DLIB_USE_CUDA
            cuda_compute(truth, output_tensor, grad, loss);
#else
            cpu_compute(truth, output_tensor, grad, loss);
#endif
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
            out << "<loss_multiclass_log_per_pixel/>\n";
        }

    private:

#ifdef DLIB_USE_CUDA
        cuda::compute_loss_multiclass_log_per_pixel cuda_compute;
#else
        cpu::compute_loss_multiclass_log_per_pixel cpu_compute;
#endif
    };

    template <typename SUBNET>
    using loss_multiclass_log_per_pixel = add_loss_layer<loss_multiclass_log_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_multiclass_log_per_pixel_weighted_
    {
    public:

        typedef dlib::weighted_label<uint16_t> weighted_label;
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

            double loss;
#ifdef DLIB_USE_CUDA
            cuda_compute(truth, output_tensor, grad, loss);
#else
            cpu_compute(truth, output_tensor, grad, loss);
#endif
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
            out << "<loss_multiclass_log_per_pixel_weighted/>\n";
        }

    private:

#ifdef DLIB_USE_CUDA
        cuda::compute_loss_multiclass_log_per_pixel_weighted cuda_compute;
#else
        cpu::compute_loss_multiclass_log_per_pixel_weighted cpu_compute;
#endif

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
                        loss += temp2*temp1;
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
            out << "<loss_mean_squared_per_pixel/>\n";
        }

    };

    template <typename SUBNET>
    using loss_mean_squared_per_pixel = add_loss_layer<loss_mean_squared_per_pixel_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template<long _num_channels>
    class loss_mean_squared_per_channel_and_pixel_
    {
    public:

        typedef std::array<matrix<float>, _num_channels> training_label_type;
        typedef std::array<matrix<float>, _num_channels> output_label_type;

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

            DLIB_CASSERT(output_tensor.k() == _num_channels, "output k = " << output_tensor.k());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const float* out_data = output_tensor.host();

            for (long i = 0; i < output_tensor.num_samples(); ++i, ++iter)
            {
                for (long k = 0; k < output_tensor.k(); ++k)
                {
                    (*iter)[k].set_size(output_tensor.nr(), output_tensor.nc());
                    for (long r = 0; r < output_tensor.nr(); ++r)
                    {
                        for (long c = 0; c < output_tensor.nc(); ++c)
                        {
                            (*iter)[k].operator()(r, c) = out_data[tensor_index(output_tensor, i, k, r, c)];
                        }
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
            DLIB_CASSERT(output_tensor.k() == _num_channels);
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                output_tensor.nc() == grad.nc() &&
                output_tensor.k() == grad.k());
            for (long idx = 0; idx < output_tensor.num_samples(); ++idx)
            {
                const_label_iterator truth_matrix_ptr = (truth + idx);
                DLIB_CASSERT((*truth_matrix_ptr).size() == _num_channels);
                for (long k = 0; k < output_tensor.k(); ++k)
                {
                    DLIB_CASSERT((*truth_matrix_ptr)[k].nr() == output_tensor.nr() &&
                        (*truth_matrix_ptr)[k].nc() == output_tensor.nc(),
                        "truth size = " << (*truth_matrix_ptr)[k].nr() << " x " << (*truth_matrix_ptr)[k].nc() << ", "
                        "output size = " << output_tensor.nr() << " x " << output_tensor.nc());
                }
            }
            double loss;
#ifdef DLIB_USE_CUDA
            cuda_compute(truth, output_tensor, grad, loss);
#else
            cpu_compute(truth, output_tensor, grad, loss);
#endif
            return loss;
        }

        friend void serialize(const loss_mean_squared_per_channel_and_pixel_& , std::ostream& out)
        {
            serialize("loss_mean_squared_per_channel_and_pixel_", out);
        }

        friend void deserialize(loss_mean_squared_per_channel_and_pixel_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_mean_squared_per_channel_and_pixel_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_mean_squared_per_channel_and_pixel_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_mean_squared_per_channel_and_pixel_& )
        {
            out << "loss_mean_squared_per_channel_and_pixel";
            return out;
        }

        friend void to_xml(const loss_mean_squared_per_channel_and_pixel_& /*item*/, std::ostream& out)
        {
            out << "<loss_mean_squared_per_channel_and_pixel/>\n";
        }

    private:

#ifdef DLIB_USE_CUDA
        cuda::compute_loss_mean_squared_per_channel_and_pixel cuda_compute;
#else
        cpu::compute_loss_mean_squared_per_channel_and_pixel cpu_compute;
#endif
    };

    template <long num_channels, typename SUBNET>
    using loss_mean_squared_per_channel_and_pixel = add_loss_layer<loss_mean_squared_per_channel_and_pixel_<num_channels>, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_dot_
    {
    public:

        typedef matrix<float,0,1> training_label_type;
        typedef matrix<float,0,1> output_label_type;

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

            for (long i = 0; i < output_tensor.num_samples(); ++i)
                *iter++ = trans(rowm(mat(output_tensor),i));
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

            const long network_output_dims = output_tensor.size()/output_tensor.num_samples();


            // The loss we output is the average loss over the mini-batch. 
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            float* g = grad.host();
            const float* out_data = output_tensor.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                DLIB_CASSERT(truth->size() == network_output_dims, "The network must output a vector with the same dimensionality as the training labels. "
                    << "\ntruth->size():       " << truth->size()
                    << "\nnetwork_output_dims: " << network_output_dims); 

                const float* t = &(*truth++)(0);

                for (long j = 0; j < network_output_dims; ++j)
                {
                    g[j] = -t[j]*scale;
                    loss -= out_data[j]*t[j];
                }

                g += network_output_dims;
                out_data += network_output_dims;
            }
            return loss*scale;
        }

        friend void serialize(const loss_dot_& , std::ostream& out)
        {
            serialize("loss_dot_", out);
        }

        friend void deserialize(loss_dot_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_dot_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_dot_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_dot_& )
        {
            out << "loss_dot";
            return out;
        }

        friend void to_xml(const loss_dot_& /*item*/, std::ostream& out)
        {
            out << "<loss_dot/>\n";
        }

    };

    template <typename SUBNET>
    using loss_dot = add_loss_layer<loss_dot_, SUBNET>;

// ----------------------------------------------------------------------------------------

    struct yolo_options
    {
    public:
        struct anchor_box_details
        {
            anchor_box_details() = default;
            anchor_box_details(unsigned long w, unsigned long h) : width(w), height(h) {}

            unsigned long width = 0;
            unsigned long height = 0;

            friend inline void serialize(const anchor_box_details& item, std::ostream& out)
            {
                int version = 0;
                serialize(version, out);
                serialize(item.width, out);
                serialize(item.height, out);
            }

            friend inline void deserialize(anchor_box_details& item, std::istream& in)
            {
                int version = 0;
                deserialize(version, in);
                deserialize(item.width, in);
                deserialize(item.height, in);
            }
        };

        yolo_options() = default;

        template <template <typename> class TAG_TYPE>
        void add_anchors(const std::vector<anchor_box_details>& boxes)
        {
            anchors[tag_id<TAG_TYPE>::id] = boxes;
        }

        // map between the stride and the anchor boxes
        std::map<int, std::vector<anchor_box_details>> anchors;
        std::vector<std::string> labels;
        double iou_ignore_threshold = 0.7;
        double iou_anchor_threshold = 1.0;
        test_box_overlap overlaps_nms = test_box_overlap(0.45, 1.0);
        bool classwise_nms = true;
        double lambda_obj = 1.0;
        double lambda_box = 1.0;
        double lambda_cls = 1.0;
        double gamma_obj = 0.0;
        double gamma_cls = 0.0;

    };

    inline void serialize(const yolo_options& item, std::ostream& out)
    {
        int version = 2;
        serialize(version, out);
        serialize(item.anchors, out);
        serialize(item.labels, out);
        serialize(item.iou_ignore_threshold, out);
        serialize(item.iou_anchor_threshold, out);
        serialize(item.classwise_nms, out);
        serialize(item.overlaps_nms, out);
        serialize(item.lambda_obj, out);
        serialize(item.lambda_box, out);
        serialize(item.lambda_cls, out);
        serialize(item.gamma_obj, out);
        serialize(item.gamma_cls, out);
    }

    inline void deserialize(yolo_options& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (!(1 <= version && version <= 2))
            throw serialization_error("Unexpected version found while deserializing dlib::yolo_options.");
        deserialize(item.anchors, in);
        deserialize(item.labels, in);
        deserialize(item.iou_ignore_threshold, in);
        deserialize(item.iou_anchor_threshold, in);
        deserialize(item.classwise_nms, in);
        deserialize(item.overlaps_nms, in);
        deserialize(item.lambda_obj, in);
        deserialize(item.lambda_box, in);
        deserialize(item.lambda_cls, in);
        if (version >= 2)
        {
            deserialize(item.gamma_obj, in);
            deserialize(item.gamma_cls, in);
        }
    }

    inline std::ostream& operator<<(std::ostream& out, const std::map<int, std::vector<yolo_options::anchor_box_details>>& anchors)
    {
        // write anchor boxes grouped by tag id
        size_t tag_count = 0;
        for (const auto& i : anchors)
        {
            const auto& tag_id = i.first;
            const auto& details = i.second;
            if (tag_count++ > 0)
                out << ";";
            out << "tag" << tag_id << ":";
            for (size_t a = 0; a < details.size(); ++a)
            {
                out << details[a].width << "x" << details[a].height;
                if (a + 1 < details.size())
                    out << ",";
            }
        }
        return out;
    }

    namespace impl
    {
        template <template <typename> class TAG_TYPE, template <typename> class... TAG_TYPES>
        struct yolo_helper_impl
        {
            constexpr static size_t tag_count()
            {
                return 1 + yolo_helper_impl<TAG_TYPES...>::tag_count();
            }

            static void list_tags(std::ostream& out)
            {
                out << "tag" << tag_id<TAG_TYPE>::id << (tag_count() > 1 ? "," : "");
                yolo_helper_impl<TAG_TYPES...>::list_tags(out);
            }

            template <typename SUBNET>
            static void tensor_to_dets (
                const tensor& input_tensor,
                const SUBNET& sub,
                const long n,
                const yolo_options& options,
                const double adjust_threshold,
                std::vector<yolo_rect>& dets
            )
            {
                yolo_helper_impl<TAG_TYPE>::tensor_to_dets(input_tensor, sub, n, options, adjust_threshold, dets);
                yolo_helper_impl<TAG_TYPES...>::tensor_to_dets(input_tensor, sub, n, options, adjust_threshold, dets);
            }

            template <
                typename const_label_iterator,
                typename SUBNET
            >
            static void tensor_to_loss (
                const tensor& input_tensor,
                const_label_iterator truth,
                SUBNET& sub,
                const long n,
                const yolo_options& options,
                double& loss
            )
            {
                yolo_helper_impl<TAG_TYPE>::tensor_to_loss(input_tensor, truth, sub, n, options, loss);
                yolo_helper_impl<TAG_TYPES...>::tensor_to_loss(input_tensor, truth, sub, n, options, loss);
            }
        };

        template <template <typename> class TAG_TYPE>
        struct yolo_helper_impl<TAG_TYPE>
        {
            constexpr static size_t tag_count() { return 1; }

            static void list_tags(std::ostream& out) { out << "tag" << tag_id<TAG_TYPE>::id; }

            template <typename SUBNET>
            static void tensor_to_dets (
                const tensor& input_tensor,
                const SUBNET& sub,
                const long n,
                const yolo_options& options,
                const double adjust_threshold,
                std::vector<yolo_rect>& dets
            )
            {
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                DLIB_CASSERT(static_cast<size_t>(output_tensor.k()) == anchors.size() * (options.labels.size() + 5));
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                const long num_feats = output_tensor.k() / anchors.size();
                const long num_classes = num_feats - 5;
                const float* const out_data = output_tensor.host();

                for (size_t a = 0; a < anchors.size(); ++a)
                {
                    const long k = a * num_feats;
                    for (long r = 0; r < output_tensor.nr(); ++r)
                    {
                        for (long c = 0; c < output_tensor.nc(); ++c)
                        {
                            const double obj = out_data[tensor_index(output_tensor, n, k + 4, r, c)];
                            if (obj > adjust_threshold)
                            {
                                // The scaling and shifting in the x and y coordinates avoids the grid sensitivity
                                // effect by allowing the network to output centers along the grid boundaries.
                                const double x = out_data[tensor_index(output_tensor, n, k + 0, r, c)] * 2.0 - 0.5;
                                const double y = out_data[tensor_index(output_tensor, n, k + 1, r, c)] * 2.0 - 0.5;
                                const double w = out_data[tensor_index(output_tensor, n, k + 2, r, c)];
                                const double h = out_data[tensor_index(output_tensor, n, k + 3, r, c)];
                                yolo_rect det(centered_drect(dpoint((x + c) * stride_x, (y + r) * stride_y),
                                                             w / (1 - w) * anchors[a].width,
                                                             h / (1 - h) * anchors[a].height));
                                for (long i = 0; i < num_classes; ++i)
                                {
                                    const double conf = obj * out_data[tensor_index(output_tensor, n, k + 5 + i, r, c)];
                                    if (conf > adjust_threshold)
                                        det.labels.emplace_back(conf, options.labels[i]);
                                }
                                if (!det.labels.empty())
                                {
                                    std::sort(det.labels.rbegin(), det.labels.rend());
                                    det.detection_confidence = det.labels[0].first;
                                    det.label = det.labels[0].second;
                                    dets.push_back(std::move(det));
                                }
                            }
                        }
                    }
                }
            }

            template <
                typename const_label_iterator,
                typename SUBNET
            >
            static void tensor_to_loss (
                const tensor& input_tensor,
                const_label_iterator truth,
                SUBNET& sub,
                const long n,
                const yolo_options& options,
                double& loss
            )
            {
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                DLIB_CASSERT(static_cast<size_t>(output_tensor.k()) == anchors.size() * (options.labels.size() + 5));
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                const long num_feats = output_tensor.k() / anchors.size();
                const long num_classes = num_feats - 5;
                const float* const out_data = output_tensor.host();
                tensor& grad = layer<TAG_TYPE>(sub).get_gradient_input();
                DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
                DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
                const drectangle input_rect = rectangle(input_tensor.nr(), input_tensor.nc());
                const auto input_area = input_rect.area();
                float* g = grad.host();

                // Compute the objectness loss for all grid cells
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        for (size_t a = 0; a < anchors.size(); ++a)
                        {
                            const long k = a * num_feats;
                            const double x = out_data[tensor_index(output_tensor, n, k + 0, r, c)] * 2.0 - 0.5;
                            const double y = out_data[tensor_index(output_tensor, n, k + 1, r, c)] * 2.0 - 0.5;
                            const double w = out_data[tensor_index(output_tensor, n, k + 2, r, c)];
                            const double h = out_data[tensor_index(output_tensor, n, k + 3, r, c)];

                            // The prediction at r, c for anchor a
                            const yolo_rect pred(centered_drect(dpoint((x + c) * stride_x, (y + r) * stride_y),
                                                                w / (1 - w) * anchors[a].width,
                                                                h / (1 - h) * anchors[a].height));

                            // Find the best IoU for all ground truth boxes
                            double best_iou = 0;
                            for (const yolo_rect& truth_box : *truth)
                            {
                                if (truth_box.ignore || !input_rect.contains(center(truth_box.rect)))
                                    continue;
                                best_iou = std::max(best_iou, box_intersection_over_union(truth_box.rect, pred.rect));
                            }

                            // Incur loss for the boxes that are below a certain IoU threshold with any truth box
                            if (best_iou < options.iou_ignore_threshold)
                            {
                                const auto o_idx = tensor_index(output_tensor, n, k + 4, r, c);
                                const double p = out_data[o_idx];
                                const double focus = std::pow(p, options.gamma_obj);
                                const double g_obj = focus * (options.gamma_obj * (1 - p) * safe_log(1 - p) + p);
                                g[o_idx] = options.lambda_obj * g_obj;
                            }
                        }
                    }
                }

                // Now find the best anchor box for each truth box
                for (const yolo_rect& truth_box : *truth)
                {
                    if (truth_box.ignore || !input_rect.contains(center(truth_box.rect)))
                        continue;
                    const auto truth_box_area = truth_box.rect.area();
                    const auto t_center = center(truth_box.rect);
                    double best_iou = 0;
                    size_t best_a = 0;
                    size_t best_tag_id = 0;
                    running_stats<double> ious;
                    for (const auto& item : options.anchors)
                    {
                        const auto tag_id = item.first;
                        const auto details = item.second;
                        for (size_t a = 0; a < details.size(); ++a)
                        {
                            const auto anchor(centered_drect(t_center, details[a].width, details[a].height));
                            const double iou = box_intersection_over_union(truth_box.rect, anchor);
                            if (iou > best_iou)
                            {
                                best_iou = iou;
                                best_a = a;
                                best_tag_id = tag_id;
                                ious.add(iou);
                            }
                        }
                    }

                    // ATSS: Adaptive Training Sample Selection
                    double iou_anchor_threshold = options.iou_anchor_threshold;
                    if (iou_anchor_threshold == 0)
                        iou_anchor_threshold = ious.mean() + ious.stddev();

                    for (size_t a = 0; a < anchors.size(); ++a)
                    {
                        // We will always backpropagate on the best anchor, regardless of its IOU.
                        // For other anchors, only if they have an IOU >= iou_anchor_threshold.
                        if (!(best_tag_id == tag_id<TAG_TYPE>::id && best_a == a))
                        {
                            if (iou_anchor_threshold >= 1)
                                continue;
                            const auto anchor(centered_drect(t_center, anchors[a].width, anchors[a].height));
                            if (box_intersection_over_union(truth_box.rect, anchor) < iou_anchor_threshold)
                                continue;
                        }

                        const long c = t_center.x() / stride_x;
                        const long r = t_center.y() / stride_y;
                        const long k = a * num_feats;

                        // Get the truth box target values
                        const double tx = t_center.x() / stride_x - c;
                        const double ty = t_center.y() / stride_y - r;
                        const double tw = truth_box.rect.width() / (anchors[a].width + truth_box.rect.width());
                        const double th = truth_box.rect.height() / (anchors[a].height + truth_box.rect.height());

                        // Scale regression error according to the truth size
                        const double scale_box = options.lambda_box * (2.0 - truth_box_area / input_area);

                        // Compute the smoothed L1 gradient for the box coordinates
                        const auto x_idx = tensor_index(output_tensor, n, k + 0, r, c);
                        const auto y_idx = tensor_index(output_tensor, n, k + 1, r, c);
                        const auto w_idx = tensor_index(output_tensor, n, k + 2, r, c);
                        const auto h_idx = tensor_index(output_tensor, n, k + 3, r, c);
                        g[x_idx] = scale_box * put_in_range(-1, 1, (out_data[x_idx] * 2.0 - 0.5 - tx));
                        g[y_idx] = scale_box * put_in_range(-1, 1, (out_data[y_idx] * 2.0 - 0.5 - ty));
                        g[w_idx] = scale_box * put_in_range(-1, 1, (out_data[w_idx] - tw));
                        g[h_idx] = scale_box * put_in_range(-1, 1, (out_data[h_idx] - th));

                        // This grid cell should detect an object
                        const auto o_idx = tensor_index(output_tensor, n, k + 4, r, c);
                        {
                            const auto p = out_data[o_idx];
                            const double focus = std::pow(1 - p, options.gamma_obj);
                            const double g_obj = focus * (options.gamma_obj * p * safe_log(p) + p - 1);
                            g[o_idx] = options.lambda_obj * g_obj;
                        }

                        // Compute the classification error using the truth weights and the focal loss
                        for (long i = 0; i < num_classes; ++i)
                        {
                            const auto c_idx = tensor_index(output_tensor, n, k + 5 + i, r, c);
                            const auto p = out_data[c_idx];
                            if (truth_box.label == options.labels[i])
                            {
                                const double focus = std::pow(1 - p, options.gamma_cls);
                                const double g_cls = focus * (options.gamma_cls * p * safe_log(p) + p - 1);
                                g[c_idx] = truth_box.detection_confidence * options.lambda_cls * g_cls;
                            }
                            else
                            {
                                const double focus = std::pow(p, options.gamma_cls);
                                const double g_cls = focus * (options.gamma_cls * (1 - p) * safe_log(1 - p) + p);
                                g[c_idx] = options.lambda_cls * g_cls;
                            }
                        }
                    }
                }

                // The loss is the squared norm of the gradient
                loss += length_squared(rowm(mat(grad), n));
            }
        };
    }

    template <template <typename> class... TAG_TYPES>
    class loss_yolo_
    {
        static void list_tags(std::ostream& out) { impl::yolo_helper_impl<TAG_TYPES...>::list_tags(out); }

    public:

        typedef std::vector<yolo_rect> training_label_type;
        typedef std::vector<yolo_rect> output_label_type;

        constexpr static size_t tag_count() { return impl::yolo_helper_impl<TAG_TYPES...>::tag_count(); }

        loss_yolo_() {};

        loss_yolo_(const yolo_options& options) : options(options) { }

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter,
            double adjust_threshold = 0.25
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
            std::vector<yolo_rect> dets_accum;
            std::vector<yolo_rect> final_dets;
            for (long i = 0; i < input_tensor.num_samples(); ++i)
            {
                dets_accum.clear();
                impl::yolo_helper_impl<TAG_TYPES...>::tensor_to_dets(input_tensor, sub, i, options, adjust_threshold, dets_accum);

                // Do non-max suppression
                std::sort(dets_accum.rbegin(), dets_accum.rend());
                final_dets.clear();
                for (size_t j = 0; j < dets_accum.size(); ++j)
                {
                    if (overlaps_any_box_nms(final_dets, dets_accum[j]))
                        continue;

                    final_dets.push_back(dets_accum[j]);
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
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
            double loss = 0;
            for (long i = 0; i < input_tensor.num_samples(); ++i)
            {
                impl::yolo_helper_impl<TAG_TYPES...>::tensor_to_loss(input_tensor, truth, sub, i, options, loss);
                ++truth;
            }
            return loss / input_tensor.num_samples();
        }

        const yolo_options& get_options() const { return options; }

        void adjust_nms(double iou_thresh, double percent_covered_thresh = 1, bool classwise = true)
        {
            options.overlaps_nms = test_box_overlap(iou_thresh, percent_covered_thresh);
            options.classwise_nms = classwise;
        }

        friend void serialize(const loss_yolo_& item, std::ostream& out)
        {
            serialize("loss_yolo_", out);
            size_t count = tag_count();
            serialize(count, out);
            serialize(item.options, out);
        }

        friend void deserialize(loss_yolo_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_yolo_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_yolo_.");
            size_t count = 0;
            deserialize(count, in);
            if (count != tag_count())
                throw serialization_error("Invalid number of detection tags " + std::to_string(count) +
                                          ", while deserializing dlib::loss_yolo_, expecting " +
                                          std::to_string(tag_count()) + " tags instead.");
            deserialize(item.options, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_yolo_& item)
        {
            out << "loss_yolo\t (";
            const auto& opts = item.options;
            out << tag_count() << " output" << (tag_count() != 1 ? "s" : "") << ":(";
            list_tags(out);
            out << ")";
            out << ", anchor_boxes:(" << opts.anchors << ")";
            out << ", " << opts.labels.size() << " label" << (opts.labels.size() != 1 ? "s" : "") << ":(";
            for (size_t i = 0; i < opts.labels.size(); ++i)
            {
                out << opts.labels[i];
                if (i + 1 < opts.labels.size())
                    out << ",";
            }
            out << ")";
            out << ", iou_ignore_threshold: " << opts.iou_ignore_threshold;
            out << ", iou_anchor_threshold: " << opts.iou_anchor_threshold;
            out << ", lambda_obj:" << opts.lambda_obj;
            out << ", lambda_box:" << opts.lambda_box;
            out << ", lambda_cls:" << opts.lambda_cls;
            out << ", gamma_obj:" << opts.gamma_obj;
            out << ", gamma_cls:" << opts.gamma_cls;
            out << ", overlaps_nms:(" << opts.overlaps_nms.get_iou_thresh() << "," << opts.overlaps_nms.get_percent_covered_thresh() << ")";
            out << ", classwise_nms:" << std::boolalpha << opts.classwise_nms;
            out << ")";
            return out;
        }

        friend void to_xml(const loss_yolo_& /*item*/, std::ostream& out)
        {
            out << "<loss_yolo/>\n";
        }

    private:

        yolo_options options;

        inline bool overlaps_any_box_nms (
            const std::vector<yolo_rect>& boxes,
            const yolo_rect& box
        ) const
        {
            for (const auto& b : boxes)
            {
                if (options.overlaps_nms(b.rect, box.rect))
                {
                    if (options.classwise_nms)
                    {
                        if (b.label == box.label)
                            return true;
                    }
                    else
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    };

    template <template <typename> class TAG_1, template <typename> class TAG_2, template <typename> class TAG_3, typename SUBNET>
    using loss_yolo = add_loss_layer<loss_yolo_<TAG_1, TAG_2, TAG_3>, SUBNET>;

// ----------------------------------------------------------------------------------------

    class loss_barlow_twins_
    {
    public:

        loss_barlow_twins_() = default;

        loss_barlow_twins_(float lambda) : lambda(lambda)
        {
            DLIB_CASSERT(lambda > 0);
        }

        template <
            typename SUBNET
        >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 2);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);

            const auto batch_size = output_tensor.num_samples() / 2;
            const auto sample_size = output_tensor.k();
            const auto offset = batch_size * sample_size;

            // Alias helpers to access the samples in the batch
            alias_tensor split(batch_size, sample_size);
            auto za = split(output_tensor);
            auto zb = split(output_tensor, offset);

            // Normalize both batches independently across the batch dimension
            const double eps = 1e-4;
            g.set_size(1, sample_size);
            g = 1;
            b.set_size(1, sample_size);
            b = 0;
            tt::batch_normalize(eps, za_norm, means_a, invstds_a, 1, rms, rvs, za, g, b);
            tt::batch_normalize(eps, zb_norm, means_b, invstds_b, 1, rms, rvs, zb, g, b);

            // Compute the empirical cross-correlation matrix
            eccm.set_size(sample_size, sample_size);
            tt::gemm(0, eccm, 1, za_norm, true, zb_norm, false);
            eccm /= batch_size;

            // Set sizes and setup auxiliary tensors
            if (!have_same_dimensions(eccm, identity))
                identity = identity_matrix<float>(sample_size);
            if (!have_same_dimensions(eccm, cdiag))
                cdiag.copy_size(eccm);
            if (!have_same_dimensions(eccm, cdiag_1))
                cdiag_1.copy_size(eccm);
            if (!have_same_dimensions(eccm, off_mask))
                off_mask = ones_matrix<float>(sample_size, sample_size) - identity_matrix<float>(sample_size);
            if (!have_same_dimensions(eccm, off_diag))
                off_diag.copy_size(eccm);
            if (!have_same_dimensions(grad, grad_input))
                grad_input.copy_size(grad);
            if (!have_same_dimensions(g_grad, g))
                g_grad.copy_size(g);
            if (!have_same_dimensions(b_grad, b))
                b_grad.copy_size(b);

            // Loss gradient, which will be used as the input of the batch normalization gradient
            auto grad_input_a = split(grad_input);
            auto grad_input_b = split(grad_input, offset);

            // Compute the loss: notation from http://www.matrixcalculus.org/
            // A = za_norm
            // B = zb_norm
            // C = eccm
            // D = off_mask: a mask that keeps only the elements outside the diagonal

            // A diagonal matrix containing the diagonal of eccm
            tt::multiply(false, cdiag, eccm, identity);
            // The diagonal of eccm minus the identity matrix
            tt::affine_transform(cdiag_1, cdiag, identity, 1, -1);

            // diagonal term: sum((diag(A' * B) - vector(1)).^2)
            // --------------------------------------------
            // 	=> d/dA = 2 * B * diag(diag(A' * B) - vector(1)) = 2 * B * diag(diag(C) - vector(1))
            // 	=> d/dB = 2 * A * diag(diag(A' * B) - vector(1)) = 2 * A * diag(diag(C) - vector(1))
            tt::gemm(0, grad_input_a, 2.0 / batch_size, zb_norm, false, cdiag_1, false);
            tt::gemm(0, grad_input_b, 2.0 / batch_size, za_norm, false, cdiag_1, false);

            // off-diag term: sum(((A'* B) .* D).^2)
            // --------------------------------
            //  => d/dA = 2 * B * ((B' * A) .* (D .* D)') = 2 * B * (C' .* (D .* D)) = 2 * B * (C' .* D)
            //  => d/dB = 2 * A * ((A' * B) .* (D .* D))  = 2 * A * (C .* (D .* D)) = 2 * A * (C .* D)
            tt::multiply(false, off_diag, eccm, off_mask);
            tt::gemm(1, grad_input_a, lambda * 2.0 / batch_size, zb_norm, false, off_diag, true);
            tt::gemm(1, grad_input_b, lambda * 2.0 / batch_size, za_norm, false, off_diag, false);

            // Compute the batch norm gradients, g and b grads are not used
            auto gza = split(grad);
            auto gzb = split(grad, offset);
            tt::batch_normalize_gradient(eps, grad_input_a, means_a, invstds_a, za, g, gza, g_grad, b_grad);
            tt::batch_normalize_gradient(eps, grad_input_b, means_b, invstds_b, zb, g, gzb, g_grad, b_grad);

            // Compute the loss: MSE between eccm and the identity matrix.
            // Off-diagonal terms are weighed by lambda.
            const double diagonal_loss = sum(squared(mat(cdiag_1)));
            const double off_diag_loss = sum(squared(mat(off_diag)));
            return diagonal_loss + lambda * off_diag_loss;
        }

        float get_lambda() const  { return lambda; }

        tensor& get_eccm() { return eccm; }
        const tensor& get_eccm() const { return eccm; }

        friend void serialize(const loss_barlow_twins_& item, std::ostream& out)
        {
            serialize("loss_barlow_twins_", out);
            serialize(item.lambda, out);
        }

        friend void deserialize(loss_barlow_twins_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "loss_barlow_twins_")
            {
                deserialize(item.lambda, in);
            }
            else
            {
                throw serialization_error("Unexpected version found while deserializing dlib::loss_barlow_twins_.  Instead found " + version);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_barlow_twins_& item)
        {
            out << "loss_barlow_twins (lambda=" << item.lambda << ")";
            return out;
        }

        friend void to_xml(const loss_barlow_twins_& item, std::ostream& out)
        {
            out << "<loss_barlow_twins lambda='" << item.lambda << "'/>\n";
        }

    private:
        float lambda = 0.0051;
        mutable resizable_tensor za_norm, means_a, invstds_a;
        mutable resizable_tensor zb_norm, means_b, invstds_b;
        mutable resizable_tensor rms, rvs, g, b;
        mutable resizable_tensor eccm, grad_input, g_grad, b_grad;
        mutable resizable_tensor cdiag, cdiag_1, identity, off_mask, off_diag;
    };

    template <typename SUBNET>
    using loss_barlow_twins = add_loss_layer<loss_barlow_twins_, SUBNET>;

}

#endif // DLIB_DNn_LOSS_H_

