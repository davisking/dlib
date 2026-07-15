// Copyright (C) 2026
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_ONNX_H_
#define DLIB_DNn_ONNX_H_

#include "core.h"
#include "input.h"
#include "layers.h"
#include "loss.h"
#include "../error.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <type_traits>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class onnx_export_input_mode
    {
        dnn_tensor,
        dlib_input_layer
    };

    struct onnx_export_options
    {
        std::vector<int64_t> input_tensor_shape;
        std::string input_name = "input";
        std::string output_name = "output";
        std::string graph_name = "dlib_network";
        int opset_version = 17;
        onnx_export_input_mode input_mode = onnx_export_input_mode::dnn_tensor;
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        namespace onnx
        {
            /*
                This dependency-free writer emits the small ONNX proto subset
                needed by the exporter.  Field numbers below match ONNX's
                ModelProto, GraphProto, NodeProto, TensorProto, ValueInfoProto,
                TypeProto, TensorShapeProto, OperatorSetIdProto, and
                AttributeProto definitions for opset 17.
            */
            const int64_t dlib_onnx_ir_version = 8;
            const int dlib_onnx_supported_opset_version = 17;

            enum tensor_data_type
            {
                FLOAT = 1,
                INT64 = 7
            };

            enum attribute_type
            {
                ATTRIBUTE_FLOAT = 1,
                ATTRIBUTE_INT = 2,
                ATTRIBUTE_STRING = 3,
                ATTRIBUTE_TENSOR = 4,
                ATTRIBUTE_INTS = 7
            };

            inline void append_varint(std::string& out, uint64_t value)
            {
                while (value >= 0x80)
                {
                    out.push_back(static_cast<char>((value & 0x7f) | 0x80));
                    value >>= 7;
                }
                out.push_back(static_cast<char>(value));
            }

            inline void append_key(std::string& out, int field_number, int wire_type)
            {
                append_varint(out, (static_cast<uint64_t>(field_number) << 3) | wire_type);
            }

            inline void append_int64(std::string& out, int field_number, int64_t value)
            {
                append_key(out, field_number, 0);
                append_varint(out, static_cast<uint64_t>(value));
            }

            inline void append_int32(std::string& out, int field_number, int32_t value)
            {
                append_int64(out, field_number, value);
            }

            inline void append_string(std::string& out, int field_number, const std::string& value)
            {
                append_key(out, field_number, 2);
                append_varint(out, value.size());
                out.append(value);
            }

            inline void append_bytes(std::string& out, int field_number, const std::string& value)
            {
                append_string(out, field_number, value);
            }

            inline void append_message(std::string& out, int field_number, const std::string& value)
            {
                append_key(out, field_number, 2);
                append_varint(out, value.size());
                out.append(value);
            }

            inline void append_float(std::string& out, int field_number, float value)
            {
                append_key(out, field_number, 5);
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(value));
                out.push_back(static_cast<char>(bits & 0xff));
                out.push_back(static_cast<char>((bits >> 8) & 0xff));
                out.push_back(static_cast<char>((bits >> 16) & 0xff));
                out.push_back(static_cast<char>((bits >> 24) & 0xff));
            }

            inline std::string make_raw_float_data(const float* data, size_t size)
            {
                std::string out;
                out.reserve(size*sizeof(float));
                for (size_t i = 0; i < size; ++i)
                {
                    uint32_t bits = 0;
                    std::memcpy(&bits, data+i, sizeof(float));
                    out.push_back(static_cast<char>(bits & 0xff));
                    out.push_back(static_cast<char>((bits >> 8) & 0xff));
                    out.push_back(static_cast<char>((bits >> 16) & 0xff));
                    out.push_back(static_cast<char>((bits >> 24) & 0xff));
                }
                return out;
            }

            inline std::string make_raw_int64_data(const int64_t* data, size_t size)
            {
                std::string out;
                out.reserve(size*sizeof(int64_t));
                for (size_t i = 0; i < size; ++i)
                {
                    const uint64_t value = static_cast<uint64_t>(data[i]);
                    for (int j = 0; j < 8; ++j)
                        out.push_back(static_cast<char>((value >> (8*j)) & 0xff));
                }
                return out;
            }

            inline std::string make_tensor_shape_dim(int64_t dim)
            {
                std::string out;
                append_int64(out, 1, dim);
                return out;
            }

            inline std::string make_tensor_shape(const std::vector<int64_t>& dims)
            {
                std::string out;
                for (auto dim : dims)
                    append_message(out, 1, make_tensor_shape_dim(dim));
                return out;
            }

            inline std::string make_tensor_type(const std::vector<int64_t>& dims)
            {
                std::string tensor_type;
                append_int32(tensor_type, 1, FLOAT);
                append_message(tensor_type, 2, make_tensor_shape(dims));

                std::string type;
                append_message(type, 1, tensor_type);
                return type;
            }

            inline std::string make_value_info(const std::string& name, const std::vector<int64_t>& dims)
            {
                std::string out;
                append_string(out, 1, name);
                append_message(out, 2, make_tensor_type(dims));
                return out;
            }

            inline std::string make_tensor(
                const std::string& name,
                const std::vector<int64_t>& dims,
                const float* data,
                size_t size
            )
            {
                std::string out;
                for (auto dim : dims)
                    append_int64(out, 1, dim);
                append_int32(out, 2, FLOAT);
                append_string(out, 8, name);
                append_bytes(out, 9, make_raw_float_data(data, size));
                return out;
            }

            inline std::string make_tensor_int64(
                const std::string& name,
                const std::vector<int64_t>& dims,
                const int64_t* data,
                size_t size
            )
            {
                std::string out;
                for (auto dim : dims)
                    append_int64(out, 1, dim);
                append_int32(out, 2, INT64);
                append_string(out, 8, name);
                append_bytes(out, 9, make_raw_int64_data(data, size));
                return out;
            }

            inline std::string make_attribute_int(const std::string& name, int64_t value)
            {
                std::string out;
                append_string(out, 1, name);
                append_int64(out, 3, value);
                append_int32(out, 20, ATTRIBUTE_INT);
                return out;
            }

            inline std::string make_attribute_float(const std::string& name, float value)
            {
                std::string out;
                append_string(out, 1, name);
                append_float(out, 2, value);
                append_int32(out, 20, ATTRIBUTE_FLOAT);
                return out;
            }

            inline std::string make_attribute_string(const std::string& name, const std::string& value)
            {
                std::string out;
                append_string(out, 1, name);
                append_string(out, 4, value);
                append_int32(out, 20, ATTRIBUTE_STRING);
                return out;
            }

            inline std::string make_attribute_ints(const std::string& name, const std::vector<int64_t>& values)
            {
                std::string out;
                append_string(out, 1, name);
                for (auto value : values)
                    append_int64(out, 8, value);
                append_int32(out, 20, ATTRIBUTE_INTS);
                return out;
            }

            inline std::string make_attribute_tensor(const std::string& name, const std::string& tensor)
            {
                std::string out;
                append_string(out, 1, name);
                append_message(out, 5, tensor);
                append_int32(out, 20, ATTRIBUTE_TENSOR);
                return out;
            }

            struct node
            {
                std::string name;
                std::string op_type;
                std::vector<std::string> inputs;
                std::vector<std::string> outputs;
                std::vector<std::string> attributes;
            };

            inline std::string make_node(const node& n)
            {
                std::string out;
                for (const auto& input : n.inputs)
                    append_string(out, 1, input);
                for (const auto& output : n.outputs)
                    append_string(out, 2, output);
                append_string(out, 3, n.name);
                append_string(out, 4, n.op_type);
                for (const auto& attr : n.attributes)
                    append_message(out, 5, attr);
                return out;
            }

            inline std::string make_opset_import(int64_t version)
            {
                std::string out;
                append_int64(out, 2, version);
                return out;
            }

            inline std::string tensor_shape_to_string(const std::vector<int64_t>& shape)
            {
                std::ostringstream sout;
                sout << "[";
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (i != 0)
                        sout << ",";
                    sout << shape[i];
                }
                sout << "]";
                return sout.str();
            }

            inline size_t element_count(const std::vector<int64_t>& shape)
            {
                size_t count = 1;
                for (auto dim : shape)
                    count *= static_cast<size_t>(dim);
                return count;
            }

            class export_context
            {
            public:
                struct tagged_value
                {
                    std::string name;
                    std::vector<int64_t> shape;
                };

                explicit export_context(const onnx_export_options& options_) : options(options_)
                {
                    current_name = options.input_name;
                    current_shape = options.input_tensor_shape;
                    graph_inputs.push_back(make_value_info(options.input_name, options.input_tensor_shape));
                }

                std::string next_value_name(const std::string& prefix)
                {
                    std::ostringstream sout;
                    sout << prefix << "_" << value_index++;
                    return sout.str();
                }

                std::string next_node_name(const std::string& prefix)
                {
                    std::ostringstream sout;
                    sout << prefix << "_" << node_index++;
                    return sout.str();
                }

                void add_initializer(
                    const std::string& name,
                    const std::vector<int64_t>& dims,
                    const float* data,
                    size_t size
                )
                {
                    if (element_count(dims) != size)
                    {
                        std::ostringstream sout;
                        sout << "Initializer " << name << " has shape " << tensor_shape_to_string(dims)
                             << " but " << size << " values.";
                        throw dlib::error(sout.str());
                    }
                    initializers.push_back(make_tensor(name, dims, data, size));
                }

                void add_initializer(
                    const std::string& name,
                    const std::vector<int64_t>& dims,
                    const std::vector<float>& data
                )
                {
                    add_initializer(name, dims, data.data(), data.size());
                }

                void add_initializer_int64(
                    const std::string& name,
                    const std::vector<int64_t>& dims,
                    const std::vector<int64_t>& data
                )
                {
                    if (element_count(dims) != data.size())
                    {
                        std::ostringstream sout;
                        sout << "Initializer " << name << " has shape " << tensor_shape_to_string(dims)
                             << " but " << data.size() << " values.";
                        throw dlib::error(sout.str());
                    }
                    initializers.push_back(make_tensor_int64(name, dims, data.data(), data.size()));
                }

                void add_node(
                    const std::string& op_type,
                    const std::vector<std::string>& inputs,
                    const std::vector<std::string>& outputs,
                    const std::vector<std::string>& attributes = std::vector<std::string>()
                )
                {
                    node n;
                    n.name = next_node_name(op_type);
                    n.op_type = op_type;
                    n.inputs = inputs;
                    n.outputs = outputs;
                    n.attributes = attributes;
                    nodes.push_back(n);
                }

                void set_tag(unsigned long id)
                {
                    tags[id] = tagged_value{current_name, current_shape};
                }

                tagged_value get_tag(unsigned long id) const
                {
                    auto found = tags.find(id);
                    if (found == tags.end())
                    {
                        std::ostringstream sout;
                        sout << "ONNX export found layer referencing missing tag" << id << ".";
                        throw dlib::error(sout.str());
                    }
                    return found->second;
                }

                void use_tag(unsigned long id)
                {
                    const auto found = get_tag(id);
                    current_name = found.name;
                    current_shape = found.shape;
                }

                void finish()
                {
                    if (current_name != options.output_name)
                    {
                        add_node("Identity", {current_name}, {options.output_name});
                        current_name = options.output_name;
                    }
                    graph_outputs.push_back(make_value_info(options.output_name, current_shape));
                }

                void save(const std::string& filename)
                {
                    std::ofstream fout(filename, std::ios::binary);
                    if (!fout)
                        throw dlib::error("Unable to open ONNX output file: " + filename);
                    save(fout);
                    if (!fout)
                        throw dlib::error("Error while writing ONNX output file: " + filename);
                }

                void save(std::ostream& out) const
                {
                    std::string graph;
                    for (const auto& n : nodes)
                        append_message(graph, 1, make_node(n));
                    append_string(graph, 2, options.graph_name);
                    for (const auto& initializer : initializers)
                        append_message(graph, 5, initializer);
                    for (const auto& input : graph_inputs)
                        append_message(graph, 11, input);
                    for (const auto& output : graph_outputs)
                        append_message(graph, 12, output);

                    std::string model;
                    append_int64(model, 1, dlib_onnx_ir_version);
                    append_string(model, 2, "dlib");
                    append_message(model, 7, graph);
                    append_message(model, 8, make_opset_import(options.opset_version));

                    out.write(model.data(), static_cast<std::streamsize>(model.size()));
                }

                onnx_export_options options;
                std::string current_name;
                std::vector<int64_t> current_shape;

            private:
                size_t value_index = 0;
                size_t node_index = 0;
                std::vector<node> nodes;
                std::vector<std::string> initializers;
                std::vector<std::string> graph_inputs;
                std::vector<std::string> graph_outputs;
                std::map<unsigned long, tagged_value> tags;
            };

            inline void validate_export_options(const onnx_export_options& options)
            {
                if (options.input_tensor_shape.size() != 4)
                    throw dlib::error("ONNX export requires a 4D input tensor shape [N,K,NR,NC].");
                for (auto dim : options.input_tensor_shape)
                {
                    if (dim <= 0)
                        throw dlib::error("ONNX export input_tensor_shape dimensions must be positive.");
                }
                if (options.input_name.empty())
                    throw dlib::error("ONNX export input_name must not be empty.");
                if (options.output_name.empty())
                    throw dlib::error("ONNX export output_name must not be empty.");
                if (options.graph_name.empty())
                    throw dlib::error("ONNX export graph_name must not be empty.");
                if (options.opset_version != dlib_onnx_supported_opset_version)
                    throw dlib::error("ONNX export currently supports opset_version 17.");
                if (options.input_mode != onnx_export_input_mode::dnn_tensor &&
                    options.input_mode != onnx_export_input_mode::dlib_input_layer)
                    throw dlib::error("ONNX export input_mode has an invalid value.");
            }

            inline std::vector<int64_t> normalize_input_shape(const onnx_export_options& options)
            {
                if (options.input_tensor_shape.empty())
                    throw dlib::error(
                        "net_to_onnx() requires options.input_tensor_shape to be [N,K,NR,NC] "
                        "unless it can be inferred from the dlib input layer."
                    );
                if (options.input_tensor_shape.size() != 4)
                    throw dlib::error("net_to_onnx() requires options.input_tensor_shape to be [N,K,NR,NC].");
                return options.input_tensor_shape;
            }

            template <typename input_layer_type>
            std::vector<int64_t> normalize_input_shape(const onnx_export_options& options, const input_layer_type&)
            {
                return normalize_input_shape(options);
            }

            template <size_t NR, size_t NC>
            std::vector<int64_t> normalize_input_shape(const onnx_export_options& options, const input_rgb_image_sized<NR, NC>&)
            {
                if (options.input_tensor_shape.empty())
                    return {1, 3, static_cast<int64_t>(NR), static_cast<int64_t>(NC)};
                return normalize_input_shape(options);
            }

            inline void validate_dlib_input_mode(const onnx_export_options& options)
            {
                if (options.input_mode == onnx_export_input_mode::dnn_tensor)
                    return;
            }

            template <size_t NR, size_t NC>
            void validate_dlib_input_mode(const onnx_export_options& options, const input_rgb_image_sized<NR, NC>&)
            {
                if (options.input_tensor_shape.size() == 4 &&
                    (options.input_tensor_shape[1] != 3 ||
                     options.input_tensor_shape[2] != static_cast<int64_t>(NR) ||
                     options.input_tensor_shape[3] != static_cast<int64_t>(NC)))
                    throw dlib::error("ONNX export input_rgb_image_sized shape must be [N,3,NR,NC] matching the dlib input layer.");
                if (options.input_mode == onnx_export_input_mode::dlib_input_layer)
                    return;
            }

            inline void validate_dlib_input_mode(const onnx_export_options& options, const input_rgb_image&)
            {
                if (options.input_tensor_shape.size() == 4 && options.input_tensor_shape[1] != 3)
                    throw dlib::error("ONNX export input_rgb_image shape must be [N,3,NR,NC].");
                if (options.input_mode == onnx_export_input_mode::dlib_input_layer)
                    return;
            }

            inline void validate_dlib_input_mode(const onnx_export_options&, const input_rgb_image_pair&)
            {
                throw dlib::error(
                    "ONNX export doesn't support input_rgb_image_pair. "
                    "Export a tensor-input inference subnet or split the pair preprocessing outside ONNX."
                );
            }

            template <typename PYRAMID_TYPE>
            void validate_dlib_input_mode(const onnx_export_options&, const input_rgb_image_pyramid<PYRAMID_TYPE>&)
            {
                throw dlib::error(
                    "ONNX export doesn't support image pyramid input layers. "
                    "Export a fixed tensor-input inference subnet or perform pyramid preprocessing outside ONNX."
                );
            }

            inline void validate_dlib_input_mode(const onnx_export_options&, const input_tensor&)
            {
            }

            template <typename net_type>
            void setup_network_for_export(net_type& net, const onnx_export_options& options, const input_tensor&)
            {
                std::vector<resizable_tensor> samples(static_cast<size_t>(options.input_tensor_shape[0]));
                for (auto& sample : samples)
                {
                    sample.set_size(1,
                                    options.input_tensor_shape[1],
                                    options.input_tensor_shape[2],
                                    options.input_tensor_shape[3]);
                    sample = 0;
                }

                resizable_tensor input;
                net.to_tensor(samples.begin(), samples.end(), input);
                net.forward(input);
            }

            template <typename net_type>
            void setup_network_for_export(net_type& net, const onnx_export_options& options, const input_rgb_image&)
            {
                std::vector<matrix<rgb_pixel>> samples(static_cast<size_t>(options.input_tensor_shape[0]));
                for (auto& sample : samples)
                {
                    sample.set_size(options.input_tensor_shape[2], options.input_tensor_shape[3]);
                    for (long r = 0; r < sample.nr(); ++r)
                        for (long c = 0; c < sample.nc(); ++c)
                            sample(r,c) = rgb_pixel(0,0,0);
                }

                resizable_tensor input;
                net.to_tensor(samples.begin(), samples.end(), input);
                net.forward(input);
            }

            template <typename net_type, size_t NR, size_t NC>
            void setup_network_for_export(net_type& net, const onnx_export_options& options, const input_rgb_image_sized<NR, NC>&)
            {
                setup_network_for_export(net, options, input_rgb_image());
            }

            template <typename net_type, typename T, long NR, long NC, typename MM, typename L>
            void setup_network_for_export(net_type& net, const onnx_export_options& options, const input<matrix<T, NR, NC, MM, L>>&)
            {
                std::vector<matrix<T, NR, NC, MM, L>> samples(static_cast<size_t>(options.input_tensor_shape[0]));
                for (auto& sample : samples)
                {
                    sample.set_size(options.input_tensor_shape[2], options.input_tensor_shape[3]);
                    sample = T();
                }

                resizable_tensor input;
                net.to_tensor(samples.begin(), samples.end(), input);
                net.forward(input);
            }

            template <typename net_type, typename T, typename MM>
            void setup_network_for_export(net_type& net, const onnx_export_options& options, const input<array2d<T, MM>>&)
            {
                std::vector<array2d<T, MM>> samples(static_cast<size_t>(options.input_tensor_shape[0]));
                for (auto& sample : samples)
                {
                    sample.set_size(options.input_tensor_shape[2], options.input_tensor_shape[3]);
                    for (long r = 0; r < sample.nr(); ++r)
                        for (long c = 0; c < sample.nc(); ++c)
                            sample[r][c] = T();
                }

                resizable_tensor input;
                net.to_tensor(samples.begin(), samples.end(), input);
                net.forward(input);
            }

            template <typename T>
            void validate_dlib_input_mode(const onnx_export_options& options, const T&)
            {
                if (options.input_mode == onnx_export_input_mode::dlib_input_layer)
                    throw dlib::error(std::string("ONNX export doesn't support this dlib input layer in dlib_input_layer mode: ") + typeid(T).name());
            }

            template <typename net_type, typename input_layer_type>
            void setup_network_for_export(net_type& net, const onnx_export_options& options, const input_layer_type&)
            {
                resizable_tensor input;
                input.set_size(options.input_tensor_shape[0],
                               options.input_tensor_shape[1],
                               options.input_tensor_shape[2],
                               options.input_tensor_shape[3]);
                input = 0;
                net.forward(input);
            }

            template <long num_filters, long nr, long nc, int stride_y, int stride_x, int padding_y, int padding_x>
            void export_layer(
                export_context& ctx,
                const con_<num_filters, nr, nc, stride_y, stride_x, padding_y, padding_x>& layer
            )
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX Conv export expected a 4D input tensor.");

                const auto& params = layer.get_layer_params();
                const long filters = layer.num_filters();
                const long input_channels = static_cast<long>(ctx.current_shape[1]);
                const long filter_nr = layer.nr();
                const long filter_nc = layer.nc();
                const size_t weight_count = static_cast<size_t>(filters)*input_channels*filter_nr*filter_nc;
                if (params.size() < weight_count)
                    throw dlib::error("ONNX Conv export found an uninitialized dlib convolution layer.");
                if (!layer.bias_is_disabled() && params.size() < weight_count + static_cast<size_t>(filters))
                    throw dlib::error("ONNX Conv export found a dlib convolution layer with missing bias parameters.");

                const std::string weight_name = ctx.next_value_name("conv_W");
                ctx.add_initializer(weight_name, {filters, input_channels, filter_nr, filter_nc}, params.host(), weight_count);

                std::vector<std::string> inputs = {ctx.current_name, weight_name};
                if (!layer.bias_is_disabled())
                {
                    const std::string bias_name = ctx.next_value_name("conv_B");
                    ctx.add_initializer(bias_name, {filters}, params.host()+weight_count, static_cast<size_t>(filters));
                    inputs.push_back(bias_name);
                }

                const std::string output_name = ctx.next_value_name("conv");
                ctx.add_node("Conv", inputs, {output_name}, {
                    make_attribute_ints("strides", {stride_y, stride_x}),
                    make_attribute_ints("pads", {layer.padding_y(), layer.padding_x(), layer.padding_y(), layer.padding_x()})
                });
                ctx.current_name = output_name;
                ctx.current_shape = {
                    ctx.current_shape[0],
                    filters,
                    (ctx.current_shape[2] + 2*layer.padding_y() - filter_nr)/stride_y + 1,
                    (ctx.current_shape[3] + 2*layer.padding_x() - filter_nc)/stride_x + 1
                };

                if (!layer.relu_is_disabled())
                {
                    const std::string relu_output = ctx.next_value_name("relu");
                    ctx.add_node("Relu", {ctx.current_name}, {relu_output});
                    ctx.current_name = relu_output;
                }
            }

            inline void export_layer(export_context& ctx, const relu_& layer)
            {
                if (layer.is_disabled())
                    return;
                const std::string output_name = ctx.next_value_name("relu");
                ctx.add_node("Relu", {ctx.current_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline std::string add_scalar_initializer(export_context& ctx, const std::string& prefix, float value);
            inline std::string add_int64_initializer(export_context& ctx, const std::string& prefix, const std::vector<int64_t>& values);
            inline std::string add_shape_initializer(export_context& ctx, const std::string& prefix, const std::vector<int64_t>& shape);

            inline void export_layer(export_context& ctx, const multiply_& layer)
            {
                const std::string scale_name = add_scalar_initializer(ctx, "multiply_value", layer.get_multiply_value());
                const std::string output_name = ctx.next_value_name("multiply");
                ctx.add_node("Mul", {ctx.current_name, scale_name}, {output_name});
                ctx.current_name = output_name;
            }

            template <typename layer_type>
            typename std::enable_if<
                std::is_base_of<multiply_, layer_type>::value &&
                !std::is_same<multiply_, layer_type>::value
            >::type export_layer(export_context& ctx, const layer_type& layer)
            {
                export_layer(ctx, static_cast<const multiply_&>(layer));
            }

            inline void export_unary_layer(export_context& ctx, const std::string& op_type, const std::string& prefix)
            {
                const std::string output_name = ctx.next_value_name(prefix);
                ctx.add_node(op_type, {ctx.current_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline std::string add_scalar_initializer(export_context& ctx, const std::string& prefix, float value)
            {
                const std::string name = ctx.next_value_name(prefix);
                std::vector<float> values(1, value);
                ctx.add_initializer(name, {1}, values);
                return name;
            }

            inline std::string make_one_element_tensor_attribute(float value)
            {
                return make_tensor(std::string(), std::vector<int64_t>(1, 1), &value, 1);
            }

            inline std::string add_int64_initializer(export_context& ctx, const std::string& prefix, const std::vector<int64_t>& values)
            {
                const std::string name = ctx.next_value_name(prefix);
                ctx.add_initializer_int64(name, {static_cast<int64_t>(values.size())}, values);
                return name;
            }

            inline std::string add_shape_initializer(export_context& ctx, const std::string& prefix, const std::vector<int64_t>& shape)
            {
                return add_int64_initializer(ctx, prefix, shape);
            }

            inline std::string add_channel_initializer(
                export_context& ctx,
                const std::string& prefix,
                const float* values,
                size_t count
            )
            {
                const std::string name = ctx.next_value_name(prefix);
                ctx.add_initializer(name, {1, static_cast<int64_t>(count), 1, 1}, values, count);
                return name;
            }

            inline void export_rgb_input_preprocessing(
                export_context& ctx,
                float avg_red,
                float avg_green,
                float avg_blue
            )
            {
                if (ctx.options.input_mode == onnx_export_input_mode::dnn_tensor)
                    return;
                if (ctx.current_shape.size() != 4 || ctx.current_shape[1] != 3)
                    throw dlib::error("ONNX dlib_input_layer mode for RGB input layers requires input_tensor_shape [N,3,NR,NC].");

                const float mean_values[3] = {avg_red, avg_green, avg_blue};
                const std::string mean_name = add_channel_initializer(ctx, "input_rgb_mean", mean_values, 3);
                const std::string scale_name = add_scalar_initializer(ctx, "input_rgb_scale", 256);
                const std::string centered_name = ctx.next_value_name("input_rgb_centered");
                const std::string output_name = ctx.next_value_name("input_rgb_preprocessed");
                ctx.add_node("Sub", {ctx.current_name, mean_name}, {centered_name});
                ctx.add_node("Div", {centered_name, scale_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_scalar_input_preprocessing(export_context& ctx, bool divide_by_256)
            {
                if (ctx.options.input_mode == onnx_export_input_mode::dnn_tensor || !divide_by_256)
                    return;

                const std::string scale_name = add_scalar_initializer(ctx, "input_scale", 256);
                const std::string output_name = ctx.next_value_name("input_preprocessed");
                ctx.add_node("Div", {ctx.current_name, scale_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_dlib_input_layer(export_context&, const input_tensor&)
            {
            }

            inline void export_dlib_input_layer(export_context& ctx, const input_rgb_image& input_layer)
            {
                export_rgb_input_preprocessing(
                    ctx,
                    input_layer.get_avg_red(),
                    input_layer.get_avg_green(),
                    input_layer.get_avg_blue()
                );
            }

            template <size_t NR, size_t NC>
            void export_dlib_input_layer(export_context& ctx, const input_rgb_image_sized<NR, NC>& input_layer)
            {
                export_rgb_input_preprocessing(
                    ctx,
                    input_layer.get_avg_red(),
                    input_layer.get_avg_green(),
                    input_layer.get_avg_blue()
                );
            }

            template <typename T, long NR, long NC, typename MM, typename L>
            void export_dlib_input_layer(export_context& ctx, const input<matrix<T, NR, NC, MM, L>>&)
            {
                typedef typename pixel_traits<T>::basic_pixel_type basic_pixel_type;
                export_scalar_input_preprocessing(ctx, std::is_same<basic_pixel_type, unsigned char>::value);
            }

            template <typename T, typename MM>
            void export_dlib_input_layer(export_context& ctx, const input<array2d<T, MM>>&)
            {
                typedef typename pixel_traits<T>::basic_pixel_type basic_pixel_type;
                export_scalar_input_preprocessing(ctx, std::is_same<basic_pixel_type, unsigned char>::value);
            }

            template <typename input_layer_type>
            void export_dlib_input_layer(export_context& ctx, const input_layer_type&)
            {
                if (ctx.options.input_mode == onnx_export_input_mode::dlib_input_layer)
                    throw dlib::error(std::string("ONNX export doesn't support this dlib input layer in dlib_input_layer mode: ") + typeid(input_layer_type).name());
            }

            inline void add_resize_node(
                export_context& ctx,
                const std::string& input_name,
                const std::vector<int64_t>& output_shape,
                const std::string& output_name,
                const std::string& mode
            )
            {
                const std::string sizes_name = add_int64_initializer(ctx, "resize_sizes", output_shape);
                // The optional roi and scales inputs are left unspecified with empty
                // input names, since ONNX allows only one of scales/sizes to be given.
                // dlib's bilinear resize maps output pixel x to input pixel
                // x*(in-1)/(out-1), which is ONNX's align_corners convention,
                // not half_pixel or asymmetric.
                ctx.add_node("Resize", {input_name, "", "", sizes_name}, {output_name}, {
                    make_attribute_string("mode", mode),
                    make_attribute_string("coordinate_transformation_mode", "align_corners")
                });
            }

            inline void validate_concat_shapes(
                const std::vector<int64_t>& reference,
                const std::vector<int64_t>& shape
            )
            {
                if (reference.size() != 4 || shape.size() != 4 ||
                    reference[0] != shape[0] ||
                    reference[2] != shape[2] ||
                    reference[3] != shape[3])
                {
                    std::ostringstream sout;
                    sout << "ONNX Concat export requires 4D tensors with equal N/NR/NC dimensions. Got "
                         << tensor_shape_to_string(reference) << " and "
                         << tensor_shape_to_string(shape) << ".";
                    throw dlib::error(sout.str());
                }
            }

            template <template<typename> class... TAG_TYPES>
            struct concat_tag_appender;

            template <>
            struct concat_tag_appender<>
            {
                static void append(export_context&, std::vector<std::string>&, std::vector<int64_t>&, bool&) {}
            };

            inline void export_reshape(export_context& ctx, const std::vector<int64_t>& output_shape, const std::string& prefix)
            {
                const std::string shape_name = add_shape_initializer(ctx, prefix + "_shape", output_shape);
                const std::string output_name = ctx.next_value_name(prefix);
                ctx.add_node("Reshape", {ctx.current_name, shape_name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            inline void export_slice_flattened(
                export_context& ctx,
                int64_t offset,
                const std::vector<int64_t>& output_shape,
                const std::string& prefix
            )
            {
                const int64_t count = static_cast<int64_t>(element_count(output_shape)/static_cast<size_t>(output_shape[0]));
                const int64_t flat_count = static_cast<int64_t>(
                    element_count(ctx.current_shape)/static_cast<size_t>(ctx.current_shape[0])
                );
                const std::vector<int64_t> flat_shape = {ctx.current_shape[0], flat_count};
                export_reshape(ctx, flat_shape, prefix + "_flatten");

                const std::string starts_name = add_int64_initializer(ctx, prefix + "_starts", {offset});
                const std::string ends_name = add_int64_initializer(ctx, prefix + "_ends", {offset + count});
                const std::string axes_name = add_int64_initializer(ctx, prefix + "_axes", {1});
                const std::string slice_name = ctx.next_value_name(prefix + "_slice");
                ctx.add_node("Slice", {ctx.current_name, starts_name, ends_name, axes_name}, {slice_name});
                ctx.current_name = slice_name;
                ctx.current_shape = {ctx.current_shape[0], count};

                export_reshape(ctx, output_shape, prefix);
            }

            inline void export_four_dimensional_slice(
                export_context& ctx,
                const std::vector<int64_t>& starts,
                const std::vector<int64_t>& ends,
                const std::vector<int64_t>& output_shape,
                const std::string& prefix
            )
            {
                const std::string starts_name = add_int64_initializer(ctx, prefix + "_starts", starts);
                const std::string ends_name = add_int64_initializer(ctx, prefix + "_ends", ends);
                const std::string axes_name = add_int64_initializer(ctx, prefix + "_axes", {0, 1, 2, 3});
                const std::string output_name = ctx.next_value_name(prefix);
                ctx.add_node("Slice", {ctx.current_name, starts_name, ends_name, axes_name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            inline void export_rms_norm_sequence(
                export_context& ctx,
                const std::string& input_name,
                const std::vector<int64_t>& input_shape,
                float eps,
                const std::string& gamma_name,
                const std::string& prefix
            )
            {
                const std::string two_name = add_scalar_initializer(ctx, prefix + "_two", 2);
                const std::string eps_name = add_scalar_initializer(ctx, prefix + "_eps", eps);
                const std::string square_name = ctx.next_value_name(prefix + "_square");
                const std::string mean_name = ctx.next_value_name(prefix + "_mean");
                const std::string add_eps_name = ctx.next_value_name(prefix + "_add_eps");
                const std::string sqrt_name = ctx.next_value_name(prefix + "_sqrt");
                const std::string normalized_name = ctx.next_value_name(prefix + "_normalized");
                const std::string output_name = ctx.next_value_name(prefix);

                ctx.add_node("Pow", {input_name, two_name}, {square_name});
                ctx.add_node("ReduceMean", {square_name}, {mean_name}, {
                    make_attribute_ints("axes", {1, 2, 3}),
                    make_attribute_int("keepdims", 1)
                });
                ctx.add_node("Add", {mean_name, eps_name}, {add_eps_name});
                ctx.add_node("Sqrt", {add_eps_name}, {sqrt_name});
                ctx.add_node("Div", {input_name, sqrt_name}, {normalized_name});
                ctx.add_node("Mul", {normalized_name, gamma_name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = input_shape;
            }

            inline std::string pad_tensor_to_shape(
                export_context& ctx,
                const std::string& input_name,
                const std::vector<int64_t>& input_shape,
                const std::vector<int64_t>& output_shape,
                const std::string& prefix
            )
            {
                if (input_shape == output_shape)
                    return input_name;
                if (input_shape.size() != 4 || output_shape.size() != 4)
                    throw dlib::error("ONNX zero padding export currently requires 4D tensors.");

                std::vector<int64_t> pads(8, 0);
                for (size_t i = 0; i < 4; ++i)
                {
                    if (input_shape[i] > output_shape[i])
                        throw dlib::error("ONNX zero padding export cannot shrink tensors.");
                    pads[i + 4] = output_shape[i] - input_shape[i];
                }

                const std::string pads_name = add_int64_initializer(ctx, prefix + "_pads", pads);
                const std::string value_name = add_scalar_initializer(ctx, prefix + "_value", 0);
                const std::string output_name = ctx.next_value_name(prefix);
                ctx.add_node("Pad", {input_name, pads_name, value_name}, {output_name}, {
                    make_attribute_string("mode", "constant")
                });
                return output_name;
            }

            template <template<typename> class TAG_TYPE, template<typename> class... REST>
            struct concat_tag_appender<TAG_TYPE, REST...>
            {
                static void append(
                    export_context& ctx,
                    std::vector<std::string>& inputs,
                    std::vector<int64_t>& output_shape,
                    bool& first
                )
                {
                    const auto tagged = ctx.get_tag(tag_id<TAG_TYPE>::id);
                    if (first)
                    {
                        output_shape = tagged.shape;
                        first = false;
                    }
                    else
                    {
                        validate_concat_shapes(output_shape, tagged.shape);
                        output_shape[1] += tagged.shape[1];
                    }
                    inputs.push_back(tagged.name);
                    concat_tag_appender<REST...>::append(ctx, inputs, output_shape, first);
                }
            };

            inline void export_layer(export_context& ctx, const sig_&)
            {
                export_unary_layer(ctx, "Sigmoid", "sigmoid");
            }

            inline void export_layer(export_context& ctx, const htan_&)
            {
                export_unary_layer(ctx, "Tanh", "tanh");
            }

            inline void export_layer(export_context& ctx, const leaky_relu_& layer)
            {
                const std::string output_name = ctx.next_value_name("leaky_relu");
                ctx.add_node("LeakyRelu", {ctx.current_name}, {output_name}, {make_attribute_float("alpha", layer.get_alpha())});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const prelu_& layer)
            {
                const auto& params = layer.get_layer_params();
                if (params.size() != 1)
                    throw dlib::error("ONNX PRelu export expected a scalar dlib prelu parameter.");
                const std::string slope_name = ctx.next_value_name("prelu_slope");
                ctx.add_initializer(slope_name, {1}, params.host(), 1);
                const std::string output_name = ctx.next_value_name("prelu");
                ctx.add_node("PRelu", {ctx.current_name, slope_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const clipped_relu_& layer)
            {
                const std::string min_name = add_scalar_initializer(ctx, "clip_min", 0);
                const std::string max_name = add_scalar_initializer(ctx, "clip_max", layer.get_ceiling());
                const std::string output_name = ctx.next_value_name("clip");
                ctx.add_node("Clip", {ctx.current_name, min_name, max_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const elu_& layer)
            {
                const std::string output_name = ctx.next_value_name("elu");
                ctx.add_node("Elu", {ctx.current_name}, {output_name}, {make_attribute_float("alpha", layer.get_alpha())});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const silu_&)
            {
                const std::string sigmoid_name = ctx.next_value_name("silu_sigmoid");
                const std::string output_name = ctx.next_value_name("silu");
                ctx.add_node("Sigmoid", {ctx.current_name}, {sigmoid_name});
                ctx.add_node("Mul", {ctx.current_name, sigmoid_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const mish_&)
            {
                const std::string softplus_name = ctx.next_value_name("mish_softplus");
                const std::string tanh_name = ctx.next_value_name("mish_tanh");
                const std::string output_name = ctx.next_value_name("mish");
                ctx.add_node("Softplus", {ctx.current_name}, {softplus_name});
                ctx.add_node("Tanh", {softplus_name}, {tanh_name});
                ctx.add_node("Mul", {ctx.current_name, tanh_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const gelu_&)
            {
                const std::string sqrt2_name = add_scalar_initializer(ctx, "gelu_sqrt2", 1.4142135623730951f);
                const std::string one_name = add_scalar_initializer(ctx, "gelu_one", 1);
                const std::string half_name = add_scalar_initializer(ctx, "gelu_half", 0.5f);
                const std::string div_name = ctx.next_value_name("gelu_div");
                const std::string erf_name = ctx.next_value_name("gelu_erf");
                const std::string add_name = ctx.next_value_name("gelu_add");
                const std::string mul_x_name = ctx.next_value_name("gelu_mul_x");
                const std::string output_name = ctx.next_value_name("gelu");
                ctx.add_node("Div", {ctx.current_name, sqrt2_name}, {div_name});
                ctx.add_node("Erf", {div_name}, {erf_name});
                ctx.add_node("Add", {erf_name, one_name}, {add_name});
                ctx.add_node("Mul", {ctx.current_name, add_name}, {mul_x_name});
                ctx.add_node("Mul", {mul_x_name, half_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const smelu_& layer)
            {
                const float beta = layer.get_beta();
                if (beta <= 0)
                    throw dlib::error("ONNX smelu export requires beta > 0.");
                const std::string beta_name = add_scalar_initializer(ctx, "smelu_beta", beta);
                const std::string neg_beta_name = add_scalar_initializer(ctx, "smelu_neg_beta", -beta);
                const std::string four_beta_name = add_scalar_initializer(ctx, "smelu_four_beta", 4*beta);
                const std::string two_name = add_scalar_initializer(ctx, "smelu_two", 2);
                const std::string zero_name = add_scalar_initializer(ctx, "smelu_zero", 0);
                const std::string gt_name = ctx.next_value_name("smelu_gt_beta");
                const std::string lt_name = ctx.next_value_name("smelu_lt_neg_beta");
                const std::string add_beta_name = ctx.next_value_name("smelu_add_beta");
                const std::string square_name = ctx.next_value_name("smelu_square");
                const std::string smooth_name = ctx.next_value_name("smelu_smooth");
                const std::string lower_name = ctx.next_value_name("smelu_lower");
                const std::string output_name = ctx.next_value_name("smelu");

                ctx.add_node("Greater", {ctx.current_name, beta_name}, {gt_name});
                ctx.add_node("Less", {ctx.current_name, neg_beta_name}, {lt_name});
                ctx.add_node("Add", {ctx.current_name, beta_name}, {add_beta_name});
                ctx.add_node("Pow", {add_beta_name, two_name}, {square_name});
                ctx.add_node("Div", {square_name, four_beta_name}, {smooth_name});
                ctx.add_node("Where", {lt_name, zero_name, smooth_name}, {lower_name});
                ctx.add_node("Where", {gt_name, ctx.current_name, lower_name}, {output_name});
                ctx.current_name = output_name;
            }

            inline void export_layer(export_context& ctx, const l2normalize_& layer)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX l2normalize export expected a 4D input tensor.");
                const auto input_shape = ctx.current_shape;
                const std::string input_name = ctx.current_name;
                const std::string two_name = add_scalar_initializer(ctx, "l2normalize_two", 2);
                const std::string eps_name = add_scalar_initializer(ctx, "l2normalize_eps", static_cast<float>(layer.get_eps()));
                const std::string square_name = ctx.next_value_name("l2normalize_square");
                const std::string sum_name = ctx.next_value_name("l2normalize_sum");
                const std::string sum_eps_name = ctx.next_value_name("l2normalize_sum_eps");
                const std::string sqrt_name = ctx.next_value_name("l2normalize_sqrt");
                const std::string output_name = ctx.next_value_name("l2normalize");
                const std::string axes_name = add_int64_initializer(ctx, "l2normalize_axes", {1, 2, 3});

                ctx.add_node("Pow", {input_name, two_name}, {square_name});
                ctx.add_node("ReduceSum", {square_name, axes_name}, {sum_name}, {
                    make_attribute_int("keepdims", 1)
                });
                ctx.add_node("Add", {sum_name, eps_name}, {sum_eps_name});
                ctx.add_node("Sqrt", {sum_eps_name}, {sqrt_name});
                ctx.add_node("Div", {input_name, sqrt_name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = input_shape;
            }

            template <long nr, long nc, int stride_y, int stride_x, int padding_y, int padding_x>
            void export_layer(
                export_context& ctx,
                const max_pool_<nr, nc, stride_y, stride_x, padding_y, padding_x>& layer
            )
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX MaxPool export expected a 4D input tensor.");
                const std::string output_name = ctx.next_value_name("maxpool");
                if (layer.nr() == 0 && layer.nc() == 0)
                {
                    ctx.add_node("GlobalMaxPool", {ctx.current_name}, {output_name});
                }
                else
                {
                    const long kernel_nr = layer.nr() == 0 ? static_cast<long>(ctx.current_shape[2]) : layer.nr();
                    const long kernel_nc = layer.nc() == 0 ? static_cast<long>(ctx.current_shape[3]) : layer.nc();
                    const int64_t output_nr = (ctx.current_shape[2] + 2*layer.padding_y() - kernel_nr)/stride_y + 1;
                    const int64_t output_nc = (ctx.current_shape[3] + 2*layer.padding_x() - kernel_nc)/stride_x + 1;
                    ctx.add_node("MaxPool", {ctx.current_name}, {output_name}, {
                        make_attribute_ints("kernel_shape", {kernel_nr, kernel_nc}),
                        make_attribute_ints("strides", {stride_y, stride_x}),
                        make_attribute_ints("pads", {layer.padding_y(), layer.padding_x(), layer.padding_y(), layer.padding_x()})
                    });
                    ctx.current_shape = {
                        ctx.current_shape[0],
                        ctx.current_shape[1],
                        output_nr,
                        output_nc
                    };
                }
                ctx.current_name = output_name;
                if (layer.nr() == 0 && layer.nc() == 0)
                {
                    ctx.current_shape = {ctx.current_shape[0], ctx.current_shape[1], 1, 1};
                }
            }

            template <long nr, long nc, int stride_y, int stride_x, int padding_y, int padding_x>
            void export_layer(
                export_context& ctx,
                const avg_pool_<nr, nc, stride_y, stride_x, padding_y, padding_x>& layer
            )
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX AveragePool export expected a 4D input tensor.");
                const std::string output_name = ctx.next_value_name("avgpool");
                if (layer.nr() == 0 && layer.nc() == 0)
                {
                    ctx.add_node("GlobalAveragePool", {ctx.current_name}, {output_name});
                }
                else
                {
                    ctx.add_node("AveragePool", {ctx.current_name}, {output_name}, {
                        make_attribute_ints("kernel_shape", {layer.nr(), layer.nc()}),
                        make_attribute_ints("strides", {stride_y, stride_x}),
                        make_attribute_ints("pads", {layer.padding_y(), layer.padding_x(), layer.padding_y(), layer.padding_x()})
                    });
                }
                ctx.current_name = output_name;
                if (layer.nr() == 0 && layer.nc() == 0)
                {
                    ctx.current_shape = {ctx.current_shape[0], ctx.current_shape[1], 1, 1};
                }
                else
                {
                    ctx.current_shape = {
                        ctx.current_shape[0],
                        ctx.current_shape[1],
                        (ctx.current_shape[2] + 2*layer.padding_y() - layer.nr())/stride_y + 1,
                        (ctx.current_shape[3] + 2*layer.padding_x() - layer.nc())/stride_x + 1
                    };
                }
            }

            template <long num_filters, long nr, long nc, int stride_y, int stride_x, int padding_y, int padding_x>
            void export_layer(
                export_context& ctx,
                const cont_<num_filters, nr, nc, stride_y, stride_x, padding_y, padding_x>& layer
            )
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX ConvTranspose export expected a 4D input tensor.");

                const auto& params = layer.get_layer_params();
                const long filters = layer.num_filters();
                const long input_channels = static_cast<long>(ctx.current_shape[1]);
                const long filter_nr = layer.nr();
                const long filter_nc = layer.nc();
                const size_t weight_count = static_cast<size_t>(input_channels)*filters*filter_nr*filter_nc;
                if (params.size() < weight_count)
                    throw dlib::error("ONNX ConvTranspose export found an uninitialized dlib transposed convolution layer.");
                if (!layer.bias_is_disabled() && params.size() < weight_count + static_cast<size_t>(filters))
                    throw dlib::error("ONNX ConvTranspose export found a dlib transposed convolution layer with missing bias parameters.");

                const std::string weight_name = ctx.next_value_name("cont_W");
                ctx.add_initializer(weight_name, {input_channels, filters, filter_nr, filter_nc}, params.host(), weight_count);

                std::vector<std::string> inputs = {ctx.current_name, weight_name};
                if (!layer.bias_is_disabled())
                {
                    const std::string bias_name = ctx.next_value_name("cont_B");
                    ctx.add_initializer(bias_name, {filters}, params.host()+weight_count, static_cast<size_t>(filters));
                    inputs.push_back(bias_name);
                }

                const std::string output_name = ctx.next_value_name("cont");
                ctx.add_node("ConvTranspose", inputs, {output_name}, {
                    make_attribute_ints("strides", {stride_y, stride_x}),
                    make_attribute_ints("pads", {layer.padding_y(), layer.padding_x(), layer.padding_y(), layer.padding_x()})
                });
                ctx.current_name = output_name;
                ctx.current_shape = {
                    ctx.current_shape[0],
                    filters,
                    stride_y*(ctx.current_shape[2]-1) + filter_nr - 2*layer.padding_y(),
                    stride_x*(ctx.current_shape[3]-1) + filter_nc - 2*layer.padding_x()
                };
            }

            template <int scale_y, int scale_x>
            void export_layer(export_context& ctx, const upsample_<scale_y, scale_x>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX Resize export expected a 4D input tensor.");
                const std::vector<int64_t> output_shape = {
                    ctx.current_shape[0],
                    ctx.current_shape[1],
                    ctx.current_shape[2]*scale_y,
                    ctx.current_shape[3]*scale_x
                };
                const std::string output_name = ctx.next_value_name("upsample");
                add_resize_node(ctx, ctx.current_name, output_shape, output_name, "linear");
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            template <long NR, long NC>
            void export_layer(export_context& ctx, const resize_to_<NR, NC>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX resize_to export expected a 4D input tensor.");
                const std::vector<int64_t> output_shape = {
                    ctx.current_shape[0],
                    ctx.current_shape[1],
                    NR,
                    NC
                };
                const std::string output_name = ctx.next_value_name("resize_to");
                add_resize_node(ctx, ctx.current_name, output_shape, output_name, "linear");
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            template <long k, long nr, long nc>
            void export_layer(export_context& ctx, const reshape_to_<k, nr, nc>& layer)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX reshape_to export expected a 4D input tensor.");
                const std::vector<int64_t> output_shape = {
                    ctx.current_shape[0],
                    layer.get_output_k(),
                    layer.get_output_nr(),
                    layer.get_output_nc()
                };
                const size_t input_values = element_count(ctx.current_shape);
                const size_t output_values = element_count(output_shape);
                if (input_values == output_values)
                {
                    export_reshape(ctx, output_shape, "reshape_to");
                }
                else if (ctx.current_shape[1] == output_shape[1])
                {
                    const std::string output_name = ctx.next_value_name("reshape_to_resize");
                    add_resize_node(ctx, ctx.current_name, output_shape, output_name, "linear");
                    ctx.current_name = output_name;
                    ctx.current_shape = output_shape;
                }
                else
                {
                    throw dlib::error("ONNX reshape_to export requires either equal element counts or spatial resize with unchanged channels.");
                }
            }

            template <unsigned long num_outputs, fc_bias_mode bias_mode>
            void export_layer(
                export_context& ctx,
                const fc_<num_outputs, bias_mode>& layer
            )
            {
                if (ctx.current_shape.size() != 2)
                {
                    const std::string flat_name = ctx.next_value_name("flatten");
                    ctx.add_node("Flatten", {ctx.current_name}, {flat_name}, {make_attribute_int("axis", 1)});
                    ctx.current_name = flat_name;
                    ctx.current_shape = {
                        ctx.current_shape[0],
                        static_cast<int64_t>(element_count(ctx.current_shape)/static_cast<size_t>(ctx.current_shape[0]))
                    };
                }

                const auto& params = layer.get_layer_params();
                const int64_t num_inputs = ctx.current_shape[1];
                const int64_t outputs = layer.get_num_outputs();
                const size_t weight_count = static_cast<size_t>(num_inputs*outputs);
                if (params.size() < weight_count)
                    throw dlib::error("ONNX Gemm export found an uninitialized dlib fc layer.");
                if (bias_mode == FC_HAS_BIAS && !layer.bias_is_disabled() && params.size() < weight_count + static_cast<size_t>(outputs))
                    throw dlib::error("ONNX Gemm export found a dlib fc layer with missing bias parameters.");

                const std::string weight_name = ctx.next_value_name("fc_W");
                ctx.add_initializer(weight_name, {num_inputs, outputs}, params.host(), weight_count);

                std::vector<std::string> inputs = {ctx.current_name, weight_name};
                if (bias_mode == FC_HAS_BIAS && !layer.bias_is_disabled())
                {
                    const std::string bias_name = ctx.next_value_name("fc_B");
                    ctx.add_initializer(bias_name, {outputs}, params.host()+weight_count, static_cast<size_t>(outputs));
                    inputs.push_back(bias_name);
                }

                const std::string output_name = ctx.next_value_name("fc");
                ctx.add_node("Gemm", inputs, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = {ctx.current_shape[0], outputs};
            }

            template <unsigned long num_outputs, linear_bias_mode bias_mode>
            void export_layer(
                export_context& ctx,
                const linear_<num_outputs, bias_mode>& layer
            )
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX linear export expected a 4D input tensor.");

                const int64_t num_inputs = ctx.current_shape[3];
                const int64_t outputs = layer.get_num_outputs();
                const auto& params = layer.get_layer_params();
                const size_t weight_count = static_cast<size_t>(num_inputs*outputs);
                if (params.size() < weight_count)
                    throw dlib::error("ONNX MatMul export found an uninitialized dlib linear layer.");
                if (bias_mode == LINEAR_HAS_BIAS && params.size() < weight_count + static_cast<size_t>(outputs))
                    throw dlib::error("ONNX MatMul export found a dlib linear layer with missing bias parameters.");

                const std::string weight_name = ctx.next_value_name("linear_W");
                ctx.add_initializer(weight_name, {num_inputs, outputs}, params.host(), weight_count);

                const std::string matmul_name = ctx.next_value_name("linear_matmul");
                ctx.add_node("MatMul", {ctx.current_name, weight_name}, {matmul_name});
                ctx.current_name = matmul_name;
                ctx.current_shape[3] = outputs;

                if (bias_mode == LINEAR_HAS_BIAS)
                {
                    const std::string bias_name = ctx.next_value_name("linear_B");
                    const std::string output_name = ctx.next_value_name("linear");
                    ctx.add_initializer(bias_name, {outputs}, params.host()+weight_count, static_cast<size_t>(outputs));
                    ctx.add_node("Add", {ctx.current_name, bias_name}, {output_name});
                    ctx.current_name = output_name;
                }
            }

            inline void export_layer(export_context& ctx, const affine_& layer)
            {
                if (layer.is_disabled())
                    return;

                alias_tensor_const_instance gamma_alias = layer.get_gamma();
                alias_tensor_const_instance beta_alias = layer.get_beta();
                const tensor& gamma = gamma_alias;
                const tensor& beta = beta_alias;
                std::vector<int64_t> param_shape;
                if (layer.get_mode() == CONV_MODE)
                    param_shape = {1, gamma.k(), 1, 1};
                else if (ctx.current_shape.size() == 2)
                    param_shape = {1, static_cast<int64_t>(gamma.size())};
                else
                    param_shape = {1, gamma.k(), gamma.nr(), gamma.nc()};

                const std::string gamma_name = ctx.next_value_name("affine_gamma");
                const std::string beta_name = ctx.next_value_name("affine_beta");
                ctx.add_initializer(gamma_name, param_shape, gamma.host(), gamma.size());
                ctx.add_initializer(beta_name, param_shape, beta.host(), beta.size());

                const std::string mul_name = ctx.next_value_name("affine_mul");
                const std::string add_name = ctx.next_value_name("affine");
                ctx.add_node("Mul", {ctx.current_name, gamma_name}, {mul_name});
                ctx.add_node("Add", {mul_name, beta_name}, {add_name});
                ctx.current_name = add_name;
            }

            template <layer_mode mode>
            void export_layer(export_context& ctx, const bn_<mode>& layer)
            {
                export_layer(ctx, affine_(layer));
            }

            inline void export_layer(export_context& ctx, const rms_norm_& layer)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX rms_norm export expected a 4D input tensor.");
                const auto& params = layer.get_layer_params();
                const int64_t channels = ctx.current_shape[1];
                if (params.size() < static_cast<size_t>(channels))
                    throw dlib::error("ONNX rms_norm export found uninitialized dlib rms_norm parameters.");

                const auto input_shape = ctx.current_shape;
                const std::string input_name = ctx.current_name;
                const std::string gamma_name = add_channel_initializer(ctx, "rms_norm_gamma", params.host(), static_cast<size_t>(channels));
                export_rms_norm_sequence(ctx, input_name, input_shape, static_cast<float>(layer.get_eps()), gamma_name, "rms_norm");
            }

            inline void export_layer(export_context& ctx, const layer_norm_& layer)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX layer_norm export expected a 4D input tensor.");
                const auto& params = layer.get_layer_params();
                const int64_t channels = ctx.current_shape[1];
                const size_t channel_count = static_cast<size_t>(channels);
                if (params.size() < channel_count*2)
                    throw dlib::error("ONNX layer_norm export found uninitialized dlib layer_norm parameters.");

                const auto input_shape = ctx.current_shape;
                const std::string input_name = ctx.current_name;
                const std::string gamma_name = add_channel_initializer(ctx, "layer_norm_gamma", params.host(), channel_count);
                const std::string beta_name = add_channel_initializer(ctx, "layer_norm_beta", params.host()+channel_count, channel_count);
                const std::string two_name = add_scalar_initializer(ctx, "layer_norm_two", 2);
                const std::string eps_name = add_scalar_initializer(ctx, "layer_norm_eps", static_cast<float>(layer.get_eps()));
                const std::string mean_name = ctx.next_value_name("layer_norm_mean");
                const std::string centered_name = ctx.next_value_name("layer_norm_centered");
                const std::string square_name = ctx.next_value_name("layer_norm_square");
                const std::string var_name = ctx.next_value_name("layer_norm_var");
                const std::string var_eps_name = ctx.next_value_name("layer_norm_var_eps");
                const std::string sqrt_name = ctx.next_value_name("layer_norm_sqrt");
                const std::string normalized_name = ctx.next_value_name("layer_norm_normalized");
                const std::string scaled_name = ctx.next_value_name("layer_norm_scaled");
                const std::string output_name = ctx.next_value_name("layer_norm");

                ctx.add_node("ReduceMean", {input_name}, {mean_name}, {
                    make_attribute_ints("axes", {1, 2, 3}),
                    make_attribute_int("keepdims", 1)
                });
                ctx.add_node("Sub", {input_name, mean_name}, {centered_name});
                ctx.add_node("Pow", {centered_name, two_name}, {square_name});
                ctx.add_node("ReduceMean", {square_name}, {var_name}, {
                    make_attribute_ints("axes", {1, 2, 3}),
                    make_attribute_int("keepdims", 1)
                });
                ctx.add_node("Add", {var_name, eps_name}, {var_eps_name});
                ctx.add_node("Sqrt", {var_eps_name}, {sqrt_name});
                ctx.add_node("Div", {centered_name, sqrt_name}, {normalized_name});
                ctx.add_node("Mul", {normalized_name, gamma_name}, {scaled_name});
                ctx.add_node("Add", {scaled_name, beta_name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = input_shape;
            }

            template <template<typename> class tag>
            void export_layer(export_context& ctx, const resize_prev_to_tagged_<tag>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX resize_prev_to_tagged export expected a 4D input tensor.");
                const auto tagged = ctx.get_tag(tag_id<tag>::id);
                if (tagged.shape.size() != 4 ||
                    tagged.shape[0] != ctx.current_shape[0])
                    throw dlib::error("ONNX resize_prev_to_tagged export requires current and tagged tensors to have equal N dimensions.");

                const std::vector<int64_t> output_shape = {
                    ctx.current_shape[0],
                    ctx.current_shape[1],
                    tagged.shape[2],
                    tagged.shape[3]
                };

                if (ctx.current_shape == output_shape)
                    return;

                const std::string output_name = ctx.next_value_name("resize_prev_to_tagged");
                add_resize_node(ctx, ctx.current_name, output_shape, output_name, "linear");
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            template <template<typename> class... TAG_TYPES>
            void export_layer(export_context& ctx, const concat_<TAG_TYPES...>&)
            {
                std::vector<std::string> inputs;
                std::vector<int64_t> output_shape;
                bool first = true;
                concat_tag_appender<TAG_TYPES...>::append(ctx, inputs, output_shape, first);
                if (inputs.empty())
                    throw dlib::error("ONNX Concat export found a concat layer with no input tags.");
                const std::string output_name = ctx.next_value_name("concat");
                ctx.add_node("Concat", inputs, {output_name}, {make_attribute_int("axis", 1)});
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            template <template<typename> class tag>
            void export_layer(export_context& ctx, const mult_prev_<tag>&)
            {
                const auto current_name = ctx.current_name;
                const auto current_shape = ctx.current_shape;
                const auto tagged = ctx.get_tag(tag_id<tag>::id);
                std::vector<int64_t> output_shape = current_shape;
                if (current_shape == tagged.shape)
                {
                    const std::string output_name = ctx.next_value_name("mult_prev");
                    ctx.add_node("Mul", {current_name, tagged.name}, {output_name});
                    ctx.current_name = output_name;
                    return;
                }

                if (current_shape.size() != 4 || tagged.shape.size() != 4)
                    throw dlib::error("ONNX mult_prev export requires equal shapes unless both inputs are 4D tensors.");
                for (size_t i = 0; i < 4; ++i)
                    output_shape[i] = std::max(current_shape[i], tagged.shape[i]);

                const std::string lhs = pad_tensor_to_shape(ctx, current_name, current_shape, output_shape, "mult_prev_lhs_pad");
                const std::string rhs = pad_tensor_to_shape(ctx, tagged.name, tagged.shape, output_shape, "mult_prev_rhs_pad");
                const std::string output_name = ctx.next_value_name("mult_prev");
                ctx.add_node("Mul", {lhs, rhs}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            template <template<typename> class tag>
            void export_layer(export_context& ctx, const multm_prev_<tag>&)
            {
                const auto current_name = ctx.current_name;
                const auto current_shape = ctx.current_shape;
                const auto tagged = ctx.get_tag(tag_id<tag>::id);
                if (current_shape.size() != 4 || tagged.shape.size() != 4 ||
                    current_shape[0] != tagged.shape[0] ||
                    current_shape[1] != tagged.shape[1] ||
                    current_shape[3] != tagged.shape[2])
                {
                    std::ostringstream sout;
                    sout << "ONNX multm_prev export requires [N,K,M,L] x [N,K,L,P] shapes. Got "
                         << tensor_shape_to_string(current_shape) << " and "
                         << tensor_shape_to_string(tagged.shape) << ".";
                    throw dlib::error(sout.str());
                }
                const std::string output_name = ctx.next_value_name("multm_prev");
                ctx.add_node("MatMul", {current_name, tagged.name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = {current_shape[0], current_shape[1], current_shape[2], tagged.shape[3]};
            }

            template <template<typename> class tag>
            void export_layer(export_context& ctx, const scale_prev_<tag>&)
            {
                const auto tagged = ctx.get_tag(tag_id<tag>::id);
                if (ctx.current_shape.size() != 4 || tagged.shape.size() != 4 ||
                    tagged.shape[0] != ctx.current_shape[0] ||
                    tagged.shape[1] != ctx.current_shape[1] ||
                    tagged.shape[2] != 1 ||
                    tagged.shape[3] != 1)
                    throw dlib::error("ONNX scale_prev export requires tagged scales with shape [N,K,1,1].");

                const std::string output_name = ctx.next_value_name("scale_prev");
                ctx.add_node("Mul", {ctx.current_name, tagged.name}, {output_name});
                ctx.current_name = output_name;
            }

            template <template<typename> class tag>
            void export_layer(export_context& ctx, const scale_<tag>&)
            {
                const auto tagged = ctx.get_tag(tag_id<tag>::id);
                if (ctx.current_shape.size() != 4 || tagged.shape.size() != 4 ||
                    ctx.current_shape[0] != tagged.shape[0] ||
                    ctx.current_shape[1] != tagged.shape[1] ||
                    ctx.current_shape[2] != 1 ||
                    ctx.current_shape[3] != 1)
                    throw dlib::error("ONNX scale export requires current scales with shape [N,K,1,1].");

                const std::string output_name = ctx.next_value_name("scale");
                ctx.add_node("Mul", {tagged.name, ctx.current_name}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = tagged.shape;
            }

            template <template<typename> class tag>
            void export_layer(export_context& ctx, const add_prev_<tag>&)
            {
                const auto current_name = ctx.current_name;
                const auto current_shape = ctx.current_shape;
                const auto tagged = ctx.get_tag(tag_id<tag>::id);
                std::vector<int64_t> output_shape = current_shape;
                if (current_shape == tagged.shape)
                {
                    const std::string output_name = ctx.next_value_name("add");
                    ctx.add_node("Add", {current_name, tagged.name}, {output_name});
                    ctx.current_name = output_name;
                    return;
                }

                if (current_shape.size() != 4 || tagged.shape.size() != 4)
                    throw dlib::error("ONNX add_prev export requires equal shapes unless both inputs are 4D tensors.");
                for (size_t i = 0; i < 4; ++i)
                    output_shape[i] = std::max(current_shape[i], tagged.shape[i]);

                const std::string lhs = pad_tensor_to_shape(ctx, current_name, current_shape, output_shape, "add_lhs_pad");
                const std::string rhs = pad_tensor_to_shape(ctx, tagged.name, tagged.shape, output_shape, "add_rhs_pad");
                const std::string output_name = ctx.next_value_name("add");
                ctx.add_node("Add", {lhs, rhs}, {output_name});
                ctx.current_name = output_name;
                ctx.current_shape = output_shape;
            }

            template <operation_mode mode>
            void export_layer(export_context& ctx, const softmax_<mode>&)
            {
                if (mode == operation_mode::CHANNEL_WISE)
                {
                    if (ctx.current_shape.size() == 2)
                    {
                        const std::string output_name = ctx.next_value_name("softmax");
                        ctx.add_node("Softmax", {ctx.current_name}, {output_name}, {make_attribute_int("axis", 1)});
                        ctx.current_name = output_name;
                        return;
                    }
                    if (ctx.current_shape.size() != 4)
                        throw dlib::error("ONNX channel-wise softmax export expected a 2D or 4D input tensor.");

                    const auto output_shape = ctx.current_shape;
                    const std::string transposed_name = ctx.next_value_name("softmax_channels_last");
                    const std::string softmax_name = ctx.next_value_name("softmax");
                    const std::string output_name = ctx.next_value_name("softmax_channels_first");
                    ctx.add_node("Transpose", {ctx.current_name}, {transposed_name}, {make_attribute_ints("perm", {0, 2, 3, 1})});
                    ctx.add_node("Softmax", {transposed_name}, {softmax_name}, {make_attribute_int("axis", 3)});
                    ctx.add_node("Transpose", {softmax_name}, {output_name}, {make_attribute_ints("perm", {0, 3, 1, 2})});
                    ctx.current_name = output_name;
                    ctx.current_shape = output_shape;
                }
                else if (mode == operation_mode::PLANE_WISE)
                {
                    if (ctx.current_shape.size() != 4)
                        throw dlib::error("ONNX softmaxm export expected a 4D input tensor.");
                    const auto output_shape = ctx.current_shape;
                    const std::vector<int64_t> flat_plane_shape = {
                        output_shape[0],
                        output_shape[1],
                        output_shape[2]*output_shape[3]
                    };
                    export_reshape(ctx, flat_plane_shape, "softmaxm_flatten");
                    const std::string output_name = ctx.next_value_name("softmax");
                    ctx.add_node("Softmax", {ctx.current_name}, {output_name}, {make_attribute_int("axis", 2)});
                    ctx.current_name = output_name;
                    ctx.current_shape = flat_plane_shape;
                    export_reshape(ctx, output_shape, "softmaxm");
                }
                else
                {
                    throw dlib::error("ONNX export doesn't support this dlib softmax mode.");
                }
            }

            inline void export_layer(export_context& ctx, const softmax_all_&)
            {
                const auto output_shape = ctx.current_shape;
                const int64_t values_per_sample = static_cast<int64_t>(
                    element_count(output_shape)/static_cast<size_t>(output_shape[0])
                );
                const std::vector<int64_t> flat_shape = {output_shape[0], values_per_sample};
                export_reshape(ctx, flat_shape, "softmax_all_flatten");
                const std::string output_name = ctx.next_value_name("softmax");
                ctx.add_node("Softmax", {ctx.current_name}, {output_name}, {make_attribute_int("axis", 1)});
                ctx.current_name = output_name;
                ctx.current_shape = flat_shape;

                export_reshape(ctx, output_shape, "softmax_all");
            }

            template <long offset, long k, long nr, long nc>
            void export_layer(export_context& ctx, const extract_<offset, k, nr, nc>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX extract export expected a 4D input tensor.");
                const std::vector<int64_t> output_shape = {ctx.current_shape[0], k, nr, nc};
                if (static_cast<size_t>(offset) + element_count(output_shape)/static_cast<size_t>(output_shape[0]) >
                    element_count(ctx.current_shape)/static_cast<size_t>(ctx.current_shape[0]))
                    throw dlib::error("ONNX extract export found an extraction range outside the input tensor.");
                export_slice_flattened(ctx, offset, output_shape, "extract");
            }

            template <long offset_k, long offset_nr, long offset_nc, long k, long nr, long nc>
            void export_layer(export_context& ctx, const slice_<offset_k, offset_nr, offset_nc, k, nr, nc>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX slice export expected a 4D input tensor.");
                const std::vector<int64_t> output_shape = {ctx.current_shape[0], k, nr, nc};
                if (offset_k + k > ctx.current_shape[1] ||
                    offset_nr + nr > ctx.current_shape[2] ||
                    offset_nc + nc > ctx.current_shape[3])
                    throw dlib::error("ONNX slice export found a slice range outside the input tensor.");
                export_four_dimensional_slice(
                    ctx,
                    {0, offset_k, offset_nr, offset_nc},
                    {ctx.current_shape[0], offset_k + k, offset_nr + nr, offset_nc + nc},
                    output_shape,
                    "slice"
                );
            }

            template <long long row_stride, long long col_stride>
            void export_layer(export_context& ctx, const reorg_<row_stride, col_stride>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX reorg export expected a 4D input tensor.");
                if (ctx.current_shape[2] % row_stride != 0 || ctx.current_shape[3] % col_stride != 0)
                    throw dlib::error("ONNX reorg export requires NR and NC to be divisible by the row and column strides.");

                const std::vector<int64_t> input_shape = ctx.current_shape;
                const std::vector<int64_t> split_shape = {
                    input_shape[0],
                    input_shape[1],
                    input_shape[2]/row_stride,
                    row_stride,
                    input_shape[3]/col_stride,
                    col_stride
                };
                const std::vector<int64_t> output_shape = {
                    input_shape[0],
                    input_shape[1]*row_stride*col_stride,
                    input_shape[2]/row_stride,
                    input_shape[3]/col_stride
                };
                export_reshape(ctx, split_shape, "reorg_split");
                const std::string transpose_name = ctx.next_value_name("reorg_transpose");
                ctx.add_node("Transpose", {ctx.current_name}, {transpose_name}, {make_attribute_ints("perm", {0, 3, 5, 1, 2, 4})});
                ctx.current_name = transpose_name;
                ctx.current_shape = {
                    input_shape[0],
                    row_stride,
                    col_stride,
                    input_shape[1],
                    input_shape[2]/row_stride,
                    input_shape[3]/col_stride
                };
                export_reshape(ctx, output_shape, "reorg");
            }

            inline void export_layer(export_context& ctx, const transpose_&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX transpose export expected a 4D input tensor.");
                const std::string output_name = ctx.next_value_name("transpose");
                ctx.add_node("Transpose", {ctx.current_name}, {output_name}, {make_attribute_ints("perm", {0, 1, 3, 2})});
                ctx.current_name = output_name;
                std::swap(ctx.current_shape[2], ctx.current_shape[3]);
            }

            inline void export_layer(export_context& ctx, const positional_encodings_& layer)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX positional_encodings export expected a 4D input tensor.");
                const auto& pe = layer.get_positional_encodings();
                if (pe.size() != element_count(ctx.current_shape))
                    throw dlib::error("ONNX positional_encodings export found uninitialized or mismatched positional encodings.");
                const std::string pe_name = ctx.next_value_name("positional_encodings");
                const std::string output_name = ctx.next_value_name("positional_encodings_add");
                ctx.add_initializer(pe_name, ctx.current_shape, pe.host(), pe.size());
                ctx.add_node("Add", {ctx.current_name, pe_name}, {output_name});
                ctx.current_name = output_name;
            }

            template <unsigned long num_embeddings, unsigned long embedding_dim>
            void export_layer(export_context& ctx, const embeddings_<num_embeddings, embedding_dim>& layer)
            {
                if (ctx.current_shape.size() != 4 || ctx.current_shape[3] != 1)
                    throw dlib::error("ONNX embeddings export expected a 4D token tensor with NC == 1.");
                const auto& embeddings = layer.get_embeddings();
                const size_t weight_count = static_cast<size_t>(layer.get_num_embeddings()*layer.get_embedding_dim());
                if (embeddings.size() != weight_count)
                    throw dlib::error("ONNX embeddings export found uninitialized dlib embedding parameters.");

                const std::vector<int64_t> gather_shape = {
                    ctx.current_shape[0],
                    ctx.current_shape[1],
                    ctx.current_shape[2],
                    1,
                    static_cast<int64_t>(layer.get_embedding_dim())
                };
                const std::vector<int64_t> output_shape = {
                    ctx.current_shape[0],
                    ctx.current_shape[1],
                    ctx.current_shape[2],
                    static_cast<int64_t>(layer.get_embedding_dim())
                };
                const std::string embeddings_name = ctx.next_value_name("embeddings_table");
                ctx.add_initializer(
                    embeddings_name,
                    {static_cast<int64_t>(layer.get_num_embeddings()), static_cast<int64_t>(layer.get_embedding_dim())},
                    embeddings.host(),
                    weight_count
                );

                const std::string ids_name = ctx.next_value_name("embeddings_ids");
                const std::string clipped_ids_name = ctx.next_value_name("embeddings_clipped_ids");
                const std::string gathered_name = ctx.next_value_name("embeddings_gather");
                const std::string valid_low_name = ctx.next_value_name("embeddings_valid_low");
                const std::string valid_high_name = ctx.next_value_name("embeddings_valid_high");
                const std::string valid_name = ctx.next_value_name("embeddings_valid");
                const std::string valid_shape_name = add_shape_initializer(
                    ctx,
                    "embeddings_valid_shape",
                    {ctx.current_shape[0], ctx.current_shape[1], ctx.current_shape[2], 1, 1}
                );
                const std::string valid_reshaped_name = ctx.next_value_name("embeddings_valid_reshaped");
                const std::string valid_float_name = ctx.next_value_name("embeddings_valid_float");
                const std::string masked_name = ctx.next_value_name("embeddings_masked");
                const std::string min_id_name = add_int64_initializer(ctx, "embeddings_min_id", {0});
                const std::string max_id_name = add_int64_initializer(
                    ctx,
                    "embeddings_max_id",
                    {static_cast<int64_t>(layer.get_num_embeddings() - 1)}
                );
                const std::string upper_bound_name = add_int64_initializer(
                    ctx,
                    "embeddings_upper_bound",
                    {static_cast<int64_t>(layer.get_num_embeddings())}
                );

                ctx.add_node("Cast", {ctx.current_name}, {ids_name}, {make_attribute_int("to", INT64)});
                ctx.add_node("Clip", {ids_name, min_id_name, max_id_name}, {clipped_ids_name});
                ctx.add_node("Gather", {embeddings_name, clipped_ids_name}, {gathered_name}, {make_attribute_int("axis", 0)});
                ctx.add_node("GreaterOrEqual", {ids_name, min_id_name}, {valid_low_name});
                ctx.add_node("Less", {ids_name, upper_bound_name}, {valid_high_name});
                ctx.add_node("And", {valid_low_name, valid_high_name}, {valid_name});
                ctx.add_node("Reshape", {valid_name, valid_shape_name}, {valid_reshaped_name});
                ctx.add_node("Cast", {valid_reshaped_name}, {valid_float_name}, {make_attribute_int("to", FLOAT)});
                ctx.add_node("Mul", {gathered_name, valid_float_name}, {masked_name});
                ctx.current_name = masked_name;
                ctx.current_shape = gather_shape;
                export_reshape(ctx, output_shape, "embeddings");
            }

            template <long diag, typename value_tag, long num, long den>
            void export_layer(export_context& ctx, const tril_<diag, value_tag, num, den>&)
            {
                if (ctx.current_shape.size() != 4)
                    throw dlib::error("ONNX tril export expected a 4D input tensor.");

                float diag_value = 0;
                if (std::is_same<value_tag, neg_infinity_tag>::value)
                    diag_value = -std::numeric_limits<float>::infinity();
                else if (std::is_same<value_tag, zero_tag>::value)
                    diag_value = 0;
                else
                    diag_value = static_cast<float>(num)/static_cast<float>(den);

                const std::string diag_name = add_int64_initializer(ctx, "tril_diag", {diag});
                if (diag_value == 0)
                {
                    const std::string lower_name = ctx.next_value_name("tril");
                    ctx.add_node("Trilu", {ctx.current_name, diag_name}, {lower_name}, {
                        make_attribute_int("upper", 0)
                    });
                    ctx.current_name = lower_name;
                    return;
                }

                const std::string shape_name = ctx.next_value_name("tril_shape");
                const std::string ones_name = ctx.next_value_name("tril_ones");
                const std::string lower_ones_name = ctx.next_value_name("tril_lower_ones");
                const std::string one_name = add_scalar_initializer(ctx, "tril_one", 1);
                const std::string condition_name = ctx.next_value_name("tril_condition");
                const std::string fill_name = ctx.next_value_name("tril_fill");
                const std::string output_name = ctx.next_value_name("tril");

                ctx.add_node("Shape", {ctx.current_name}, {shape_name});
                ctx.add_node("ConstantOfShape", {shape_name}, {ones_name}, {
                    make_attribute_tensor("value", make_one_element_tensor_attribute(1))
                });
                ctx.add_node("Trilu", {ones_name, diag_name}, {lower_ones_name}, {
                    make_attribute_int("upper", 0)
                });
                ctx.add_node("Equal", {lower_ones_name, one_name}, {condition_name});
                ctx.add_node("ConstantOfShape", {shape_name}, {fill_name}, {
                    make_attribute_tensor("value", make_one_element_tensor_attribute(diag_value))
                });
                ctx.add_node("Where", {condition_name, ctx.current_name, fill_name}, {output_name});
                ctx.current_name = output_name;
            }

            template <typename layer_type>
            typename std::enable_if<!std::is_base_of<multiply_, layer_type>::value>::type export_layer(export_context&, const layer_type&)
            {
                throw dlib::error(std::string("ONNX export doesn't support this dlib layer: ") + typeid(layer_type).name());
            }

            class export_visitor
            {
            public:
                explicit export_visitor(export_context& ctx_) : ctx(ctx_) {}

                template <typename input_layer_type>
                void operator()(size_t, const input_layer_type& input_layer) const
                {
                    validate_dlib_input_mode(ctx.options, input_layer);
                    export_dlib_input_layer(ctx, input_layer);
                }

                template <typename LOSS_DETAILS, typename SUBNET>
                void operator()(size_t, const add_loss_layer<LOSS_DETAILS, SUBNET>&) const
                {
                    // Loss layers are inference-only pass-throughs in ONNX export.
                }

                template <unsigned long ID, typename SUBNET, typename enabled>
                void operator()(size_t, const add_tag_layer<ID, SUBNET, enabled>&) const
                {
                    ctx.set_tag(ID);
                }

                template <template<typename> class TAG_TYPE, typename SUBNET>
                void operator()(size_t, const add_skip_layer<TAG_TYPE, SUBNET>&) const
                {
                    ctx.use_tag(tag_id<TAG_TYPE>::id);
                }

                template <typename LAYER_DETAILS, typename SUBNET, typename enabled>
                void operator()(size_t, const add_layer<LAYER_DETAILS, SUBNET, enabled>& net) const
                {
                    export_layer(ctx, net.layer_details());
                }

            private:
                export_context& ctx;
            };

            template <typename net_type>
            void export_network(export_context& ctx, net_type& net)
            {
                visit_layers_backwards(net, export_visitor(ctx));
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    void net_to_onnx (
        net_type& net,
        std::ostream& out,
        const onnx_export_options& options = onnx_export_options()
    )
    {
        onnx_export_options normalized_options = options;
        normalized_options.input_tensor_shape = impl::onnx::normalize_input_shape(options, net.input_layer());

        impl::onnx::validate_export_options(normalized_options);
        impl::onnx::validate_dlib_input_mode(normalized_options, net.input_layer());
        impl::onnx::setup_network_for_export(net, normalized_options, net.input_layer());

        impl::onnx::export_context ctx(normalized_options);
        impl::onnx::export_network(ctx, net);
        ctx.finish();
        ctx.save(out);
        if (!out)
            throw dlib::error("Error while writing ONNX model to output stream.");
    }

    template <typename net_type>
    void net_to_onnx (
        net_type& net,
        const std::string& filename,
        const onnx_export_options& options = onnx_export_options()
    )
    {
        std::ofstream fout(filename, std::ios::binary);
        if (!fout)
            throw dlib::error("Unable to open ONNX output file: " + filename);
        net_to_onnx(net, fout, options);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_ONNX_H_
