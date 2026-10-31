// Copyright (C) 2026
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_ONNX_ABSTRACT_H_
#ifdef DLIB_DNn_ONNX_ABSTRACT_H_

#include <cstdint>
#include <iosfwd>
#include <string>
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
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object contains options for exporting a dlib DNN to ONNX.

        THREAD SAFETY
            Objects of this type are ordinary value objects.

        input_tensor_shape
            The fixed NCHW tensor shape [N,K,NR,NC] used for export: number of
            samples, channels, rows, and columns.  All dimensions must be positive.
            For input_rgb_image_sized<NR,NC>, an empty shape selects [1,3,NR,NC].
            Other input layers require an explicit shape.

        input_mode
            dnn_tensor:
                The ONNX graph input is the already-normalized NCHW tensor that
                can be passed to net.forward().

            dlib_input_layer:
                The ONNX graph input is a fixed-shape float NCHW tensor in the
                native value domain of the supported dlib input layer, and supported
                dlib input preprocessing is emitted into the graph.  For
                input_rgb_image and input_rgb_image_sized this means raw RGB
                channel values are converted by subtracting the layer means and
                dividing by 256.  For input<matrix<T>> and input<array2d<T>>,
                unsigned char pixel values are divided by 256 and other scalar
                values are passed through.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    void net_to_onnx (
        net_type& net,
        std::ostream& out,
        const onnx_export_options& options = onnx_export_options()
    );
    /*!
        requires
            - options.input_tensor_shape == [N,K,NR,NC], unless this shape can
              be inferred from net.input_layer().
            - all dimensions in options.input_tensor_shape are positive and match
              net's input layer, including its channel count and any fixed row or
              column dimensions.  RGB image input layers require K == 3.
            - options.input_name, options.output_name, and options.graph_name
              are non-empty.
            - options.opset_version == 17.
        ensures
            - Writes an ONNX ModelProto representing net to out.
            - The ONNX graph input is a fixed-shape float NCHW tensor, including
              a fixed number of samples.
            - Loss layers are exported as inference pass-throughs.  The graph
              returns the tensor entering the loss layer, without applying its
              label or detection decoding.
            - Runs net on zero-valued samples, applying its input preprocessing,
              to initialize parameters and determine tensor shapes.  This can
              modify net's parameters and intermediate outputs.
        throws
            - dlib::error if net contains an unsupported layer or out reports
              a failed stream state after writing the model.
            - Exceptions raised by out are propagated.

        Supported layer families:
            - input_tensor, input_rgb_image, input_rgb_image_sized, input<matrix<T>>,
              and input<array2d<T>> in dnn_tensor mode and dlib_input_layer
              mode.
            - con_, cont_, fc_, linear_, affine_, bn_ converted through affine_,
              relu_, prelu_, leaky_relu_, sig_, htan_, clipped_relu_, elu_,
              gelu_, smelu_, silu_, mish_, multiply_ and layers derived from
              multiply_, max_pool_, avg_pool_,
              avg_pool_everything, upsample_, resize_to_, reshape_to_,
              resize_prev_to_tagged_, add_prev_, mult_prev_ with fixed 4D
              zero-padded shapes, l2normalize_,
              multm_prev_ with compatible fixed matrix dimensions, scale_,
              scale_prev_, concat_, tag layers, skip layers, channel-wise
              softmax_, softmaxm, softmax_all_, extract_, slice_, reorg_,
              transpose_, layer_norm_, rms_norm_, embeddings_,
              positional_encodings_, tril_, tril_mask, and tril_diag where
              fixed input shapes make the exported ONNX graph exact.

        Unsupported layer families include:
            - input_rgb_image_pair, image pyramid inputs, dropout_, detection
              postprocessing/loss output decoding, adaptive_computation_time_,
              and custom user layers.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    void net_to_onnx (
        net_type& net,
        const std::string& filename,
        const onnx_export_options& options = onnx_export_options()
    );
    /*!
        requires
            - The requirements of net_to_onnx(net, out, options) are satisfied.
        ensures
            - Opens filename for binary output and writes the ONNX model to it,
              with the same export behavior as net_to_onnx(net, out, options).
        throws
            - dlib::error if the file cannot be opened or written, or net contains
              an unsupported layer.  If the file was opened, its previous contents
              have been discarded, even when export fails.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_ONNX_ABSTRACT_H_
