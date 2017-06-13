
#include <dlib/xml_parser.h>
#include <dlib/matrix.h>
#include <fstream>
#include <vector>
#include <stack>
#include <set>
#include <dlib/string.h>

using namespace std;
using namespace dlib;


// ----------------------------------------------------------------------------------------

// Only these computational layers have parameters
const std::set<string> comp_tags_with_params = {"fc", "fc_no_bias", "con", "affine_con", "affine_fc", "affine", "prelu"};

struct layer
{
    string type; // comp, loss, or input
    int idx;

    matrix<long,4,1> output_tensor_shape; // (N,K,NR,NC)

    string detail_name; // The name of the tag inside the layer tag. e.g. fc, con, max_pool, input_rgb_image.
    std::map<string,double> attributes;
    matrix<float> params;
    long tag_id = -1;   // If this isn't -1 then it means this layer was tagged, e.g. wrapped with tag2<> giving tag_id==2
    long skip_id = -1;  // If this isn't -1 then it means this layer draws its inputs from
                        // the most recent layer with tag_id==skip_id rather than its immediate predecessor. 

    double attribute (const string& key) const
    {
        auto i = attributes.find(key);
        if (i != attributes.end())
            return i->second;
        else
            throw dlib::error("Layer doesn't have the requested attribute '" + key + "'.");
    }

    string caffe_layer_name() const 
    { 
        if (type == "input")
            return "data";
        else
            return detail_name+to_string(idx);
    }
};

// ----------------------------------------------------------------------------------------

std::vector<layer> parse_dlib_xml(
    const matrix<long,4,1>& input_tensor_shape, 
    const string& xml_filename
);

// ----------------------------------------------------------------------------------------

template <typename iterator>
const layer& find_layer (
    iterator i,
    long tag_id
)
/*!
    requires
        - i is a reverse iterator pointing to a layer in the list of layers produced by parse_dlib_xml().
        - i is not an input layer.
    ensures
        - if (tag_id == -1) then
            - returns the previous layer (i.e. closer to the input) to layer i.
        - else
            - returns the previous layer (i.e. closer to the input) to layer i with the
              given tag_id.
!*/
{
    if (tag_id == -1)
    {
        return *(i-1);
    }
    else
    {
        while(true)
        {
            i--;
            // if we hit the end of the network before we found what we were looking for
            if (i->tag_id == tag_id)
                return *i;
            if (i->type == "input")
                throw dlib::error("Network definition is bad, a layer wanted to skip back to a non-existing layer.");
        }
    }
}

template <typename iterator>
const layer& find_input_layer (iterator i) { return find_layer(i, i->skip_id); }

template <typename iterator>
string find_layer_caffe_name (
    iterator i,
    long tag_id
)
{
    return find_layer(i,tag_id).caffe_layer_name();
}

template <typename iterator>
string find_input_layer_caffe_name (iterator i) { return find_input_layer(i).caffe_layer_name(); }

// ----------------------------------------------------------------------------------------

template <typename iterator>
void compute_caffe_padding_size_for_pooling_layer(
    const iterator& i,
    long& pad_x,
    long& pad_y
)
/*!
    requires
        - i is a reverse iterator pointing to a layer in the list of layers produced by parse_dlib_xml().
        - i is not an input layer.
    ensures
        - Caffe is funny about how it computes the output sizes from pooling layers.
          Rather than using the normal formula for output row/column sizes used by all the
          other layers (and what dlib uses everywhere), 
            floor((bottom_size + 2*pad - kernel_size) / stride) + 1
          it instead uses:
            ceil((bottom_size + 2*pad - kernel_size) / stride) + 1

          These are the same except when the stride!=1.  In that case we need to figure out
          how to change the padding value so that the output size of the caffe padding
          layer will match the output size of the dlib padding layer.   That is what this
          function does.
!*/
{
    const long dlib_output_nr = i->output_tensor_shape(2);
    const long dlib_output_nc = i->output_tensor_shape(3);
    const long bottom_nr = find_input_layer(i).output_tensor_shape(2);
    const long bottom_nc = find_input_layer(i).output_tensor_shape(3);
    const long padding_x = (long)i->attribute("padding_x");
    const long padding_y = (long)i->attribute("padding_y");
    const long stride_x = (long)i->attribute("stride_x");
    const long stride_y = (long)i->attribute("stride_y");
    long kernel_w = i->attribute("nc");
    long kernel_h = i->attribute("nr");

    if (kernel_w == 0)
        kernel_w = bottom_nc;
    if (kernel_h == 0)
        kernel_h = bottom_nr;

    
    // The correct padding for caffe could be anything in the range [0,padding_x].  So
    // check what gives the correct output size and use that.
    for (pad_x = 0; pad_x <= padding_x; ++pad_x)
    {
        long caffe_out_size = ceil((bottom_nc + 2.0*pad_x - kernel_w)/(double)stride_x) + 1;
        if (caffe_out_size == dlib_output_nc)
            break;
    }
    if (pad_x == padding_x+1)
    {
        std::ostringstream sout;
        sout << "No conversion between dlib pooling layer parameters and caffe pooling layer parameters found for layer " << to_string(i->idx) << endl;
        sout << "dlib_output_nc: " << dlib_output_nc << endl;
        sout << "bottom_nc:      " << bottom_nc << endl;
        sout << "padding_x:      " << padding_x << endl;
        sout << "stride_x:       " << stride_x << endl;
        sout << "kernel_w:       " << kernel_w << endl;
        sout << "pad_x:          " << pad_x << endl;
        throw dlib::error(sout.str());
    }

    for (pad_y = 0; pad_y <= padding_y; ++pad_y)
    {
        long caffe_out_size = ceil((bottom_nr + 2.0*pad_y - kernel_h)/(double)stride_y) + 1;
        if (caffe_out_size == dlib_output_nr)
            break;
    }
    if (pad_y == padding_y+1)
    {
        std::ostringstream sout;
        sout << "No conversion between dlib pooling layer parameters and caffe pooling layer parameters found for layer " << to_string(i->idx) << endl;
        sout << "dlib_output_nr: " << dlib_output_nr << endl;
        sout << "bottom_nr:      " << bottom_nr << endl;
        sout << "padding_y:      " << padding_y << endl;
        sout << "stride_y:       " << stride_y << endl;
        sout << "kernel_h:       " << kernel_h << endl;
        sout << "pad_y:          " << pad_y << endl;
        throw dlib::error(sout.str());
    }
}

// ----------------------------------------------------------------------------------------

void convert_dlib_xml_to_caffe_python_code(
    const string& xml_filename,
    const long N,
    const long K,
    const long NR,
    const long NC
)
{
    const string out_filename = left_substr(xml_filename,".") + "_dlib_to_caffe_model.py";
    const string out_weights_filename = left_substr(xml_filename,".") + "_dlib_to_caffe_model.weights";
    cout << "Writing python part of model to " << out_filename << endl;
    cout << "Writing weights part of model to " << out_weights_filename << endl;
    ofstream fout(out_filename);
    fout.precision(9);
    const auto layers = parse_dlib_xml({N,K,NR,NC}, xml_filename);


    fout << "#\n";
    fout << "# !!! This file was automatically generated by dlib's tools/convert_dlib_nets_to_caffe utility.     !!!\n";
    fout << "# !!! It contains all the information from a dlib DNN network and lets you save it as a cafe model. !!!\n";
    fout << "#\n";
    fout << "import caffe " << endl;
    fout << "from caffe import layers as L, params as P" << endl;
    fout << "import numpy as np" << endl;

    // dlib nets don't commit to a batch size, so just use 1 as the default
    fout << "\n# Input tensor dimensions" << endl;
    fout << "input_batch_size = " << N << ";" << endl;
    if (layers.back().detail_name == "input_rgb_image")
    {
        fout << "input_num_channels = 3;" << endl;
        fout << "input_num_rows = "<<NR<<";" << endl;
        fout << "input_num_cols = "<<NC<<";" << endl;
        if (K != 3)
            throw dlib::error("The dlib model requires input tensors with NUM_CHANNELS==3, but the dtoc command line specified NUM_CHANNELS=="+to_string(K));
    }
    else if (layers.back().detail_name == "input_rgb_image_sized")
    {
        fout << "input_num_channels = 3;" << endl;
        fout << "input_num_rows = " << layers.back().attribute("nr") << ";" << endl;
        fout << "input_num_cols = " << layers.back().attribute("nc") << ";" << endl;
        if (NR != layers.back().attribute("nr"))
            throw dlib::error("The dlib model requires input tensors with NUM_ROWS=="+to_string((long)layers.back().attribute("nr"))+", but the dtoc command line specified NUM_ROWS=="+to_string(NR));
        if (NC != layers.back().attribute("nc"))
            throw dlib::error("The dlib model requires input tensors with NUM_COLUMNS=="+to_string((long)layers.back().attribute("nc"))+", but the dtoc command line specified NUM_COLUMNS=="+to_string(NC));
        if (K != 3)
            throw dlib::error("The dlib model requires input tensors with NUM_CHANNELS==3, but the dtoc command line specified NUM_CHANNELS=="+to_string(K));
    }
    else if (layers.back().detail_name == "input")
    {
        fout << "input_num_channels = 1;" << endl;
        fout << "input_num_rows = "<<NR<<";" << endl;
        fout << "input_num_cols = "<<NC<<";" << endl;
        if (K != 1)
            throw dlib::error("The dlib model requires input tensors with NUM_CHANNELS==1, but the dtoc command line specified NUM_CHANNELS=="+to_string(K));
    }
    else
    {
        throw dlib::error("No known transformation from dlib's " + layers.back().detail_name + " layer to caffe.");
    }
    fout << endl;
    fout << "# Call this function to write the dlib DNN model out to file as a pair of caffe\n";
    fout << "# definition and weight files.  You can then use the network by loading it with\n";
    fout << "# this statement: \n";
    fout << "#    net = caffe.Net(def_file, weights_file, caffe.TEST);\n";
    fout << "#\n";
    fout << "def save_as_caffe_model(def_file, weights_file):\n";
    fout << "    with open(def_file, 'w') as f: f.write(str(make_netspec()));\n";
    fout << "    net = caffe.Net(def_file, caffe.TEST);\n";
    fout << "    set_network_weights(net);\n";
    fout << "    net.save(weights_file);\n\n";
    fout << "###############################################################################\n";
    fout << "#         EVERYTHING BELOW HERE DEFINES THE DLIB MODEL PARAMETERS             #\n";
    fout << "###############################################################################\n\n\n";


    // -----------------------------------------------------------------------------------
    //  The next block of code outputs python code that defines the network architecture. 
    // -----------------------------------------------------------------------------------

    fout << "def make_netspec():" << endl;
    fout << "    # For reference, the only \"documentation\" about caffe layer parameters seems to be this page:\n";
    fout << "    # https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto\n" << endl;
    fout << "    n = caffe.NetSpec(); " << endl;
    fout << "    n.data,n.label = L.MemoryData(batch_size=input_batch_size, channels=input_num_channels, height=input_num_rows, width=input_num_cols, ntop=2)" << endl;
    // iterate the layers starting with the input layer
    for (auto i = layers.rbegin(); i != layers.rend(); ++i)
    {
        // skip input and loss layers
        if (i->type == "loss" || i->type == "input")
            continue;


        if (i->detail_name == "con")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.Convolution(n." << find_input_layer_caffe_name(i);
            fout << ", num_output=" << i->attribute("num_filters");
            fout << ", kernel_w=" << i->attribute("nc");
            fout << ", kernel_h=" << i->attribute("nr");
            fout << ", stride_w=" << i->attribute("stride_x");
            fout << ", stride_h=" << i->attribute("stride_y");
            fout << ", pad_w=" << i->attribute("padding_x");
            fout << ", pad_h=" << i->attribute("padding_y");
            fout << ");\n";
        }
        else if (i->detail_name == "relu")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.ReLU(n." << find_input_layer_caffe_name(i);
            fout << ");\n";
        }
        else if (i->detail_name == "sig")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.Sigmoid(n." << find_input_layer_caffe_name(i);
            fout << ");\n";
        }
        else if (i->detail_name == "prelu")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.PReLU(n." << find_input_layer_caffe_name(i);
            fout << ", channel_shared=True"; 
            fout << ");\n";
        }
        else if (i->detail_name == "max_pool")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.Pooling(n." << find_input_layer_caffe_name(i);
            fout << ", pool=P.Pooling.MAX"; 
            if (i->attribute("nc")==0)
            {
                fout << ", global_pooling=True";
            }
            else
            {
                fout << ", kernel_w=" << i->attribute("nc");
                fout << ", kernel_h=" << i->attribute("nr");
            }

            fout << ", stride_w=" << i->attribute("stride_x");
            fout << ", stride_h=" << i->attribute("stride_y");
            long pad_x, pad_y;
            compute_caffe_padding_size_for_pooling_layer(i, pad_x, pad_y);
            fout << ", pad_w=" << pad_x;
            fout << ", pad_h=" << pad_y;
            fout << ");\n";
        }
        else if (i->detail_name == "avg_pool")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.Pooling(n." << find_input_layer_caffe_name(i);
            fout << ", pool=P.Pooling.AVE"; 
            if (i->attribute("nc")==0)
            {
                fout << ", global_pooling=True";
            }
            else
            {
                fout << ", kernel_w=" << i->attribute("nc");
                fout << ", kernel_h=" << i->attribute("nr");
            }
            if (i->attribute("padding_x") != 0 || i->attribute("padding_y") != 0)
            {
                throw dlib::error("dlib and caffe implement pooling with non-zero padding differently, so you can't convert a "
                    "network with such pooling layers.");
            }

            fout << ", stride_w=" << i->attribute("stride_x");
            fout << ", stride_h=" << i->attribute("stride_y");
            long pad_x, pad_y;
            compute_caffe_padding_size_for_pooling_layer(i, pad_x, pad_y);
            fout << ", pad_w=" << pad_x;
            fout << ", pad_h=" << pad_y;
            fout << ");\n";
        }
        else if (i->detail_name == "fc")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.InnerProduct(n." << find_input_layer_caffe_name(i);
            fout << ", num_output=" << i->attribute("num_outputs");
            fout << ", bias_term=True";
            fout << ");\n";
        }
        else if (i->detail_name == "fc_no_bias")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.InnerProduct(n." << find_input_layer_caffe_name(i);
            fout << ", num_output=" << i->attribute("num_outputs");
            fout << ", bias_term=False";
            fout << ");\n";
        }
        else if (i->detail_name == "bn_con" || i->detail_name == "bn_fc")
        {
            throw dlib::error("Conversion from dlib's batch norm layers to caffe's isn't supported.  Instead, "
                "you should put your dlib network into 'test mode' by switching batch norm layers to affine layers. "
                "Then you can convert that 'test mode' network to caffe.");
        }
        else if (i->detail_name == "affine_con")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.Scale(n." << find_input_layer_caffe_name(i);
            fout << ", bias_term=True";
            fout << ");\n";
        }
        else if (i->detail_name == "affine_fc")
        {
            fout << "    n." << i->caffe_layer_name() << " = L.Scale(n." << find_input_layer_caffe_name(i);
            fout << ", bias_term=True";
            fout << ");\n";
        }
        else if (i->detail_name == "add_prev")
        {
            auto in_shape1 = find_input_layer(i).output_tensor_shape;
            auto in_shape2 = find_layer(i,i->attribute("tag")).output_tensor_shape;
            if (in_shape1 != in_shape2)
            {
                // if only the number of channels differs then we will use a dummy layer to
                // pad with zeros.  But otherwise we will throw an error.
                if (in_shape1(0) == in_shape2(0) && 
                    in_shape1(2) == in_shape2(2) && 
                    in_shape1(3) == in_shape2(3))
                {
                    fout << "    n." << i->caffe_layer_name() << "_zeropad = L.DummyData(num=" << in_shape1(0);
                    fout << ", channels="<<std::abs(in_shape1(1)-in_shape2(1));
                    fout << ", height="<<in_shape1(2);
                    fout << ", width="<<in_shape1(3);
                    fout << ");\n";

                    string smaller_layer = find_input_layer_caffe_name(i);
                    string bigger_layer = find_layer_caffe_name(i, i->attribute("tag"));
                    if (in_shape1(1) > in_shape2(1))
                        swap(smaller_layer, bigger_layer);

                    fout << "    n." << i->caffe_layer_name() << "_concat = L.Concat(n." << smaller_layer;
                    fout << ", n." << i->caffe_layer_name() << "_zeropad";
                    fout << ");\n";

                    fout << "    n." << i->caffe_layer_name() << " = L.Eltwise(n." << i->caffe_layer_name() << "_concat";
                    fout << ", n." << bigger_layer;
                    fout << ", operation=P.Eltwise.SUM";
                    fout << ");\n";
                }
                else
                {
                    std::ostringstream sout;
                    sout << "The dlib network contained an add_prev layer (layer idx " << i->idx << ") that adds two previous ";
                    sout << "layers with different output tensor dimensions.  Caffe's equivalent layer, Eltwise, doesn't support ";
                    sout << "adding layers together with different dimensions.  In the special case where the only difference is "; 
                    sout << "in the number of channels, this converter program will add a dummy layer that outputs a tensor full of zeros ";
                    sout << "and concat it appropriately so this will work.  However, this network you are converting has tensor dimensions ";
                    sout << "different in values other than the number of channels.  In particular, here are the two tensor shapes (batch size, channels, rows, cols): ";
                    std::ostringstream sout2;
                    sout2 << wrap_string(sout.str()) << endl;
                    sout2 << trans(in_shape1);
                    sout2 << trans(in_shape2);
                    throw dlib::error(sout2.str());
                }
            }
            else
            {
                fout << "    n." << i->caffe_layer_name() << " = L.Eltwise(n." << find_input_layer_caffe_name(i);
                fout << ", n." << find_layer_caffe_name(i, i->attribute("tag"));
                fout << ", operation=P.Eltwise.SUM";
                fout << ");\n";
            }
        }
        else
        {
            throw dlib::error("No known transformation from dlib's " + i->detail_name + " layer to caffe.");
        }
    }
    fout << "    return n.to_proto();\n\n" << endl;


    // -----------------------------------------------------------------------------------
    //  The next block of code outputs python code that populates all the filter weights.
    // -----------------------------------------------------------------------------------

    ofstream fweights(out_weights_filename, ios::binary);
    fout << "def set_network_weights(net):\n";
    fout << "    # populate network parameters\n";
    fout << "    f = open('"<<out_weights_filename<<"', 'rb');\n";
    // iterate the layers starting with the input layer
    for (auto i = layers.rbegin(); i != layers.rend(); ++i)
    {
        // skip input and loss layers
        if (i->type == "loss" || i->type == "input")
            continue;


        if (i->detail_name == "con")
        {
            const long num_filters = i->attribute("num_filters");
            matrix<float> weights = trans(rowm(i->params,range(0,i->params.size()-num_filters-1)));
            matrix<float> biases  = trans(rowm(i->params,range(i->params.size()-num_filters, i->params.size()-1)));
            fweights.write((char*)&weights(0,0), weights.size()*sizeof(float));
            fweights.write((char*)&biases(0,0), biases.size()*sizeof(float));

            // main filter weights
            fout << "    p = np.fromfile(f, dtype='float32', count="<<weights.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";

            // biases
            fout << "    p = np.fromfile(f, dtype='float32', count="<<biases.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][1].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][1].data[:] = p;\n";
        }
        else if (i->detail_name == "fc")
        {
            matrix<float> weights = trans(rowm(i->params, range(0,i->params.nr()-2))); 
            matrix<float> biases  = rowm(i->params, i->params.nr()-1); 
            fweights.write((char*)&weights(0,0), weights.size()*sizeof(float));
            fweights.write((char*)&biases(0,0), biases.size()*sizeof(float));

            // main filter weights
            fout << "    p = np.fromfile(f, dtype='float32', count="<<weights.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";

            // biases
            fout << "    p = np.fromfile(f, dtype='float32', count="<<biases.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][1].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][1].data[:] = p;\n";
        }
        else if (i->detail_name == "fc_no_bias")
        {
            matrix<float> weights = trans(i->params); 
            fweights.write((char*)&weights(0,0), weights.size()*sizeof(float));

            // main filter weights
            fout << "    p = np.fromfile(f, dtype='float32', count="<<weights.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";
        }
        else if (i->detail_name == "affine_con" || i->detail_name == "affine_fc")
        {
            const long dims = i->params.size()/2;
            matrix<float> gamma = trans(rowm(i->params,range(0,dims-1)));
            matrix<float> beta  = trans(rowm(i->params,range(dims, 2*dims-1)));
            fweights.write((char*)&gamma(0,0), gamma.size()*sizeof(float));
            fweights.write((char*)&beta(0,0), beta.size()*sizeof(float));

            // set gamma weights
            fout << "    p = np.fromfile(f, dtype='float32', count="<<gamma.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";

            // set beta weights 
            fout << "    p = np.fromfile(f, dtype='float32', count="<<beta.size()<<");\n"; 
            fout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][1].data.shape;\n";
            fout << "    net.params['"<<i->caffe_layer_name()<<"'][1].data[:] = p;\n";
        }
        else if (i->detail_name == "prelu")
        {
            const double param = i->params(0);

            // main filter weights
            fout << "    tmp = net.params['"<<i->caffe_layer_name()<<"'][0].data.view();\n";
            fout << "    tmp.shape = 1;\n";
            fout << "    tmp[0] = "<<param<<";\n";
        }
    }

}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 6)
    {
        cout << "To use this program, give it an xml file generated by dlib::net_to_xml() " << endl;
        cout << "and then 4 numbers that indicate the input tensor size.  It will convert " << endl;
        cout << "the xml file into a python file that outputs a caffe model containing the dlib model." << endl;
        cout << "For example, you might run this program like this: " << endl;
        cout << "   ./dtoc lenet.xml 1 1 28 28" << endl;
        cout << "would convert the lenet.xml model into a caffe model with an input tensor of shape(1,1,28,28)" << endl;
        cout << "where the shape values are (num samples in batch, num channels, num rows, num columns)." << endl;
        return 0;
    }

    const long N = sa = argv[2];
    const long K = sa = argv[3];
    const long NR = sa = argv[4];
    const long NC = sa = argv[5];

    convert_dlib_xml_to_caffe_python_code(argv[1], N, K, NR, NC);

    return 0;
}
catch(std::exception& e)
{
    cout << "\n\n*************** ERROR CONVERTING TO CAFFE ***************\n" << e.what() << endl;
    return 1;
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

class doc_handler : public document_handler
{
public:
    std::vector<layer> layers;
    bool seen_first_tag = false;

    layer next_layer;
    std::stack<string> current_tag;
    long tag_id = -1;


    virtual void start_document (
    ) 
    { 
        layers.clear(); 
        seen_first_tag = false;
        tag_id = -1;
    }

    virtual void end_document (
    ) { }

    virtual void start_element ( 
        const unsigned long line_number,
        const std::string& name,
        const dlib::attribute_list& atts
    )
    {
        if (!seen_first_tag)
        {
            if (name != "net")
                throw dlib::error("The top level XML tag must be a 'net' tag.");
            seen_first_tag = true;
        }

        if (name == "layer")
        {
            next_layer = layer();
            if (atts["type"] == "skip")
            {
                // Don't make a new layer, just apply the tag id to the previous layer
                if (layers.size() == 0)
                    throw dlib::error("A skip layer was found as the first layer, but the first layer should be an input layer.");
                layers.back().skip_id = sa = atts["id"];
                
                // We intentionally leave next_layer empty so the end_element() callback
                // don't add it as another layer when called.
            }
            else if (atts["type"] == "tag")
            {
                // Don't make a new layer, just remember the tag id so we can apply it on
                // the next layer.
                tag_id = sa = atts["id"];
                
                // We intentionally leave next_layer empty so the end_element() callback
                // don't add it as another layer when called.
            }
            else
            {
                next_layer.idx = sa = atts["idx"];
                next_layer.type = atts["type"];
                if (tag_id != -1)
                {
                    next_layer.tag_id = tag_id;
                    tag_id = -1;
                }
            }
        }
        else if (current_tag.size() != 0 && current_tag.top() == "layer")
        {
            next_layer.detail_name = name;
            // copy all the XML tag's attributes into the layer struct
            atts.reset();
            while (atts.move_next())
                next_layer.attributes[atts.element().key()] = sa = atts.element().value();
        }

        current_tag.push(name);
    }

    virtual void end_element ( 
        const unsigned long line_number,
        const std::string& name
    )
    {
        current_tag.pop();
        if (name == "layer" && next_layer.type.size() != 0)
            layers.push_back(next_layer);
    }

    virtual void characters ( 
        const std::string& data
    )
    {
        if (current_tag.size() == 0)
            return;

        if (comp_tags_with_params.count(current_tag.top()) != 0)
        {
            istringstream sin(data);
            sin >> next_layer.params;
        }

    }

    virtual void processing_instruction (
        const unsigned long line_number,
        const std::string& target,
        const std::string& data
    )
    {
    }
};

// ----------------------------------------------------------------------------------------

void compute_output_tensor_shapes(const matrix<long,4,1>& input_tensor_shape, std::vector<layer>& layers)
{
    DLIB_CASSERT(layers.back().type == "input");
    layers.back().output_tensor_shape = input_tensor_shape;
    for (auto i = ++layers.rbegin(); i != layers.rend(); ++i)
    {
        const auto input_shape = find_input_layer(i).output_tensor_shape;
        if (i->type == "comp")
        {
            if (i->detail_name == "fc" || i->detail_name == "fc_no_bias")
            {
                long num_outputs = i->attribute("num_outputs");
                i->output_tensor_shape = {input_shape(0), num_outputs, 1, 1};
            }
            else if (i->detail_name == "con")
            {
                long num_filters = i->attribute("num_filters");
                long filter_nc = i->attribute("nc");
                long filter_nr = i->attribute("nr");
                long stride_x = i->attribute("stride_x");
                long stride_y = i->attribute("stride_y");
                long padding_x = i->attribute("padding_x");
                long padding_y = i->attribute("padding_y");
                long nr = 1+(input_shape(2) + 2*padding_y - filter_nr)/stride_y;
                long nc = 1+(input_shape(3) + 2*padding_x - filter_nc)/stride_x;
                i->output_tensor_shape = {input_shape(0), num_filters, nr, nc};
            }
            else if (i->detail_name == "max_pool" || i->detail_name == "avg_pool")
            {
                long filter_nc = i->attribute("nc");
                long filter_nr = i->attribute("nr");
                long stride_x = i->attribute("stride_x");
                long stride_y = i->attribute("stride_y");
                long padding_x = i->attribute("padding_x");
                long padding_y = i->attribute("padding_y");
                if (filter_nc != 0)
                {
                    long nr = 1+(input_shape(2) + 2*padding_y - filter_nr)/stride_y;
                    long nc = 1+(input_shape(3) + 2*padding_x - filter_nc)/stride_x;
                    i->output_tensor_shape = {input_shape(0), input_shape(1), nr, nc};
                }
                else // if we are filtering the whole input down to one thing
                {
                    i->output_tensor_shape = {input_shape(0), input_shape(1), 1, 1};
                }
            }
            else if (i->detail_name == "add_prev")
            {
                auto aux_shape = find_layer(i, i->attribute("tag")).output_tensor_shape;
                for (long j = 0; j < input_shape.size(); ++j)
                    i->output_tensor_shape(j) = std::max(input_shape(j), aux_shape(j));
            }
            else
            {
                i->output_tensor_shape = input_shape;
            }
        }
        else
        {
            i->output_tensor_shape = input_shape;
        }

    }
}

// ----------------------------------------------------------------------------------------

std::vector<layer> parse_dlib_xml(
    const matrix<long,4,1>& input_tensor_shape, 
    const string& xml_filename
)
{
    doc_handler dh;
    parse_xml(xml_filename, dh);
    if (dh.layers.size() == 0)
        throw dlib::error("No layers found in XML file!");

    if (dh.layers.back().type != "input")
        throw dlib::error("The network in the XML file is missing an input layer!");

    compute_output_tensor_shapes(input_tensor_shape, dh.layers);

    return dh.layers;
}

// ----------------------------------------------------------------------------------------

