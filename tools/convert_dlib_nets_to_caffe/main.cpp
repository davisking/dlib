
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

    string detail_name; // The name of the tag inside the layer tag. e.g. fc, con, max_pool, input_rgb_image.
    std::map<string,double> attributes;
    matrix<double> params;
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
    const string& xml_filename
);

// ----------------------------------------------------------------------------------------

template <typename iterator>
string find_layer_caffe_name (
    iterator i,
    long tag_id
)
/*!
    requires
        - i is an iterator pointing to a layer in the list of layers produced by parse_dlib_xml().
        - i is not an input layer.
    ensures
        - if (tag_id == -1) then
            - returns the caffe string name for the previous layer to layer i.
        - else
            - returns the caffe string name for the previous layer to layer i with the given tag_id.
!*/
{
    if (tag_id == -1)
    {
        return (i-1)->caffe_layer_name();
    }
    else
    {
        while(true)
        {
            i--;
            // if we hit the end of the network before we found what we were looking for
            if (i->tag_id == tag_id)
                return i->caffe_layer_name();
            if (i->type == "input")
                throw dlib::error("Network definition is bad, a layer wanted to skip back to a non-existing layer.");
        }
    }
}

template <typename iterator>
string find_input_layer_caffe_name (iterator i) { return find_layer_caffe_name(i, i->skip_id); }

// ----------------------------------------------------------------------------------------

template <typename EXP>
void print_as_np_array(std::ostream& out, const matrix_exp<EXP>& m)
{
    out << "np.array([";
    for (auto x : m)
        out << x << ",";
    out << "], dtype='float32')";
}

// ----------------------------------------------------------------------------------------

void convert_dlib_xml_to_cafffe_python_code(
    const string& xml_filename
)
{
    const auto layers = parse_dlib_xml(xml_filename);

    cout << "import caffe " << endl;
    cout << "from caffe import layers as L, params as P" << endl;
    cout << "import numpy as np" << endl;

    // dlib nets don't commit to a batch size, so just use 1 as the default
    cout << "batch_size = 1;" << endl;
    if (layers.back().detail_name == "input_rgb_image")
    {
        cout << "input_nr = 28; #WARNING, the source dlib network didn't commit to a specific input size, so we put 28 here as a default." << endl;
        cout << "input_nc = 28; #WARNING, the source dlib network didn't commit to a specific input size, so we put 28 here as a default." << endl;
        cout << "input_k = 3;" << endl;
    }
    else if (layers.back().detail_name == "input_rgb_image_sized")
    {
        cout << "input_nr = " << layers.back().attribute("nr") << ";" << endl;
        cout << "input_nc = " << layers.back().attribute("nc") << ";" << endl;
        cout << "input_k = 3;" << endl;
    }
    else if (layers.back().detail_name == "input")
    {
        cout << "input_nr = 28; #WARNING, the source dlib network didn't commit to a specific input size, so we put 28 here as a default." << endl;
        cout << "input_nc = 28; #WARNING, the source dlib network didn't commit to a specific input size, so we put 28 here as a default." << endl;
        cout << "input_k = 1;" << endl;
    }
    else
    {
        throw dlib::error("No known transformation from dlib's " + layers.back().detail_name + " layer to caffe.");
    }

    cout << "def make_netspec():" << endl;
    cout << "    # For reference, the only \"documentation\" about caffe layer parameters seems to be this page:\n";
    cout << "    # https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto\n" << endl;
    cout << "    n = caffe.NetSpec(); " << endl;
    cout << "    n.data,n.label = L.MemoryData(batch_size=batch_size, channels=input_k, height=input_nr, width=input_nc, ntop=2)" << endl;
    // iterate the layers starting with the input layer
    for (auto i = layers.rbegin(); i != layers.rend(); ++i)
    {
        // skip input and loss layers
        if (i->type == "loss" || i->type == "input")
            continue;


        if (i->detail_name == "con")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.Convolution(n." << find_input_layer_caffe_name(i);
            cout << ", num_output=" << i->attribute("num_filters");
            cout << ", kernel_w=" << i->attribute("nc");
            cout << ", kernel_h=" << i->attribute("nr");
            cout << ", stride_w=" << i->attribute("stride_x");
            cout << ", stride_h=" << i->attribute("stride_y");
            cout << ", pad_w=" << i->attribute("padding_x");
            cout << ", pad_h=" << i->attribute("padding_y");
            cout << ");\n";
        }
        else if (i->detail_name == "relu")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.ReLU(n." << find_input_layer_caffe_name(i);
            cout << ");\n";
        }
        else if (i->detail_name == "max_pool")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.Pooling(n." << find_input_layer_caffe_name(i);
            cout << ", pool=P.Pooling.MAX"; 
            if (i->attribute("nc")==0)
            {
                cout << ", global_pooling=True";
            }
            else
            {
                cout << ", kernel_w=" << i->attribute("nc");
                cout << ", kernel_h=" << i->attribute("nr");
            }
            if (i->attribute("padding_x") != 0 || i->attribute("padding_y") != 0)
            {
                throw dlib::error("dlib and caffe implement pooling with non-zero padding differently, so you can't convert a "
                    "network with such pooling layers.");
            }

            cout << ", stride_w=" << i->attribute("stride_x");
            cout << ", stride_h=" << i->attribute("stride_y");
            cout << ", pad_w=" << i->attribute("padding_x");
            cout << ", pad_h=" << i->attribute("padding_y");
            cout << ");\n";
        }
        else if (i->detail_name == "avg_pool")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.Pooling(n." << find_input_layer_caffe_name(i);
            cout << ", pool=P.Pooling.AVE"; 
            if (i->attribute("nc")==0)
            {
                cout << ", global_pooling=True";
            }
            else
            {
                cout << ", kernel_w=" << i->attribute("nc");
                cout << ", kernel_h=" << i->attribute("nr");
            }
            if (i->attribute("padding_x") != 0 || i->attribute("padding_y") != 0)
            {
                throw dlib::error("dlib and caffe implement pooling with non-zero padding differently, so you can't convert a "
                    "network with such pooling layers.");
            }

            cout << ", stride_w=" << i->attribute("stride_x");
            cout << ", stride_h=" << i->attribute("stride_y");
            cout << ", pad_w=" << i->attribute("padding_x");
            cout << ", pad_h=" << i->attribute("padding_y");
            cout << ");\n";
        }
        else if (i->detail_name == "fc")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.InnerProduct(n." << find_input_layer_caffe_name(i);
            cout << ", num_output=" << i->attribute("num_outputs");
            cout << ", bias_term=True";
            cout << ");\n";
        }
        else if (i->detail_name == "fc_no_bias")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.InnerProduct(n." << find_input_layer_caffe_name(i);
            cout << ", num_output=" << i->attribute("num_outputs");
            cout << ", bias_term=False";
            cout << ");\n";
        }
        else if (i->detail_name == "bn_con" || i->detail_name == "bn_fc")
        {
            throw dlib::error("Conversion from dlib's batch norm layers to caffe's isn't supported.  Instead, "
                "you should put your network into 'test mode' by switching batch norm layers to affine layers.");
        }
        else if (i->detail_name == "affine_con")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.Scale(n." << find_input_layer_caffe_name(i);
            cout << ", axis=1";
            cout << ", bias_term=True";
            cout << ");\n";
        }
        else if (i->detail_name == "affine_fc")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.Scale(n." << find_input_layer_caffe_name(i);
            cout << ", axis=3";
            cout << ", bias_term=True";
            cout << ");\n";
        }
        else if (i->detail_name == "add_prev")
        {
            cout << "    n." << i->caffe_layer_name() << " = L.Eltwise(n." << find_input_layer_caffe_name(i);
            cout << ", n." << find_layer_caffe_name(i, i->attribute("tag"));
            cout << ", operation=P.Eltwise.SUM";
            cout << ");\n";
        }
        else
        {
            throw dlib::error("No known transformation from dlib's " + i->detail_name + " layer to caffe.");
        }
    }
    cout << "    return n.to_proto();\n\n" << endl;


    // -------------------------
    // -------------------------


    cout << "def save_as_caffe_model(def_file, weights_file):\n";
    cout << "    with open(def_file, 'w') as f: f.write(str(make_netspec()));\n";
    cout << "    net = caffe.Net(def_file, caffe.TEST);\n";
    cout << "    set_network_weights(net);\n";
    cout << "    net.save(weights_file);\n\n";


    // -------------------------
    // -------------------------


    cout << "def set_network_weights(net):\n";
    cout << "    # populate network parameters\n";
    // iterate the layers starting with the input layer
    for (auto i = layers.rbegin(); i != layers.rend(); ++i)
    {
        // skip input and loss layers
        if (i->type == "loss" || i->type == "input")
            continue;


        if (i->detail_name == "con")
        {
            const long num_filters = i->attribute("num_filters");
            matrix<double> weights = trans(rowm(i->params,range(0,i->params.size()-num_filters-1)));
            matrix<double> biases  = trans(rowm(i->params,range(i->params.size()-num_filters, i->params.size()-1)));

            // main filter weights
            cout << "    p = "; print_as_np_array(cout,weights); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";

            // biases
            cout << "    p = "; print_as_np_array(cout,biases); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][1].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][1].data[:] = p;\n";
        }
        else if (i->detail_name == "fc")
        {
            matrix<double> weights = trans(rowm(i->params, range(0,i->params.nr()-2))); 
            matrix<double> biases  = rowm(i->params, i->params.nr()-1); 

            // main filter weights
            cout << "    p = "; print_as_np_array(cout,weights); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";

            // biases
            cout << "    p = "; print_as_np_array(cout,biases); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][1].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][1].data[:] = p;\n";
        }
        else if (i->detail_name == "fc_no_bias")
        {
            matrix<double> weights = trans(i->params); 

            // main filter weights
            cout << "    p = "; print_as_np_array(cout,weights); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";
        }
        else if (i->detail_name == "affine_con" || i->detail_name == "affine_fc")
        {
            const long dims = i->params.size()/2;
            matrix<double> gamma = trans(rowm(i->params,range(0,dims-1)));
            matrix<double> beta  = trans(rowm(i->params,range(dims, 2*dims-1)));

            // set gamma weights
            cout << "    p = "; print_as_np_array(cout,gamma); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][0].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][0].data[:] = p;\n";

            // set beta weights 
            cout << "    p = "; print_as_np_array(cout,beta); cout << ";\n";
            cout << "    p.shape = net.params['"<<i->caffe_layer_name()<<"'][1].data.shape;\n";
            cout << "    net.params['"<<i->caffe_layer_name()<<"'][1].data[:] = p;\n";
        }
    }

}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    cout.precision(9);
    // TODO, write out to multiple files or just process one file at a time.  
    for (int i = 1; i < argc; ++i)
        convert_dlib_xml_to_cafffe_python_code(argv[i]);

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

std::vector<layer> parse_dlib_xml(
    const string& xml_filename
)
{
    doc_handler dh;
    parse_xml(xml_filename, dh);
    if (dh.layers.size() == 0)
        throw dlib::error("No layers found in XML file!");

    if (dh.layers.back().type != "input")
        throw dlib::error("The network in the XML file is missing an input layer!");

    return dh.layers;
}

// ----------------------------------------------------------------------------------------

