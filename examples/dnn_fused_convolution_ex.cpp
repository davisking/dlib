#include "resnet.h"

#include <dlib/dnn.h>
#include <dlib/image_io.h>

using namespace std;
using namespace dlib;

using fms = chrono::duration<float, milli>;

ostream& operator<<(ostream& out, const tensor& t)
{
    out << t.num_samples() << 'x' << t.k() << 'x' << t.nr() << 'x' << t.nc();
    return out;
}

int main(const int argc, const char** argv)
try
{
    resnet::infer_50 net1;
    std::vector<std::string> labels;
    deserialize("resnet50_1000_imagenet_classifier.dnn") >> net1 >> labels;

    resnet::infer_50 net2;
    net2 = net1;

    std::vector<matrix<rgb_pixel>> image(1);
    load_image(image.front(), "elephant.jpg");
    resizable_tensor x;
    net1.to_tensor(image.begin(), image.end(), x);
    cout << x << endl;

    resizable_tensor out1 = net1.forward(x);
    const auto& label1 = labels[index_of_max(mat(out1))];
    resizable_tensor probs(out1);
    tt::softmax(probs, out1);
    cout << "pred1: " << label1 << " (" << max(mat(probs)) << ")" << endl;

    {
        running_stats<float> rs;
        for (int i = 0; i < 1000; ++i)
        {
            const auto t0 = chrono::steady_clock::now();
            net1.forward(x);
            const auto t1 = chrono::steady_clock::now();
            rs.add(chrono::duration_cast<fms>(t1 - t0).count());
        }
        cout << "affine: " << rs.mean() << " ± " << rs.stddev() << " ms" << endl;
        net1.clean();
        ostringstream sout;
        serialize(net1, sout);
        cout << "size: " << sout.str().size() / 1024.0 / 1024.0 << " MiB" << endl;
    }

    // fuse the convolutions in the network
    fuse_layers(net2);
    resizable_tensor out2 = net2.forward(x);
    const auto& label2 = labels[index_of_max(mat(out2))];
    tt::softmax(probs, out2);
    cout << "pred2: " << label2 << " (" << max(mat(probs)) << ")" << endl;

    {
        running_stats<float> rs;
        for (int i = 0; i < 1000; ++i)
        {
            const auto t0 = chrono::steady_clock::now();
            net2.forward(x);
            const auto t1 = chrono::steady_clock::now();
            rs.add(chrono::duration_cast<fms>(t1 - t0).count());
        }
        cout << "fused:  " << rs.mean() << " ± " << rs.stddev() << " ms" << endl;
        net2.clean();
        ostringstream sout;
        serialize(net2, sout);
        cout << "size: " << sout.str().size() / 1024.0 / 1024.0 << " MiB" << endl;
    }

    cout << "max abs difference: " << max(abs(mat(out1) - mat(out2))) << endl;
    DLIB_CASSERT(max(abs(mat(out1) - mat(out2))) < 1e-2);
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
