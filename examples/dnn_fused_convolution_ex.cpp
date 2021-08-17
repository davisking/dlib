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
    resnet::infer_50 net;
    std::vector<std::string> labels;
    if (argc > 1)
        deserialize(argv[1]) >> net >> labels;
    else
        return EXIT_FAILURE;
    if (argc > 2)
    {
        cout << "fusing layers...\n";
        fuse_layers(net);
        serialize("resnet50_1000_imagenet_classifier-fused.dnn") << net << labels;
    }

    matrix<rgb_pixel> image;
    load_image(image, "elephant.jpg");
    resizable_tensor x;
    net.to_tensor(&image, &image + 1, x);
    cout << x << endl;

    resizable_tensor out = net.forward(x);
    const auto& label = labels[index_of_max(mat(out))];
    resizable_tensor probs(out);
    tt::softmax(probs, out);
    cout << "pred1: " << label << " (" << max(mat(probs)) << ")" << endl;
    {
        running_stats<float> rs;
        for (int i = 0; i < 1000; ++i)
        {
            const auto t0 = chrono::steady_clock::now();
            net.forward(x);
            const auto t1 = chrono::steady_clock::now();
            rs.add(chrono::duration_cast<fms>(t1 - t0).count());
        }
        cout << "affine: " << rs.mean() << " ± " << rs.stddev() << " ms" << endl;
        net.clean();
        ostringstream sout;
        serialize(net, sout);
        cout << "size: " << sout.str().size() / 1024.0 / 1024.0 << " MiB" << endl;
    }
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
