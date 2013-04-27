#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

tuple get_training_data()
{
    typedef matrix<double,0,1> sample_type;
    std::vector<sample_type> samples;
    std::vector<double> labels;

    sample_type samp(3);
    samp = 1,2,3;
    samples.push_back(samp);
    labels.push_back(+1);
    samp = -1,-2,-3;
    samples.push_back(samp);
    labels.push_back(-1);

    return make_tuple(samples, labels);
}

void bind_other()
{
    def("get_training_data",get_training_data);
}

