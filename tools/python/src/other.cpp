#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <dlib/data_io.h>
#include <dlib/sparse_vector.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

typedef std::vector<std::pair<unsigned long,double> > sparse_vect;

tuple get_training_data()
{
    typedef matrix<double,0,1> sample_type;
    std::vector<sample_type> samples;
    std::vector<double> labels;

    sample_type samp(3);

    for (int i = 0; i < 10; ++i)
    {
        samp = 1,2,3;
        samples.push_back(samp);
        labels.push_back(+1);

        samp = -1,-2,-3;
        samples.push_back(samp);
        labels.push_back(-1);
    }

    return make_tuple(samples, labels);
}

void _make_sparse_vector (
    sparse_vect& v
)
{
    make_sparse_vector_inplace(v);
}

void _make_sparse_vector2 (
    std::vector<sparse_vect>& v
)
{
    for (unsigned long i = 0; i < v.size(); ++i)
        make_sparse_vector_inplace(v[i]);
}

tuple _load_libsvm_formatted_data (
    const std::string& file_name
) 
{ 
    std::vector<sparse_vect> samples;
    std::vector<double> labels;
    load_libsvm_formatted_data(file_name, samples, labels); 
    return make_tuple(samples, labels);
}

void _save_libsvm_formatted_data (
    const std::string& file_name,
    const std::vector<sparse_vect>& samples,
    const std::vector<double>& labels
) { save_libsvm_formatted_data(file_name, samples, labels); }

void bind_other()
{
    def("get_training_data",get_training_data);
    def("make_sparse_vector", _make_sparse_vector , "This function modifies its argument so that it is a properly sorted sparse vector.");
    def("make_sparse_vector", _make_sparse_vector2 , "This function modifies a sparse_vectors object so that all elements it contains are properly sorted sparse vectors.");
    def("load_libsvm_formatted_data",_load_libsvm_formatted_data);
    def("save_libsvm_formatted_data",_save_libsvm_formatted_data);
}

