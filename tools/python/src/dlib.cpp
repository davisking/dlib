
#include <boost/python.hpp>

void bind_matrix();
void bind_vector();
void bind_svm_c_trainer();
void bind_decision_functions();
void bind_basic_types();
void bind_other();


BOOST_PYTHON_MODULE(dlib)
{
    bind_matrix();
    bind_vector();
    bind_svm_c_trainer();
    bind_decision_functions();
    bind_basic_types();
    bind_other();
}
