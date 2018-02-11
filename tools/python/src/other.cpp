// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/data_io.h>
#include <dlib/sparse_vector.h>
#include <dlib/optimization.h>
#include <dlib/statistics/running_gradient.h>

using namespace dlib;
using namespace std;
namespace py = pybind11;

typedef std::vector<std::pair<unsigned long,double> > sparse_vect;


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

py::tuple _load_libsvm_formatted_data(
    const std::string& file_name
) 
{ 
    std::vector<sparse_vect> samples;
    std::vector<double> labels;
    load_libsvm_formatted_data(file_name, samples, labels); 
    return py::make_tuple(samples, labels);
}

void _save_libsvm_formatted_data (
    const std::string& file_name,
    const std::vector<sparse_vect>& samples,
    const std::vector<double>& labels
) 
{ 
    pyassert(samples.size() == labels.size(), "Invalid inputs");
    save_libsvm_formatted_data(file_name, samples, labels); 
}

// ----------------------------------------------------------------------------------------

py::list _max_cost_assignment (
    const matrix<double>& cost
)
{
    if (cost.nr() != cost.nc())
        throw dlib::error("The input matrix must be square.");

    // max_cost_assignment() only works with integer matrices, so convert from
    // double to integer.
    const double scale = (std::numeric_limits<dlib::int64>::max()/1000)/max(abs(cost));
    matrix<dlib::int64> int_cost = matrix_cast<dlib::int64>(round(cost*scale));
    return vector_to_python_list(max_cost_assignment(int_cost));
}

double _assignment_cost (
    const matrix<double>& cost,
    const py::list& assignment
)
{
    return assignment_cost(cost, python_list_to_vector<long>(assignment));
}

// ----------------------------------------------------------------------------------------

size_t py_count_steps_without_decrease (
    py::object arr,
    double probability_of_decrease
) 
{ 
    DLIB_CASSERT(0.5 < probability_of_decrease && probability_of_decrease < 1);
    return count_steps_without_decrease(python_list_to_vector<double>(arr), probability_of_decrease); 
}

// ----------------------------------------------------------------------------------------

size_t py_count_steps_without_decrease_robust (
    py::object arr,
    double probability_of_decrease,
    double quantile_discard
) 
{ 
    DLIB_CASSERT(0.5 < probability_of_decrease && probability_of_decrease < 1);
    DLIB_CASSERT(0 <= quantile_discard && quantile_discard <= 1);
    return count_steps_without_decrease_robust(python_list_to_vector<double>(arr), probability_of_decrease, quantile_discard); 
}

// ----------------------------------------------------------------------------------------

double probability_that_sequence_is_increasing (
    py::object arr
)
{
    DLIB_CASSERT(len(arr) > 2);
    return probability_gradient_greater_than(python_list_to_vector<double>(arr), 0);
}

// ----------------------------------------------------------------------------------------

void hit_enter_to_continue()
{
    std::cout << "Hit enter to continue";
    std::cin.get();
}

// ----------------------------------------------------------------------------------------

void bind_other(py::module &m)
{
    m.def("max_cost_assignment", _max_cost_assignment, py::arg("cost"),
"requires    \n\
    - cost.nr() == cost.nc()    \n\
      (i.e. the input must be a square matrix)    \n\
ensures    \n\
    - Finds and returns the solution to the following optimization problem:    \n\
    \n\
        Maximize: f(A) == assignment_cost(cost, A)    \n\
        Subject to the following constraints:    \n\
            - The elements of A are unique. That is, there aren't any     \n\
              elements of A which are equal.      \n\
            - len(A) == cost.nr()    \n\
    \n\
    - Note that this function converts the input cost matrix into a 64bit fixed    \n\
      point representation.  Therefore, you should make sure that the values in    \n\
      your cost matrix can be accurately represented by 64bit fixed point values.    \n\
      If this is not the case then the solution my become inaccurate due to    \n\
      rounding error.  In general, this function will work properly when the ratio    \n\
      of the largest to the smallest value in cost is no more than about 1e16.   " 
        );

    m.def("assignment_cost", _assignment_cost, py::arg("cost"),py::arg("assignment"),
"requires    \n\
    - cost.nr() == cost.nc()    \n\
      (i.e. the input must be a square matrix)    \n\
    - for all valid i:    \n\
        - 0 <= assignment[i] < cost.nr()    \n\
ensures    \n\
    - Interprets cost as a cost assignment matrix. That is, cost[i][j]     \n\
      represents the cost of assigning i to j.      \n\
    - Interprets assignment as a particular set of assignments. That is,    \n\
      i is assigned to assignment[i].    \n\
    - returns the cost of the given assignment. That is, returns    \n\
      a number which is:    \n\
        sum over i: cost[i][assignment[i]]   " 
        );

    m.def("make_sparse_vector", _make_sparse_vector , 
"This function modifies its argument so that it is a properly sorted sparse vector.    \n\
This means that the elements of the sparse vector will be ordered so that pairs    \n\
with smaller indices come first.  Additionally, there won't be any pairs with    \n\
identical indices.  If such pairs were present in the input sparse vector then    \n\
their values will be added together and only one pair with their index will be    \n\
present in the output.   " 
        );
    m.def("make_sparse_vector", _make_sparse_vector2 , 
        "This function modifies a sparse_vectors object so that all elements it contains are properly sorted sparse vectors.");

    m.def("load_libsvm_formatted_data",_load_libsvm_formatted_data, py::arg("file_name"),
"ensures    \n\
    - Attempts to read a file of the given name that should contain libsvm    \n\
      formatted data.  The data is returned as a tuple where the first tuple    \n\
      element is an array of sparse vectors and the second element is an array of    \n\
      labels.    " 
    );

    m.def("save_libsvm_formatted_data",_save_libsvm_formatted_data, py::arg("file_name"), py::arg("samples"), py::arg("labels"),
"requires    \n\
    - len(samples) == len(labels)    \n\
ensures    \n\
    - saves the data to the given file in libsvm format   " 
    );

    m.def("hit_enter_to_continue", hit_enter_to_continue, 
        "Asks the user to hit enter to continue and pauses until they do so.");




    m.def("count_steps_without_decrease",py_count_steps_without_decrease, py::arg("time_series"), py::arg("probability_of_decrease")=0.51,
"requires \n\
    - time_series must be a one dimensional array of real numbers.  \n\
    - 0.5 < probability_of_decrease < 1 \n\
ensures \n\
    - If you think of the contents of time_series as a potentially noisy time \n\
      series, then this function returns a count of how long the time series has \n\
      gone without noticeably decreasing in value.  It does this by scanning along \n\
      the elements, starting from the end (i.e. time_series[-1]) to the beginning, \n\
      and checking how many elements you need to examine before you are confident \n\
      that the series has been decreasing in value.  Here, \"confident of decrease\" \n\
      means the probability of decrease is >= probability_of_decrease.   \n\
    - Setting probability_of_decrease to 0.51 means we count until we see even a \n\
      small hint of decrease, whereas a larger value of 0.99 would return a larger \n\
      count since it keeps going until it is nearly certain the time series is \n\
      decreasing. \n\
    - The max possible output from this function is len(time_series). \n\
    - The implementation of this function is done using the dlib::running_gradient \n\
      object, which is a tool that finds the least squares fit of a line to the \n\
      time series and the confidence interval around the slope of that line.  That \n\
      can then be used in a simple statistical test to determine if the slope is \n\
      positive or negative." 
    /*!
        requires
            - time_series must be a one dimensional array of real numbers. 
            - 0.5 < probability_of_decrease < 1
        ensures
            - If you think of the contents of time_series as a potentially noisy time
              series, then this function returns a count of how long the time series has
              gone without noticeably decreasing in value.  It does this by scanning along
              the elements, starting from the end (i.e. time_series[-1]) to the beginning,
              and checking how many elements you need to examine before you are confident
              that the series has been decreasing in value.  Here, "confident of decrease"
              means the probability of decrease is >= probability_of_decrease.  
            - Setting probability_of_decrease to 0.51 means we count until we see even a
              small hint of decrease, whereas a larger value of 0.99 would return a larger
              count since it keeps going until it is nearly certain the time series is
              decreasing.
            - The max possible output from this function is len(time_series).
            - The implementation of this function is done using the dlib::running_gradient
              object, which is a tool that finds the least squares fit of a line to the
              time series and the confidence interval around the slope of that line.  That
              can then be used in a simple statistical test to determine if the slope is
              positive or negative.
    !*/
    );

    m.def("count_steps_without_decrease_robust",py_count_steps_without_decrease_robust, py::arg("time_series"), py::arg("probability_of_decrease")=0.51, py::arg("quantile_discard")=0.1,
"requires \n\
    - time_series must be a one dimensional array of real numbers.  \n\
    - 0.5 < probability_of_decrease < 1 \n\
    - 0 <= quantile_discard <= 1 \n\
ensures \n\
    - This function behaves just like \n\
      count_steps_without_decrease(time_series,probability_of_decrease) except that \n\
      it ignores values in the time series that are in the upper quantile_discard \n\
      quantile.  So for example, if the quantile discard is 0.1 then the 10% \n\
      largest values in the time series are ignored." 
    /*!
        requires
            - time_series must be a one dimensional array of real numbers. 
            - 0.5 < probability_of_decrease < 1
            - 0 <= quantile_discard <= 1
        ensures
            - This function behaves just like
              count_steps_without_decrease(time_series,probability_of_decrease) except that
              it ignores values in the time series that are in the upper quantile_discard
              quantile.  So for example, if the quantile discard is 0.1 then the 10%
              largest values in the time series are ignored.
    !*/
    );

    m.def("probability_that_sequence_is_increasing",probability_that_sequence_is_increasing, py::arg("time_series"),
        "returns the probability that the given sequence of real numbers is increasing in value over time.");
}

