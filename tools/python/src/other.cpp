// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/data_io.h>
#include <dlib/sparse_vector.h>
#include <dlib/optimization.h>
#include <dlib/statistics/running_gradient.h>
#include <dlib/filtering.h>

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

string print_momentum_filter(const momentum_filter& r)
{
    std::ostringstream sout;
    sout << "momentum_filter(";
    sout << "measurement_noise="<<r.get_measurement_noise();
    sout << ", typical_acceleration="<<r.get_typical_acceleration();
    sout << ", max_measurement_deviation="<<r.get_max_measurement_deviation();
    sout << ")";
    return sout.str();
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

    {
        typedef momentum_filter type;
        py::class_<type>(m, "momentum_filter",
            R"asdf( 
                This object is a simple tool for filtering a single scalar value that
                measures the location of a moving object that has some non-trivial
                momentum.  Importantly, the measurements are noisy and the object can
                experience sudden unpredictable accelerations.  To accomplish this
                filtering we use a simple Kalman filter with a state transition model of:

                    position_{i+1} = position_{i} + velocity_{i} 
                    velocity_{i+1} = velocity_{i} + some_unpredictable_acceleration

                and a measurement model of:
                    
                    measured_position_{i} = position_{i} + measurement_noise

                Where some_unpredictable_acceleration and measurement_noise are 0 mean Gaussian 
                noise sources with standard deviations of get_typical_acceleration() and
                get_measurement_noise() respectively.

                To allow for really sudden and large but infrequent accelerations, at each
                step we check if the current measured position deviates from the predicted
                filtered position by more than get_max_measurement_deviation()*get_measurement_noise() 
                and if so we adjust the filter's state to keep it within these bounds.
                This allows the moving object to undergo large unmodeled accelerations, far
                in excess of what would be suggested by get_typical_acceleration(), without
                then experiencing a long lag time where the Kalman filter has to "catch
                up" to the new position.  )asdf"
        )
        .def(py::init<double,double,double>(), py::arg("measurement_noise"), py::arg("typical_acceleration"), py::arg("max_measurement_deviation"))
        .def("measurement_noise",   &type::get_measurement_noise)
        .def("typical_acceleration", &type::get_typical_acceleration)
        .def("max_measurement_deviation", &type::get_max_measurement_deviation)
        .def("__call__", &type::operator())
        .def("__repr__", print_momentum_filter)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    m.def("find_optimal_momentum_filter",
        [](const py::object sequence, const double smoothness) {
            return find_optimal_momentum_filter(python_list_to_vector<double>(sequence), smoothness);
        },
        py::arg("sequence"),
        py::arg("smoothness")=1,
        R"asdf(requires
            - sequences.size() != 0
            - for all valid i: sequences[i].size() > 4
            - smoothness >= 0
        ensures
            - This function finds the "optimal" settings of a momentum_filter based on
              recorded measurement data stored in sequences.  Here we assume that each
              vector in sequences is a complete track history of some object's measured
              positions.  What we do is find the momentum_filter that minimizes the
              following objective function:
                 sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1])
                 Where i is a time index.
              The sum runs over all the data in sequences.  So what we do is find the
              filter settings that produce smooth filtered trajectories but also produce
              filtered outputs that are as close to the measured positions as possible.
              The larger the value of smoothness the less jittery the filter outputs will
              be, but they might become biased or laggy if smoothness is set really high.)asdf"
    );
}

