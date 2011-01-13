// The contents of this file are in the public domain. 
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt (in dlib)
/*
    This is a command line program that can try different regression 
    algorithms on a libsvm-formatted data set.
*/
#include <iostream>
#include <map>
#include <vector>
#include <math.h>
#include <float.h>

#include "dlib/cmd_line_parser.h"
#include "dlib/data_io.h"
#include "dlib/mlp.h"
#include "dlib/svm.h"

using namespace dlib;

typedef dlib::cmd_line_parser<char>::check_1a_c clp;
typedef std::map<unsigned long, double> sparse_sample_type;
typedef dlib::matrix< sparse_sample_type::value_type::second_type,0,1
        > dense_sample_type;

/* exp10() is not in C/C++ standard */
double
exp10_ (double m)
{
    return exp (2.3025850929940456840179914546844 * m);
}

/* ---------------------------------------------------------------------
   option_range class
   --------------------------------------------------------------------- */
struct option_range {
public:
    bool log_range;
    float min_value;
    float max_value;
    float incr;
public:
    option_range () {
    log_range = false;
    min_value = 0;
    max_value = 100;
    incr = 10;
    }
    void set_option (clp& parser, std::string const& option, 
    float default_val);
    float get_min_value ();
    float get_max_value ();
    float get_next_value (float curr_val);
};

void
option_range::set_option (
    clp& parser,
    std::string const& option, 
    float default_val
)
{
    int rc;

    /* No option specified */
    if (!parser.option (option)) {
    log_range = 0;
    min_value = default_val;
    max_value = default_val;
    incr = 1;
    return;
    }

    /* Range specified */
    rc = sscanf (parser.option(option).argument().c_str(), "%f:%f:%f", 
    &min_value, &incr, &max_value);
    if (rc == 3) {
    log_range = 1;
    return;
    }

    /* Single value specified */
    if (rc == 1) {
    log_range = 0;
    max_value = min_value;
    incr = 1;
    return;
    }

    else {
    std::cerr << "Error parsing option" << option << "\n";
    exit (-1);
    }
}

float 
option_range::get_min_value ()
{
    if (log_range) {
    return exp10_ (min_value);
    } else {
    return min_value;
    }
}

float 
option_range::get_max_value ()
{
    if (log_range) {
    return exp10_ (max_value);
    } else {
    return max_value;
    }
}

float 
option_range::get_next_value (float curr_value)
{
    if (log_range) {
    curr_value = log10 (curr_value);
    curr_value += incr;
    curr_value = exp10_ (curr_value);
    } else {
    curr_value += incr;
    }
    return curr_value;
}

/* ---------------------------------------------------------------------
   global functions
   --------------------------------------------------------------------- */
static void
parse_args (clp& parser, int argc, char* argv[])
{
    try {
    // Algorithm-independent options
        parser.add_option ("a",
        "Choose the learning algorithm: {krls,krr,mlp,svr}.",1);
        parser.add_option ("h","Display this help message.");
        parser.add_option ("help","Display this help message.");
        parser.add_option ("k",
        "Learning kernel (for krls,krr,svr methods): {lin,rbk}.",1);
        parser.add_option ("in","A libsvm-formatted file to test.",1);
        parser.add_option ("normalize",
        "Normalize the sample inputs to zero-mean unit variance?");
        parser.add_option ("train-best",
        "Train and save a network using best parameters", 1);

    // Algorithm-specific options
        parser.add_option ("rbk-gamma",
        "Width of radial basis kernels: {float}.",1);
        parser.add_option ("krls-tolerance",
        "Numerical tolerance of krls linear dependency test: {float}.",1);
        parser.add_option ("mlp-hidden-units",
        "Number of hidden units in mlp: {integer}.",1);
        parser.add_option ("mlp-num-iterations",
        "Number of epochs to train the mlp: {integer}.",1);
        parser.add_option ("svr-c",
        "SVR regularization parameter \"C\": "
        "{float}.",1);
        parser.add_option ("svr-epsilon-insensitivity",
        "SVR fitting tolerance parameter: "
        "{float}.",1);
        parser.add_option ("verbose", "Use verbose trainers");

    // Parse the command line arguments
        parser.parse(argc,argv);

    // Check that options aren't given multiple times
        const char* one_time_opts[] = {"a", "h", "help", "in"};
        parser.check_one_time_options(one_time_opts);

        // Check if the -h option was given
        if (parser.option("h") || parser.option("help")) {
        std::cout << "Usage: dlib_test [-a algorithm] --in input_file\n";
            parser.print_options(std::cout);
        std::cout << std::endl;
        exit (0);
        }

    // Check that an input file was given
        if (!parser.option("in")) {
        std::cout 
        << "Error in command line:\n"
        << "You must specify an input file with the --in option.\n"
        << "\nTry the -h option for more information\n";
        exit (0);
    }
    }
    catch (std::exception& e) {
        // Catch cmd_line_parse_error exceptions and print usage message.
    std::cout << e.what() << std::endl;
    exit (1);
    }
    catch (...) {
    std::cout << "Some error occurred" << std::endl;
    }
}

static const char*
get_kernel (
    clp& parser
)
{
    const char* kernel = "rbk";
    if (parser.option ("k")) {
    kernel = parser.option("k").argument().c_str();
    }
    return kernel;
}

static void
get_rbk_gamma (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    option_range& range
) {
    float default_gamma = 3.0 / compute_mean_squared_distance (
    randomly_subsample (dense_samples, 2000));
    range.set_option (parser, "rbk-gamma", default_gamma);
}

static void
get_krls_tolerance (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples, 
    option_range& range
)
{
    float default_krls_tolerance = 0.001;
    range.set_option (parser, "krls-tolerance", default_krls_tolerance);
}

static double
get_mlp_hidden_units (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    int num_hidden = 5;
    if (parser.option ("mlp-hidden-units")) {
    num_hidden = sa = parser.option("mlp-hidden-units").argument();
    }
    return num_hidden;
}

static double
get_mlp_num_iterations (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    int num_iterations = 5000;
    if (parser.option ("mlp-num-iterations")) {
    num_iterations = sa = parser.option("mlp-num-iterations").argument();
    }
    return num_iterations;
}

static void
get_svr_c (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples, 
    option_range& range
)
{
    float default_svr_c = 1000.;
    range.set_option (parser, "svr-c", default_svr_c);
}

static double
get_svr_epsilon_insensitivity (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    // Epsilon-insensitive regression means we do regression but stop 
    // trying to fit a data point once it is "close enough" to its 
    // target value.  This parameter is the value that controls what 
    // we mean by "close enough".  In this case, I'm saying I'm happy 
    // if the resulting regression function gets within 0.001 of the 
    // target value.
    double epsilon_insensitivity = 0.001;
    if (parser.option ("svr-epsilon-insensitivity")) {
    epsilon_insensitivity 
        = sa = parser.option("svr-epsilon-insensitivity").argument();
    }
    return epsilon_insensitivity;
}

static void
krls_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    option_range gamma_range, krls_tol_range;

    get_rbk_gamma (parser, dense_samples, gamma_range);
    get_krls_tolerance (parser, dense_samples, krls_tol_range);

    // Split into training set and testing set
    float training_pct = 0.8;
    unsigned int training_samples = (unsigned int) floor (
    training_pct * dense_samples.size());

    for (float krls_tol = krls_tol_range.get_min_value(); 
     krls_tol <= krls_tol_range.get_max_value();
     krls_tol = krls_tol_range.get_next_value (krls_tol))
    {
    for (float gamma = gamma_range.get_min_value(); 
         gamma <= gamma_range.get_max_value();
         gamma = gamma_range.get_next_value (gamma))
    {
        krls<kernel_type> net (kernel_type(gamma), krls_tol);

        // Krls doesn't seem to come with any batch training function
        for (unsigned int j = 0; j < training_samples; j++) {
        net.train (dense_samples[j], labels[j]);
        }

        // Test the performance (sorry, no cross-validation)
        double total_err = 0.0;
        for (unsigned int j = training_samples + 1; 
         j < dense_samples.size(); j++)
        {
        double diff = net(dense_samples[j]) - labels[j];
        total_err += diff * diff;
        }

        double testset_error = total_err 
        / (dense_samples.size() - training_samples);
        printf ("%3.6f %3.6f %3.9f\n", krls_tol, gamma, testset_error);
    }
    }
}

static void
krr_rbk_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;
    option_range gamma_range;
    double best_gamma = DBL_MAX;
    float best_loo = FLT_MAX;

    get_rbk_gamma (parser, dense_samples, gamma_range);

    for (float gamma = gamma_range.get_min_value(); 
     gamma <= gamma_range.get_max_value();
     gamma = gamma_range.get_next_value (gamma))
    {
    // LOO cross validation
    double loo_error;

    if (parser.option("verbose")) {
        trainer.set_search_lambdas(logspace(-9, 4, 100));
        trainer.be_verbose();
    }
    trainer.set_kernel (kernel_type (gamma));
    trainer.train (dense_samples, labels, loo_error);
    if (loo_error < best_loo) {
        best_loo = loo_error;
        best_gamma = gamma;
    }
    printf ("10^%f %9.6f\n", log10(gamma), loo_error);
    }

    printf ("Best result: gamma=10^%f (%g), loo_error=%9.6f\n",
    log10(best_gamma), best_gamma, best_loo);
    if (parser.option("train-best")) {
    printf ("Training network with best parameters\n");
    trainer.set_kernel (kernel_type (best_gamma));
    decision_function<kernel_type> best_network = 
        trainer.train (dense_samples, labels);

    std::ofstream fout (parser.option("train-best").argument().c_str(), 
        std::ios::binary);
    serialize (best_network, fout);
    fout.close();
    }
}

static void
krr_lin_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef linear_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;

    // LOO cross validation
    double loo_error;
    trainer.train(dense_samples, labels, loo_error);
    std::cout << "mean squared LOO error: " << loo_error << std::endl;
}

static void
krr_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    const char* kernel = get_kernel (parser);

    if (!strcmp (kernel, "lin")) {
    krr_lin_test (parser, dense_samples, labels);
    } else if (!strcmp (kernel, "rbk")) {
    krr_rbk_test (parser, dense_samples, labels);
    } else {
    fprintf (stderr, "Unknown kernel type: %s\n", kernel);
    exit (-1);
    }
}

static void
mlp_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    // Create a multi-layer perceptron network.
    const int num_input = dense_samples[0].size();
    int num_hidden = get_mlp_hidden_units (parser, dense_samples);
    printf ("Creating ANN with size (%d, %d)\n", num_input, num_hidden);
    mlp::kernel_1a_c net (num_input, num_hidden);

    // Dlib barfs if output values are not normalized to [0,1]
    double label_min = *(std::min_element (labels.begin(), labels.end()));
    double label_max = *(std::max_element (labels.begin(), labels.end()));
    std::vector<double>::iterator it;
    for (it = labels.begin(); it != labels.end(); it++) {
    (*it) = ((*it) - label_min) / (label_max - label_min);
    }

    // Split into training set and testing set
    float training_pct = 0.8;
    unsigned int training_samples = (unsigned int) floor (
    training_pct * dense_samples.size());

    // Dlib doesn't seem to come with any batch training functions for mlp.
    // Also, note that only backprop is supported.
    int num_iterations = get_mlp_num_iterations (parser, dense_samples);
    for (int i = 0; i < num_iterations; i++) {
    for (unsigned int j = 0; j < training_samples; j++) {
        net.train (dense_samples[j], labels[j]);
    }
    }

    // Test the performance (sorry, no cross-validation) */
    double total_err = 0.0;
    for (unsigned int j = training_samples + 1; j < dense_samples.size(); j++)
    {
    double diff = net(dense_samples[j]) - labels[j];
    diff = diff * (label_max - label_min);
    total_err += diff * diff;
    }
    std::cout 
    << "MSE (no cross-validation): " 
    << total_err / (dense_samples.size() - training_samples) << std::endl;
}

static void
svr_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    svr_trainer<kernel_type> trainer;
    option_range gamma_range, svr_c_range;

    get_rbk_gamma (parser, dense_samples, gamma_range);
    get_svr_c (parser, dense_samples, svr_c_range);

    double epsilon_insensitivity = get_svr_epsilon_insensitivity (
    parser, dense_samples);
    trainer.set_epsilon_insensitivity (epsilon_insensitivity);

    for (float svr_c = svr_c_range.get_min_value(); 
     svr_c <= svr_c_range.get_max_value();
     svr_c = svr_c_range.get_next_value (svr_c))
    {
    trainer.set_c (svr_c);
    for (float gamma = gamma_range.get_min_value(); 
         gamma <= gamma_range.get_max_value();
         gamma = gamma_range.get_next_value (gamma))
    {
        double cv_error;
        trainer.set_kernel (kernel_type (gamma));
        cv_error = cross_validate_regression_trainer (trainer, 
        dense_samples, labels, 10);
        printf ("%3.6f %3.6f %3.9f\n", svr_c, gamma, cv_error);
    }
    }
}

int 
main (int argc, char* argv[])
{
    clp parser;

    parse_args(parser, argc, argv);

    const clp::option_type& option_alg = parser.option("a");
    const clp::option_type& option_in = parser.option("in");

    std::vector<sparse_sample_type> sparse_samples;
    std::vector<double> labels;

    load_libsvm_formatted_data (
    option_in.argument(), 
    sparse_samples, 
    labels
    );

    if (sparse_samples.size() < 1) {
    std::cout 
        << "Sorry, I couldn't find any samples in your data set.\n"
        << "Aborting the operation.\n";
    exit (0);
    }

    std::vector<dense_sample_type> dense_samples;
    dense_samples = sparse_to_dense (sparse_samples);

    /* GCS FIX: The sparse_to_dense converter adds an extra column, 
       because libsvm files are indexed starting with "1". */
    std::cout 
    << "Loaded " << sparse_samples.size() << " samples"
    << std::endl
    << "Each sample has size " << sparse_samples[0].size() 
    << std::endl
    << "Each dense sample has size " << dense_samples[0].size() 
    << std::endl;

    // Normalize inputs to N(0,1)
    if (parser.option ("normalize")) {
    vector_normalizer<dense_sample_type> normalizer;
    normalizer.train (dense_samples);
    for (unsigned long i = 0; i < dense_samples.size(); ++i) {
        dense_samples[i] = normalizer (dense_samples[i]);
    }
    }

    // Randomize the order of the samples, labels
    randomize_samples (dense_samples, labels);

    if (!option_alg) {
    // Do KRR if user didn't specify an algorithm
    std::cout << "No algorithm specified, default to KRR\n";
    krr_test (parser, dense_samples, labels);
    }
    else if (option_alg.argument() == "krls") {
    krls_test (parser, dense_samples, labels);
    }
    else if (option_alg.argument() == "krr") {
    krr_test (parser, dense_samples, labels);
    }
    else if (option_alg.argument() == "mlp") {
    mlp_test (parser, dense_samples, labels);
    }
    else if (option_alg.argument() == "svr") {
    svr_test (parser, dense_samples, labels);
    }
    else {
    fprintf (stderr, 
        "Error, algorithm \"%s\" is unknown.\n"
        "Please use -h to see the command line options\n",
        option_alg.argument().c_str());
    exit (-1);
    }
}
