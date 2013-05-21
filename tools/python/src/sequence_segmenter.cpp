
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include "serialize_pickle.h"
#include <dlib/svm_threaded.h>
#include "pyassert.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/args.hpp>

using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> sample_type; 
typedef std::vector<std::pair<unsigned long,double> > sparse_vect;
typedef std::vector<std::pair<unsigned long, unsigned long> > ranges;

// ----------------------------------------------------------------------------------------

template <typename samp_type, bool BIO, bool high_order, bool nonnegative>
class segmenter_feature_extractor
{

public:
    typedef std::vector<samp_type> sequence_type;
    const static bool use_BIO_model = BIO;
    const static bool use_high_order_features = high_order;
    const static bool allow_negative_weights = nonnegative;


    unsigned long _num_features;
    unsigned long _window_size;

    segmenter_feature_extractor(
    ) : _num_features(0), _window_size(0) {}

    segmenter_feature_extractor(
        unsigned long _num_features_,
        unsigned long _window_size_
    ) : _num_features(_num_features_), _window_size(_window_size_) {}

    unsigned long num_features(
    ) const { return _num_features; }

    unsigned long window_size(
    ) const {return _window_size; }

    template <typename feature_setter>
    void get_features (
        feature_setter& set_feature,
        const std::vector<sample_type>& x,
        unsigned long position
    ) const
    {
        for (long i = 0; i < x[position].size(); ++i)
        {
            set_feature(i, x[position](i));
        }
    }

    template <typename feature_setter>
    void get_features (
        feature_setter& set_feature,
        const std::vector<sparse_vect>& x,
        unsigned long position
    ) const
    {
        for (long i = 0; i < x[position].size(); ++i)
        {
            set_feature(x[position][i].first, x[position][i].second);
        }
    }

    friend void serialize(const segmenter_feature_extractor& item, std::ostream& out)
    {
        dlib::serialize(item._num_features, out);
        dlib::serialize(item._window_size, out);
    }
    friend void deserialize(segmenter_feature_extractor& item, std::istream& in)
    {
        dlib::deserialize(item._num_features, in);
        dlib::deserialize(item._window_size, in);
    }
};

// ----------------------------------------------------------------------------------------

struct segmenter_type
{
    segmenter_type() : mode(0)
    { }

    ranges segment_sequence (
        const std::vector<sample_type>& x
    ) const 
    {
        return ranges();
    }

    const matrix<double,0,1>& get_weights() 
    { 
        switch(mode)
        {
            case 0: return segmenter0.get_weights(); 
            case 1: return segmenter1.get_weights(); 
            case 2: return segmenter2.get_weights(); 
            case 3: return segmenter3.get_weights(); 
            case 4: return segmenter4.get_weights(); 
            case 5: return segmenter5.get_weights(); 
            case 6: return segmenter6.get_weights(); 
            case 7: return segmenter7.get_weights(); 
        }
    }

    friend void serialize (const segmenter_type& item, std::ostream& out) 
    {
        serialize(item.mode, out);
        switch(item.mode)
        {
            case 0: serialize(item.segmenter0, out); break;
            case 1: serialize(item.segmenter1, out); break;
            case 2: serialize(item.segmenter2, out); break;
            case 3: serialize(item.segmenter3, out); break;
            case 4: serialize(item.segmenter4, out); break;
            case 5: serialize(item.segmenter5, out); break;
            case 6: serialize(item.segmenter6, out); break;
            case 7: serialize(item.segmenter7, out); break;
        }
    }
    friend void deserialize (segmenter_type& item, std::istream& in)
    {
        deserialize(item.mode, in);
        switch(item.mode)
        {
            case 0: deserialize(item.segmenter0, in); break;
            case 1: deserialize(item.segmenter1, in); break;
            case 2: deserialize(item.segmenter2, in); break;
            case 3: deserialize(item.segmenter3, in); break;
            case 4: deserialize(item.segmenter4, in); break;
            case 5: deserialize(item.segmenter5, in); break;
            case 6: deserialize(item.segmenter6, in); break;
            case 7: deserialize(item.segmenter7, in); break;
        }
    }

    int mode;

    typedef segmenter_feature_extractor<sample_type, true, true, true>  fe0;
    typedef segmenter_feature_extractor<sample_type, true, true, false> fe1;
    typedef segmenter_feature_extractor<sample_type, true, false,true>  fe2;
    typedef segmenter_feature_extractor<sample_type, true, false,false> fe3;
    typedef segmenter_feature_extractor<sample_type, false,true, true>  fe4;
    typedef segmenter_feature_extractor<sample_type, false,true, false> fe5;
    typedef segmenter_feature_extractor<sample_type, false,false,true>  fe6;
    typedef segmenter_feature_extractor<sample_type, false,false,false> fe7;
    sequence_segmenter<fe0> segmenter0;
    sequence_segmenter<fe1> segmenter1;
    sequence_segmenter<fe2> segmenter2;
    sequence_segmenter<fe3> segmenter3;
    sequence_segmenter<fe4> segmenter4;
    sequence_segmenter<fe5> segmenter5;
    sequence_segmenter<fe6> segmenter6;
    sequence_segmenter<fe7> segmenter7;

    typedef segmenter_feature_extractor<sparse_vect, true, true, true>  fe8;
    typedef segmenter_feature_extractor<sparse_vect, true, true, false> fe9;
    typedef segmenter_feature_extractor<sparse_vect, true, false,true>  fe10;
    typedef segmenter_feature_extractor<sparse_vect, true, false,false> fe11;
    typedef segmenter_feature_extractor<sparse_vect, false,true, true>  fe12;
    typedef segmenter_feature_extractor<sparse_vect, false,true, false> fe13;
    typedef segmenter_feature_extractor<sparse_vect, false,false,true>  fe14;
    typedef segmenter_feature_extractor<sparse_vect, false,false,false> fe15;
    sequence_segmenter<fe8> segmenter8;
    sequence_segmenter<fe9> segmenter9;
    sequence_segmenter<fe10> segmenter10;
    sequence_segmenter<fe11> segmenter11;
    sequence_segmenter<fe12> segmenter12;
    sequence_segmenter<fe13> segmenter13;
    sequence_segmenter<fe14> segmenter14;
    sequence_segmenter<fe15> segmenter15;
};


// ----------------------------------------------------------------------------------------

struct segmenter_params
{
    segmenter_params()
    {
        use_BIO_model = true;
        use_high_order_features = true;
        allow_negative_weights = true;
        window_size = 5;
        num_threads = 4;
        epsilon = 0.1;
        max_cache_size = 40;
        C = 100;
    }

    bool use_BIO_model;
    bool use_high_order_features;
    bool allow_negative_weights;
    unsigned long window_size;
    unsigned long num_threads;
    double epsilon;
    unsigned long max_cache_size;
    bool be_verbose;
    double C;
};

// ----------------------------------------------------------------------------------------

template <typename T>
void configure_trainer (
    const std::vector<std::vector<sample_type> >& samples,
    structural_sequence_segmentation_trainer<T>& trainer,
    const segmenter_params& params
)
{
    pyassert(samples.size() != 0, "Invalid arguments.  You must give some training sequences.");
    pyassert(samples[0].size() != 0, "Invalid arguments. You can't have zero length training sequences.");
    const long dims = samples[0][0].size();

    trainer = structural_sequence_segmentation_trainer<T>(T(dims, params.window_size));
    trainer.set_num_threads(params.num_threads);
    trainer.set_epsilon(params.epsilon);
    trainer.set_max_cache_size(params.max_cache_size);
    trainer.set_c(params.C);
    if (params.be_verbose)
        trainer.be_verbose();
}

// ----------------------------------------------------------------------------------------

segmenter_type train_dense (
    const std::vector<std::vector<sample_type> >& samples,
    const std::vector<ranges>& segments,
    segmenter_params params
)
{
    pyassert(is_sequence_segmentation_problem(samples, segments), "Invalid inputs");

    int mode = 0;
    if (params.use_BIO_model)
        mode = mode*2 + 1;
    else 
        mode = mode*2;
    if (params.use_high_order_features)
        mode = mode*2 + 1;
    else 
        mode = mode*2;
    if (params.allow_negative_weights)
        mode = mode*2 + 1;
    else 
        mode = mode*2;

    segmenter_type res;
    res.mode = mode;
    switch(mode)
    {
        case 0: { structural_sequence_segmentation_trainer<segmenter_type::fe0> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter0 = trainer.train(samples, segments);
                } break;
        case 1: { structural_sequence_segmentation_trainer<segmenter_type::fe1> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter1 = trainer.train(samples, segments);
                } break;
        case 2: { structural_sequence_segmentation_trainer<segmenter_type::fe2> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter2 = trainer.train(samples, segments);
                } break;
        case 3: { structural_sequence_segmentation_trainer<segmenter_type::fe3> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter3 = trainer.train(samples, segments);
                } break;
        case 4: { structural_sequence_segmentation_trainer<segmenter_type::fe4> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter4 = trainer.train(samples, segments);
                } break;
        case 5: { structural_sequence_segmentation_trainer<segmenter_type::fe5> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter5 = trainer.train(samples, segments);
                } break;
        case 6: { structural_sequence_segmentation_trainer<segmenter_type::fe6> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter6 = trainer.train(samples, segments);
                } break;
        case 7: { structural_sequence_segmentation_trainer<segmenter_type::fe7> trainer;
                  configure_trainer(samples, trainer, params);
                  res.segmenter7 = trainer.train(samples, segments);
                } break;
    }


    return res;
}

// ----------------------------------------------------------------------------------------

void bind_sequence_segmenter()
{
    class_<segmenter_params>("segmenter_params",
"This class is used to define all the optional parameters to the    \n\
train_sequence_segmenter() routine.   ")
        .add_property("use_BIO_model", &segmenter_params::use_BIO_model)
        .add_property("use_high_order_features", &segmenter_params::use_high_order_features)
        .add_property("allow_negative_weights", &segmenter_params::allow_negative_weights)
        .add_property("window_size", &segmenter_params::window_size)
        .add_property("num_threads", &segmenter_params::num_threads)
        .add_property("epsilon", &segmenter_params::epsilon)
        .add_property("max_cache_size", &segmenter_params::max_cache_size)
        .add_property("C", &segmenter_params::C);

    class_<segmenter_type> ("segmenter_type")
        .def("segment_sequence", &segmenter_type::segment_sequence)
        .def_pickle(serialize_pickle<segmenter_type>());

    using boost::python::arg;
    def("train_sequence_segmenter", train_dense, (arg("samples"), arg("segments"), arg("params")=segmenter_params()));
}




