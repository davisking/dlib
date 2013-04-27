#ifndef DLIB_SERIALIZE_PiCKLE_H__
#define DLIB_SERIALIZE_PiCKLE_H__

#include <dlib/serialize.h>
#include <boost/python.hpp>
#include <sstream>

template <typename T>
struct serialize_pickle : boost::python::pickle_suite
{
    static boost::python::tuple getstate(
        const T& item 
    )
    {
        using namespace dlib;
        std::ostringstream sout;
        serialize(item, sout);
        return boost::python::make_tuple(sout.str());
    }

    static void setstate(
        T& item, 
        boost::python::tuple state
    )
    {
        using namespace dlib;
        using namespace boost::python;
        if (len(state) != 1)
        {
            PyErr_SetObject(PyExc_ValueError,
                ("expected 1-item tuple in call to __setstate__; got %s"
                 % state).ptr()
            );
            throw_error_already_set();
        }

        std::string& data = extract<std::string&>(state[0]);
        std::istringstream sin(data);
        deserialize(item, sin);
    }
};

#endif // DLIB_SERIALIZE_PiCKLE_H__

