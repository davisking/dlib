

#define pyassert(_exp,_message)                                             \
    {if ( !(_exp) )                                                         \
    {                                                                       \
        PyErr_SetString( PyExc_ValueError, _message );                      \
        boost::python::throw_error_already_set();                           \
    }}                                                                      
