// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_KALMAN_FiLTER_ABSTRACT_H__
#ifdef DLIB_KALMAN_FiLTER_ABSTRACT_H__


namespace dlib
{

// ----------------------------------------------------------------------------------------

    void serialize (
        const kalman_filter& item, 
        std::ostream& out 
    );   
    /*!
        provides serialization support 
    !*/

    void deserialize (
        kalman_filter& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KALMAN_FiLTER_ABSTRACT_H__


