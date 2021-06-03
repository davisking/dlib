// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_uNSERIALIZE_ABSTRACT_Hh_
#ifdef DLIB_uNSERIALIZE_ABSTRACT_Hh_

#include "../serialize.h"
#include <iostream>

namespace dlib
{
    class unserialize : public std::istream
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a tool that allows you to effectively put an object you just
                deserialized from a stream back into the stream.  Its use is best
                illustrated via an example.  

                void example(std::istream& in)
                {
                    // Suppose that in contains serialized copies of three "some_type"
                    // objects.  You could read them as follows:
                    some_type obj1, obj2, obj3;

                    deserialize(obj1, in); // reads obj1 from stream.
                    deserialize(obj2, in); // reads obj2 from stream.

                    unserialize in2(obj2, in); // make the in2 stream that has obj2 at its front.
                    deserialize(obj2, in2); // reads obj2 from stream again.
                    deserialize(obj3, in2); // reads obj3 from stream.
                }

                The reason unserialize is useful is because it allows you to peek at the
                next object in a stream and potentially do something different based on
                what object is coming next, but still allowing subsequent deserialize()
                statements to be undisturbed by the fact that you peeked at the data.
        !*/

    public:

        template <typename T>
        unserialize (
            const T& item,
            std::istream& in 
        );
        /*!
            requires
                - T must be serializable 
            ensures
                - The bytes in this stream begin with a serialized copy of item followed
                  immediately by the bytes in the given istream.
        !*/
    };
}

#endif // DLIB_uNSERIALIZE_ABSTRACT_Hh_


