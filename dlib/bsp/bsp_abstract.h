// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BsP_ABSTRACT_H__
#ifdef DLIB_BsP_ABSTRACT_H__

#include "../noncopyable.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class bsp_context : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS

            THREAD SAFETY
                This object is not thread-safe.  This means you must serialize all access
                to it using an appropriate mutex or other synchronization mechanism if it
                is to be accessed from multiple threads. 
        !*/

    public:

        template <typename T>
        void send(
            const T& item,
            unsigned long target_node_id
        );
        /*!
            requires
                - item is serializable 
                - target_node_id < number_of_nodes()
                - target_node_id != node_id()
            ensures
                - sends a copy of item to the node with the given id.
        !*/

        template <typename T>
        void broadcast (
            const T& item
        );
        /*!
            ensures
                - sends a copy of item to all other processing nodes.
        !*/

        unsigned long node_id (
        ) const; 
        /*!
            ensures
                - Returns the id of the current processing node.  That is, 
                  returns a number N such that:
                    - N < number_of_nodes()
                    - N == the node id of the processing node that called
                      node_id().
        !*/

        unsigned long number_of_nodes (
        ) const; 
        /*!
            ensures
                - returns the number of processing nodes participating in the
                  BSP computation.
        !*/

        template <typename T>
        bool receive (
            T& item
        );
        /*!
            ensures
                - if (this function returns true) then
                    - #item == the next message which was sent to the calling processing
                      node.
                - else
                    - There were no other messages to receive and all other processing
                      nodes are blocked on calls to receive().
        !*/

        template <typename T>
        bool receive (
            T& item,
            unsigned long& sending_node_id
        ); 
        /*!
            ensures
                - if (this function returns true) then
                    - #item == the next message which was sent to the calling processing
                      node.
                    - #sending_node_id == the node id of the node that sent this message.
                    - #sending_node_id < number_of_nodes()
                - else
                    - There were no other messages to receive and all other processing
                      nodes are blocked on calls to receive().
        !*/

        void receive (
        );
        /*!
            ensures
                - simply waits for all other nodes to become blocked
                  on calls to receive() or to terminate (i.e. waits for
                  other nodes to be in a state that can't send messages).
            throws
                - socket_error:
                  This exception is thrown if a message is received before this function
                  would otherwise return.  
        !*/

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct_type
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type funct
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type funct,
        ARG1 arg1
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type funct,
        ARG1 arg1,
        ARG2 arg2
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3,
        typename ARG4
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3,
        ARG4 arg4
    );

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct_type
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type funct
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type funct,
        ARG1 arg1
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type funct,
        ARG1 arg1,
        ARG2 arg2
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3
    );

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3,
        typename ARG4
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3,
        ARG4 arg4
    );

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BsP_ABSTRACT_H__

