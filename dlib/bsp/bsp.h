// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BsP_H__
#define DLIB_BsP_H__

#include "bsp_abstract.h"
#include "../sockets.h"
#include "../array.h"
#include "../smart_pointers.h"
#include "../sockstreambuf.h"
#include "../string.h"
#include "../serialize.h"
#include "../map.h"
#include <deque>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        struct bsp_con
        {
            bsp_con(
                const std::pair<std::string,unsigned short>& dest
            ) : 
                con(connect(dest.first,dest.second)),
                buf(con),
                stream(&buf)
            {}

            bsp_con(
               scoped_ptr<connection>& conptr 
            ) : 
                buf(conptr),
                stream(&buf)
            {
                // make sure we own the connection
                conptr.swap(con);
            }

            scoped_ptr<connection> con;
            sockstreambuf::kernel_2a buf;
            std::iostream stream;
        };

        typedef dlib::map<unsigned long, scoped_ptr<bsp_con> >::kernel_1a_c map_id_to_con;

        void connect_all (
            map_id_to_con& cons,
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            unsigned long node_id
        );
        /*!
            ensures
                - creates connections to all the given hosts and stores them into cons
        !*/

        void send_out_connection_orders (
            map_id_to_con& cons,
            const std::vector<std::pair<std::string,unsigned short> >& hosts
        );

        void listen_and_connect_all(
            unsigned long& node_id,
            map_id_to_con& cons,
            unsigned short port
        );
    }

// ----------------------------------------------------------------------------------------

    class bsp : noncopyable
    {

    public:

        template <typename T>
        void send(
            const T& item,
            unsigned long target_node_id
        ) 
        /*!
            requires
                - item is serializable 
                - target_node_id < number_of_nodes()
                - target_node_id != node_id()
            ensures
                - sends a copy of item to the node with the given id.
        !*/
        {
            std::ostringstream sout;
            serialize(item, sout);
            send_data(sout.str(), target_node_id);
        }

        template <typename T>
        void broadcast (
            const T& item
        ) 
        /*!
            ensures
                - sends a copy of item to all other processing nodes.
        !*/
        {
            std::ostringstream sout;
            serialize(item, sout);
            for (unsigned long i = 0; i < number_of_nodes(); ++i)
            {
                if (i == node_id())
                    continue;
                send_data(sout.str(), i);
            }
        }

        unsigned long node_id (
        ) const { return _node_id; }
        /*!
            ensures
                - Returns the id of the current processing node.  That is, 
                  returns a number N such that:
                    - N < number_of_nodes()
                    - N == the node id of the processing node that called
                      node_id().
        !*/

        unsigned long number_of_nodes (
        ) const { return _cons.size()+1; }
        /*!
            ensures
                - returns the number of processing nodes participating in the
                  BSP computation.
        !*/

        template <typename T>
        bool receive (
            T& item
        ) 
        /*!
            ensures
                - if (this function returns true) then
                    - #item == the next message which was sent to the calling processing
                      node.
                - else
                    - There were no other messages to receive and all other processing
                      nodes are blocked on calls to receive().
        !*/
        {
            unsigned long sending_node_id;
            return receive(item, sending_node_id);
        }

        template <typename T>
        bool receive (
            T& item,
            unsigned long& sending_node_id
        ) 
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
        {
            shared_ptr<std::string> temp;
            if (receive_data(temp, sending_node_id))
            {
                std::istringstream sin(*temp);
                deserialize(item, sin);
                return true;
            }
            else
            {
                return false;
            }
        }

        ~bsp();

    private:

        bsp();

        bsp(
            unsigned long node_id_,
            impl::map_id_to_con& cons_
        );

        bool receive_data (
            shared_ptr<std::string>& item,
            unsigned long& sending_node_id
        );

        void send_to_master_node (
            char msg
        );

        void notify_everyone_if_all_blocked(
        );
        /*!
            requires
                - class_mutex is locked
            ensures
                - sends out notifications to all the nodes if we are all blocked on receive.  This
                  will cause all receive calls to unblock and return false.
        !*/

        void read_thread (
            impl::bsp_con* con,
            unsigned long sender_id
        );


        void check_for_errors();

        void send_data(
            const std::string& item,
            unsigned long target_node_id
        );
        /*!
            requires
                - target_node_id < number_of_nodes()
                - target_node_id != node_id()
            ensures
                - sends a copy of item to the node with the given id.
        !*/



        rmutex class_mutex; // used to lock any class members touched from more than one thread. 
        std::string error_message;
        bool read_thread_terminated; // true if any of our connections goes down.
        unsigned long outstanding_messages;
        unsigned long num_waiting_nodes;
        rsignaler buf_not_empty; // used to signal when msg_buffer isn't empty
        std::deque<shared_ptr<std::string> > msg_buffer;
        std::deque<unsigned long> msg_sender_id;

        impl::map_id_to_con& _cons;
        const unsigned long _node_id;
        array<scoped_ptr<thread_function> > threads;

    // -----------------------------------

        template <
            typename funct_type
            >
        friend void bsp_connect (
            funct_type& funct,
            const std::vector<std::pair<std::string,unsigned short> >& hosts
        );

        template <
            typename funct_type,
            typename ARG1
            >
        friend void bsp_connect (
            funct_type& funct,
            ARG1 arg1,
            const std::vector<std::pair<std::string,unsigned short> >& hosts
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2
            >
        friend void bsp_connect (
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            const std::vector<std::pair<std::string,unsigned short> >& hosts
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2,
            typename ARG3
            >
        friend void bsp_connect (
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            ARG3 arg3,
            const std::vector<std::pair<std::string,unsigned short> >& hosts
        );

    // -----------------------------------

        template <
            typename funct_type
            >
        friend void bsp_listen (
            funct_type& funct,
            unsigned short listening_port
        );

        template <
            typename funct_type,
            typename ARG1
            >
        friend void bsp_listen (
            funct_type& funct,
            ARG1 arg1,
            unsigned short listening_port
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2
            >
        friend void bsp_listen (
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            unsigned short listening_port
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2,
            typename ARG3
            >
        friend void bsp_listen (
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            ARG3 arg3,
            unsigned short listening_port
        );

    // -----------------------------------

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct_type
        >
    void bsp_connect (
        funct_type& funct,
        const std::vector<std::pair<std::string,unsigned short> >& hosts
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp obj(node_id, cons);
        funct(obj);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1
        >
    void bsp_connect (
        funct_type& funct,
        ARG1 arg1,
        const std::vector<std::pair<std::string,unsigned short> >& hosts
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp obj(node_id, cons);
        funct(obj,arg1);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2
        >
    void bsp_connect (
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        const std::vector<std::pair<std::string,unsigned short> >& hosts
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp obj(node_id, cons);
        funct(obj,arg1,arg2);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3
        >
    void bsp_connect (
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3,
        const std::vector<std::pair<std::string,unsigned short> >& hosts
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp obj(node_id, cons);
        funct(obj,arg1,arg2,arg3);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct_type
        >
    void bsp_listen (
        funct_type& funct,
        unsigned short listening_port
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp obj(node_id, cons);
        funct(obj);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1
        >
    void bsp_listen (
        funct_type& funct,
        ARG1 arg1,
        unsigned short listening_port
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp obj(node_id, cons);
        funct(obj,arg1);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2
        >
    void bsp_listen (
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        unsigned short listening_port
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp obj(node_id, cons);
        funct(obj,arg1,arg2);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3
        >
    void bsp_listen (
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3,
        unsigned short listening_port
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp obj(node_id, cons);
        funct(obj,arg1,arg2,arg3);
        obj.check_for_errors();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "bsp.cpp"
#endif

#endif // DLIB_BsP_H__

