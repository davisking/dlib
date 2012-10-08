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
#include <vector>

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
                stream(&buf),
                terminated(false)
            {}

            bsp_con(
               scoped_ptr<connection>& conptr 
            ) : 
                buf(conptr),
                stream(&buf),
                terminated(false)
            {
                // make sure we own the connection
                conptr.swap(con);
            }

            scoped_ptr<connection> con;
            sockstreambuf::kernel_2a buf;
            std::iostream stream;
            bool terminated;
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

    class bsp_context : noncopyable
    {

    public:

        template <typename T>
        void send(
            const T& item,
            unsigned long target_node_id
        ) 
        {
            std::ostringstream sout;
            serialize(item, sout);
            send_data(sout.str(), target_node_id);
        }

        template <typename T>
        void broadcast (
            const T& item
        ) 
        {
            std::ostringstream sout;
            serialize(item, sout);
            for (unsigned long i = 0; i < number_of_nodes(); ++i)
            {
                // Don't send to yourself.
                if (i == node_id())
                    continue;

                send_data(sout.str(), i);
            }
        }

        unsigned long node_id (
        ) const { return _node_id; }

        unsigned long number_of_nodes (
        ) const { return _cons.size()+1; }

        void receive (
        )
        {
            unsigned long id;
            shared_ptr<std::string> temp;
            if (receive_data(temp,id))
                throw dlib::socket_error("Call to bsp_context::receive() got an unexpected message.");
        }

        template <typename T>
        bool receive (
            T& item
        ) 
        {
            unsigned long sending_node_id;
            return receive(item, sending_node_id);
        }

        template <typename T>
        bool receive (
            T& item,
            unsigned long& sending_node_id
        ) 
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

        ~bsp_context();

    private:

        bsp_context();

        bsp_context(
            unsigned long node_id_,
            impl::map_id_to_con& cons_
        );

        void close_all_connections_gracefully();
        /*!
            ensures
                - closes all the connections to other nodes and lets them know that
                  we are terminating normally rather than as the result of some kind
                  of error.
        !*/

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
        bool read_thread_terminated_improperly; // true if any of our connections goes down.
        unsigned long outstanding_messages;
        unsigned long num_waiting_nodes;
        unsigned long num_terminated_nodes;
        rsignaler buf_not_empty; // used to signal when msg_buffer isn't empty
        rsignaler terminated_signal; 
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
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            funct_type& funct
        );

        template <
            typename funct_type,
            typename ARG1
            >
        friend void bsp_connect (
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            funct_type& funct,
            ARG1 arg1
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2
            >
        friend void bsp_connect (
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2,
            typename ARG3
            >
        friend void bsp_connect (
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            ARG3 arg3
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2,
            typename ARG3,
            typename ARG4
            >
        friend void bsp_connect (
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            ARG3 arg3,
            ARG4 arg4
        );

    // -----------------------------------

        template <
            typename funct_type
            >
        friend void bsp_listen (
            unsigned short listening_port,
            funct_type& funct
        );

        template <
            typename funct_type,
            typename ARG1
            >
        friend void bsp_listen (
            unsigned short listening_port,
            funct_type& funct,
            ARG1 arg1
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2
            >
        friend void bsp_listen (
            unsigned short listening_port,
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2,
            typename ARG3
            >
        friend void bsp_listen (
            unsigned short listening_port,
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            ARG3 arg3
        );

        template <
            typename funct_type,
            typename ARG1,
            typename ARG2,
            typename ARG3,
            typename ARG4
            >
        friend void bsp_listen (
            unsigned short listening_port,
            funct_type& funct,
            ARG1 arg1,
            ARG2 arg2,
            ARG3 arg3,
            ARG4 arg4
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
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type& funct
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp_context obj(node_id, cons);
        funct(obj);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type& funct,
        ARG1 arg1
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp_context obj(node_id, cons);
        funct(obj,arg1);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp_context obj(node_id, cons);
        funct(obj,arg1,arg2);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3
        >
    void bsp_connect (
        const std::vector<std::pair<std::string,unsigned short> >& hosts,
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp_context obj(node_id, cons);
        funct(obj,arg1,arg2,arg3);
        obj.close_all_connections_gracefully();
    }

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
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3,
        ARG4 arg4
    )
    {
        impl::map_id_to_con cons;
        const unsigned long node_id = 0;
        connect_all(cons, hosts, node_id);
        send_out_connection_orders(cons, hosts);
        bsp_context obj(node_id, cons);
        funct(obj,arg1,arg2,arg3,arg4);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct_type
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type& funct
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp_context obj(node_id, cons);
        funct(obj);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type& funct,
        ARG1 arg1
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp_context obj(node_id, cons);
        funct(obj,arg1);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp_context obj(node_id, cons);
        funct(obj,arg1,arg2);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct_type,
        typename ARG1,
        typename ARG2,
        typename ARG3
        >
    void bsp_listen (
        unsigned short listening_port,
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp_context obj(node_id, cons);
        funct(obj,arg1,arg2,arg3);
        obj.close_all_connections_gracefully();
    }

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
        funct_type& funct,
        ARG1 arg1,
        ARG2 arg2,
        ARG3 arg3,
        ARG4 arg4
    )
    {
        impl::map_id_to_con cons;
        unsigned long node_id;
        listen_and_connect_all(node_id, cons, listening_port);
        bsp_context obj(node_id, cons);
        funct(obj,arg1,arg2,arg3,arg4);
        obj.close_all_connections_gracefully();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "bsp.cpp"
#endif

#endif // DLIB_BsP_H__

