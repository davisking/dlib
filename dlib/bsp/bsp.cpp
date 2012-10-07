// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "bsp.h"
#include "../ref.h"

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

namespace dlib
{

    namespace impl
    {

        struct hostinfo
        {
            hostinfo() {}
            hostinfo (
                const std::string& ip_,
                unsigned short port_,
                unsigned long node_id_
            ) : 
                ip(ip_),
                port(port_),
                node_id(node_id_)
            {
            }

            std::string ip;
            unsigned short port;
            unsigned long node_id;
        };

        void connect_all (
            map_id_to_con& cons,
            const std::vector<std::pair<std::string,unsigned short> >& hosts,
            unsigned long node_id
        )
        {
            cons.clear();
            for (unsigned long i = 0; i < hosts.size(); ++i)
            {
                scoped_ptr<bsp_con> con(new bsp_con(hosts[i]));
                serialize(node_id, con->stream); // tell the other end our node_id
                unsigned long id = i+1;
                cons.add(id, con);
            }
        }

        void connect_all_hostinfo (
            map_id_to_con& cons,
            const std::vector<hostinfo>& hosts,
            unsigned long node_id
        )
        {
            cons.clear();
            for (unsigned long i = 0; i < hosts.size(); ++i)
            {
                scoped_ptr<bsp_con> con(new bsp_con(make_pair(hosts[i].ip,hosts[i].port)));
                serialize(node_id, con->stream); // tell the other end our node_id
                con->stream.flush();
                unsigned long id = hosts[i].node_id;
                cons.add(id, con);
            }
        }


        void serialize (
            const hostinfo& item,
            std::ostream& out
        )
        {
            dlib::serialize(item.ip, out);
            dlib::serialize(item.port, out);
            dlib::serialize(item.node_id, out);
        }

        void deserialize (
            hostinfo& item,
            std::istream& in
        )
        {
            dlib::deserialize(item.ip, in);
            dlib::deserialize(item.port, in);
            dlib::deserialize(item.node_id, in);
        }

        void send_out_connection_orders (
            map_id_to_con& cons,
            const std::vector<std::pair<std::string,unsigned short> >& hosts
        )
        {
            // tell everyone their node ids
            cons.reset();
            while (cons.move_next())
            {
                dlib::serialize(cons.element().key(), cons.element().value()->stream);
            }

            // now tell them who to connect to
            std::vector<hostinfo> targets; 
            for (unsigned long i = 0; i < hosts.size(); ++i)
            {
                hostinfo info(hosts[i].first, hosts[i].second, i+1);

                dlib::serialize(targets, cons[info.node_id]->stream);
                targets.push_back(info);

                // let the other host know how many incoming connections to expect
                const unsigned long num = hosts.size()-targets.size();
                dlib::serialize(num, cons[info.node_id]->stream);
                cons[info.node_id]->stream.flush();
            }
        }

    // ------------------------------------------------------------------------------------

        // These control bytes are sent before each message nodes send to each other.
        const static char MESSAGE_HEADER         = 0;
        const static char WAITING_ON_RECEIVE     = 1;
        const static char NOT_WAITING_ON_RECEIVE = 2;
        const static char ALL_NODES_WAITING      = 3;
        const static char SENT_MESSAGE           = 4;
        const static char GOT_MESSAGE            = 5;
        const static char NODE_TERMINATE         = 6;

    // ------------------------------------------------------------------------------------

        void listen_and_connect_all(
            unsigned long& node_id,
            map_id_to_con& cons,
            unsigned short port
        )
        {
            cons.clear();
            scoped_ptr<listener> list;
            const int status = create_listener(list, port);
            if (status == PORTINUSE)
            {
                throw socket_error("Unable to create listening port " + cast_to_string(port) +
                                   ".  The port is already in use");
            }
            else if (status != 0)
            {
                throw socket_error("Unable to create listening port " + cast_to_string(port) );
            }

            scoped_ptr<connection> con;
            if (list->accept(con))
            {
                throw socket_error("Error occurred while accepting new connection");
            }

            scoped_ptr<bsp_con> temp(new bsp_con(con));

            unsigned long remote_node_id;
            dlib::deserialize(remote_node_id, temp->stream);
            dlib::deserialize(node_id, temp->stream);
            std::vector<hostinfo> targets; 
            dlib::deserialize(targets, temp->stream);
            unsigned long num_incoming_connections;
            dlib::deserialize(num_incoming_connections, temp->stream);

            cons.add(remote_node_id,temp);

            // make a thread that will connect to all the targets
            map_id_to_con cons2;
            thread_function thread(impl::connect_all_hostinfo, ref(cons2), ref(targets), node_id);

            // accept any incoming connections
            for (unsigned long i = 0; i < num_incoming_connections; ++i)
            {
                // If it takes more than 10 seconds for the other nodes to connect to us
                // then something has gone horribly wrong and it almost certainly will
                // never connect at all.  So just give up if that happens.
                const unsigned long timeout_milliseconds = 10000;
                if (list->accept(con, timeout_milliseconds))
                {
                    throw socket_error("Error occurred while accepting new connection");
                }

                temp.reset(new bsp_con(con));

                dlib::deserialize(remote_node_id, temp->stream);
                cons.add(remote_node_id,temp);
            }


            // put all the connections created by the thread into cons
            thread.wait();
            while (cons2.size() > 0)
            {
                unsigned long id;
                scoped_ptr<bsp_con> temp;
                cons2.remove_any(id,temp);
                cons.add(id,temp);
            }
        }
    }
    
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                          IMPLEMENTATION OF bsp_context OBJECT MEMBERS
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void bsp_context::
    close_all_connections_gracefully(
    )
    {
        if (_node_id == 0)
        {
            // Wait for all the other nodes to terminate before we do anything since
            // we are the controller node.
            receive();
        }

        _cons.reset();
        while (_cons.move_next())
        {
            // tell the other end that we are intentionally dropping the connection
            serialize(impl::NODE_TERMINATE,_cons.element().value()->stream);
            _cons.element().value()->stream.flush();
            _cons.element().value()->con->shutdown();
        }

        check_for_errors();
    }

// ----------------------------------------------------------------------------------------

    bsp_context::
    ~bsp_context()
    {
        _cons.reset();
        while (_cons.move_next())
        {
            _cons.element().value()->con->shutdown();
        }


        // this will wait for all the threads to terminate
        threads.clear();
    }

// ----------------------------------------------------------------------------------------

    bsp_context::
    bsp_context(
        unsigned long node_id_,
        impl::map_id_to_con& cons_
    ) :
        read_thread_terminated_improperly(false),
        outstanding_messages(0),
        num_waiting_nodes(0),
        buf_not_empty(class_mutex),
        _cons(cons_),
        _node_id(node_id_)
    {
        // spawn a bunch of read threads, one for each connection
        member_function_pointer<impl::bsp_con*, unsigned long>::kernel_1a_c mfp;
        mfp.set(*this, &bsp_context::read_thread);
        _cons.reset();
        while (_cons.move_next())
        {
            scoped_ptr<thread_function> ptr(new thread_function(mfp,
                                                                _cons.element().value().get(),
                                                                _cons.element().key()));
            threads.push_back(ptr);
        }

    }

// ----------------------------------------------------------------------------------------

    bool bsp_context::
    receive_data (
        shared_ptr<std::string>& item,
        unsigned long& sending_node_id
    ) 
    {
        using namespace impl;
        // If there aren't any other nodes then you will never receive anything.
        if (_cons.size() == 0)
            return false;

        {
            auto_mutex lock(class_mutex);
            if (msg_buffer.size() == 0)
            {
                send_to_master_node(WAITING_ON_RECEIVE);
                while (msg_buffer.size() == 0 && !read_thread_terminated_improperly)
                {
                    buf_not_empty.wait();
                }
                if (read_thread_terminated_improperly)
                {
                    throw dlib::socket_error("A connection between processing nodes has been lost.");
                }
                send_to_master_node(NOT_WAITING_ON_RECEIVE);
            }

            sending_node_id = msg_sender_id.front();
            msg_sender_id.pop_front();
            item = msg_buffer.front();
            msg_buffer.pop_front();
        }

        // if this is a message from another node rather than the
        // "everyone is blocked on receive() message".
        if (item)
        {
            send_to_master_node(GOT_MESSAGE);
            return true;
        }
        else
        {
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    send_to_master_node (
        char msg
    )
    {
        using namespace impl;
        // if we aren't the special controlling node then send the
        // controller a message.
        if (_cons.is_in_domain(0))
        {
            serialize(msg, _cons[0]->stream);
            _cons[0]->stream.flush();
        }
        else if (_node_id == 0) // if this is the master node
        {
            // since we are the master node we will just modify our state directly
            auto_mutex lock(class_mutex);
            switch(msg)
            {
                case WAITING_ON_RECEIVE: {
                    ++num_waiting_nodes;
                    notify_everyone_if_all_blocked();
                } break;

                case NOT_WAITING_ON_RECEIVE: {
                    --num_waiting_nodes;
                } break;

                case SENT_MESSAGE: {
                    ++outstanding_messages;
                } break;

                case GOT_MESSAGE: {
                    --outstanding_messages;
                } break;

                default:
                    DLIB_CASSERT(false,"this should not happen");
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    notify_everyone_if_all_blocked(
    )
    {
        using namespace impl;
        // if all the nodes are blocked on receive() and there aren't any
        // messages in flight.
        if (_node_id == 0 && num_waiting_nodes == number_of_nodes() && outstanding_messages == 0)
        {
            // send notifications
            _cons.reset();
            while (_cons.move_next())
            {
                try
                {
                    // Skip connections to nodes that have already terminated their
                    // execution.
                    if (_cons.element().value()->terminated == false)
                    {
                        serialize(ALL_NODES_WAITING, _cons.element().value()->stream);
                        _cons.element().value()->stream.flush();
                        if (!_cons.element().value()->stream)
                            throw dlib::error("Error writing data to TCP connection");
                    }
                }
                catch (std::exception& e)
                {
                    const connection* const con = _cons.element().value()->con.get();
                    std::ostringstream sout;
                    sout << "An exception occurred in the controlling node while it was trying to communicate with a listening node.\n";
                    sout << "  Listening processing node address:   " << con->get_foreign_ip() << ":" << con->get_foreign_port() << std::endl;
                    sout << "  Controlling processing node address: " << con->get_local_ip() << ":" << con->get_local_port() << std::endl;
                    sout << "  Error message in the exception: " << e.what() << std::endl;
                    error_message = sout.str();
                }
            }

            // unblock the control node itself
            shared_ptr<std::string> msg;
            msg_buffer.push_back(msg);
            msg_sender_id.push_back(0);
            buf_not_empty.signal();
        }
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    read_thread (
        impl::bsp_con* con,
        unsigned long sender_id
    )
    {
        try
        {
            using namespace impl;
            while (con->stream.peek() != EOF)
            {
                char header;
                deserialize(header, con->stream);
                switch (header)
                {
                    case MESSAGE_HEADER: {
                        shared_ptr<std::string> msg(new std::string);
                        deserialize(*msg, con->stream);

                        auto_mutex lock(class_mutex);
                        msg_buffer.push_back(msg);
                        msg_sender_id.push_back(sender_id);
                        buf_not_empty.signal();
                    } break;

                    case WAITING_ON_RECEIVE: {
                        auto_mutex lock(class_mutex);
                        ++num_waiting_nodes;
                        notify_everyone_if_all_blocked();
                    } break;

                    case NOT_WAITING_ON_RECEIVE: {
                        auto_mutex lock(class_mutex);
                        --num_waiting_nodes;
                    } break;

                    case ALL_NODES_WAITING: {
                        // put something into the message buffer that lets 
                        // receive() know to return false.  We do this using
                        // a null msg pointer.
                        auto_mutex lock(class_mutex);
                        shared_ptr<std::string> msg;
                        msg_buffer.push_back(msg);
                        msg_sender_id.push_back(sender_id);
                        buf_not_empty.signal();
                    } break;

                    case SENT_MESSAGE: {
                        auto_mutex lock(class_mutex);
                        ++outstanding_messages;
                    } break;

                    case GOT_MESSAGE: {
                        auto_mutex lock(class_mutex);
                        --outstanding_messages;
                    } break;

                    case NODE_TERMINATE: {
                        auto_mutex lock(class_mutex);
                        if (_node_id == 0)
                        {
                            // a terminating node is basically the same as a node that waits forever.
                            _cons[sender_id]->terminated = true;
                            ++num_waiting_nodes; 
                            notify_everyone_if_all_blocked();
                        }
                        return;
                    } break;
                }
            }
        }
        catch (std::exception& e)
        {
            std::ostringstream sout;
            sout << "An exception was thrown while attempting to receive a message from processing node " << sender_id << ".\n";
            sout << "  Sending processing node address:   " << con->con->get_foreign_ip() << ":" << con->con->get_foreign_port() << std::endl;
            sout << "  Receiving processing node address: " << con->con->get_local_ip() << ":" << con->con->get_local_port() << std::endl;
            sout << "  Error message in the exception: " << e.what() << std::endl;
            auto_mutex lock(class_mutex);
            error_message = sout.str();
        }

        auto_mutex lock(class_mutex);
        read_thread_terminated_improperly = true;
        buf_not_empty.signal();
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    check_for_errors()
    {
        auto_mutex lock(class_mutex);
        if (error_message.size() != 0)
            throw dlib::socket_error(error_message);
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    send_data(
        const std::string& item,
        unsigned long target_node_id
    ) 
    {
        using namespace impl;
        if (_cons[target_node_id]->terminated)
            throw socket_error("Attempt to send a message to a node that has terminated.");

        serialize(MESSAGE_HEADER, _cons[target_node_id]->stream);
        serialize(item, _cons[target_node_id]->stream);
        _cons[target_node_id]->stream.flush();
        send_to_master_node(SENT_MESSAGE);
    }

// ----------------------------------------------------------------------------------------

}

