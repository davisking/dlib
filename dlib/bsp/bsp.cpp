// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "bsp.h"
#include "../ref.h"
#include <stack>

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

namespace dlib
{

    namespace impl1
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
            unsigned long node_id,
            std::string& error_string 
        )
        {
            cons.clear();
            for (unsigned long i = 0; i < hosts.size(); ++i)
            {
                try
                {
                    scoped_ptr<bsp_con> con(new bsp_con(make_pair(hosts[i].ip,hosts[i].port)));
                    serialize(node_id, con->stream); // tell the other end our node_id
                    con->stream.flush();
                    unsigned long id = hosts[i].node_id;
                    cons.add(id, con);
                }
                catch (std::exception& e)
                {
                    std::ostringstream sout;
                    sout << "Could not connect to " << hosts[i].ip << ":" << hosts[i].port;
                    error_string = sout.str();
                    break;
                }
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
            std::string error_string;
            thread_function thread(connect_all_hostinfo, ref(cons2), ref(targets), node_id, ref(error_string));
            if (error_string.size() != 0)
                throw socket_error(error_string);

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
// ----------------------------------------------------------------------------------------

    namespace impl2
    {
        // These control bytes are sent before each message nodes send to each other.

        // denotes a normal content message.
        const static char MESSAGE_HEADER            = 0; 

        // sent back to sender, means message was returned by receive().
        const static char GOT_MESSAGE               = 1; 

        // broadcast when a node goes into a state where it has no outstanding sent
        // messages (i.e. it received GOT_MESSAGE for all its sent messages) and is waiting
        // on receive().
        const static char IN_WAITING_STATE          = 2; 

        // broadcast when no longer in IN_WAITING_STATE state.
        const static char NOT_IN_WAITING_STATE      = 3; 

        // broadcast when a node terminates itself. 
        const static char NODE_TERMINATE            = 4; 

        // broadcast when a node finds out that all non-terminated nodes are in the
        // IN_WAITING_STATE state.  sending this message puts a node into the
        // SEE_ALL_IN_WAITING_STATE where it will wait until it gets this message from all
        // others and then return from receive() once this happens.
        const static char SEE_ALL_IN_WAITING_STATE  = 5;


        const static char READ_ERROR                = 6;

    // ------------------------------------------------------------------------------------

        void read_thread (
            impl1::bsp_con* con,
            unsigned long node_id,
            unsigned long sender_id,
            impl1::thread_safe_deque& msg_buffer
        )
        {
            try
            {
                while(true)
                {
                    impl1::msg_data msg;
                    deserialize(msg.msg_type, con->stream);
                    msg.sender_id = sender_id;

                    if (msg.msg_type == MESSAGE_HEADER)
                    {
                        msg.data.reset(new std::string);
                        deserialize(*msg.data, con->stream);
                    }

                    msg_buffer.push_and_consume(msg);

                    if (msg.msg_type == NODE_TERMINATE)
                        break;
                }
            }
            catch (std::exception& e)
            {
                std::ostringstream sout;
                sout << "An exception was thrown while attempting to receive a message from processing node " << sender_id << ".\n";
                sout << "  Sending processing node address:   " << con->con->get_foreign_ip() << ":" << con->con->get_foreign_port() << std::endl;
                sout << "  Receiving processing node address: " << con->con->get_local_ip() << ":" << con->con->get_local_port() << std::endl;
                sout << "  Receiving processing node id:      " << node_id << std::endl;
                sout << "  Error message in the exception:    " << e.what() << std::endl;

                impl1::msg_data msg;
                msg.sender_id = sender_id;
                msg.msg_type = READ_ERROR;
                msg.data.reset(new std::string);
                *msg.data = sout.str();

                msg_buffer.push_and_consume(msg);
            }
            catch (...)
            {
                std::ostringstream sout;
                sout << "An exception was thrown while attempting to receive a message from processing node " << sender_id << ".\n";
                sout << "  Sending processing node address:   " << con->con->get_foreign_ip() << ":" << con->con->get_foreign_port() << std::endl;
                sout << "  Receiving processing node address: " << con->con->get_local_ip() << ":" << con->con->get_local_port() << std::endl;
                sout << "  Receiving processing node id:      " << node_id << std::endl;

                impl1::msg_data msg;
                msg.sender_id = sender_id;
                msg.msg_type = READ_ERROR;
                msg.data.reset(new std::string);
                *msg.data = sout.str();

                msg_buffer.push_and_consume(msg);
            }
        }

    // ------------------------------------------------------------------------------------

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
        _cons.reset();
        while (_cons.move_next())
        {
            // tell the other end that we are intentionally dropping the connection
            serialize(impl2::NODE_TERMINATE,_cons.element().value()->stream);
            _cons.element().value()->stream.flush();
        }

        impl1::msg_data msg;
        // now wait for all the other nodes to terminate
        while (num_terminated_nodes < _cons.size() )
        {
            if (!msg_buffer.pop(msg))
                throw dlib::socket_error("Error reading from msg_buffer in dlib::bsp_context.");

            if (msg.msg_type == impl2::NODE_TERMINATE)
                ++num_terminated_nodes;
            else if (msg.msg_type == impl2::READ_ERROR)
                throw dlib::socket_error(*msg.data);
            else if (msg.msg_type == impl2::GOT_MESSAGE)
                --outstanding_messages;
        }

        if (outstanding_messages != 0)
        {
            std::ostringstream sout;
            sout << "A BSP job was allowed to terminate before all sent messages have been received.\n";
            sout << "There are at least " << outstanding_messages << " messages still in flight.   Make sure all sent messages\n";
            sout << "have a corresponding call to receive().";
            throw dlib::socket_error(sout.str());
        }
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

        msg_buffer.disable();

        // this will wait for all the threads to terminate
        threads.clear();
    }

// ----------------------------------------------------------------------------------------

    bsp_context::
    bsp_context(
        unsigned long node_id_,
        impl1::map_id_to_con& cons_
    ) :
        outstanding_messages(0),
        num_waiting_nodes(0),
        num_terminated_nodes(0),
        _cons(cons_),
        _node_id(node_id_)
    {
        // spawn a bunch of read threads, one for each connection
        _cons.reset();
        while (_cons.move_next())
        {
            scoped_ptr<thread_function> ptr(new thread_function(&impl2::read_thread,
                                                                _cons.element().value().get(),
                                                                _node_id,
                                                                _cons.element().key(),
                                                                ref(msg_buffer)));
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
        if (outstanding_messages == 0)
            broadcast_byte(impl2::IN_WAITING_STATE);

        unsigned long num_in_see_all_in_waiting_state = 0;
        bool sent_see_all_in_waiting_state = false;
        std::stack<impl1::msg_data> buf;

        while (true)
        {
            // if there aren't any nodes left to give us messages then return right now.
            if (num_terminated_nodes == _cons.size())
                return false;

            // if all running nodes are currently blocking forever on receive_data()
            if (outstanding_messages == 0 && num_terminated_nodes + num_waiting_nodes == _cons.size())
            {
                num_waiting_nodes = 0;
                sent_see_all_in_waiting_state = true;
                broadcast_byte(impl2::SEE_ALL_IN_WAITING_STATE);
            }

            impl1::msg_data data;
            if (!msg_buffer.pop(data))
                throw dlib::socket_error("Error reading from msg_buffer in dlib::bsp_context.");

            if (sent_see_all_in_waiting_state)
            {
                // Once we have gotten one SEE_ALL_IN_WAITING_STATE, all we care about is
                // getting the rest of them.  So the effect of this code is to always move
                // any SEE_ALL_IN_WAITING_STATE messages to the front of the message queue.
                if (data.msg_type != impl2::SEE_ALL_IN_WAITING_STATE)
                {
                    buf.push(data);
                    continue;
                }
            }

            switch(data.msg_type)
            {
                case impl2::MESSAGE_HEADER: {
                    item = data.data;
                    sending_node_id = data.sender_id;

                    // if we would have send the IN_WAITING_STATE message before getting to
                    // this point then let other nodes know that we aren't waiting anymore.
                    if (outstanding_messages == 0)
                        broadcast_byte(impl2::NOT_IN_WAITING_STATE);

                    send_byte(impl2::GOT_MESSAGE, data.sender_id);

                    return true;

                } break;

                case impl2::IN_WAITING_STATE: {
                    ++num_waiting_nodes;
                } break;

                case impl2::NOT_IN_WAITING_STATE: {
                    --num_waiting_nodes;
                } break;

                case impl2::GOT_MESSAGE: {
                    --outstanding_messages;
                    if (outstanding_messages == 0)
                        broadcast_byte(impl2::IN_WAITING_STATE);
                } break;

                case impl2::NODE_TERMINATE: {
                    ++num_terminated_nodes;
                    _cons[data.sender_id]->terminated = true;
                    if (num_terminated_nodes == _cons.size())
                    {
                        return false;
                    }
                } break;

                case impl2::SEE_ALL_IN_WAITING_STATE: {
                    ++num_in_see_all_in_waiting_state;
                    if (num_in_see_all_in_waiting_state + num_terminated_nodes == _cons.size())
                    {
                        // put stuff from buf back into msg_buffer
                        while (buf.size() != 0)
                        {
                            msg_buffer.push_front(buf.top());
                            buf.pop();
                        }
                        return false;
                    }
                } break;

                case impl2::READ_ERROR: {
                    throw dlib::socket_error(*data.data);
                } break;

                default: {
                    throw dlib::socket_error("Unknown message received by dlib::bsp_context");
                } break;
            } // end switch()
        } // end while (true)
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    send_byte (
        char val,
        unsigned long target_node_id
    )
    {
        serialize(val, _cons[target_node_id]->stream);
        _cons[target_node_id]->stream.flush();
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    broadcast_byte (
        char val
    )
    {
        for (unsigned long i = 0; i < number_of_nodes(); ++i)
        {
            // don't send to yourself or to terminated nodes
            if (i == node_id() || _cons[i]->terminated)
                continue;

            send_byte(val,i);
        }
    }

// ----------------------------------------------------------------------------------------

    void bsp_context::
    send_data(
        const std::string& item,
        unsigned long target_node_id
    ) 
    {
        using namespace impl2;
        if (_cons[target_node_id]->terminated)
            throw socket_error("Attempt to send a message to a node that has terminated.");

        serialize(MESSAGE_HEADER, _cons[target_node_id]->stream);
        serialize(item, _cons[target_node_id]->stream);
        _cons[target_node_id]->stream.flush();

        ++outstanding_messages;
    }

// ----------------------------------------------------------------------------------------

}

