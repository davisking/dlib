// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SERVER_KERNEL_1_
#define DLIB_SERVER_KERNEL_1_

#include "server_kernel_abstract.h"

#include "../threads.h"
#include "../sockets.h"
#include <string>
#include "../algs.h"
#include "../set.h"
#include "../logger.h"
#include "../smart_pointers.h"


namespace dlib
{

    // These forward declarations are here so we can use them in the typedefs in the server
    // class.  The reason for this is for backwards compatibility with previous versions of
    // dlib.
    class server_http;
    class server_iostream;

    class server
    {


        /*!
            INITIAL VALUE
                listening_port          == 0
                listening_ip            == ""
                running                 == false
                shutting_down           == false
                cons.size()             == 0
                listening_port_mutex    == a mutex
                listening_ip_mutex      == a mutex
                running_mutex           == a mutex
                running_signaler        == a signaler associated with running_mutex
                shutting_down_mutex     == a mutex
                cons_mutex              == a mutex
                thread_count            == 0
                thread_count_mutex      == a mutex
                thread_count_signaler   == a signaler associated with thread_count_mutex
                thread_count_zero       == a signaler associated with thread_count_mutex
                max_connections         == 1000 
                max_connections_mutex   == a mutex for max_connections
             
            CONVENTION
                listening_port          == get_listening_port()
                listening_ip            == get_listening_ip()
                running                 == is_running()
                shutting_down           == true while clear() is running.  this
                                           bool is used to tell the thread blocked on
                                           accept that it should terminate
                cons                    == a set containing all open connections
                listening_port_mutex    == a mutex for listening_port
                listening_ip_mutex      == a mutex for listening_ip
                running_mutex           == a mutex for running
                running_signaler        == a signaler for running and
                                           is associated with running_mutex.  it is 
                                           used to signal when running is false
                shutting_down_mutex     == a mutex for shutting_down
                cons_mutex              == a mutex for cons
                thread_count            == the number of threads currently running
                thread_count_mutex      == a mutex for thread_count
                thread_count_signaler   == a signaler for thread_count and
                                           is associated with thread_count_mutex.  it
                                           is used to signal when thread_count is
                                           decremented  
                thread_count_zero       == a signaler for thread_count and
                                           is associated with thread_count_mutex.  it
                                           is used to signal when thread_count becomes
                                           zero
                max_connections         == get_max_connections()
                max_connections_mutex   == a mutex for max_connections
        !*/
        

        typedef set<connection*>::kernel_1a set_of_connections;

        // this structure is used to pass parameters to new threads
        struct param
        {
            param (
                server& server_,
                connection& new_connection_
            ) :
                the_server(server_),
                new_connection(new_connection_)
            {}

            server& the_server;
            connection& new_connection;
        };
        


        public:

            // These typedefs are here for backward compatibility with previous versions of dlib
            typedef     server kernel_1a;
            typedef     server kernel_1a_c;
            typedef     server_iostream iostream_1a;
            typedef     server_iostream iostream_1a_c;
            typedef     server_http http_1a;
            typedef     server_http http_1a_c;

            server(
            );

            virtual ~server(
            ); 

            void clear(
            );

            void start (
            );

            bool is_running ( 
            ) const;

            const std::string get_listening_ip (
            ) const;

            int get_listening_port (
            ) const;

            void set_listening_port (
                int port
            );

            void set_listening_ip (
                const std::string& ip
            );

            void set_max_connections (
                int max
            );

            int get_max_connections (
            ) const;

            void start_async (
            );

        private:

            void start_async_helper (
            );

            void start_accepting_connections (
            );

            void open_listening_socket (
            );

            virtual void on_connect (
                connection& new_connection
            )=0;

            virtual void on_listening_port_assigned (
            ) {}

            const static logger sdlog;

            static void service_connection(
                void* item
            );
            /*!
                requires
                    item is a pointer to a param struct
                ensures
                    services the new connection
                    will take care of closing the connection and 
                    adding the connection to cons when it first starts and
                    remove the connection from cons and signal that it has 
                    done so when it ends
            !*/

            // data members
            int listening_port;
            std::string listening_ip;
            bool running;
            bool shutting_down;
            set_of_connections cons;
            mutex listening_port_mutex;
            mutex listening_ip_mutex;
            rmutex running_mutex;
            rsignaler running_signaler;
            mutex shutting_down_mutex;
            mutex cons_mutex;
            int thread_count;
            mutex thread_count_mutex;
            signaler thread_count_signaler;
            int max_connections;
            mutex max_connections_mutex;
            signaler thread_count_zero;
            scoped_ptr<thread_function> async_start_thread;
            scoped_ptr<listener> sock;


            // restricted functions
            server(server&);   
            server& operator= (
                server&
                );    
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    server::
    server (
    ) :
        listening_port(0),
        running(false),
        shutting_down(false),
        running_signaler(running_mutex),
        thread_count(0),
        thread_count_signaler(thread_count_mutex),
        max_connections(1000),
        thread_count_zero(thread_count_mutex)
    {
    }

// ----------------------------------------------------------------------------------------

    server::
    ~server (
    )
    {
        clear();
    }

// ----------------------------------------------------------------------------------------

    int server::
    get_max_connections (
    ) const
    {
        max_connections_mutex.lock();
        int temp = max_connections;
        max_connections_mutex.unlock();
        return temp;
    }

// ----------------------------------------------------------------------------------------

    void server::
    set_max_connections (
        int max
    ) 
    {
        // make sure requires clause is not broken
        DLIB_CASSERT( 
            max >= 0 ,
            "\tvoid server::set_max_connections"
            << "\n\tmax == " << max
            << "\n\tthis: " << this
            );

        max_connections_mutex.lock();
        max_connections = max;
        max_connections_mutex.unlock();
    }

// ----------------------------------------------------------------------------------------

    void server::
    clear (
    )
    {
        // signal that we are shutting down
        shutting_down_mutex.lock();
        shutting_down = true;
        shutting_down_mutex.unlock();



        max_connections_mutex.lock();
        listening_port_mutex.lock();
        listening_ip_mutex.lock();
        listening_ip = "";        
        listening_port = 0;
        max_connections = 1000;
        listening_port_mutex.unlock();
        listening_ip_mutex.unlock();
        max_connections_mutex.unlock();


        // tell all the connections to shut down
        cons_mutex.lock();
        connection* temp;
        while (cons.size() > 0)
        {
            cons.remove_any(temp);
            temp->shutdown();
        }
        cons_mutex.unlock();


        // wait for all the connections to shut down 
        thread_count_mutex.lock();
        while (thread_count > 0)
        {
            thread_count_zero.wait();
        }
        thread_count_mutex.unlock();
        



        // wait for the listener to close
        running_mutex.lock();
        while (running == true)
        {
            running_signaler.wait();
        }
        running_mutex.unlock();



        // signal that the shutdown is complete
        shutting_down_mutex.lock();
        shutting_down = false;
        shutting_down_mutex.unlock();
    }

// ----------------------------------------------------------------------------------------

    void server::
    start_async_helper (
    )
    {
        try
        {
            start_accepting_connections();
        }
        catch (std::exception& e)
        {
            sdlog << LERROR << e.what();
        }
    }

// ----------------------------------------------------------------------------------------

    void server::
    start_async (
    )
    {
        auto_mutex lock(running_mutex);
        if (running)
            return;

        // Any exceptions likely to be thrown by the server are going to be
        // thrown when trying to bind the port.  So calling this here rather
        // than in the thread we are about to make will cause start_async()
        // to report errors back to the user in a very straight forward way.
        open_listening_socket();

        member_function_pointer<>::kernel_1a mfp;
        mfp.set(*this,&server::start_async_helper);
        async_start_thread.reset(new thread_function(mfp));
    }

// ----------------------------------------------------------------------------------------

    void server::
    open_listening_socket (
    )
    {
        if (!sock)
        {
            int status = create_listener(sock,listening_port,listening_ip);

            // if there was an error then clear this object
            if (status < 0)
            {
                max_connections_mutex.lock();
                listening_port_mutex.lock();
                listening_ip_mutex.lock();
                listening_ip = "";        
                listening_port = 0;
                max_connections = 1000;
                listening_port_mutex.unlock();
                listening_ip_mutex.unlock();
                max_connections_mutex.unlock();
            }



            // throw an exception for the error
            if (status == PORTINUSE)
            {
                throw dlib::socket_error(
                    EPORT_IN_USE,
                    "error occurred in server::start()\nport already in use"
                );
            }
            else if (status == OTHER_ERROR)
            {
                throw dlib::socket_error(
                    "error occurred in server::start()\nunable to create listener"
                );            
            }
        }

        running_mutex.lock();
        running = true;
        running_mutex.unlock();
    }

// ----------------------------------------------------------------------------------------

    void server::
    start (
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT( 
              this->is_running() == false,
            "\tvoid server::start"
            << "\n\tis_running() == " << this->is_running() 
            << "\n\tthis: " << this
            );

        start_accepting_connections();

    }

// ----------------------------------------------------------------------------------------

    void server::
    start_accepting_connections (
    )
    {
        open_listening_socket();

        // determine the listening port
        bool port_assigned = false;
        listening_port_mutex.lock();
        if (listening_port == 0)
        {
            port_assigned = true;
            listening_port = sock->get_listening_port();
        }
        listening_port_mutex.unlock();
        if (port_assigned)
            on_listening_port_assigned();
        


        int status = 0;

        connection* client;
        bool exit = false;
        while ( true )
        {


            // accept the next connection
            status = sock->accept(client,1000);


            // if there was an error then quit the loop
            if (status == OTHER_ERROR)
            {
                break;
            }

            shutting_down_mutex.lock();
            // if we are shutting down then signal that we should quit the loop
            exit = shutting_down;
            shutting_down_mutex.unlock();  


            // if we should be shutting down 
            if (exit)
            {
                // if a connection was opened then close it
                if (status == 0)
                    delete client;
                break;
            }



            // if the accept timed out
            if (status == TIMEOUT)
            {
                continue;       
            }





            // add this new connection to cons
            cons_mutex.lock();
            connection* client_temp = client;
            try{cons.add(client_temp);}
            catch(...)
            { 
                sock.reset();;
                delete client;
                cons_mutex.unlock();

                // signal that we are not running start() anymore
                running_mutex.lock();
                running = false;
                running_signaler.broadcast();
                running_mutex.unlock();               
                

                clear(); 
                throw;
            }
            cons_mutex.unlock();


            // make a param structure
            param* temp = 0;
            try{
            temp = new param (
                            *this,
                            *client
                            );
            } catch (...) 
            {
                sock.reset();
                delete client;
                running_mutex.lock();
                running = false;
                running_signaler.broadcast();
                running_mutex.unlock();
                clear(); 
                throw;
            }


            // if create_new_thread failed
            if (!create_new_thread(service_connection,temp))
            {
                delete temp;
                // close the listening socket
                sock.reset();

                // close the new connection and remove it from cons
                cons_mutex.lock();
                connection* ctemp;
                if (cons.is_member(client))
                {
                    cons.remove(client,ctemp);
                }
                delete client;
                cons_mutex.unlock();


                // signal that the listener has closed
                running_mutex.lock();
                running = false;
                running_signaler.broadcast();
                running_mutex.unlock();

                // make sure the object is cleared
                clear();

                // throw the exception
                throw dlib::thread_error(
                    ECREATE_THREAD,
                    "error occurred in server::start()\nunable to start thread"
                    );    
            }
            // if we made the new thread then update thread_count
            else
            {
                // increment the thread count
                thread_count_mutex.lock();
                ++thread_count;
                if (thread_count == 0)
                    thread_count_zero.broadcast();
                thread_count_mutex.unlock();
            }


            

            // check if we have hit the maximum allowed number of connections
            max_connections_mutex.lock();
            // if max_connections is zero or the loop is ending then skip this
            if (max_connections != 0)
            {
                // wait for thread_count to be less than max_connections
                thread_count_mutex.lock();
                while (thread_count >= max_connections)
                {
                    max_connections_mutex.unlock();
                    thread_count_signaler.wait();
                    max_connections_mutex.lock();     

                    // if we are shutting down the quit the loop
                    shutting_down_mutex.lock();
                    exit = shutting_down;
                    shutting_down_mutex.unlock();
                    if (exit)
                        break;
                }
                thread_count_mutex.unlock();
            }
            max_connections_mutex.unlock();

            if (exit)
            {
                break;
            }
        } //while ( true )


        // close the socket
        sock.reset();

        // signal that the listener has closed
        running_mutex.lock();
        running = false;
        running_signaler.broadcast();
        running_mutex.unlock();

        // if there was an error with accept then throw an exception
        if (status == OTHER_ERROR)
        {
            // make sure the object is cleared
            clear();

            // throw the exception
            throw dlib::socket_error(
             "error occurred in server::start()\nlistening socket returned error"
                );            
        }
    }

// ----------------------------------------------------------------------------------------

    bool server::
    is_running (
    ) const
    {
        running_mutex.lock();
        bool temp = running;
        running_mutex.unlock();
        return temp;
    }

// ----------------------------------------------------------------------------------------

    const std::string server::
    get_listening_ip (
    ) const
    {
        listening_ip_mutex.lock();
        std::string ip(listening_ip);
        listening_ip_mutex.unlock();
        return ip;
    }

// ----------------------------------------------------------------------------------------

    int server::
    get_listening_port (
    ) const
    {
        listening_port_mutex.lock();
        int port = listening_port;
        listening_port_mutex.unlock();        
        return port;
    }

// ----------------------------------------------------------------------------------------

    void server::
    set_listening_port (
        int port
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT( 
            ( port >= 0 &&
              this->is_running() == false ),
            "\tvoid server::set_listening_port"
            << "\n\tport         == " << port
            << "\n\tis_running() == " << this->is_running() 
            << "\n\tthis: " << this
            );

        listening_port_mutex.lock();
        listening_port = port;
        listening_port_mutex.unlock();
    }

// ----------------------------------------------------------------------------------------

    void server::
    set_listening_ip (
        const std::string& ip
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT( 
            ( ( is_ip_address(ip) || ip == "" ) &&
              this->is_running() == false ),
            "\tvoid server::set_listening_ip"
            << "\n\tip           == " << ip
            << "\n\tis_running() == " << this->is_running() 
            << "\n\tthis: " << this
            );

        listening_ip_mutex.lock();
        listening_ip = ip;
        listening_ip_mutex.unlock();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // static member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    const logger server::sdlog("dlib.server");

    void server::
    service_connection(
        void* item
    )
    {
        param& p = *static_cast<param*>(item);


        p.the_server.on_connect(p.new_connection);


        // remove this connection from cons and close it
        p.the_server.cons_mutex.lock();
        connection* temp;
        if (p.the_server.cons.is_member(&p.new_connection))
            p.the_server.cons.remove(&p.new_connection,temp);
        try{ close_gracefully(&p.new_connection); } 
        catch (...) { sdlog << LERROR << "close_gracefully() threw"; } 
        p.the_server.cons_mutex.unlock();

        // decrement the thread count and signal if it is now zero
        p.the_server.thread_count_mutex.lock();
        --p.the_server.thread_count;
        p.the_server.thread_count_signaler.broadcast();
        if (p.the_server.thread_count == 0)
            p.the_server.thread_count_zero.broadcast();
        p.the_server.thread_count_mutex.unlock();

        delete &p;


    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SERVER_KERNEL_1_

