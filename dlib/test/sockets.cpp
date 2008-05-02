// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/sockets.h>
#include <dlib/server.h>
#include <dlib/misc_api.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    dlib::mutex m;
    dlib::signaler s(m);
    const char magic_num = 42;
    const int min_bytes_sent = 10000;
    int thread_count;
    int thread_count2;
    int assigned_port;

    logger dlog("test.sockets");

// ----------------------------------------------------------------------------------------

    class serv : public server::kernel_1a_c
    {
    public:
        serv (
        ) :
            error_occurred (false)
        {}

        void on_listening_port_assigned (
        )
        {
            auto_mutex M(m);
            assigned_port = get_listening_port();
            s.broadcast();
        }


        void on_connect (
            connection& con
        )
        {
            dlog << LINFO << "in serv::on_connect(): got new connection";
            int status;
            int count = 0;
            char buf[100];
            while ((status = con.read(buf,sizeof(buf))) > 0)
            {
                for (int i = 0; i < status; ++i)
                {
                    if (buf[i] != magic_num)
                    {
                        tag = 4.0;
                        error_occurred = true;
                    }
                }
                count += status;
            }
            if (count != min_bytes_sent)
            {
                tag = 5.0;
                error_occurred = true;
            }
            dlog << LINFO << "in serv::on_connect(): on_connect ending";
        }

        bool error_occurred;
        double tag;
    };

// ----------------------------------------------------------------------------------------

    void thread_proc (
        void* param
    )
    {
        serv& srv = *reinterpret_cast<serv*>(param);
        try
        {
            dlog << LTRACE << "enter thread";
            {
                auto_mutex M(m);
                while (assigned_port == 0)
                    s.wait();
            }

            int status;
            connection* con;
            string hostname;
            string ip;
            status = get_local_hostname(hostname);
            if (status)
            {
                srv.tag = 1.0;
                srv.error_occurred = true;
                srv.clear();
                dlog << LWARN << "leaving thread, line: " << __LINE__;
                return;
            }

            status = hostname_to_ip(hostname,ip);
            if (status)
            {
                srv.tag = 2.0;
                srv.error_occurred = true;
                srv.clear();
                dlog << LWARN << "leaving thread, line: " << __LINE__;
                return;
            }

            dlog << LTRACE << "try to connect to the server at port " << srv.get_listening_port();
            status = create_connection(con,srv.get_listening_port(),ip);
            if (status)
            {
                srv.tag = 3.0;
                srv.error_occurred = true;
                srv.clear();
                dlog << LWARN << "leaving thread, line: " << __LINE__;
                return;
            }

            dlog << LTRACE << "sending magic_num to server";
            int i;
            for (i = 0; i < min_bytes_sent; ++i)
            {
                con->write(&magic_num,1); 
            }

            dlog << LTRACE << "shutting down connection to server";
            close_gracefully(con);
            dlog << LTRACE << "finished calling close_gracefully() on the connection";

            auto_mutex M(m);
            --thread_count;
            s.broadcast();
            while (thread_count > 0)
                s.wait();

            srv.clear();

            --thread_count2;
            s.broadcast();
        }
        catch (exception& e)
        {
            srv.error_occurred = true;
            dlog << LERROR << "exception thrown in thread_proc(): " << e.what();
            cout << "exception thrown in thread_proc(): " << e.what();
        }
        dlog << LTRACE << "exit thread";
    }

    void sockets_test (
    )
    /*!
        requires
            - sockets is an implementation of sockets/sockets_kernel_abstract.h 
              is instantiated with int
        ensures
            - runs tests on sockets for compliance with the specs
    !*/
    {        

        dlog << LTRACE << "starting test";
        serv srv;

        thread_count = 10;
        assigned_port = 0;
        thread_count2 = thread_count;


        dlog << LTRACE << "spawning threads";
        int num = thread_count;
        for (int i = 0; i < num; ++i)
        {
            create_new_thread(thread_proc,&srv);
        }



        dlog << LTRACE << "calling srv.start()";
        srv.start();
        dlog << LTRACE << "srv.start() just ended.";

        {
            auto_mutex M(m);
            while (thread_count2 > 0)
                s.wait();
        }
        if (srv.error_occurred)
        {
            dlog << LDEBUG << "tag: " << srv.tag;
        }

        dlog << LTRACE << "ending successful test";
        DLIB_CASSERT( !srv.error_occurred,""); 
    }

// ----------------------------------------------------------------------------------------


    class sockets_tester : public tester
    {
    public:
        sockets_tester (
        ) :
            tester ("test_sockets",
                    "Runs tests on the sockets component.")
        {}

        void perform_test (
        )
        {
            sockets_test();
        }
    } a;

}

