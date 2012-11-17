// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IOSOCKSTrEAM_H__
#define DLIB_IOSOCKSTrEAM_H__

#include "iosockstream_abstract.h"

#include <iostream>
#include "../sockstreambuf.h"
#include "../smart_pointers_thread_safe.h"
#include "../timeout.h"

namespace dlib
{

// ---------------------------------------------------------------------------------------- 

    class iosockstream : public std::iostream
    {
    public:

        iosockstream()
        {
            tie(this);
        }

        iosockstream( 
            const network_address& addr
        ) 
        { 
            tie(this); 
            open(addr); 
        }

        iosockstream( 
            const network_address& addr,
            unsigned long timeout 
        ) 
        { 
            tie(this); 
            open(addr, timeout); 
        }

        ~iosockstream()
        {
            close();
        }

        void open (
            const network_address& addr
        )
        {
            close();
            con.reset(connect(addr));
            buf.reset(new sockstreambuf(con.get()));
            rdbuf(buf.get());
            clear();
        }

        void open (
            const network_address& addr,
            unsigned long timeout
        )
        {
            close(timeout);
            con.reset(connect(addr.host_address, addr.port, timeout));
            buf.reset(new sockstreambuf(con.get()));
            rdbuf(buf.get());
            clear();
        }

        void close(
            unsigned long timeout = 10000
        )
        {
            rdbuf(0);
            try
            {
                if (buf)
                {
                    dlib::timeout t(*con,&connection::shutdown,timeout);

                    // This will flush the sockstreambuf and also destroy it.
                    buf.reset();

                    if(con->shutdown_outgoing())
                    {
                        // there was an error so just close it now and return
                        con->shutdown();
                    }
                    else
                    {
                        char junk[100];
                        // wait for the other end to close their side
                        while (con->read(junk,sizeof(junk)) > 0);
                    }
                }
            }
            catch (...)
            {
                con.reset();
                throw;
            }
            con.reset();
        }

        void terminate_connection_after_timeout (
            unsigned long timeout
        )
        {
            if (con)
            {
                con_timeout.reset(new dlib::timeout(*this,&iosockstream::terminate_connection,timeout,con));
            }
        }

    private:

        void terminate_connection(
            shared_ptr_thread_safe<connection> thecon
        )
        {
            thecon->shutdown();
        }

        scoped_ptr<timeout> con_timeout;
        shared_ptr_thread_safe<connection> con;
        scoped_ptr<sockstreambuf> buf;

    };

// ---------------------------------------------------------------------------------------- 

}


#endif // DLIB_IOSOCKSTrEAM_H__


