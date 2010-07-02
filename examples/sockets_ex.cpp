// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the sockets and
    server components from the dlib C++ Library.

    This is a simple echo server.  It listens on port 1234 for incoming
    connections and just echos back any data it receives.  

*/




#include "dlib/sockets.h"
#include "dlib/server.h"
#include "dlib/ref.h" // for ref()
#include <iostream>

using namespace dlib;
using namespace std;



class serv : public server::kernel_1a_c
{

    void on_connect  (
        connection& con
    )
    {
        char ch;
        while (con.read(&ch,1) > 0)
        {
            // we are just reading one char at a time and writing it back
            // to the connection.  If there is some problem writing the char
            // then we quit the loop.  
            if (con.write(&ch,1) != 1)
                break;
        }
    }

};


void thread(serv& our_server)
{
    try
    {
        // Start the server.  start() blocks until the server is shutdown
        // by a call to clear()
        our_server.start();
    }
    catch (socket_error& e)
    {
        cout << "Socket error while starting server: " << e.what() << endl;
    }
    catch (exception& e)
    {
        cout << "Error while starting server: " << e.what() << endl;
    }
}


int main()
{
    try
    {
        serv our_server;

        // set up the server object we have made
        our_server.set_listening_port(1234);
        our_server.set_max_connections(1000);

        // create a thread that will start the server.   The ref() here allows us to pass 
        // our_server into the threaded function by reference.
        thread_function t(thread, ref(our_server));

        cout << "Press enter to end this program" << endl;
        cin.get();
        // this will cause the server to shut down 
        our_server.clear();
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
    catch (...)
    {
        cout << "Some error occurred" << endl;
    }
}

