/*

    This is an example illustrating the use of the sockets and
    sockstreambuf components from the dlib C++ Library.

    This program simply connects to www.google.com at port 80 and requests the
    main Google web page.  It then prints what it gets back from Google to the 
    screen.


    For those of you curious about HTTP check out the excellent introduction at
    http://www.jmarshall.com/easy/http/
*/

#include "dlib/sockets.h"
#include "dlib/sockstreambuf.h"
#include <iostream>

using namespace std;
using namespace dlib;

int main()
{
    try
    {
        // Connect to Google's web server which listens on port 80.  If this
        // fails it will throw a dlib::socket_error exception.
        connection* con = connect("www.google.com",80);


        {
            // create a stream buffer for our connection
            sockstreambuf::kernel_2a buf(con);
            // now stick that stream buffer into an iostream object
            iostream stream(&buf);
            // this command causes the iostream to flush its output buffers
            // whenever someone makes a read request. 
            stream.tie(&stream);

            // now we make the HTTP GET request for the main Google page.
            stream << "GET / HTTP/1.0\r\n"
                << "\r\n";

            // Here we print each character we get back one at a time. 
            int ch = stream.get();
            while (ch != EOF)
            {
                cout << (char)ch;
                ch = stream.get();
            }

            // at the end of this scope buf will be destructed and flush 
            // anything it still contains to the connection.  Thus putting
            // this scope here makes it safe to call close_gracefully() next.
            // If we just called close_gracefully() before buf was destructed
            // then buf would try to flush its data to a closed connection
            // which would be an error.
        }

        // Don't forget to close the connection.  Not doing so will
        // cause a resource leak.  And once it is closed the con pointer
        // is invalid so don't touch it.
        close_gracefully(con);
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
}


