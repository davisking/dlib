// Copyright (C) 2006  Davis E. King (davis@dlib.net), Steven Van Ingelgem
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SERVER_HTTp_ABSTRACT_
#ifdef DLIB_SERVER_HTTp_ABSTRACT_


#include "server_iostream_abstract.h"
#include <iostream>
#include <sstream>
#include <string>

namespace dlib
{

    template <
        typename server_base
        >
    class server_http : public server_base 
    {

        /*!
            REQUIREMENTS ON server_base 
                is an implementation of server/server_iostream_abstract.h

            WHAT THIS EXTENSION DOES FOR SERVER IOSTREAM
                This extension turns the server object into a simple HTTP server.
                It only handles HTTP GET and POST requests and each incoming request triggers the
                on_request() callback.  

            COOKIE STRINGS
                The strings returned in the cookies key_value_map should be of the following form:
                    key:   cookie_name
                    value: cookie contents; expires=Fri, 31-Dec-2010 23:59:59 GMT; path=/; domain=.example.net

                You don't have to supply all the extra cookie arguments.  So if you just want to
                set a cookie that will expire when the client's browser is closed you can 
                just say something like incoming.cookies["cookie_name"] = "cookie contents";

            HTTP HEADERS
                The HTTP headers in the incoming.headers and outgoing.headers are the name/value pairs
                of HTTP headers.  For example, the HTTP header "Content-Type: text/html" would be
                encoded such that outgoing.headers["Content-Type"] == "text/html". 

                Also note that if you wish to change the content type of your response to the 
                client you may do so by setting the "Content-Type" header to whatever you like. 
                However, setting this field manually is not necessary as it will default to 
                "text/html" if you don't explicitly set it to something.
        !*/

    public:

        template <typename Key, typename Value>
        class constmap : public std::map<Key, Value>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is simply an extension to the std::map that allows you 
                    to use the operator[] accessor with a constant map.  
            !*/
        public:

            const Value& operator[](
                const Key& k
            ) const;
            /*!
                ensures
                    - if (this->find(k) != this->end()) then
                        - This map contains the given key
                        - return the value associated with the given key
                    - else
                        - return a default initialized Value object
            !*/

            Value& operator[](
                const Key& k
            ) { return std::map<Key, Value>::operator [](k); }
            /*!
                ensures
                    - This function does the same thing as the normal std::map operator[]
                      function.
                    - if (this->find(k) != this->end()) then
                        - This map contains the given key
                        - return the value associated with the given key
                    - else
                        - Adds a new entry into the map that is associated with the
                          given key.  The new entry will be default initialized and
                          this function returns a reference to it.
            !*/ 
        };

        typedef constmap<std::string, std::string> key_value_map;

        struct incoming_things 
        {
            std::string path;
            std::string request_type;
            std::string content_type;
            std::string body;

            key_value_map queries;
            key_value_map cookies;
            key_value_map headers;

            std::string foreign_ip;
            unsigned short foreign_port;
            std::string local_ip;
            unsigned short local_port;
        };

        struct outgoing_things 
        {
            key_value_map  cookies;
            key_value_map  headers;
            unsigned short http_return;
            std::string    http_return_status;
        };

    private:

        virtual const std::string on_request (
            const incoming_things& incoming,
            outgoing_things& outgoing
        ) = 0;
        /*!
            requires
                - on_request() is called when there is an HTTP GET or POST request to be serviced 
                - on_request() is run in its own thread 
                - is_running() == true 
                - the number of current on_request() functions running < get_max_connection() 
                - in incoming: 
                    - incoming.path == the path being requested by this request
                    - incoming.request_type == the type of request, GET or POST
                    - incoming.content_type == the content type associated with this request
                    - incoming.body == a string that contains the data that was posted back to the
                      web server by the client (e.g. The string has the length specified by the
                      Content-Length header).
                    - incoming.queries == a map that contains all the key/value pairs in the query 
                      string of this request.  The key and value strings of the query string will
                      have been decoded back into their original form before being sent to this
                      function (i.e. '+' decoded back to ' ' and "%hh" into its corresponding 
                      ascii value.  So the URL-encoding is decoded automatically)
                    - incoming.cookies == The set of cookies that came from the client along with 
                      this request.  The cookies will have been decoded back to normal form
                      from the URL-encoding.
                    - incoming.headers == a map that contains all the incoming HTTP headers 
                      from the client web browser.  
                    - incoming.foreign_ip == the foreign ip address for this request 
                    - incoming.foreign_port == the foreign port number for this request
                    - incoming.local_ip == the IP of the local interface this request is coming in on 
                    - incoming.local_port == the local port number this request is coming in on 
                - in outgoing:
                    - outgoing.cookies.size() == 0
                    - outgoing.headers.size() == 0
                    - outgoing.http_return == 200
                    - outgoing.http_return_status == "OK"
            ensures
                - This function returns the HTML page to be displayed as the response to this request. 
                - this function will not call clear()  
                - #outgoing.cookies == a set of new cookies to pass back to the client along
                  with the result of this request.  (Note that URL-encoding is automatically applied 
                  so you don't have to do it yourself)
                - #outgoing.headers == a set of additional headers you wish to appear in the
                  HTTP response to this request.  (This may be empty, the minimum needed headers
                  will be added automatically if you don't set them)
                - outgoing.http_return and outgoing.http_return_status may be set to override the 
                  default HTTP return code of 200 OK
            throws
                - does not throw any exceptions
        !*/

    };

}

#endif // DLIB_SERVER_HTTp_ABSTRACT_ 



