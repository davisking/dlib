// Copyright (C) 2006  Davis E. King (davis@dlib.net), Steven Van Ingelgem
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SERVER_HTTp_1_
#define DLIB_SERVER_HTTp_1_


#include "server_iostream_abstract.h"
#include "server_http_abstract.h"
#include <iostream>
#include <sstream>
#include <string>
#include "../logger.h"
#include "../string.h"

#ifdef  __INTEL_COMPILER
// ignore the bogus warning about hiding on_connect()
#pragma warning (disable: 1125)
#endif

#if _MSC_VER
#  pragma warning( disable: 4503 )
#endif // _MSC_VER


namespace dlib
{

    template <
        typename server_base
        >
    class server_http_1 : public server_base 
    {

        /*!
            CONVENTION
                this extension doesn't add any new state to this object.
        !*/


    public:

        template <typename Key, typename Value>
        class constmap : public std::map<Key, Value>
        {
        public:
            const Value& operator[](const Key& k) const
            {
                static const Value dummy = Value();

                typename std::map<Key, Value>::const_iterator ci = std::map<Key, Value>::find(k);

                if ( ci == this->end() )
                    return dummy;
                else
                    return ci->second;
            }

            Value& operator[](const Key& k)
            {
                return std::map<Key, Value>::operator [](k);
            }
        };

        typedef constmap< std::string, std::string > key_value_map;


        struct incoming_things 
        {
            incoming_things() : foreign_port(0), local_port(0) {}

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
            outgoing_things() : http_return(200) { }

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

        unsigned char to_hex( unsigned char x ) const 
        {
            return x + (x > 9 ? ('A'-10) : '0');
        }

        const std::string urlencode( const std::string& s ) const 
        {
            std::ostringstream os;

            for ( std::string::const_iterator ci = s.begin(); ci != s.end(); ++ci )
            {
                if ( (*ci >= 'a' && *ci <= 'z') ||
                     (*ci >= 'A' && *ci <= 'Z') ||
                     (*ci >= '0' && *ci <= '9') )
                { // allowed
                    os << *ci;
                }
                else if ( *ci == ' ')
                {
                    os << '+';
                }
                else
                {
                    os << '%' << to_hex(*ci >> 4) << to_hex(*ci % 16);
                }
            }

            return os.str();
        }

        unsigned char from_hex (
            unsigned char ch
        ) const
        {
            if (ch <= '9' && ch >= '0')
                ch -= '0';
            else if (ch <= 'f' && ch >= 'a')
                ch -= 'a' - 10;
            else if (ch <= 'F' && ch >= 'A')
                ch -= 'A' - 10;
            else 
                ch = 0;
            return ch;
        }

        const std::string urldecode (
            const std::string& str
        ) const
        {
            using namespace std;
            string result;
            string::size_type i;
            for (i = 0; i < str.size(); ++i)
            {
                if (str[i] == '+')
                {
                    result += ' ';
                }
                else if (str[i] == '%' && str.size() > i+2)
                {
                    const unsigned char ch1 = from_hex(str[i+1]);
                    const unsigned char ch2 = from_hex(str[i+2]);
                    const unsigned char ch = (ch1 << 4) | ch2;
                    result += ch;
                    i += 2;
                }
                else
                {
                    result += str[i];
                }
            }
            return result;
        }

        void parse_url(std::string word, key_value_map& queries)
        /*!
            Parses the query string of a URL.  word should be the stuff that comes
            after the ? in the query URL.
        !*/
        {
            std::string::size_type pos;

            for (pos = 0; pos < word.size(); ++pos)
            {
                if (word[pos] == '&')
                    word[pos] = ' ';
            }

            std::istringstream sin(word);
            sin >> word;
            while (sin)
            {
                pos = word.find_first_of("=");
                if (pos != std::string::npos)
                {
                    std::string key = urldecode(word.substr(0,pos));
                    std::string value = urldecode(word.substr(pos+1));

                    queries[key] = value;
                }
                sin >> word;
            }
        }

        void on_connect (
            std::istream& in,
            std::ostream& out,
            const std::string& foreign_ip,
            const std::string& local_ip,
            unsigned short foreign_port,
            unsigned short local_port,
            uint64
        )
        {
            bool my_fault = true;

            using namespace std;

            try
            {
                incoming_things incoming;
                outgoing_things outgoing;

                incoming.foreign_ip   = foreign_ip;
                incoming.foreign_port = foreign_port;
                incoming.local_ip     = local_ip;
                incoming.local_port   = local_port;

                in >> incoming.request_type;

                // get the path
                in >> incoming.path;

                key_value_map& incoming_headers = incoming.headers;
                key_value_map& cookies          = incoming.cookies;
                std::string& path               = incoming.path;
                std::string& content_type       = incoming.content_type;
                unsigned long content_length = 0;

                string line;
                getline(in,line);
                string first_part_of_header;
                string::size_type position_of_double_point;
                // now loop over all the incoming_headers
                while (line.size() > 2)
                {
                    position_of_double_point = line.find_first_of(':');
                    if ( position_of_double_point != string::npos )
                    {
                        first_part_of_header = dlib::trim(line.substr(0, position_of_double_point));

                        if ( !incoming_headers[first_part_of_header].empty() )
                            incoming_headers[ first_part_of_header ] += " ";
                        incoming_headers[first_part_of_header] += dlib::trim(line.substr(position_of_double_point+1));

                        // look for Content-Type:
                        if (line.size() > 14 && strings_equal_ignore_case(line, "Content-Type:", 13))
                        {
                            content_type = line.substr(14);
                            if (content_type[content_type.size()-1] == '\r')
                                content_type.erase(content_type.size()-1);
                        }
                        // look for Content-Length:
                        else if (line.size() > 16 && strings_equal_ignore_case(line, "Content-Length:", 15))
                        {
                            istringstream sin(line.substr(16));
                            sin >> content_length;
                            if (!sin)
                                content_length = 0;
                        }
                        // look for any cookies
                        else if (line.size() > 6 && strings_equal_ignore_case(line, "Cookie:", 7))
                        {
                            string::size_type pos = 6;
                            string key, value;
                            bool seen_key_start = false;
                            bool seen_equal_sign = false;
                            while (pos + 1 < line.size())
                            {
                                ++pos;
                                // ignore whitespace between cookies
                                if (!seen_key_start && line[pos] == ' ')
                                    continue;

                                seen_key_start = true;
                                if (!seen_equal_sign) 
                                {
                                    if (line[pos] == '=')
                                    {
                                        seen_equal_sign = true;
                                    }
                                    else
                                    {
                                        key += line[pos];
                                    }
                                }
                                else
                                {
                                    if (line[pos] == ';')
                                    {
                                        cookies[urldecode(key)] = urldecode(value);
                                        seen_equal_sign = false;
                                        seen_key_start = false;
                                        key.clear();
                                        value.clear();
                                    }
                                    else
                                    {
                                        value += line[pos];
                                    }
                                }
                            }
                            if (key.size() > 0)
                            {
                                cookies[urldecode(key)] = urldecode(value);
                                key.clear();
                                value.clear();
                            }
                        }
                    } // no ':' in it!
                    getline(in,line);
                } // while (line.size() > 2 )

                // If there is data being posted back to us then load it into the incoming.body
                // string.
                if ( content_length > 0 )
                {
                    incoming.body.resize(content_length);
                    in.read(&incoming.body[0],content_length);
                }

                // If there is data being posted back to us as a query string then
                // pick out the queries using parse_url.
                if (strings_equal_ignore_case(incoming.request_type, "POST") && 
                    strings_equal_ignore_case(left_substr(content_type,";"), "application/x-www-form-urlencoded"))
                {
                    parse_url(incoming.body, incoming.queries);
                }

                string::size_type pos = path.find_first_of("?");
                if (pos != string::npos)
                {
                    parse_url(path.substr(pos+1), incoming.queries);
                }


                my_fault = false;
                key_value_map& new_cookies      = outgoing.cookies;
                key_value_map& response_headers = outgoing.headers;

                // Set some defaults
                outgoing.http_return        = 200;
                outgoing.http_return_status = "OK";

                // if there wasn't a problem with the input stream at some point
                // then lets trigger this request callback.
                std::string result;
                if (in)
                    result = on_request(incoming, outgoing);
                my_fault = true;

                // only send this header if the user hasn't told us to send another kind
                bool has_content_type(false),
                     has_location(false);
                for( typename key_value_map::const_iterator ci = response_headers.begin(); ci != response_headers.end(); ++ci )
                {
                    if ( !has_content_type && strings_equal_ignore_case(ci->first , "content-type") )
                    {
                        has_content_type = true;
                    }
                    else if ( !has_location && strings_equal_ignore_case(ci->first , "location") )
                    {
                        has_location = true;
                    }
                }

                if ( has_location )
                {
                    outgoing.http_return = 302;
                }

                if ( !has_content_type )
                {
                    response_headers["Content-Type"] = "text/html";
                }

                {
                    ostringstream os;
                    os << result.size();

                    response_headers["Content-Length"] = os.str();
                }

                out << "HTTP/1.0 " << outgoing.http_return << " " << outgoing.http_return_status << "\r\n";

                // Set any new headers
                for( typename key_value_map::const_iterator ci = response_headers.begin(); ci != response_headers.end(); ++ci )
                {
                    out << ci->first << ": " << ci->second << "\r\n";
                }

                // set any cookies 
                for( typename key_value_map::const_iterator ci = new_cookies.begin(); ci != new_cookies.end(); ++ci )
                {
                    out << "Set-Cookie: " << urlencode(ci->first) << '=' << urlencode(ci->second) << "\r\n";
                }
                out << "\r\n" << result;
            }
            catch (std::bad_alloc&)
            {
                dlog << LERROR << "We ran out of memory in server_http::on_connect()";
                // If this is an escaped exception from on_request then let it fly! 
                // Seriously though, this way it is obvious to the user that something bad happened
                // since they probably won't have the dlib logger enabled.
                if (!my_fault)
                    throw;
            }

        }

        const static logger dlog;
    };

    template <
        typename server_base
        >
    const logger server_http_1<server_base>::dlog("dlib.server");
}

#endif // DLIB_SERVER_HTTp_1_




