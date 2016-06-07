// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SUBPROCeSS_STREAM_H_
#define DLIB_SUBPROCeSS_STREAM_H_

#include <utility>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <dlib/matrix.h>

namespace dlib
{

// --------------------------------------------------------------------------------------

    // Call dlib's serialize and deserialize by default.   The point of this version of
    // serailize is to do something fast that normally we wouldn't do, like directly copy
    // memory.  This is safe since this is an interprocess communication happening the same
    // machine.
    template <typename T> void interprocess_serialize ( const T& item, std::ostream& out) { serialize(item, out); } 
    template <typename T> void interprocess_deserialize (T& item, std::istream& in) { deserialize(item, in); } 

    // But have overloads for direct memory copies for some types since this is faster than
    // their default serialization.
    template <typename T, long NR, long NC, typename MM, typename L>
    void interprocess_serialize(const dlib::matrix<T,NR,NC,MM,L>& item, std::ostream& out)
    {
        dlib::serialize(item.nr(), out);
        dlib::serialize(item.nc(), out);
        if (item.size() != 0)
            out.write((const char*)&item(0,0), sizeof(T)*item.size());
        if (!out)
            throw dlib::serialization_error("Error writing matrix to interprocess iostream.");
    }

    template <typename T, long NR, long NC, typename MM, typename L>
    void interprocess_deserialize(dlib::matrix<T,NR,NC,MM,L>& item, std::istream& in)
    {
        long nr, nc;
        dlib::deserialize(nr, in);
        dlib::deserialize(nc, in);
        item.set_size(nr,nc);
        if (item.size() != 0)
            in.read((char*)&item(0,0), sizeof(T)*item.size());
        if (!in)
            throw dlib::serialization_error("Error reading matrix from interprocess iostream.");
    }

// ----------------------------------------------------------------------------------------

    inline void send_to_parent_process() {std::cout.flush();}
    template <typename U, typename ...T>
    void send_to_parent_process(U&& arg1, T&& ...args)
    /*!
        ensures
            - sends all the arguments to send_to_parent_process() to standard output (and
              hence to the parent process) by serializing them with
              interprocess_serialize().
    !*/
    {
        interprocess_serialize(arg1, std::cout);
        send_to_parent_process(std::forward<T>(args)...);
        if (!std::cout)
            throw dlib::error("Error sending object to parent process.");
    }

    inline void receive_from_parent_process() {}
    template <typename U, typename ...T>
    void receive_from_parent_process(U&& arg1, T&& ...args)
    /*!
        ensures
            - receives all the arguments to receive_from_parent_process() from standard
              input (and hence from the parent process) by deserializing them with
              interprocess_deserialize().
    !*/
    {
        interprocess_deserialize(arg1, std::cin);
        receive_from_parent_process(std::forward<T>(args)...);
        if (!std::cin)
            throw dlib::error("Error receiving object from parent process.");
    }


// ----------------------------------------------------------------------------------------

    class filestreambuf;

    class subprocess_stream : public std::iostream
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a tool for spawning a subprocess and communicating with it through
                that processes standard input, output, and error.  Here is an example: 

                    subprocess_stream s("/usr/bin/echo");
                    s << "echo me this!";
                    string line;
                    getline(s, line);
                    cout << line << endl;
                    s.wait();

                That example runs echo, sends it some text, gets it back, and prints it to
                the screen.  Then it waits for the subprocess to finish.
        !*/

    public:

        explicit subprocess_stream(
            const char* program_name
        );
        /*!
            ensures
                - spawns a sub process by executing the file with the given program_name.
        !*/

        ~subprocess_stream(
        );
        /*!
            ensures
                - calls wait().  Note that the destructor never throws even though wait() can. 
                If an exception is thrown by wait() it is just logged to std::cerr.
        !*/

        void wait(
        );
        /*!
            ensures
                - closes the standard input of the child process and then waits for the
                  child to terminate.  
                - If the child returns an error (by returning != 0 from its main) or
                  outputs to its standard error then wait() throws a dlib::error() with the
                  standard error output in it.
        !*/

        int get_child_pid() const { return child_pid; }
        /*!
            ensures
                - returns the PID of the child process
        !*/

        template <typename U, typename ...T>
        void send(U&& arg1, T&& ...args)
        /*!
            ensures
                - sends all the arguments to send() to the subprocess by serializing them
                  with interprocess_serialize().
        !*/
        {
            interprocess_serialize(arg1, *this);
            send(std::forward<T>(args)...);
            if (!this->good())
            {
                std::ostringstream sout;
                sout << stderr.rdbuf();
                throw dlib::error("Error sending object to child process.\n" + sout.str());
            }
        }
        void send() {this->flush();}

        template <typename U, typename ...T>
        void receive(U&& arg1, T&& ...args)
        /*!
            ensures
                - receives all the arguments to receive() to the subprocess by deserializing
                  them with interprocess_deserialize().
        !*/
        {
            interprocess_deserialize(arg1, *this);
            receive(std::forward<T>(args)...);
            if (!this->good())
            {
                std::ostringstream sout;
                sout << stderr.rdbuf();
                throw dlib::error("Error receiving object from child process.\n" + sout.str() );
            }
        }
        void receive() {}


    private:

        void send_eof(); 

        class cpipe 
        {
        private:
            int fd[2];
        public:
            cpipe()  { if (pipe(fd)) throw dlib::error("Failed to create pipe"); }
            ~cpipe() { close(); }
            int read_fd() const { return fd[0]; }
            int write_fd() const { return fd[1]; }
            void close() { ::close(fd[0]); ::close(fd[1]); }
        };

        cpipe write_pipe;
        cpipe read_pipe;
        cpipe err_pipe;
        bool wait_called = false;
        std::unique_ptr<filestreambuf> inout_buf; 
        std::unique_ptr<filestreambuf> err_buf;
        int child_pid = -1;
        std::istream stderr;
    };
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_SUBPROCeSS_STREAM_H_

