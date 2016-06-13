// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SUBPROCeSS_STREAM_H_
#define DLIB_SUBPROCeSS_STREAM_H_

#include <utility>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <dlib/matrix.h>
#include <sys/types.h>
#include <sys/socket.h>


namespace dlib
{

// --------------------------------------------------------------------------------------

    // Call dlib's serialize and deserialize by default.   The point of this version of
    // serialize is to do something fast that normally we wouldn't do, like directly copy
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

    namespace impl{ std::ostream& get_data_ostream(); }

    inline void send_to_parent_process() {impl::get_data_ostream().flush();}
    template <typename U, typename ...T>
    void send_to_parent_process(U&& arg1, T&& ...args)
    /*!
        ensures
            - sends all the arguments to send_to_parent_process() to the parent process by
              serializing them with interprocess_serialize().
    !*/
    {
        interprocess_serialize(arg1, impl::get_data_ostream());
        send_to_parent_process(std::forward<T>(args)...);
        if (!impl::get_data_ostream())
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

    class subprocess_stream 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a tool for spawning a subprocess and communicating with it through
                its standard input, output, and error.  Here is an example: 

                    subprocess_stream s("/usr/bin/some_program");
                    s.send(obj1, obj2, obj3);
                    s.receive(obj4, obj5);
                    s.wait(); // wait for sub process to terminate 

                Then in the sub process you would have:

                    receive_from_parent_process(obj1, obj2, obj3);
                    // do stuff
                    cout << "echo this text to parent cout" << endl;
                    send_to_parent_process(obj4, obj5);


                Additionally, if the sub process writes to its standard out then that will
                be echoed to std::cout in the parent process.  Also, the communication of
                send()/receive() calls between the parent and child happens all on the
                standard input file descriptor.  So you can't really use std::cin for
                anything inside the child process as that would interfere with
                receive_from_parent_process() and send_to_parent_process().
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
            interprocess_serialize(arg1, iosub);
            send(std::forward<T>(args)...);
            if (!iosub)
            {
                std::ostringstream sout;
                sout << stderr.rdbuf();
                throw dlib::error("Error sending object to child process.\n" + sout.str());
            }
        }
        void send() {iosub.flush();}

        template <typename U, typename ...T>
        void receive(U&& arg1, T&& ...args)
        /*!
            ensures
                - receives all the arguments to receive() to the subprocess by deserializing
                  them with interprocess_deserialize().
        !*/
        {
            interprocess_deserialize(arg1, iosub);
            receive(std::forward<T>(args)...);
            if (!iosub)
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
            cpipe() { if (socketpair(AF_LOCAL, SOCK_STREAM, 0, fd)) throw dlib::error("Failed to create pipe"); }
            ~cpipe() { close(); }
            int parent_fd() const { return fd[0]; }
            int child_fd() const { return fd[1]; }
            void close() { ::close(fd[0]); ::close(fd[1]); }
        };

        cpipe data_pipe;
        cpipe stdout_pipe;
        cpipe stderr_pipe;
        bool wait_called = false;
        std::unique_ptr<filestreambuf> inout_buf; 
        std::unique_ptr<filestreambuf> err_buf;
        int child_pid = -1;
        std::istream stderr;
        std::iostream iosub;
    };
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_SUBPROCeSS_STREAM_H_

