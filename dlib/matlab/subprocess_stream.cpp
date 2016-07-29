// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "subprocess_stream.h"

#include <sstream>
#include <utility>
#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/select.h>
#include "call_matlab.h"

using namespace std;

// ----------------------------------------------------------------------------------------

namespace dlib
{

// ----------------------------------------------------------------------------------------

    void make_fd_non_blocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

// ----------------------------------------------------------------------------------------

    // Block until fd is ready to read,  while also echoing whatever is in fd_printf to
    // cout.
    int read_echoing_select(int fd, int fd_printf)
    {
        // run until fd has data ready
        while(fd_printf >= 0)
        {
            fd_set rfds;
            int retval;

            while(true)
            {
                FD_ZERO(&rfds);
                FD_SET(fd, &rfds);
                FD_SET(fd_printf, &rfds);

                // select times out every second just so we can check for matlab ctrl+c.
                struct timeval tv;
                tv.tv_sec = 1;
                tv.tv_usec = 0;

                try{check_for_matlab_ctrl_c();} catch(...) { return 1; }
                retval = select(std::max(fd,fd_printf)+1, &rfds, NULL, NULL, &tv);
                try{check_for_matlab_ctrl_c();} catch(...) { return 1; }
                if (retval == 0) // keep going if it was just a timeout.
                    continue;
                else if (retval == -1 && errno == EINTR)
                    continue;

                break;
            }

            if (retval == -1)
            {
                return 1;
            }
            else 
            {
                if (FD_ISSET(fd,&rfds))
                {
                    return 0;
                }
                else
                {
                    char buf[1024];
                    int num = read(fd_printf,buf, sizeof(buf)-1);
                    if (num == -1)
                        return 1;
                    if (num > 0)
                    {
                        buf[num] = 0;
                        cout << buf << flush;
                    }
                }
            }
        }
        return 0;
    }

    int write_echoing_select(int fd, int fd_printf)
    {
        // run until fd has data ready
        while(fd_printf >= 0)
        {
            fd_set rfds, wfds;
            int retval;
            while(true)
            {
                FD_ZERO(&rfds);
                FD_ZERO(&wfds);
                FD_SET(fd, &wfds);
                FD_SET(fd_printf, &rfds);

                // select times out every second just so we can check for matlab ctrl+c.
                struct timeval tv;
                tv.tv_sec = 1;
                tv.tv_usec = 0;

                try{check_for_matlab_ctrl_c();} catch(...) { return 1; }
                retval = select(std::max(fd,fd_printf)+1, &rfds, &wfds, NULL, &tv);
                try{check_for_matlab_ctrl_c();} catch(...) { return 1; }
                if (retval == 0) // keep going if it was just a timeout.
                    continue;
                else if (retval == -1 && errno == EINTR)
                    continue;

                break;
            }

            if (retval == -1)
            {
                return 1;
            }
            else 
            {
                if (FD_ISSET(fd,&wfds))
                {
                    return 0;
                }
                else
                {
                    char buf[1024];
                    int num = read(fd_printf,buf, sizeof(buf)-1);
                    if (num == -1)
                        return 1;
                    if (num > 0)
                    {
                        buf[num] = 0;
                        cout << buf << flush;
                    }
                }
            }
        }
        return 0;
    }

// ----------------------------------------------------------------------------------------

    class filestreambuf : public std::streambuf
    {
        /*!
                    INITIAL VALUE
                        - fd == the file descriptor we read from.
                        - in_buffer == an array of in_buffer_size bytes
                        - out_buffer == an array of out_buffer_size bytes

                    CONVENTION
                        - in_buffer == the input buffer used by this streambuf
                        - out_buffer == the output buffer used by this streambuf
                        - max_putback == the maximum number of chars to have in the put back buffer.
        !*/

    public:

        filestreambuf (
            int fd_,
            int fd_printf_
        ) :
            fd(fd_),
            fd_printf(fd_printf_),
            out_buffer(0),
            in_buffer(0)
        {
            init();
        }

        virtual ~filestreambuf (
        )
        {
            sync();
            delete [] out_buffer;
            delete [] in_buffer;
        }

        int sync (
        )
        {
            if (flush_out_buffer() == EOF)
            {
                // an error occurred
                return -1;
            }
            return 0;
        }
    protected:

        void init (
        )
        {
            try
            {
                out_buffer = new char[out_buffer_size];
                in_buffer = new char[in_buffer_size];
            }
            catch (...)
            {
                if (out_buffer) delete [] out_buffer;
                throw;
            }
            setp(out_buffer, out_buffer + (out_buffer_size-1));
            setg(in_buffer+max_putback, 
                in_buffer+max_putback, 
                in_buffer+max_putback);
        }

        int flush_out_buffer (
        )
        {
            int num = static_cast<int>(pptr()-pbase());
            const int num_written = num;
            char* buf = out_buffer;
            while(num != 0)
            {
                if(write_echoing_select(fd, fd_printf))
                    return EOF;
                int status = write(fd,buf,num);
                if (status < 0)
                {
                    // the write was not successful so return EOF 
                    return EOF;
                } 
                num -= status;
                buf += status;
            }
            pbump(-num_written);
            return num_written;
        }

        // output functions
        int_type overflow (
            int_type c
        )
        {
            if (c != EOF)
            {
                *pptr() = c;
                pbump(1);
            }
            if (flush_out_buffer() == EOF)
            {
                // an error occurred
                return EOF;
            }
            return c;
        }


        std::streamsize xsputn (
            const char* s,
            std::streamsize num
        )
        {
            // Add a sanity check here 
            DLIB_ASSERT(num >= 0,
                "\tstd::streamsize filestreambuf::xsputn"
                << "\n\tThe number of bytes to write can't be negative"
                << "\n\tnum:  " << num 
                << "\n\tthis: " << this
            );

            std::streamsize space_left = static_cast<std::streamsize>(epptr()-pptr());
            if (num <= space_left)
            {
                std::memcpy(pptr(),s,static_cast<size_t>(num));
                pbump(static_cast<int>(num));
                return num;
            }
            else
            {
                std::memcpy(pptr(),s,static_cast<size_t>(space_left));
                s += space_left;
                pbump(space_left);
                std::streamsize num_left = num - space_left;

                if (flush_out_buffer() == EOF)
                {
                    // the write was not successful so return that 0 bytes were written
                    return 0;
                }

                if (num_left < out_buffer_size)
                {
                    std::memcpy(pptr(),s,static_cast<size_t>(num_left));
                    pbump(num_left);
                    return num;
                }
                else
                {
                    while(num_left != 0)
                    {
                        if(write_echoing_select(fd, fd_printf))
                            return EOF;
                        int status = write(fd,s,num_left);
                        if (status < 0)
                        {
                            // the write was not successful so return that 0 bytes were written
                            return 0;
                        } 
                        num_left -= status;
                        s += status;
                    }
                    return num;
                }
            }
        }

        // input functions
        int_type underflow( 
        )
        {
            if (gptr() < egptr())
            {
                return static_cast<unsigned char>(*gptr());
            }

            int num_put_back = static_cast<int>(gptr() - eback());
            if (num_put_back > max_putback)
            {
                num_put_back = max_putback;
            }

            // copy the putback characters into the putback end of the in_buffer
            std::memmove(in_buffer+(max_putback-num_put_back), gptr()-num_put_back, num_put_back);


            if (read_echoing_select(fd, fd_printf))
                return EOF;
            int num = read(fd,in_buffer+max_putback, in_buffer_size-max_putback);
            if (num <= 0)
            {
                // an error occurred or the connection is over which is EOF
                return EOF;
            }

            // reset in_buffer pointers
            setg (in_buffer+(max_putback-num_put_back),
                in_buffer+max_putback,
                in_buffer+max_putback+num);

            return static_cast<unsigned char>(*gptr());
        }

        std::streamsize xsgetn (
            char_type* s, 
            std::streamsize n
        )
        { 
            std::streamsize temp = n;
            while (n > 0)
            {
                int num = static_cast<int>(egptr() - gptr());
                if (num >= n)
                {
                    // copy data from our buffer 
                    std::memcpy(s, gptr(), static_cast<size_t>(n));
                    gbump(static_cast<int>(n));
                    return temp;
                }

                // read more data into our buffer  
                if (num == 0)
                {
                    if (underflow() == EOF)
                        break;
                    continue;
                }

                // copy all the data from our buffer 
                std::memcpy(s, gptr(), num);
                n -= num;
                gbump(num);
                s += num;
            }
            return temp-n;       
        }

    private:

        // member data
        int  fd;
        int  fd_printf;
        static const std::streamsize max_putback = 4;
        static const std::streamsize out_buffer_size = 10000;
        static const std::streamsize in_buffer_size = 10000;
        char* out_buffer;
        char* in_buffer;

    };

    namespace impl
    {
        int get_data_fd()
        {
            char* env_fd = getenv("DLIB_SUBPROCESS_DATA_FD");
            DLIB_CASSERT(env_fd != 0,"");
            return atoi(env_fd);
        }

        std::iostream& get_data_iostream()
        {
            static filestreambuf dbuff(get_data_fd(), -1);
            static iostream out(&dbuff);
            return out;
        }
    }

// ---------------------------------------------------------------------------------------- 

    subprocess_stream::
    subprocess_stream(const char* program_name) : stderr(NULL), iosub(NULL)
    {
        if (access(program_name, F_OK))
            throw dlib::error("Error: '" + std::string(program_name) + "' file does not exist.");
        if (access(program_name, X_OK))
            throw dlib::error("Error: '" + std::string(program_name) + "' file is not executable.");

        child_pid = fork();
        if (child_pid == -1) 
            throw dlib::error("Failed to start child process"); 

        if (child_pid == 0) 
        {   
            // In child process
            dup2(stdout_pipe.child_fd(), STDOUT_FILENO);
            dup2(stderr_pipe.child_fd(),  STDERR_FILENO);
            stdout_pipe.close();
            stderr_pipe.close();

            char* argv[] = {(char*)program_name, nullptr};
            char* cudadevs = getenv("CUDA_VISIBLE_DEVICES");
            if (cudadevs)
            {
                std::ostringstream sout;
                sout << "DLIB_SUBPROCESS_DATA_FD="<<data_pipe.child_fd();
                std::string extra = sout.str();

                std::string extra2 = std::string("CUDA_VISIBLE_DEVICES=") + cudadevs;
                char* envp[] = {(char*)extra.c_str(), (char*)extra2.c_str(), nullptr};
                execve(argv[0], argv, envp);
            }
            else
            {
                std::ostringstream sout;
                sout << "DLIB_SUBPROCESS_DATA_FD="<<data_pipe.child_fd();
                std::string extra = sout.str();
                char* envp[] = {(char*)extra.c_str(), nullptr};
                execve(argv[0], argv, envp);
            }


            // If launching the child didn't work then bail immediately so the parent
            // process has no chance to get tweaked out (*cough* MATLAB *cough*).
            _Exit(1);
        }
        else 
        {
            // In parent process
            close(data_pipe.child_fd());
            close(stdout_pipe.child_fd());
            close(stderr_pipe.child_fd());
            make_fd_non_blocking(data_pipe.parent_fd());
            make_fd_non_blocking(stdout_pipe.parent_fd());
            make_fd_non_blocking(stderr_pipe.parent_fd());
            inout_buf = std::unique_ptr<filestreambuf>(new filestreambuf(data_pipe.parent_fd(), stdout_pipe.parent_fd()));
            err_buf = std::unique_ptr<filestreambuf>(new filestreambuf(stderr_pipe.parent_fd(), stdout_pipe.parent_fd()));
            iosub.rdbuf(inout_buf.get());
            stderr.rdbuf(err_buf.get());
            iosub.tie(&iosub);
            stderr.tie(&iosub);
        }
    }

// ----------------------------------------------------------------------------------------

    subprocess_stream::
    ~subprocess_stream() 
    {
        try
        {
            wait();
        }
        catch (dlib::error& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

// ----------------------------------------------------------------------------------------

    void subprocess_stream::
    wait() 
    {
        if (!wait_called)
        {
            wait_called = true;
            send_eof();

            std::ostringstream sout;
            sout << stderr.rdbuf();

            try{check_for_matlab_ctrl_c();} catch(...) 
            { 
                kill(child_pid, SIGTERM);
            }

            int status;
            waitpid(child_pid, &status, 0);
            if (status)
                throw dlib::error("Child process terminated with an error.\n" + sout.str());

            if (sout.str().size() != 0)
                throw dlib::error("Child process terminated with an error.\n" + sout.str());
        }
    }

// ----------------------------------------------------------------------------------------

    void subprocess_stream::
    send_eof() { inout_buf->sync();  ::close(data_pipe.parent_fd()); }

// ----------------------------------------------------------------------------------------

}


