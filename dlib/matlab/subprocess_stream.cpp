// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "subprocess_stream.h"

#include <sstream>
#include <utility>
#include <iostream>
#include <cstdio>
#include <sys/wait.h>

// ----------------------------------------------------------------------------------------

namespace dlib
{

    class filestreambuf : public std::streambuf
    {
        /*!
                    INITIAL VALUE
                        - fd_in == the file descriptor we read from.
                        - fd_out == the file descriptor we write to.
                        - in_buffer == an array of in_buffer_size bytes
                        - out_buffer == an array of out_buffer_size bytes

                    CONVENTION
                        - in_buffer == the input buffer used by this streambuf
                        - out_buffer == the output buffer used by this streambuf
                        - max_putback == the maximum number of chars to have in the put back buffer.
        !*/

    public:

        filestreambuf (
            int fd_in_,
            int fd_out_ 
        ) :
            fd_in(fd_in_),
            fd_out(fd_out_),
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
            if (write(fd_out,out_buffer,num) != num)
            {
                // the write was not successful so return EOF 
                return EOF;
            } 
            pbump(-num);
            return num;
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
                    if (write(fd_out,s,num_left) != num_left)
                    {
                        // the write was not successful so return that 0 bytes were written
                        return 0;
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


            int num = read(fd_in,in_buffer+max_putback, in_buffer_size-max_putback);
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
        int  fd_in;
        int  fd_out;
        static const std::streamsize max_putback = 4;
        static const std::streamsize out_buffer_size = 10000;
        static const std::streamsize in_buffer_size = 10000;
        char* out_buffer;
        char* in_buffer;

    };

// ---------------------------------------------------------------------------------------- 

    subprocess_stream::
    subprocess_stream(const char* program_name) : stderr(NULL), std::iostream(NULL)
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
            dup2(write_pipe.read_fd(), STDIN_FILENO);
            dup2(read_pipe.write_fd(), STDOUT_FILENO);
            dup2(err_pipe.write_fd(),  STDERR_FILENO);
            write_pipe.close(); 
            read_pipe.close();
            err_pipe.close();

            char* argv[] = {(char*)program_name, nullptr};
            char* envp[] = {nullptr};

            execve(argv[0], argv, envp);
            // If launching the child didn't work then bail immediately so the parent
            // process has no chance to get tweaked out (*cough* MATLAB *cough*).
            _Exit(1);
        }
        else 
        {
            // In parent process
            close(write_pipe.read_fd());
            close(read_pipe.write_fd());
            close(err_pipe.write_fd());
            inout_buf = std::unique_ptr<filestreambuf>(new filestreambuf(read_pipe.read_fd(), write_pipe.write_fd()));
            err_buf = std::unique_ptr<filestreambuf>(new filestreambuf(err_pipe.read_fd(), 0));
            this->rdbuf(inout_buf.get());
            stderr.rdbuf(err_buf.get());
            this->tie(this);
            stderr.tie(this);
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
    send_eof() { inout_buf->sync();  ::close(write_pipe.write_fd()); }

// ----------------------------------------------------------------------------------------

}


