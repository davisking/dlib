// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/bsp.h>
#include <dlib/threads.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.bsp");


    template <typename funct>
    struct callfunct_helper
    {
        callfunct_helper (
            funct f_,
            int port_,
            bool& error_occurred_
        ) :f(f_), port(port_), error_occurred(error_occurred_) {}

        funct f;
        int port;
        bool& error_occurred;

        void operator() (
        ) const
        {
            try
            {
                bsp_listen(port, f);
            }
            catch (exception& e)
            {
                dlog << LERROR << "error calling bsp_listen(): " << e.what();
                error_occurred = true;
            }
        }
    };

    template <typename funct>
    callfunct_helper<funct> callfunct(funct f, int port, bool& error_occurred)
    {
        return callfunct_helper<funct>(f,port,error_occurred);

    }

// ----------------------------------------------------------------------------------------

    void sum_array_driver (
        bsp_context& obj,
        const std::vector<int>& v,
        int& result
    )
    {
        obj.broadcast(v);

        result = 0;
        int val;
        while(obj.receive(val))
            result += val;
    }

    void sum_array_other (
        bsp_context& obj
    )
    {
        std::vector<int> v;
        obj.receive(v);

        int sum = 0;
        for (unsigned long i = 0; i < v.size(); ++i)
            sum += v[i];

        obj.send(sum, 0);


    }


    void dotest1()
    {
        dlog << LINFO << "start dotest1()";
        print_spinner();
        bool error_occurred = false;
        {
            thread_function t1(callfunct(sum_array_other, 12345, error_occurred));
            thread_function t2(callfunct(sum_array_other, 12346, error_occurred));
            thread_function t3(callfunct(sum_array_other, 12347, error_occurred));
            std::vector<int> v;
            int true_value = 0;
            for (int i = 0; i < 10; ++i)
            {
                v.push_back(i);
                true_value += i;
            }

            // wait a little bit for the threads to start up
            dlib::sleep(200);

            try
            {
                int result;
                std::vector<std::pair<std::string,unsigned short> > hosts;
                hosts.push_back(make_pair("127.0.0.1",12345));
                hosts.push_back(make_pair("127.0.0.1",12346));
                hosts.push_back(make_pair("127.0.0.1",12347));
                bsp_connect(hosts, sum_array_driver, dlib::ref(v), dlib::ref(result));

                dlog << LINFO << "result: "<< result;
                dlog << LINFO << "should be: "<< 3*true_value; 
                DLIB_TEST(result == 3*true_value);
            }
            catch (std::exception& e)
            {
                dlog << LERROR << "error during bsp_context: " << e.what();
                DLIB_TEST(false);
            }
        }
        DLIB_TEST(error_occurred == false);
    }

// ----------------------------------------------------------------------------------------

    template <unsigned long id>
    void test2_job(bsp_context& obj)
    {
        if (obj.node_id() == id)
            dlib::sleep(100);
    }

    template <unsigned long id>
    void dotest2()
    {
        dlog << LINFO << "start dotest2()";
        print_spinner();
        bool error_occurred = false;
        {
            thread_function t1(callfunct(test2_job<id>, 12345, error_occurred));
            thread_function t2(callfunct(test2_job<id>, 12346, error_occurred));
            thread_function t3(callfunct(test2_job<id>, 12347, error_occurred));

            // wait a little bit for the threads to start up
            dlib::sleep(200);

            try
            {
                std::vector<std::pair<std::string,unsigned short> > hosts;
                hosts.push_back(make_pair("127.0.0.1",12345));
                hosts.push_back(make_pair("127.0.0.1",12346));
                hosts.push_back(make_pair("127.0.0.1",12347));
                bsp_connect(hosts, test2_job<id>);
            }
            catch (std::exception& e)
            {
                dlog << LERROR << "error during bsp_context: " << e.what();
                DLIB_TEST(false);
            }

        }
        DLIB_TEST(error_occurred == false);
    }

// ----------------------------------------------------------------------------------------

    void test3_job_driver(bsp_context& obj, int& result)
    {

        obj.broadcast(obj.node_id());

        int accum = 0;
        int temp = 0;
        while(obj.receive(temp))
            accum += temp;

        // send to node 1 so it can sum everything
        if (obj.node_id() != 1)
            obj.send(accum, 1);

        while(obj.receive(temp))
            accum += temp;

        // Now hop the accum values along the nodes until the value from node 1 gets to
        // node 0.
        obj.send(accum, (obj.node_id()+1)%obj.number_of_nodes());
        DLIB_TEST(obj.receive(accum));
        obj.send(accum, (obj.node_id()+1)%obj.number_of_nodes());
        DLIB_TEST(obj.receive(accum));
        obj.send(accum, (obj.node_id()+1)%obj.number_of_nodes());
        DLIB_TEST(obj.receive(accum));

        // this whole block is a noop since it doesn't end up doing anything.
        for (int k = 0; k < 100; ++k)
        {
            dlog << LINFO << "k: " << k;
            for (int i = 0; i < 4; ++i)
            {
                obj.send(accum, (obj.node_id()+1)%obj.number_of_nodes());
                DLIB_TEST(obj.receive(accum));
            }
        }


        dlog << LINFO << "TERMINATE";
        if (obj.node_id() == 0)
            result = accum;
    }


    void test3_job(bsp_context& obj)
    {
        int junk;
        test3_job_driver(obj, junk);
    }


    void dotest3()
    {
        dlog << LINFO << "start dotest3()";
        print_spinner();
        bool error_occurred = false;
        {
            thread_function t1(callfunct(test3_job, 12345, error_occurred));
            thread_function t2(callfunct(test3_job, 12346, error_occurred));
            thread_function t3(callfunct(test3_job, 12347, error_occurred));

            // wait a little bit for the threads to start up
            dlib::sleep(200);

            try
            {
                std::vector<std::pair<std::string,unsigned short> > hosts;
                hosts.push_back(make_pair("127.0.0.1",12345));
                hosts.push_back(make_pair("127.0.0.1",12346));
                hosts.push_back(make_pair("127.0.0.1",12347));
                int result = 0;
                const int expected =  1+2+3 + 0+2+3 + 0+1+3 + 0+1+2;
                bsp_connect(hosts, test3_job_driver, dlib::ref(result));

                dlog << LINFO << "result: " << result;
                dlog << LINFO << "should be: " << expected;
                DLIB_TEST(result == expected);
            }
            catch (std::exception& e)
            {
                dlog << LERROR << "error during bsp_context: " << e.what();
                DLIB_TEST(false);
            }

        }
        DLIB_TEST(error_occurred == false);
    }

// ----------------------------------------------------------------------------------------

    class bsp_tester : public tester
    {

    public:
        bsp_tester (
        ) :
            tester ("test_bsp",
                    "Runs tests on the BSP components.")
        {}

        void perform_test (
        )
        {
            dotest1();
            dotest2<0>();
            dotest2<1>();
            dotest2<2>();
            dotest3();
        }
    } a;

}

