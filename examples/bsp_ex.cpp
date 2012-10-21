// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the Bulk Synchronous Parallel 
    processing tools from the dlib C++ Library.


*/





#include "dlib/cmd_line_parser.h"
#include "dlib/bsp.h"
#include "dlib/matrix.h"

#include <iostream>

typedef dlib::cmd_line_parser<char>::check_1a_c clp_parser;

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

double f ( double x)
{
    return std::pow(x-2.0, 2.0);
}

// ----------------------------------------------------------------------------------------

void bsp_job_node_0 (
    bsp_context& context,
    double& min_value,
    double& optimal_x
)
{
    double left = -100;
    double right = 100;

    min_value = std::numeric_limits<double>::infinity();
    double interval_width = std::abs(right-left);

    for (int i = 0; i < 10000; ++i)
    {
        context.broadcast(left);
        context.broadcast(right);

        for (unsigned int k = 1; k < context.number_of_nodes(); ++k)
        {
            std::pair<double,double> val;
            context.receive(val);
            if (val.second < min_value)
            {
                min_value = val.second;
                optimal_x = val.first;
            }
        }

        interval_width *= 0.5;
        left  = optimal_x - interval_width/2;
        right = optimal_x + interval_width/2;
    }
}

// ----------------------------------------------------------------------------------------

void bsp_job_other_nodes (
    bsp_context& context,
    long grid_resolution
)
{
    double left, right;
    while (context.try_receive(left))
    {
        context.receive(right);

        const double l = (context.node_id()-1)/(context.number_of_nodes()-1.0);
        const double r = context.node_id()    /(context.number_of_nodes()-1.0);

        const double width = right-left;
        const matrix<double> values_to_check = linspace(left+l*width, left+r*width, grid_resolution);

        double best_x;
        double best_val = std::numeric_limits<double>::infinity();
        for (long j = 0; j < values_to_check.size(); ++j)
        {
            double temp = f(values_to_check(j));
            if (temp < best_val)
            {
                best_val = temp;
                best_x = values_to_check(j);
            }
        }

        context.send(make_pair(best_x, best_val), 0);
    }
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        clp_parser parser;
        parser.add_option("h","Display this help message.");
        parser.add_option("l","Run as a listening BSP node.",1);

        parser.parse(argc, argv);
        parser.check_option_arg_range("l", 1, 65535);

        if (parser.option("h"))
        {
            // display all the command line options
            cout << "Usage: bsp_ex (-l port | <list of hosts>)\n";
            parser.print_options(cout); 
            cout << endl;
            return 0;
        }


        if (parser.option("l"))
        {
            const unsigned short listening_port = get_option(parser, "l", 0);
            cout << "Listening in port " << listening_port << endl;
            const long grid_resolution = 100;
            bsp_listen(listening_port, bsp_job_other_nodes, grid_resolution);
        }
        else
        {
            if (parser.number_of_arguments() == 0)
            {
                cout << "You must give some listening BSP nodes as arguments to this program!" << endl;
                return 0;
            }

            std::vector<network_address> hosts;
            for (unsigned long i = 0; i < parser.number_of_arguments(); ++i)
                hosts.push_back(parser[i]);

            double min_value, optimal_x;
            bsp_connect(hosts, bsp_job_node_0, dlib::ref(min_value), dlib::ref(optimal_x));

            cout << "optimal_x: "<< optimal_x << endl;
            cout << "min_value: "<< min_value << endl;
        }

    }
    catch (std::exception& e)
    {
        cout << "error in main(): " << e.what() << endl;
    }
}

