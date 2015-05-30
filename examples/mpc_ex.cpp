// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

*/



#include <dlib/gui_widgets.h>
#include <dlib/control.h>
#include <dlib/image_transforms.h>


using namespace std;
using namespace dlib;

//  ----------------------------------------------------------------------------

int main()
{
    // state is x, y, x_vel, y_vel
    matrix<double,4,4> A;
    A = 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    matrix<double,4,2> B;
    B = 0, 0,
        0, 0,
        1, 0,
        0, 1;

    matrix<double,4,1> C;
    C = 0,
        0,
        0,
        0.1;

    matrix<double,4,1> Q;
    Q = 1, 1, 0, 0;

    matrix<double,2,1> R, lower, upper;
    R = 1, 1;
    lower = -0.5, -0.5;
    upper =  0.5,  0.5;

    mpc<4,2,30> controller(A,B,C,Q,R,lower,upper);

    dlib::rand rnd;
    matrix<double,4,1> target;
    target = rnd.get_random_double()*400,rnd.get_random_double()*400,0,0;
    controller.set_target(target);


    matrix<double,4,1> current_state;
    current_state = 200,200,0,0;

    matrix<rgb_pixel> world(400,400);
    image_window win;

    int iter = 0;
    while(!win.is_closed())
    {
        matrix<double,2,1> action = controller(current_state);

        assign_all_pixels(world, rgb_pixel(255,255,255));
        const dpoint pos = point(current_state(0),current_state(1));
        const dpoint goal = point(target(0),target(1));
        draw_solid_circle(world, goal, 9, rgb_pixel(100,255,100));
        draw_solid_circle(world, pos, 7, 0);
        draw_line(world, pos, pos-50*action, rgb_pixel(255,0,0));

        current_state = A*current_state + B*action + C;
        win.set_image(world);
        dlib::sleep(100);

        // Every 100 iterations change the target to some other random location. 
        ++iter;
        if (iter > 100)
        {
            iter = 0;
            target = rnd.get_random_double()*400,rnd.get_random_double()*400,0,0;
            controller.set_target(target);
        }
    }
}

//  ----------------------------------------------------------------------------

