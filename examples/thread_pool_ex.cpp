// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the thread_pool 
    object from the dlib C++ Library.


    In this example we will crate a thread pool with 3 threads and then show a
    few different ways to send tasks to the pool.
*/


#include <dlib/threads.h>
#include <dlib/misc_api.h>  // for dlib::sleep
#include <dlib/logger.h>
#include <vector>

using namespace dlib;

// We will be using the dlib logger object to print messages in this example
// because its output is timestamped and labeled with the thread that the log
// message came from.  This will make it easier to see what is going on in this
// example.  Here we make an instance of the logger.  See the logger
// documentation and examples for detailed information regarding its use.
logger dlog("main");


// Here we make an instance of the thread pool object.  You could also use the
// global dlib::default_thread_pool(), which automatically selects the number of
// threads based on your hardware.  But here let's make our own.
thread_pool tp(3);

// ----------------------------------------------------------------------------------------

class test
{
    /*
        The thread_pool accepts "tasks" from the user and schedules them for 
        execution in one of its threads when one becomes available.  Each task 
        is just a request to call a function.  So here we create a class called 
        test with a few member functions, which we will have the thread pool call 
        as tasks.
    */
public:

    void mytask()
    {
        dlog << LINFO << "mytask start";

        dlib::future<int> var;

        var = 1;

        // Here we ask the thread pool to call this->subtask() and this->subtask2().
        // Note that calls to add_task() will return immediately if there is an 
        // available thread.  However, if there isn't a thread ready then
        // add_task() blocks until there is such a thread.  Also, note that if
        // mytask() is executed within the thread pool then calls to add_task()
        // will execute the requested task within the calling thread in cases
        // where the thread pool is full.  This means it is always safe to spawn
        // subtasks from within another task, which is what we are doing here.
        tp.add_task(*this,&test::subtask,var); // schedule call to this->subtask(var) 
        tp.add_task(*this,&test::subtask2);    // schedule call to this->subtask2() 

        // Since var is a future, this line will wait for the test::subtask task to 
        // finish before allowing us to access the contents of var.  Then var will 
        // return the integer it contains.  In this case result will be assigned 
        // the value 2 since var was incremented by subtask().
        int result = var;
        dlog << LINFO << "var = " << result;

        // Wait for all the tasks we have started to finish.  Note that
        // wait_for_all_tasks() only waits for tasks which were started by the
        // calling thread.  So you don't have to worry about other unrelated
        // parts of your application interfering.  In this case it just waits
        // for subtask2() to finish.
        tp.wait_for_all_tasks();

        dlog << LINFO << "mytask end" ;
    }

    void subtask(int& a)
    {
        dlib::sleep(200);
        a = a + 1;
        dlog << LINFO << "subtask end ";
    }

    void subtask2()
    {
        dlib::sleep(300);
        dlog << LINFO << "subtask2 end ";
    }

};

// ----------------------------------------------------------------------------------------

int main() try
{
    // tell the logger to print out everything
    dlog.set_level(LALL);


    dlog << LINFO << "schedule a few tasks";

    test taskobj;
    // Schedule the thread pool to call taskobj.mytask().  Note that all forms of
    // add_task() pass in the task object by reference.  This means you must make sure,
    // in this case, that taskobj isn't destructed until after the task has finished
    // executing.
    tp.add_task(taskobj, &test::mytask);

    // This behavior of add_task() enables it to guarantee that no memory allocations
    // occur after the thread_pool has been constructed, so long as the user doesn't
    // call any of the add_task_by_value() routines.  The future object also doesn't
    // perform any memory allocations or contain any system resources such as mutex
    // objects.  If you don't care about memory allocations then you will likely find
    // the add_task_by_value() interface more convenient to use, which is shown below.



    // If we call add_task_by_value() we pass task objects to a thread pool by value.
    // So in this case we don't have to worry about keeping our own instance of the
    // task.  Here we create a lambda function and pass it right in and everything
    // works like it should.
    dlib::future<int> num = 3;
    tp.add_task_by_value([](int& val){val += 7;}, num);  // adds 7 to num
    int result = num.get();
    dlog << LINFO << "result = " << result;   // prints result = 10


    // dlib also contains dlib::async(), which is essentially identical to std::async()
    // except that it launches tasks to a dlib::thread_pool (using add_task_by_value)
    // rather than starting an unbounded number of threads.  As an example, here we
    // make 10 different tasks, each assigns a different value into the elements of the
    // vector vect.
    std::vector<std::future<unsigned long>> vect(10);
    for (unsigned long i = 0; i < vect.size(); ++i)
        vect[i] = dlib::async(tp, [i]() { return i*i; });
    // Print the results
    for (unsigned long i = 0; i < vect.size(); ++i)
        dlog << LINFO << "vect["<<i<<"]: " << vect[i].get();


    // Finally, it's usually a good idea to wait for all your tasks to complete.
    // Moreover, if any of your tasks threw an exception then waiting for the tasks
    // will rethrow the exception in the calling context, allowing you to handle it in
    // your local thread.  Also, if you don't wait for the tasks and there is an
    // exception and you allow the thread pool to be destructed your program will be
    // terminated.  So don't ignore exceptions :)
    tp.wait_for_all_tasks();


    /* A possible run of this program might produce the following output (the first
       column is the time the log message occurred and the value in [] is the thread
       id for the thread that generated the log message):

    0 INFO  [0] main: schedule a few tasks
    0 INFO  [1] main: task start
    0 INFO  [0] main: result = 10
  200 INFO  [2] main: subtask end 
  200 INFO  [1] main: var = 2
  200 INFO  [0] main: vect[0]: 0
  200 INFO  [0] main: vect[1]: 1
  200 INFO  [0] main: vect[2]: 4
  200 INFO  [0] main: vect[3]: 9
  200 INFO  [0] main: vect[4]: 16
  200 INFO  [0] main: vect[5]: 25
  200 INFO  [0] main: vect[6]: 36
  200 INFO  [0] main: vect[7]: 49
  200 INFO  [0] main: vect[8]: 64
  200 INFO  [0] main: vect[9]: 81
  300 INFO  [3] main: subtask2 end 
  300 INFO  [1] main: task end
    */
}
catch(std::exception& e)
{
    std::cout << e.what() << std::endl;
}


