/*

    This is an example illustrating the use of the dir_nav component from the dlib C++ Library.
    It prints a listing of all directories and files in the users
    current working directory or the directory specified on the command line.  

*/


#include <iostream>
#include <iomanip>
#include "dlib/dir_nav.h"
#include "dlib/queue.h"
#include "dlib/static_set.h"

using namespace std;
using namespace dlib;

typedef queue<directory>::kernel_2a queue_of_dirs;
typedef queue<file>::kernel_2a queue_of_files;
typedef static_set<file>::kernel_1a set_of_files;
typedef static_set<directory>::kernel_1a set_of_dirs;


int main(int argc, char** argv)
{
    try
    {
        string loc;
        if (argc == 2)
            loc = argv[1];
        else
            loc = ".";  // if no argument is given then use the current working dir.
  
        directory test(loc);

        queue_of_dirs dirs;
        queue_of_files files;
        set_of_dirs sorted_dirs;
        set_of_files sorted_files;

        cout << "directory: " << test.name() << endl;
        cout << "full path: " << test.full_name() << endl;        
        cout << "is root:   " << ((test.is_root())?"yes":"no") << endl;
        
        // get all directories and files in test
        test.get_dirs(dirs);
        test.get_files(files);

        // load the dirs and files into static_sets.  This
        // seems weird but a static_set can be enumerated in sorted order
        // so this way we can print everything in sorted order.  This
        // static_set also uses a median of three quick sort so the sorting 
        // should be very fast.
        sorted_files.load(files);
        sorted_dirs.load(dirs);

        cout << "\n\n\n";

        // print all the subdirectories
        while (sorted_dirs.move_next())
            cout << "        <DIR>    " << sorted_dirs.element().name() << "\n";

        // print all the subfiles
        while (sorted_files.move_next())
            cout << setw(13) << sorted_files.element().size() << "    " << sorted_files.element().name() << "\n";


        cout << "\n\nnumber of dirs:  " << sorted_dirs.size() << endl;
        cout << "number of files: " << sorted_files.size() << endl;

    }
    catch (file::file_not_found e)
    {
        cout << "file not found or accessable: " << e.info << endl;
    }
    catch (directory::dir_not_found e)
    {
        cout << "dir not found or accessable: " << e.info << endl;
    }
    catch (directory::listing_error e)
    {
        cout << "listing error: " << e.info << endl;
    }
}


