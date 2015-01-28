// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

/*
*/


#include <iostream>
#include <dlib/sqlite.h>
#include <dlib/matrix.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------
bool table_exists (
    database& db,
    const std::string& tablename
)
{
    return query_int(db, "select count(*) from sqlite_master where name = '"+tablename+"'")==1;
}

int main() try
{

    database db("stuff.db");

    if (!table_exists(db,"davis"))
        db.exec("create table davis (name, age, data)");

    statement st(db, "insert into davis VALUES(?,?,?)");

    string name = "davis";
    int age = 32;
    matrix<double> m = randm(3,3);

    st.bind(1, name);
    st.bind(2, age);
    st.bind(3, m);
    st.exec();



    statement st2(db, "select * from davis");
    st2.exec();
    while(st2.move_next())
    {
        string name;
        int age;
        matrix<double> m;
        st2.get_column(0, name);
        st2.get_column(1, age);
        st2.get_column(2, m);
        cout << name << " " << age << "\n" << m << endl << endl;
    }

}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------


