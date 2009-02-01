
/*
    This is an example illustrating the use of the matrix object 
    from the dlib C++ Library.

    This file also contains a discussion of the template expression
    technique and how it is used by this library.

*/


#include <iostream>
#include "dlib/matrix.h"

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main()
{
    // Lets begin this example by using the library to solve a simple 
    // linear system.
    // 
    // We will find the value of x such that y = M*x where
    //
    //      3.5
    // y =  1.2
    //      7.8
    //
    // and M is
    //
    //      54.2   7.4   12.1
    // M =  1      2     3
    //      5.9    0.05  1


    // First lets declare these 3 matrices.
    // This declares a matrix that contains doubles and has 3 rows and 1 column.
    matrix<double,3,1> y;
    // Make a 3 by 3 matrix of doubles for the M matrix.
    matrix<double,3,3> M;
    // Make a matrix of doubles that has unknown dimensions (the dimensions are
    // decided at runtime unlike the above two matrices which are bound at compile
    // time).  We could declare x the same way as y but I'm doing it differently
    // for the purposes of illustration.
    matrix<double> x;
    
    // You may be wondering why someone would want to specify the size of a matrix
    // at compile time when you don't have to.  The reason is two fold.  First,
    // there is often a substantial performance improvement, especially for small
    // matrices, because the compiler is able to perform loop unrolling if it knows
    // the sizes of matrices.  Second, the dlib::matrix object checks these compile
    // time sizes to ensure that the matrices are being used correctly.  For example,
    // if you attempt to compile the expression y = M; or x = y*y; you will get
    // a compiler error on those lines since those are not legal matrix operations.
    // So if you know the size of a matrix at compile time then it is always a good
    // idea to let the compiler know about it.




    // now we need to initialize the y and M matrices and we can do so like this:
    M = 54.2,  7.4,  12.1,
        1,     2,    3,
        5.9,   0.05, 1;

    y = 3.5,  
        1.2,    
        7.8;


    // the solution can be obtained now by multiplying the inverse of M with y
    x = inv(M)*y;

    cout << "x: \n" << x << endl;

    // We can check that it really worked by plugging x back into the original equation 
    // and subtracting y to see if we get a column vector with values all very close
    // to zero (Which is what happens.  Also, the values may not be exactly zero because 
    // there may be some numerical error and round off).
    cout << "M*x - y: \n" << M*x - y << endl;


    // Also note that we can create run-time sized column or row vectors like so
    matrix<double,0,1> runtime_sized_column_vector;
    matrix<double,1,0> runtime_sized_row_vector;
    // and then they are sized by saying
    runtime_sized_column_vector.set_size(3);

    // Similarly, the x matrix can be resized by calling set_size(num rows, num columns).  For example
    x.set_size(3,4);  // x now has 3 rows and 4 columns.



    // The elements of a matrix are accessed using the () operator like so
    cout << M(0,1) << endl;
    // The above expression prints out the value 7.4.  That is, the value of
    // the element at row 0 and column 1.


    // Let's compute the sum of elements in the M matrix.
    double M_sum = 0;
    // loop over all the rows
    for (long r = 0; r < M.nr(); ++r)
    {
        // loop over all the columns
        for (long c = 0; c < M.nc(); ++c)
        {
            M_sum += M(r,c);
        }
    }
    cout << "sum of all elements in M is " << M_sum << endl;

    // The above code is just to show you how to loop over the elements of a matrix.  An 
    // easier way to find this sum is to do the following:
    cout << "sum of all elements in M is " << sum(M) << endl;


    // If we have a matrix that is a row or column vector.  That is, it contains either 
    // a single row or a single column then we know that any access is always either 
    // to row 0 or column 0 so we can omit that 0 and use the following syntax.
    cout << y(1) << endl;
    // The above expression prints out the value 1.2




    // ---------------------------------  Comparison with MATLAB ---------------------------------
    // Here I list a set of Matlab commands and their equivalent expressions using the dlib matrix.

    matrix<double> A, B, C, D, E;
    matrix<int> Aint;
    matrix<long> Blong;

    // MATLAB: A = eye(3)
    A = identity_matrix<double>(3);

    // MATLAB: B = ones(3,4)
    B = uniform_matrix<double>(3,4, 1);

    // MATLAB: C = 1.4*A
    C = 1.4*A;

    // MATLAB: D = A.*C
    D = pointwise_multiply(A,C);

    // MATLAB: E = A * B
    E = A*B;

    // MATLAB: E = A + B
    E = A + C;

    // MATLAB: E = E'
    E = trans(E);  // Note that if you want a conjugate transpose then you need to say conj(trans(E))

    // MATLAB: E = B' * B
    E = trans(B)*B;

    double var;
    // MATLAB: var = A(1,2)
    var = A(0,1); // dlib::matrix is 0 indexed rather than starting at 1 like Matlab.

    // MATLAB: C = round(C)
    C = round(C);

    // MATLAB: C = floor(C)
    C = floor(C);

    // MATLAB: C = ceil(C)
    C = ceil(C);

    // MATLAB: C = diag(B)
    C = diag(B);

    // MATLAB: B = cast(A, "int32")
    Aint = matrix_cast<int>(A);

    // MATLAB: A = B(1,:)
    A = rowm(B,0);

    // MATLAB: A = B([1:2],:)
    A = rowm(B,range(0,1));

    // MATLAB: A = B(:,1)
    A = colm(B,0);

    // MATLAB: A = [1:5]'
    Blong = range(1,5);

    // MATLAB: A = [1:5]
    Blong = trans(range(1,5));

    // MATLAB: A = [1:2:5]
    Blong = trans(range(1,2,5));

    // MATLAB: A = B([1:3], [1:2])
    A = subm(B, range(0,2), range(0,1));
    // or equivalently
    A = subm(B, rectangle(0,0,1,2));


    // MATLAB: A = B([1:3], [1:2:4])
    A = subm(B, range(0,2), range(0,2,3));

    // MATLAB: B(:,:) = 5
    B = 5;
    // or equivalently
    set_all_elements(B,5);


    // MATLAB: B([1:2],[1,2]) = 7
    set_subm(B,range(0,1), range(0,1)) = 7;

    // MATLAB: B([1:3],[2:3]) = A
    set_subm(B,range(0,2), range(1,2)) = A;

    // MATLAB: B(:,1) = 4
    set_colm(B,0) = 4;

    // MATLAB: B(:,[1:2]) = 4
    set_colm(B,range(0,1)) = 4;

    // MATLAB: B(:,1) = B(:,2)
    set_colm(B,0) = colm(B,1);

    // MATLAB: B(1,:) = 4
    set_rowm(B,0) = 4;

    // MATLAB: B(1,:) = B(2,:)
    set_rowm(B,0) = rowm(B,1);

    // MATLAB: var = det(E' * E)
    var = det(trans(E)*E);

    // MATLAB: C = pinv(E)
    C = pinv(E);

    // MATLAB: C = inv(E)
    C = inv(E);

    // MATLAB: [A,B,C] = svd(E)
    svd(E,A,B,C);

    // MATLAB: A = chol(E,'lower') 
    A = chol(E);

    // MATLAB: var = min(min(A))
    var = min(A);



    // -------------------------  Template Expressions -----------------------------
    // Now I will discuss the "template expressions" technique and how it is 
    // used in the matrix object.  First consider the following expression:
    x = y + y;

    /*
        Normally this expression results in machine code that looks, at a high 
        level, like the following:
            temp = y + y;
            x = temp

        Temp is a temporary matrix returned by the overloaded + operator.  
        temp then contains the result of adding y to itself.  The assignment 
        operator copies the value of temp into x and temp is then destroyed while 
        the blissful C++ user never sees any of this.

        This is, however, totally inefficient.  In the process described above 
        you have to pay for the cost of constructing a temporary matrix object
        and allocating its memory.  Then you pay the additional cost of copying
        it over to x.  It also gets worse when you have more complex expressions
        such as x = round(y + y + y + M*y) which would involve the creation and copying 
        of 5 temporary matrices.  
        
        All these inefficiencies are removed by using the template expressions 
        technique.  The exact details of how the technique is performed are well
        outside the scope of this example but the basic idea is as follows.  Instead
        of having operators and functions return temporary matrix objects you 
        return a special object that represents the expression you wish to perform.

        So consider the expression x = y + y again.  With dlib::matrix what happens
        is the expression y+y returns a matrix_exp object instead of a temporary matrix.
        The construction of a matrix_exp does not allocate any memory or perform any 
        computations.  The matrix_exp however has an interface that looks just like a 
        dlib::matrix object and when you ask it for the value of one of its elements 
        it computes that value on the spot.  Only in the assignment operator does
        someone ask the matrix_exp for these values so this avoids the use of any
        temporary matrices.  Thus the statement x = y + y is equivalent to the following 
        code:
            // loop over all elements in y matrix
            for (long r = 0; r < y.nr(); ++r)
                for (long c = 0; c < y.nc(); ++c)
                    x(r,c) = y(r,c) + y(r,c);  
                
       
        This technique works for expressions of arbitrary complexity.  So if you 
        typed x = round(y + y + y + M*y) it would involve no temporary matrices being 
        created at all.  Each operator takes and returns only matrix_exp objects.
        Thus, no computations are performed until the assignment operator requests
        the values from the matrix_exp it receives as input. 
            




        There is, however, a slight complication in all of this.  It is for statements 
        that involve the multiplication of a complex matrix_exp such as the following:
    */
        x = M*(M+M+M+M+M+M+M);
    /*
        According to the discussion above, this statement would compute the value of 
        M*(M+M+M+M+M+M+M) totally without any temporary matrix objects.  This sounds 
        good but we should take a closer look.  Consider that the + operator is 
        invoked 6 times.  This means we have something like this:

        x = M * (matrix_exp representing M+M+M+M+M+M+M);

        M is being multiplied with a quite complex matrix_exp.  Now recall that when 
        you ask a matrix_exp what the value of any of its elements are it computes 
        their values *right then*.  
        
        If you think on what is involved in performing a matrix multiply you will 
        realize that each element of a matrix is accessed M.nr() times.  In the 
        case of our above expression the cost of accessing an element of the 
        matrix_exp on the right hand side is the cost of doing 6 addition operations. 

        Thus, it would be faster to assign M+M+M+M+M+M+M to a temporary matrix and then
        multiply that by M.  This is exactly what the dlib::matrix does under the covers.  
        This is because it is able to spot expressions where the introduction of a 
        temporary is needed to speed up the computation and it will automatically do this 
        for you.  



        
        Another complication that is dealt with automatically is aliasing.  Consider
        the following expressions:
           (1)  M = M + M
           (2)  B = M * M. 
           (3)  M = M * M. 

        Expressions (1) and (3) are an example of aliasing and expression (3) is also
        an example of destructive aliasing.  

        Expression (1) can and does operate without introducing any temporaries even though
        there is aliasing present in the expression.  The result is loaded straight into M 
        using the template expression techniques described above.  Expression (2) also 
        operates without any temporaries being introduced since there isn't any aliasing at all.
        Expression (3) however contains destructive aliasing.  This is because we can't 
        change any of the values in the M matrix without corrupting the ultimate result of
        the matrix multiply.  So we need to introduce a temporary.  These situations are
        dealt with by dlib::matrix automatically.  Moreover, it can tell the different between
        simple aliasing and destructive aliasing and will only introduce temporaries when
        they are necessary.
    */
}

// ----------------------------------------------------------------------------------------


