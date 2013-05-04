// Numerical Integration method based on the adaptive Simpson method in
// Gander, W. and W. Gautschi, "Adaptive Quadrature – Revisited,"
// BIT, Vol. 40, 2000, pp. 84-101

// Test functions taken from Pedro Gonnet's dissertation at ETH: 
// Adaptive Quadrature Re-Revisited
// http://e-collection.library.ethz.ch/eserv/eth:65/eth-65-02.pdf

#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

//***************************************************************//
//*Begin definitions of test functions //

//Initial Test Function
double f(double x)
{
    return pow(x,0.5);
}

// The Lyness - Kaganove test functions from page 167 of Gonnet's thesis.

// lambda in [0,1], alpha in [-0.5,0], x in [0,1]
double LK1(double x, double lambda, double alpha)
{
    return pow(abs(x-lambda),alpha);
}

// lambda in [0,1], alpha in [0,1], x in [0,1]
double LK2(double x, double lambda, double alpha)
{
    if(x > lambda)
    {
        return 0;
    }
    else
    {
        return pow(e, alpha*x);
    }
}

// lambda in [0,1], alpha in [0,4], x in [0,1]
double LK3(double x, double lambda, double alpha)
{
    return pow(e,-alpha*abs(x-lambda));
}

// lambda in [1,2], alpha in [-6,-3], x in [1,2]
double LK4(double x, double lambda, double alpha)
{
    return pow(10,alpha)/((x-lambda)*(x-lambda)+pow(10,alpha));
}

// lambda_i in [1,2], alpha in [-5,-3], x in [1,2]
double LK5(double x, double lambda1, double lambda2, double lambda3, double lambda4, double alpha)
{
    return   pow(10,alpha)/((x-lambda1)*(x-lambda1)+pow(10,alpha)) 
           + pow(10,alpha)/((x-lambda2)*(x-lambda2)+pow(10,alpha))
           + pow(10,alpha)/((x-lambda3)*(x-lambda3)+pow(10,alpha))
           + pow(10,alpha)/((x-lambda4)*(x-lambda4)+pow(10,alpha));
}

// lambda in [0,1], alpha in [1.8,2], x in [0,1]
double LK6(double x, double lambda, double alpha)
{
    double beta = pow(10,alpha)/max(lambda*lambda,(1-lambda)*(1-lambda)); 
    return 2*beta*cos(beta*(x-lambda)*(x-lambda));
}

// Test Battery from reference [33] and p. 168 of Gonnet's thesis.

// x in [0,1]
double GG1(double x)
{
    return pow(e,x);
}

// x in [0,1]
double GG2(double x)
{
    if(x > 0.3)
    {
        return 1.0;
    }
    else
    {
        return 0;
    }
}

// x in [0,1]
double GG3(double x)
{
    return pow(x,0.5);
}

// x in [0,1]
double GG4(double x)
{
    return 22/25*cosh(x)-cos(x);
}

// x in [-1,1]
double GG5(double x)
{
    return 1/(pow(x,4) + pow(x,2) + 0.9);
}

// x in [0,1]
double GG6(double x)
{
    return pow(x,1.5);
}

// x in [0,1]
double GG7(double x)
{
    return pow(x,-0.5);
}

// x in [0,1]
double GG8(double x)
{
    return 1/(1 + pow(x,4));
}

// x in [0,1]
double GG9(double x)
{
    return 2/(2 + sin(10*pi*x));
}

// x in [0,1]
double GG10(double x)
{
    return 1/(1+x);
}

// x in [0,1]
double GG11(double x)
{
    1/(1 + pow(e,x));
}

// x in [0,1]
double GG12(double x)
{
    return x/(pow(e,x)-1);
}

// x in [0.1, 1]
double GG13(double x)
{
    return sin(100.0*pi*x)/(pi*x);
}

// x in [0, 10]
double GG14(double x)
{
    return sqrt(50)*pow(e,-50.0*pi*x*x);
}

// x in [0, 10]
double GG15(double x)
{
    return 25.0*pow(e,-25.0*x);
}

// x in [0, 10]
double GG16(double x)
{
    return 50.0/(pi*(2500.0*x*x+1));
}

// x in [0.01, 1]
double GG17(double x)
{
    return 50.0*pow((sin(50.0*pi*x)/(50.0*pi*x)),2);
}

// x in [0, pi]
double GG18(double x)
{
    return cos(cos(x)+3*sin(x)+2*cos(2*x)+3*cos(3*x));
}

// x in [0,1]
double GG19(double x)
{
    return log10(x);
}

// x in [-1,1]
double GG20(double x)
{
    return 1/(1.005+x*x);
}

// x in [0,1]
double GG21(double x)
{
    return 1/cosh(20.0*(x-1/5)) + 1/cosh(400.0*(x-2/5)) + 1/cosh(8000.0*(x-3/5));
}

// x in [0,1]
double GG22(double x)
{
    return 4*pi*pi*x*sin(20.0*pi*x)*cos(2*pi*x);
}

// x in [0,1]
double GG23(double x)
{
    return 1/(1+(230*x-30)*(230*x-30));
}

// x in [0,3]
double GG24(double x)
{
    return floor(pow(e,x));
}

// x in [0,5]
double GG25(double x)
{
    if(x < 1)
    {
        return (x + 1);
    }
    else if(x >= 1 && x <= 3)
    {
        return 3 - x;
    }
    else
    {
        return 2;
    }
}

// Returns double machine precision
// Taken from Wikipedia en.wikipedia.org/wiki/Machine_epsilon
template<typename float_t, typename int_t>
float_t machine_eps()
{
    union
    {
        float_t f;
        int_t   i;
    }   one, one_plus, little, last_little;
 
    one.f    = 1.0;
    little.f = 1.0;
    last_little.f = little.f;

    while(true)
    {
        one_plus.f = one.f;
        one_plus.f += little.f;
 
        if( one.i != one_plus.i )
        {
            last_little.f = little.f;
            little.f /= 2.0;
        }
        else
        {
            return last_little.f;
        }
    }
}

// Main Integration Function.
// Supporting Integration Function
template <typename T, typename funct>
T AdaptSimpstp(const funct& f, T a, T b, T fa, T fm, T fb, T is)
{
    T m = (a + b)/2;
    T h = (b - a)/4;
    T fml = f(a + h);
    T fmr = f(b - h);
    T i1 = h/1.5*(fa+4*fm+fb);
    T i2 = h/3.0*(fa+4*(fml+fmr)+2*fm+fb);
    i1 = (16.0*i2 - i1)/15.0;
    T Q = 0;

    if((is+(i1-i2) == is) || (m <= a) || (b <= m))
    {
        if((m <= a) || (b <= m))
        {
            cout << "INT ERR" << endl;
        }
    
        Q = i1;
    }
    else 
    {
       Q = AdaptSimpstp(f, a, m, fa, fml, fm, is) + AdaptSimpstp(f,m,b,fm,fmr,fb,is); 
    }

    return Q;
}

// Main integration function. 
// f -- function to integrate, 
// a -- left end point
// b -- right end point
// tol -- error tolerance 

template <typename T, typename funct>
T AdaptSimp(const funct& f, T a, T b, T tol)
{
    T eps = machine_eps<T, uint64_t>();

    if(tol < eps)
    {
        tol = eps;
    }

    const T ba = b-a;
    const T fa = f(a);
    const T fb = f(b);
    const T fm = f((a+b)/2);

    T is =ba/8*(fa+fb+fm+ f(a + 0.9501*ba) + f(a + 0.2311*ba) + f(a + 0.6068*ba)
                           + f(a + 0.4860*ba) + f(a + 0.8913*ba));

    if(is == 0)
    {
        is = b-a;
    }

    is = is*tol/eps;
    
    T tstvl = AdaptSimpstp(f, a, b, fa, fm, fb, is);

    return tstvl;

}

// Examples 
int main()
{

typedef double T;

T tol = 1e-10;
T a = 0;
T b = 5;

T tstvl2 = AdaptSimp(&f, a, b, tol);

cout << "Integral Value is: " << std::setprecision(18) << tstvl2 << endl;

return 0;

}

