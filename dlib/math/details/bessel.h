// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

//  Copyright (c) 2007, 2013 John Maddock
//  Copyright Christopher Kormanyos 2013.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef DLIB_MATH_DETAIL_BESSEL
#define DLIB_MATH_DETAIL_BESSEL

#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include "../../numeric_constants.h"
#include "../../assert.h"

namespace dlib
{
    namespace detail
    {
#if __cpp_lib_math_special_functions
        using std::cyl_bessel_i;
        using std::cyl_bessel_j;
#else
        constexpr unsigned int BESSEL_NITERATIONS = 250;

        template <class T>
        inline int sign(const T& z)
        {
            return (z == 0) ? 0 : std::signbit(z) ? -1 : 1;
        }

        template <class T> struct max_factorial {};
        template <> struct max_factorial<float>         { static constexpr unsigned value = 34; };
        template <> struct max_factorial<double>        { static constexpr unsigned value = 170; };
        template <> struct max_factorial<long double>   { static constexpr unsigned value = 170; };

        template <class T>
        inline T factorial(unsigned i)
        {
            static_assert(!std::is_integral<T>::value, "bad template type");

            if(i <= max_factorial<T>::value)
            {
                switch(i)
                {
                    case 0: return 1;
                    case 1: return 1;
                    case 2: return 2;
                    case 3: return 6;
                    case 4: return 24;
                    case 5: return 120;
                    case 6: return 720;
                    case 7: return 5040;
                    case 8: return 40320;
                    case 9: return 362880;
                    case 10: return 3628800;
                    case 11: return 39916800;
                    case 12: return 479001600;
                    //I can't be bothered. The boost library has literally gone up to 100.
                    //You can tell that boost authors don't have a day job because if they did,
                    //they would realise that this is overkill and has zero impact on real-life code.
                    default: return i * factorial<T>(i-1);
                }
            }
            else
            {
                T result = std::tgamma(static_cast<T>(i+1));
                if(result > std::numeric_limits<T>::max())
                    throw std::runtime_error("factorial() overflow");
                return std::floor(result + 0.5f);
            }
        }

        template <class T>
        struct cyl_bessel_i_small_z
        {
            typedef T result_type;

            cyl_bessel_i_small_z(T v_, T z_) : k(0), v(v_), mult(z_*z_/4)
            {
                term = 1;
            }

            T operator()()
            {
                T result = term;
                ++k;
                term *= mult / k;
                term /= k + v;
                return result;
            }
        private:
            unsigned k;
            T v;
            T term;
            T mult;
        };

        template <class T>
        struct bessel_j_small_z_series_term
        {
            typedef T result_type;

            bessel_j_small_z_series_term(T v_, T x)
                    : N(0), v(v_)
            {
                mult = x / 2;
                mult *= -mult;
                term = 1;
            }
            T operator()()
            {
                T r = term;
                ++N;
                term *= mult / (N * (N + v));
                return r;
            }
        private:
            unsigned N;
            T v;
            T mult;
            T term;
        };

        template <class T>
        struct bessel_y_small_z_series_term_a
        {
            typedef T result_type;

            bessel_y_small_z_series_term_a(T v_, T x)
                    : N(0), v(v_)
            {
                mult = x / 2;
                mult *= -mult;
                term = 1;
            }
            T operator()()
            {
                T r = term;
                ++N;
                term *= mult / (N * (N - v));
                return r;
            }
        private:
            unsigned N;
            T v;
            T mult;
            T term;
        };

        template <class T>
        struct bessel_y_small_z_series_term_b
        {
            typedef T result_type;

            bessel_y_small_z_series_term_b(T v_, T x)
                    : N(0), v(v_)
            {
                mult = x / 2;
                mult *= -mult;
                term = 1;
            }
            T operator()()
            {
                T r = term;
                ++N;
                term *= mult / (N * (N + v));
                return r;
            }
        private:
            unsigned N;
            T v;
            T mult;
            T term;
        };

        template <class Functor, class U, class V>
        inline typename Functor::result_type sum_series(Functor& func, const U& factor, unsigned int& max_terms, const V& init_value = 0)
        {
            typedef typename Functor::result_type result_type;

            unsigned int counter = max_terms;

            result_type result = init_value;
            result_type next_term;
            do{
                next_term = func();
                result += next_term;
            }
            while((std::fabs(factor * result) < std::fabs(next_term)) && --counter);

            // set max_terms to the actual number of terms of the series evaluated:
            max_terms = max_terms - counter;

            return result;
        }

        template <class T>
        inline T bessel_i_small_z_series(T v, T x)
        {
            T prefix;
            if(v < max_factorial<T>::value)
            {
                prefix = pow(x / 2, v) / std::tgamma(v + 1);
            }
            else
            {
                prefix = v * log(x / 2) - std::lgamma(v + 1);
                prefix = exp(prefix);
            }
            if(prefix == 0)
                return prefix;

            cyl_bessel_i_small_z<T> s(v, x);

            unsigned int max_iter = BESSEL_NITERATIONS;
            T result = sum_series(s, std::numeric_limits<T>::epsilon(), max_iter, 0);
            return prefix * result;
        }

        template <class T>
        inline T bessel_j_small_z_series(T v, T x)
        {
            using namespace std;
            T prefix;
            if(v < max_factorial<T>::value)
            {
                prefix = pow(x / 2, v) / std::tgamma(v+1);
            }
            else
            {
                prefix = v * log(x / 2) - std::lgamma(v+1);
                prefix = exp(prefix);
            }
            if(0 == prefix)
                return prefix;

            bessel_j_small_z_series_term<T> s(v, x);
            unsigned int max_iter = BESSEL_NITERATIONS;
            T result = sum_series(s, std::numeric_limits<T>::epsilon(), max_iter, 0);
            //policies::check_series_iterations<T>("boost::math::bessel_j_small_z_series<%1%>(%1%,%1%)", max_iter, pol);
            return prefix * result;
        }

        template <class T>
        inline T bessel_y_small_z_series(T v, T x, T* pscale)
        {
            constexpr static const T log_max_value = log((std::numeric_limits<T>::max)());

            T prefix;
            T gam;
            T p = log(x / 2);
            T scale = 1;
            bool need_logs = (v >= max_factorial<T>::value) || (log_max_value / v < fabs(p));
            if(!need_logs)
            {
                gam = std::tgamma(v);
                p = pow(x / 2, v);
                if(std::numeric_limits<T>::infinity() * p < gam)
                {
                    scale /= gam;
                    gam = 1;
                    if(std::numeric_limits<T>::infinity() * p < gam)
                    {
                        return -std::numeric_limits<T>::infinity();
                        //return -policies::raise_overflow_error<T>(function, 0, pol);
                    }
                }
                prefix = -gam / (pi * p);
            }
            else
            {
                gam = std::lgamma(v);
                p = v * p;
                prefix = gam - log(pi) - p;
                if(log_max_value < prefix)
                {
                    prefix -= log(std::numeric_limits<T>::infinity() / 4);
                    scale /= (std::numeric_limits<T>::infinity() / 4);
                    if(log_max_value < prefix)
                    {
                        return -std::numeric_limits<T>::infinity();
                        //return -policies::raise_overflow_error<T>(function, 0, pol);
                    }
                }
                prefix = -exp(prefix);
            }
            bessel_y_small_z_series_term_a<T> s(v, x);
            unsigned int max_iter = BESSEL_NITERATIONS;
            *pscale = scale;
            T result = sum_series(s, std::numeric_limits<T>::epsilon(), max_iter, 0);

            //policies::check_series_iterations<T>("boost::math::bessel_y_small_z_series<%1%>(%1%,%1%)", max_iter, pol);
            result *= prefix;

            if(!need_logs)
            {
                prefix = std::tgamma(-v) * std::cos(pi*v) * p / pi;
            }
            else
            {
                int sgn = sign(std::tgamma(-v));
                prefix = std::lgamma(-v) + p;
                prefix = exp(prefix) * sgn / pi;
            }
            bessel_y_small_z_series_term_b<T> s2(v, x);
            max_iter = BESSEL_NITERATIONS;
            T b = sum_series(s2, std::numeric_limits<T>::epsilon(), max_iter, 0);
            result -= scale * prefix * b;
            return result;
        }

        // Calculate K(v, x) and K(v+1, x) by method analogous to
        // Temme, Journal of Computational Physics, vol 21, 343 (1976)
        template <typename T>
        int temme_ik(T v, T x, T* K, T* K1)
        {
            using namespace std;

            T f, h, p, q, coef, sum, sum1, tolerance;
            T a, b, c, d, sigma, gamma1, gamma2;
            unsigned long k;

            // |x| <= 2, Temme series converge rapidly
            // |x| > 2, the larger the |x|, the slower the convergence
            DLIB_ASSERT(abs(x) <= 2);
            DLIB_ASSERT(abs(v) <= 0.5f);

            T gp = std::tgamma(v+1)-1;
            T gm = std::tgamma(-v+1)-1;

            a = log(x / 2);
            b = exp(v * a);
            sigma = -a * v;
            c = abs(v) < std::numeric_limits<T>::epsilon() ?
                T(1) : T(std::sin(v*pi) / (v * pi));
            d = abs(sigma) < std::numeric_limits<T>::epsilon() ?
                T(1) : T(sinh(sigma) / sigma);
            gamma1 = abs(v) < std::numeric_limits<T>::epsilon() ?
                     T(-e) : T((0.5f / v) * (gp - gm) * c);
            gamma2 = (2 + gp + gm) * c / 2;

            // initial values
            p = (gp + 1) / (2 * b);
            q = (1 + gm) * b / 2;
            f = (cosh(sigma) * gamma1 + d * (-a) * gamma2) / c;
            h = p;
            coef = 1;
            sum = coef * f;
            sum1 = coef * h;

            // series summation
            tolerance = std::numeric_limits<T>::epsilon();
            for (k = 1; k < BESSEL_NITERATIONS; ++k)
            {
                f = (k * f + p + q) / (k*k - v*v);
                p /= k - v;
                q /= k + v;
                h = p - k * f;
                coef *= x * x / (4 * k);
                sum += coef * f;
                sum1 += coef * h;
                if (abs(coef * f) < abs(sum) * tolerance)
                {
                    break;
                }
            }

            *K = sum;
            *K1 = 2 * sum1 / x;

            return 0;
        }

        // Evaluate continued fraction fv = I_(v+1) / I_v, derived from
        // Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73
        template <typename T>
        int CF1_ik(T v, T x, T* fv)
        {
            using namespace std;
            
            T C, D, f, a, b, delta, tiny, tolerance;
            unsigned long k;

            // |x| <= |v|, CF1_ik converges rapidly
            // |x| > |v|, CF1_ik needs O(|x|) iterations to converge

            // modified Lentz's method, see
            // Lentz, Applied Optics, vol 15, 668 (1976)
            tolerance = 2 * std::numeric_limits<T>::epsilon();
            tiny = sqrt(std::numeric_limits<T>::min());
            C = f = tiny;                           // b0 = 0, replace with tiny
            D = 0;
            for (k = 1; k < BESSEL_NITERATIONS; ++k)
            {
                a = 1;
                b = 2 * (v + k) / x;
                C = b + a / C;
                D = b + a * D;
                if (C == 0) { C = tiny; }
                if (D == 0) { D = tiny; }
                D = 1 / D;
                delta = C * D;
                f *= delta;
                if (abs(delta - 1) <= tolerance)
                {
                    break;
                }
            }

            *fv = f;

            return 0;
        }
        
        // Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
        // z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x), see
        // Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
        template <typename T>
        int CF2_ik(T v, T x, T* Kv, T* Kv1)
        {
            using namespace std;

            T S, C, Q, D, f, a, b, q, delta, tolerance, current, prev;
            unsigned long k;

            // |x| >= |v|, CF2_ik converges rapidly
            // |x| -> 0, CF2_ik fails to converge

            DLIB_ASSERT(abs(x) > 1);

            // Steed's algorithm, see Thompson and Barnett,
            // Journal of Computational Physics, vol 64, 490 (1986)
            tolerance = std::numeric_limits<T>::epsilon();
            a = v * v - 0.25f;
            b = 2 * (x + 1);                              // b1
            D = 1 / b;                                    // D1 = 1 / b1
            f = delta = D;                                // f1 = delta1 = D1, coincidence
            prev = 0;                                     // q0
            current = 1;                                  // q1
            Q = C = -a;                                   // Q1 = C1 because q1 = 1
            S = 1 + Q * delta;                            // S1

            for (k = 2; k < BESSEL_NITERATIONS; ++k)     // starting from 2
            {
                // continued fraction f = z1 / z0
                a -= 2 * (k - 1);
                b += 2;
                D = 1 / (b + a * D);
                delta *= b * D - 1;
                f += delta;

                // series summation S = 1 + \sum_{n=1}^{\infty} C_n * z_n / z_0
                q = (prev - (b - 2) * current) / a;
                prev = current;
                current = q;                        // forward recurrence for q
                C *= -a / k;
                Q += C * q;
                S += Q * delta;
                //
                // Under some circumstances q can grow very small and C very
                // large, leading to under/overflow.  This is particularly an
                // issue for types which have many digits precision but a narrow
                // exponent range.  A typical example being a "double double" type.
                // To avoid this situation we can normalise q (and related prev/current)
                // and C.  All other variables remain unchanged in value.  A typical
                // test case occurs when x is close to 2, for example cyl_bessel_k(9.125, 2.125).
                //
                if(q < std::numeric_limits<T>::epsilon())
                {
                    C *= q;
                    prev /= q;
                    current /= q;
                    q = 1;
                }

                // S converges slower than f
                if (abs(Q * delta) < abs(S) * tolerance)
                {
                    break;
                }
            }

            constexpr static const T log_max_value = log((std::numeric_limits<T>::max)());

            if(x >= log_max_value)
                *Kv = exp(0.5f * log(pi / (2 * x)) - x - log(S));
            else
                *Kv = sqrt(pi / (2 * x)) * exp(-x) / S;
            *Kv1 = *Kv * (0.5f + v + x + (v * v - 0.25f) * f) / x;

            return 0;
        }

        template <class T>
        T asymptotic_bessel_i_large_x(T v, T x)
        {
            using namespace std;

            T s = 1;
            T mu = 4 * v * v;
            T ex = 8 * x;
            T num = mu - 1;
            T denom = ex;

            s -= num / denom;

            num *= mu - 9;
            denom *= ex * 2;
            s += num / denom;

            num *= mu - 25;
            denom *= ex * 3;
            s -= num / denom;

            // Try and avoid overflow to the last minute:
            T e = exp(x/2);

            s = e * (e * s / sqrt(2 * x * pi));

            return std::isfinite(s) ? s : std::numeric_limits<T>::infinity();
        }

        enum{
            need_i = 1,
            need_k = 2,
            need_j = 4,
            need_y = 8
        };

        // Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
        // Temme, Journal of Computational Physics, vol 19, 324 (1975)
        template <typename T>
        int bessel_ik(T v, T x, T* I, T* K, int kind)
        {
            // Kv1 = K_(v+1), fv = I_(v+1) / I_v
            // Ku1 = K_(u+1), fu = I_(u+1) / I_u
            T u, Iv, Kv, Kv1, Ku, Ku1, fv;
            T W, current, prev, next;
            bool reflect = false;
            unsigned n, k;
            int org_kind = kind;

            if (v < 0)
            {
                reflect = true;
                v = -v;                             // v is non-negative from here
                kind |= need_k;
            }
            n = std::lround(v);
            u = v - n;                              // -1/2 <= u < 1/2

            if (x == 0)
            {
                Iv = (v == 0) ? static_cast<T>(1) : static_cast<T>(0);
                Kv = std::numeric_limits<T>::quiet_NaN(); // any value will do

                if(reflect && (kind & need_i))
                {
                    T z = (u + n % 2);
                    Iv = std::sin(z*pi) == 0 ? Iv : std::numeric_limits<T>::infinity();
                }

                *I = Iv;
                *K = Kv;
                return 0;
            }

            // x is positive until reflection
            W = 1 / x;                                 // Wronskian
            if (x <= 2)                                // x in (0, 2]
            {
                temme_ik(u, x, &Ku, &Ku1);             // Temme series
            }
            else                                       // x in (2, \infty)
            {
                CF2_ik(u, x, &Ku, &Ku1);               // continued fraction CF2_ik
            }

            prev = Ku;
            current = Ku1;
            T scale = 1;
            T scale_sign = 1;
            for (k = 1; k <= n; k++)                   // forward recurrence for K
            {
                T fact = 2 * (u + k) / x;
                if((std::numeric_limits<T>::max() - fabs(prev)) / fact < fabs(current))
                {
                    prev /= current;
                    scale /= current;
                    scale_sign *= sign(current);
                    current = 1;
                }
                next = fact * current + prev;
                prev = current;
                current = next;
            }
            Kv = prev;
            Kv1 = current;

            if(kind & need_i)
            {
                T lim = (4 * v * v + 10) / (8 * x);
                lim *= lim;
                lim *= lim;
                lim /= 24;
                if((lim < std::numeric_limits<T>::epsilon() * 10) && (x > 100))
                {
                    // x is huge compared to v, CF1 may be very slow
                    // to converge so use asymptotic expansion for large
                    // x case instead.  Note that the asymptotic expansion
                    // isn't very accurate - so it's deliberately very hard
                    // to get here - probably we're going to overflow:
                    Iv = asymptotic_bessel_i_large_x(v, x);
                }
                else if((v > 0) && (x / v < 0.25))
                {
                    Iv = bessel_i_small_z_series(v, x);
                }
                else
                {
                    CF1_ik(v, x, &fv);                         // continued fraction CF1_ik
                    Iv = scale * W / (Kv * fv + Kv1);                  // Wronskian relation
                }
            }
            else
                Iv = std::numeric_limits<T>::quiet_NaN(); // any value will do

            if (reflect)
            {
                T z = (u + n % 2);
                T fact = (2 / pi) * (std::sin(pi*z) * Kv);
                if(fact == 0)
                    *I = Iv;
                else if(std::numeric_limits<T>::max() * scale < fact)
                    *I = (org_kind & need_i) ? T(sign(fact) * scale_sign * std::numeric_limits<T>::infinity()) : T(0);
                else
                    *I = Iv + fact / scale;   // reflection formula
            }
            else
            {
                *I = Iv;
            }
            if(std::numeric_limits<T>::max() * scale < Kv)
                *K = (org_kind & need_k) ? T(sign(Kv) * scale_sign * std::numeric_limits<T>::infinity()) : T(0);
            else
                *K = Kv / scale;

            return 0;
        }

        template <class T>
        T bessel_yn_small_z(int n, T z, T* scale)
        {
            //
            // See http://functions.wolfram.com/Bessel-TypeFunctions/BesselY/06/01/04/01/02/
            //
            // Note that when called we assume that x < epsilon and n is a positive integer.
            //
            using namespace std;
            DLIB_ASSERT(n >= 0);
            DLIB_ASSERT(z < std::numeric_limits<T>::epsilon());

            if(n == 0)
            {
                return (2 / pi) * (log(z / 2) +  e);
            }
            else if(n == 1)
            {
                return (z / pi) * log(z / 2)
                       - 2 / (pi * z)
                       - (z / (2 * pi)) * (1 - 2 * e);
            }
            else if(n == 2)
            {
                return (z * z) / (4 * pi) * log(z / 2)
                       - (4 / (pi * z * z))
                       - ((z * z) / (8 * pi)) * (T(3)/2 - 2 * e);
            }
            else
            {
                T p = pow(z / 2, n);
                T result = -((factorial<T>(n - 1) / pi));
                if(p * std::numeric_limits<T>::max() < result)
                {
                    T div = std::numeric_limits<T>::max() / 8;
                    result /= div;
                    *scale /= div;
                    if(p * std::numeric_limits<T>::max() < result)
                    {
                        return - std::numeric_limits<T>::infinity();
                        //return -policies::raise_overflow_error<T>("bessel_yn_small_z<%1%>(%1%,%1%)", 0, pol);
                    }
                }
                return result / p;
            }
        }

        template <class T>
        inline bool asymptotic_bessel_large_x_limit(int v, const T& x)
        {
            //
            // Determines if x is large enough compared to v to take the asymptotic
            // forms above.  From A&S 9.2.28 we require:
            //    v < x * eps^1/8
            // and from A&S 9.2.29 we require:
            //    v^12/10 < 1.5 * x * eps^1/10
            // using the former seems to work OK in practice with broadly similar
            // error rates either side of the divide for v < 10000.
            // At double precision eps^1/8 ~= 0.01.
            //
            DLIB_ASSERT(v >= 0);
            return (v ? v : 1) < x * 0.004f;
        }

        template <class T>
        inline T asymptotic_bessel_amplitude(T v, T x)
        {
            // Calculate the amplitude of J(v, x) and Y(v, x) for large
            // x: see A&S 9.2.28.
            T s = 1;
            T mu = 4 * v * v;
            T txq = 2 * x;
            txq *= txq;

            s += (mu - 1) / (2 * txq);
            s += 3 * (mu - 1) * (mu - 9) / (txq * txq * 8);
            s += 15 * (mu - 1) * (mu - 9) * (mu - 25) / (txq * txq * txq * 8 * 6);

            return std::sqrt(s * 2 / (pi * x));
        }

        template <class T>
        T asymptotic_bessel_phase_mx(T v, T x)
        {
            //
            // Calculate the phase of J(v, x) and Y(v, x) for large x.
            // See A&S 9.2.29.
            // Note that the result returned is the phase less (x - PI(v/2 + 1/4))
            // which we'll factor in later when we calculate the sines/cosines of the result:
            //
            T mu = 4 * v * v;
            T denom = 4 * x;
            T denom_mult = denom * denom;

            T s = 0;
            s += (mu - 1) / (2 * denom);
            denom *= denom_mult;
            s += (mu - 1) * (mu - 25) / (6 * denom);
            denom *= denom_mult;
            s += (mu - 1) * (mu * mu - 114 * mu + 1073) / (5 * denom);
            denom *= denom_mult;
            s += (mu - 1) * (5 * mu * mu * mu - 1535 * mu * mu + 54703 * mu - 375733) / (14 * denom);
            return s;
        }

        template <class T>
        inline T asymptotic_bessel_y_large_x_2(T v, T x)
        {
            using namespace std;
            // Get the phase and amplitude:
            T ampl = asymptotic_bessel_amplitude(v, x);
            T phase = asymptotic_bessel_phase_mx(v, x);
            //
            // Calculate the sine of the phase, using
            // sine/cosine addition rules to factor in
            // the x - PI(v/2 + 1/4) term not added to the
            // phase when we calculated it.
            //
            T cx = cos(x);
            T sx = sin(x);
            T ci = (v / 2 + 0.25f);
            T si = sin(pi*(v / 2 + 0.25f));
            T sin_phase = sin(phase) * (cx * ci + sx * si) + cos(phase) * (sx * ci - cx * si);
            return sin_phase * ampl;
        }

        template <class T>
        inline T asymptotic_bessel_j_large_x_2(T v, T x)
        {
            // See A&S 9.2.19.
            using namespace std;
            // Get the phase and amplitude:
            T ampl = asymptotic_bessel_amplitude(v, x);
            T phase = asymptotic_bessel_phase_mx(v, x);
            //
            // Calculate the sine of the phase, using
            // sine/cosine addition rules to factor in
            // the x - PI(v/2 + 1/4) term not added to the
            // phase when we calculated it.
            //
            T cx = cos(x);
            T sx = sin(x);
            T ci = cos(pi*(v / 2 + 0.25f));
            T si = sin(pi*(v / 2 + 0.25f));
            T sin_phase = cos(phase) * (cx * ci + sx * si) - sin(phase) * (sx * ci - cx * si);
            return sin_phase * ampl;
        }

        template <class T>
        bool hankel_PQ(T v, T x, T* p, T* q)
        {
            using namespace std;
            T tolerance = 2 * std::numeric_limits<T>::epsilon();
            *p = 1;
            *q = 0;
            T k = 1;
            T z8 = 8 * x;
            T sq = 1;
            T mu = 4 * v * v;
            T term = 1;
            bool ok = true;
            do
            {
                term *= (mu - sq * sq) / (k * z8);
                *q += term;
                k += 1;
                sq += 2;
                T mult = (sq * sq - mu) / (k * z8);
                ok = fabs(mult) < 0.5f;
                term *= mult;
                *p += term;
                k += 1;
                sq += 2;
            }
            while((fabs(term) > tolerance * *p) && ok);
            return ok;
        }

        // Calculate Y(v, x) and Y(v+1, x) by Temme's method, see
        // Temme, Journal of Computational Physics, vol 21, 343 (1976)
        template <typename T>
        int temme_jy(T v, T x, T* Y, T* Y1)
        {
            using namespace std;
            T g, h, p, q, f, coef, sum, sum1, tolerance;
            T a, d, e, sigma;
            unsigned long k;

            DLIB_ASSERT(fabs(v) <= 0.5f);  // precondition for using this routine

            T gp = std::tgamma(v+1)-1;
            T gm = std::tgamma(-v+1)-1;
            T spv = std::sin(pi*v);
            T spv2 = std::sin(pi*v/2);
            T xp = pow(x/2, v);

            a = log(x / 2);
            sigma = -a * v;
            d = abs(sigma) < std::numeric_limits<T>::epsilon() ?
                T(1) : sinh(sigma) / sigma;
            e = abs(v) < std::numeric_limits<T>::epsilon() ? T(v*pi*pi / 2)
                                             : T(2 * spv2 * spv2 / v);

            T g1 = (v == 0) ? T(-euler) : T((gp - gm) / ((1 + gp) * (1 + gm) * 2 * v));
            T g2 = (2 + gp + gm) / ((1 + gp) * (1 + gm) * 2);
            T vspv = (fabs(v) < std::numeric_limits<T>::epsilon()) ? T(1/pi) : T(v / spv);
            f = (g1 * cosh(sigma) - g2 * a * d) * 2 * vspv;

            p = vspv / (xp * (1 + gm));
            q = vspv * xp / (1 + gp);

            g = f + e * q;
            h = p;
            coef = 1;
            sum = coef * g;
            sum1 = coef * h;

            T v2 = v * v;
            T coef_mult = -x * x / 4;

            // series summation
            tolerance = std::numeric_limits<T>::epsilon();
            for (k = 1; k < BESSEL_NITERATIONS; ++k)
            {
                f = (k * f + p + q) / (k*k - v2);
                p /= k - v;
                q /= k + v;
                g = f + e * q;
                h = p - k * g;
                coef *= coef_mult / k;
                sum += coef * g;
                sum1 += coef * h;
                if (abs(coef * g) < abs(sum) * tolerance)
                {
                    break;
                }
            }
            //policies::check_series_iterations<T>("boost::math::bessel_jy<%1%>(%1%,%1%) in temme_jy", k, pol);
            *Y = -sum;
            *Y1 = -2 * sum1 / x;

            return 0;
        }

        // Evaluate continued fraction fv = J_(v+1) / J_v, see
        // Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73
        template <typename T>
        int CF1_jy(T v, T x, T* fv, int* sign)
        {
            using namespace std;
            T C, D, f, a, b, delta, tiny, tolerance;
            unsigned long k;
            int s = 1;

            // |x| <= |v|, CF1_jy converges rapidly
            // |x| > |v|, CF1_jy needs O(|x|) iterations to converge

            // modified Lentz's method, see
            // Lentz, Applied Optics, vol 15, 668 (1976)
            tolerance = 2 * std::numeric_limits<T>::epsilon();
            tiny = sqrt(std::numeric_limits<T>::min());
            C = f = tiny;                           // b0 = 0, replace with tiny
            D = 0;
            for (k = 1; k < BESSEL_NITERATIONS * 100; ++k)
            {
                a = -1;
                b = 2 * (v + k) / x;
                C = b + a / C;
                D = b + a * D;
                if (C == 0) { C = tiny; }
                if (D == 0) { D = tiny; }
                D = 1 / D;
                delta = C * D;
                f *= delta;
                if (D < 0) { s = -s; }
                if (abs(delta - 1) < tolerance)
                { break; }
            }
            //policies::check_series_iterations<T>("boost::math::bessel_jy<%1%>(%1%,%1%) in CF1_jy", k / 100, pol);
            *fv = -f;
            *sign = s;                              // sign of denominator

            return 0;
        }

        template <typename T>
        int CF2_jy(T v, T x, T* p, T* q)
        {
            using namespace std;

            T Cr, Ci, Dr, Di, fr, fi, a, br, bi, delta_r, delta_i, temp;
            T tiny;
            unsigned long k;

            // |x| >= |v|, CF2_jy converges rapidly
            // |x| -> 0, CF2_jy fails to converge

            // modified Lentz's method, complex numbers involved, see
            // Lentz, Applied Optics, vol 15, 668 (1976)
            T tolerance = 2 * std::numeric_limits<T>::epsilon();
            tiny = sqrt(std::numeric_limits<T>::min());
            Cr = fr = -0.5f / x;
            Ci = fi = 1;
            //Dr = Di = 0;
            T v2 = v * v;
            a = (0.25f - v2) / x; // Note complex this one time only!
            br = 2 * x;
            bi = 2;
            temp = Cr * Cr + 1;
            Ci = bi + a * Cr / temp;
            Cr = br + a / temp;
            Dr = br;
            Di = bi;
            if (fabs(Cr) + fabs(Ci) < tiny) { Cr = tiny; }
            if (fabs(Dr) + fabs(Di) < tiny) { Dr = tiny; }
            temp = Dr * Dr + Di * Di;
            Dr = Dr / temp;
            Di = -Di / temp;
            delta_r = Cr * Dr - Ci * Di;
            delta_i = Ci * Dr + Cr * Di;
            temp = fr;
            fr = temp * delta_r - fi * delta_i;
            fi = temp * delta_i + fi * delta_r;
            for (k = 2; k < BESSEL_NITERATIONS; ++k)
            {
                a = k - 0.5f;
                a *= a;
                a -= v2;
                bi += 2;
                temp = Cr * Cr + Ci * Ci;
                Cr = br + a * Cr / temp;
                Ci = bi - a * Ci / temp;
                Dr = br + a * Dr;
                Di = bi + a * Di;
                if (fabs(Cr) + fabs(Ci) < tiny) { Cr = tiny; }
                if (fabs(Dr) + fabs(Di) < tiny) { Dr = tiny; }
                temp = Dr * Dr + Di * Di;
                Dr = Dr / temp;
                Di = -Di / temp;
                delta_r = Cr * Dr - Ci * Di;
                delta_i = Ci * Dr + Cr * Di;
                temp = fr;
                fr = temp * delta_r - fi * delta_i;
                fi = temp * delta_i + fi * delta_r;
                if (fabs(delta_r - 1) + fabs(delta_i) < tolerance)
                    break;
            }
            //policies::check_series_iterations<T>("boost::math::bessel_jy<%1%>(%1%,%1%) in CF2_jy", k, pol);
            *p = fr;
            *q = fi;

            return 0;
        }

        // Compute J(v, x) and Y(v, x) simultaneously by Steed's method, see
        // Barnett et al, Computer Physics Communications, vol 8, 377 (1974)
        template <typename T>
        int bessel_jy(T v, T x, T* J, T* Y, int kind)
        {
            using namespace std;
            DLIB_ASSERT(x >= 0);

            T u, Jv, Ju, Yv, Yv1, Yu, Yu1(0), fv, fu;
            T W, p, q, gamma, current, prev, next;
            bool reflect = false;
            unsigned n, k;
            int s;
            int org_kind = kind;
            T cp = 0;
            T sp = 0;

            if (v < 0)
            {
                reflect = true;
                v = -v;                             // v is non-negative from here
            }
            if (v > static_cast<T>((std::numeric_limits<int>::max)()))
            {
                *J = *Y = std::numeric_limits<T>::infinity();
                return 1;
            }
            n = std::lround(v);
            u = v - n;                              // -1/2 <= u < 1/2

            if(reflect)
            {
                T z = (u + n % 2);
                cp = std::cos(z*pi);
                sp = std::sin(z*pi);
                if(u != 0)
                    kind = need_j|need_y;               // need both for reflection formula
            }

            if(x == 0)
            {
                if(v == 0)
                    *J = 1;
                else if((u == 0) || !reflect)
                    *J = 0;
                else if(kind & need_j)
                    *J = std::numeric_limits<T>::infinity(); // complex infinity
                else
                    *J = std::numeric_limits<T>::quiet_NaN();  // any value will do, not using J.

                if((kind & need_y) == 0)
                    *Y = std::numeric_limits<T>::quiet_NaN();  // any value will do, not using Y.
                else if(v == 0)
                    *Y = -std::numeric_limits<T>::infinity();
                else
                    *Y = std::numeric_limits<T>::infinity();
                return 1;
            }

            // x is positive until reflection
            W = T(2) / (x * pi);               // Wronskian
            T Yv_scale = 1;
            if(((kind & need_y) == 0) && ((x < 1) || (v > x * x / 4) || (x < 5)))
            {
                //
                // This series will actually converge rapidly for all small
                // x - say up to x < 20 - but the first few terms are large
                // and divergent which leads to large errors :-(
                //
                Jv = bessel_j_small_z_series(v, x);
                Yv = std::numeric_limits<T>::quiet_NaN();
            }
            else if((x < 1) && (u != 0) && (log(std::numeric_limits<T>::epsilon() / 2) > v * log((x/2) * (x/2) / v)))
            {
                // Evaluate using series representations.
                // This is particularly important for x << v as in this
                // area temme_jy may be slow to converge, if it converges at all.
                // Requires x is not an integer.
                if(kind&need_j)
                    Jv = bessel_j_small_z_series(v, x);
                else
                    Jv = std::numeric_limits<T>::quiet_NaN();
                if((org_kind&need_y && (!reflect || (cp != 0)))
                   || (org_kind & need_j && (reflect && (sp != 0))))
                {
                    // Only calculate if we need it, and if the reflection formula will actually use it:
                    Yv = bessel_y_small_z_series(v, x, &Yv_scale);
                }
                else
                    Yv = std::numeric_limits<T>::quiet_NaN();
            }
            else if((u == 0) && (x < std::numeric_limits<T>::epsilon()))
            {
                // Truncated series evaluation for small x and v an integer,
                // much quicker in this area than temme_jy below.
                if(kind&need_j)
                    Jv = bessel_j_small_z_series(v, x);
                else
                    Jv = std::numeric_limits<T>::quiet_NaN();
                if((org_kind&need_y && (!reflect || (cp != 0)))
                   || (org_kind & need_j && (reflect && (sp != 0))))
                {
                    // Only calculate if we need it, and if the reflection formula will actually use it:
                    Yv = bessel_yn_small_z(n, x, &Yv_scale);
                }
                else
                    Yv = std::numeric_limits<T>::quiet_NaN();
            }
            else if(asymptotic_bessel_large_x_limit(v, x))
            {
                if(kind&need_y)
                {
                    Yv = asymptotic_bessel_y_large_x_2(v, x);
                }
                else
                    Yv = std::numeric_limits<T>::quiet_NaN(); // any value will do, we're not using it.
                if(kind&need_j)
                {
                    Jv = asymptotic_bessel_j_large_x_2(v, x);
                }
                else
                    Jv = std::numeric_limits<T>::quiet_NaN(); // any value will do, we're not using it.
            }
            else if((x > 8) && hankel_PQ(v, x, &p, &q))
            {
                //
                // Hankel approximation: note that this method works best when x 
                // is large, but in that case we end up calculating sines and cosines
                // of large values, with horrendous resulting accuracy.  It is fast though
                // when it works....
                //
                // Normally we calculate sin/cos(chi) where:
                //
                // chi = x - fmod(T(v / 2 + 0.25f), T(2)) * boost::math::pi;
                //
                // But this introduces large errors, so use sin/cos addition formulae to
                // improve accuracy:
                //
                T mod_v = fmod(T(v / 2 + 0.25f), T(2));
                T sx = sin(x);
                T cx = cos(x);
                T sv = sin(pi*mod_v);
                T cv = cos(pi*mod_v);

                T sc = sx * cv - sv * cx; // == sin(chi);
                T cc = cx * cv + sx * sv; // == cos(chi);
                T chi = sqrt_2 / (sqrt(pi) * sqrt(x)); //sqrt(2 / (boost::math::pi * x));
                Yv = chi * (p * sc + q * cc);
                Jv = chi * (p * cc - q * sc);
            }
            else if (x <= 2)                           // x in (0, 2]
            {
                if(temme_jy(u, x, &Yu, &Yu1))             // Temme series
                {
                    // domain error:
                    *J = *Y = Yu;
                    return 1;
                }
                prev = Yu;
                current = Yu1;
                T scale = 1;
                //policies::check_series_iterations<T>(function, n, pol);
                for (k = 1; k <= n; k++)            // forward recurrence for Y
                {
                    T fact = 2 * (u + k) / x;
                    if((std::numeric_limits<T>::max() - fabs(prev)) / fact < fabs(current))
                    {
                        scale /= current;
                        prev /= current;
                        current = 1;
                    }
                    next = fact * current - prev;
                    prev = current;
                    current = next;
                }
                Yv = prev;
                Yv1 = current;
                if(kind&need_j)
                {
                    CF1_jy(v, x, &fv, &s);                 // continued fraction CF1_jy
                    Jv = scale * W / (Yv * fv - Yv1);           // Wronskian relation
                }
                else
                    Jv = std::numeric_limits<T>::quiet_NaN(); // any value will do, we're not using it.
                Yv_scale = scale;
            }
            else                                    // x in (2, \infty)
            {
                // Get Y(u, x):

                T ratio;
                CF1_jy(v, x, &fv, &s);
                // tiny initial value to prevent overflow
                T init = sqrt(std::numeric_limits<T>::min());
                prev = fv * s * init;
                current = s * init;
                if(v < max_factorial<T>::value)
                {
                    //policies::check_series_iterations<T>(function, n, pol);
                    for (k = n; k > 0; k--)             // backward recurrence for J
                    {
                        next = 2 * (u + k) * current / x - prev;
                        prev = current;
                        current = next;
                    }
                    ratio = (s * init) / current;     // scaling ratio
                    // can also call CF1_jy() to get fu, not much difference in precision
                    fu = prev / current;
                }
                else
                {
                    //
                    // When v is large we may get overflow in this calculation
                    // leading to NaN's and other nasty surprises:
                    //
                    //policies::check_series_iterations<T>(function, n, pol);
                    bool over = false;
                    for (k = n; k > 0; k--)             // backward recurrence for J
                    {
                        T t = 2 * (u + k) / x;
                        if((t > 1) && (std::numeric_limits<T>::max() / t < current))
                        {
                            over = true;
                            break;
                        }
                        next = t * current - prev;
                        prev = current;
                        current = next;
                    }
                    if(!over)
                    {
                        ratio = (s * init) / current;     // scaling ratio
                        // can also call CF1_jy() to get fu, not much difference in precision
                        fu = prev / current;
                    }
                    else
                    {
                        ratio = 0;
                        fu = 1;
                    }
                }
                CF2_jy(u, x, &p, &q);                  // continued fraction CF2_jy
                T t = u / x - fu;                   // t = J'/J
                gamma = (p - t) / q;
                //
                // We can't allow gamma to cancel out to zero competely as it messes up
                // the subsequent logic.  So pretend that one bit didn't cancel out
                // and set to a suitably small value.  The only test case we've been able to
                // find for this, is when v = 8.5 and x = 4*PI.
                //
                if(gamma == 0)
                {
                    gamma = u * std::numeric_limits<T>::epsilon() / x;
                }
                Ju = sign(current) * sqrt(W / (q + gamma * (p - t)));

                Jv = Ju * ratio;                    // normalization

                Yu = gamma * Ju;
                Yu1 = Yu * (u/x - p - q/gamma);

                if(kind&need_y)
                {
                    // compute Y:
                    prev = Yu;
                    current = Yu1;
                    //policies::check_series_iterations<T>(function, n, pol);
                    for (k = 1; k <= n; k++)            // forward recurrence for Y
                    {
                        T fact = 2 * (u + k) / x;
                        if((std::numeric_limits<T>::max() - fabs(prev)) / fact < fabs(current))
                        {
                            prev /= current;
                            Yv_scale /= current;
                            current = 1;
                        }
                        next = fact * current - prev;
                        prev = current;
                        current = next;
                    }
                    Yv = prev;
                }
                else
                    Yv = std::numeric_limits<T>::quiet_NaN(); // any value will do, we're not using it.
            }

            if (reflect)
            {
                if((sp != 0) && (std::numeric_limits<T>::max() * fabs(Yv_scale) < fabs(sp * Yv)))
                    *J = org_kind & need_j ? T(-sign(sp) * sign(Yv) * sign(Yv_scale) * std::numeric_limits<T>::infinity()) : T(0);
                else
                    *J = cp * Jv - (sp == 0 ? T(0) : T((sp * Yv) / Yv_scale));     // reflection formula
                if((cp != 0) && (std::numeric_limits<T>::max() * fabs(Yv_scale) < fabs(cp * Yv)))
                    *Y = org_kind & need_y ? T(-sign(cp) * sign(Yv) * sign(Yv_scale) * std::numeric_limits<T>::infinity()) : T(0);
                else
                    *Y = (sp != 0 ? sp * Jv : T(0)) + (cp == 0 ? T(0) : T((cp * Yv) / Yv_scale));
            }
            else
            {
                *J = Jv;
                if(std::numeric_limits<T>::max() * fabs(Yv_scale) < fabs(Yv))
                    *Y = org_kind & need_y ? T(sign(Yv) * sign(Yv_scale) * std::numeric_limits<T>::infinity()) : T(0);
                else
                    *Y = Yv / Yv_scale;
            }

            return 0;
        }

        template<typename R>
        R cyl_bessel_i(R nu, R x)
        {
            if (nu < R{0} || x < R{0})
                throw std::runtime_error("Bad argument in cyl_bessel_i.");
            else if (std::isnan(nu) || std::isnan(x))
                return std::numeric_limits<R>::quiet_NaN();

            if(x == 0)
            {
                return (nu == 0) ? R{1} : R{0};
            }
            else if(nu == R(0.5))
            {
                // common special case
                return std::sqrt(2 / (x * pi)) * std::sinh(x);
            }
            else if((nu > 0) && (x / nu < 0.25))
            {
                return bessel_i_small_z_series(nu, x);
            }
            else
            {
                R I, K;
                bessel_ik(nu, x, &I, &K, need_i);
                return I;
            }
        }

        template<typename R>
        R cyl_bessel_j(R nu, R x)
        {
            if (nu < R{0} || x < R{0})
                throw std::runtime_error("Bad argument in cyl_bessel_j.");
            else if (std::isnan(nu) || std::isnan(x))
                return std::numeric_limits<R>::quiet_NaN();

            /*! Special case !*/
            else if (x == R{0})
                return nu == R{0} ? R{1} : R{0};

            R j, y;
            bessel_jy(nu, x, &j, &y, need_j);
            return j;
        }
#endif
    }
}

#endif //DLIB_MATH_DETAIL_BESSEL
