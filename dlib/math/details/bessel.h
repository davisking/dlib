// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_DETAIL_BESSEL
#define DLIB_MATH_DETAIL_BESSEL

#include <cmath>
#include <limits>
#include <stdexcept>

namespace dlib
{
    namespace detail
    {
        template <typename R>
        void bessel_ik(R nu, R x, R& Inu, R& Knu, R& Ipnu, R& Kpnu)
        {
            if (x == R{0})
            {
                if (nu == R{0})
                {
                    Inu = R{1};
                    Ipnu = R{0};
                }
                else if (nu == R{1})
                {
                    Inu = R{0};
                    Ipnu = R(0.5L);
                }
                else
                {
                    Inu = R{0};
                    Ipnu = R{0};
                }
                Knu = std::numeric_limits<R>::infinity();
                Kpnu = -std::numeric_limits<R>::infinity();
                return;
            }

            const R eps         = std::numeric_limits<R>::epsilon();
            const R fp_min      = R(10) * std::numeric_limits<R>::epsilon();
            const int max_iter  = 15000;
            const R x_min       = R{2};

            const int nl = static_cast<int>(nu + R(0.5L));

            const R mu  = nu - nl;
            const R mu2 = mu * mu;
            const R xi  = R{1} / x;
            const R xi2 = R{2} * xi;
            R h         = nu * xi;

            if ( h < fp_min )
                h = fp_min;

            R b = xi2 * nu;
            R d = R{0};
            R c = h;
            int i;
            for ( i = 1; i <= max_iter; ++i )
            {
                b += xi2;
                d = R{1} / (b + d);
                c = b + R{1} / c;
                const R del = c * d;
                h *= del;
                if (std::abs(del - R{1}) < eps)
                    break;
            }
            if (i > max_iter)
                throw std::runtime_error("Argument x too large "
                                               "in bessel_ik; "
                                               "try asymptotic expansion.");
            R Inul  = fp_min;
            R Ipnul = h * Inul;
            R Inul1 = Inul;
            R Ipnu1 = Ipnul;
            R fact  = nu * xi;

            for (int l = nl; l >= 1; --l)
            {
                const R Inutemp = fact * Inul + Ipnul;
                fact -= xi;
                Ipnul = fact * Inutemp + Inul;
                Inul = Inutemp;
            }
            R f = Ipnul / Inul;
            R Kmu, Knu1;
            if (x < x_min)
            {
                const R x2 = x / R{2};
                const R pimu = 3.1415926535897932384626433832795029L * mu;
                const R fact = (std::abs(pimu) < eps ? R{1} : pimu / std::sin(pimu));
                R d = -std::log(x2);
                R e = mu * d;
                const R fact2 = (std::abs(e) < eps ? R{1} : std::sinh(e) / e);
                R gam1, gam2, gampl, gammi;
                gamma_temme(mu, gam1, gam2, gampl, gammi);
                R ff = fact
                           * (gam1 * std::cosh(e) + gam2 * fact2 * d);
                R sum = ff;
                e = std::exp(e);
                R p = e / (R{2} * gampl);
                R q = R{1} / (R{2} * e * gammi);
                R c = R{1};
                d = x2 * x2;
                R sum1 = p;
                int i;
                for (i = 1; i <= max_iter; ++i)
                {
                    ff = (i * ff + p + q) / (i * i - mu2);
                    c *= d / i;
                    p /= i - mu;
                    q /= i + mu;
                    const R del = c * ff;
                    sum += del;
                    const R del1 = c * (p - i * ff);
                    sum1 += del1;
                    if (std::abs(del) < eps * std::abs(sum))
                        break;
                }
                if (i > max_iter)
                    throw std::runtime_error("Bessel k series failed to converge "
                                                   "in bessel_ik.");
                Kmu = sum;
                Knu1 = sum1 * xi2;
            }
            else
            {
                R b = R{2} * (R{1} + x);
                R d = R{1} / b;
                R delh = d;
                R h = delh;
                R q1 = R{0};
                R q2 = R{1};
                R a1 = R(0.25L) - mu2;
                R q = c = a1;
                R a = -a1;
                R s = R{1} + q * delh;
                int i;
                for (i = 2; i <= max_iter; ++i)
                {
                    a -= 2 * (i - 1);
                    c = -a * c / i;
                    const R qnew = (q1 - b * q2) / a;
                    q1 = q2;
                    q2 = qnew;
                    q += c * qnew;
                    b += R{2};
                    d = R{1} / (b + a * d);
                    delh = (b * d - R{1}) * delh;
                    h += delh;
                    const R dels = q * delh;
                    s += dels;
                    if ( std::abs(dels / s) < eps )
                        break;
                }
                if (i > max_iter)
                    throw std::runtime_error("Steed's method failed "
                                                   "in bessel_ik.");
                h = a1 * h;
                Kmu = std::sqrt(3.1415926535897932384626433832795029L / (R{2} * x))
                        * std::exp(-x) / s;
                Knu1 = Kmu * (mu + x + R(0.5L) - h) * xi;
            }

            R Kpmu = mu * xi * Kmu - Knu1;
            R Inumu = xi / (f * Kmu - Kpmu);
            Inu = Inumu * Inul1 / Inul;
            Ipnu = Inumu * Ipnu1 / Inul;
            for ( i = 1; i <= nl; ++i )
            {
                const R Knutemp = (mu + i) * xi2 * Knu1 + Kmu;
                Kmu = Knu1;
                Knu1 = Knutemp;
            }
            Knu = Kmu;
            Kpnu = nu * xi * Kmu - Knu1;
        }
        
        template<typename R>
        R cyl_bessel_ij_series(R nu, R x, R sgn, unsigned int max_iter)
        {
            if (x == R{0})
                return nu == R{0} ? R{1} : R{0};

            const R x2      = x / R{2};
            const R xx4     = sgn * x2 * x2;
            const R fact    = std::exp(nu * std::log(x2) - std::lgamma(nu + R{1}));

            R Jn  = 1;
            R term = 1;

            for (unsigned int i = 1; i < max_iter; ++i)
            {
                term *= xx4 / (R{i} * (nu + R{i}));
                Jn   += term;
                if (std::abs(term / Jn) < std::numeric_limits<R>::epsilon())
                    break;
            }

            return fact * Jn;
        }

        template <typename R>
        void bessel_jn(R nu, R x, R& Jnu, R& Nnu, R& Jpnu, R& Npnu)
        {
            if (x == R{0})
            {
                if (nu == R{0})
                {
                    Jnu = R{1};
                    Jpnu = R{0};
                }
                else if (nu == R{1})
                {
                    Jnu = R{0};
                    Jpnu = R(0.5L);
                }
                else
                {
                    Jnu = R{0};
                    Jpnu = R{0};
                }
                Nnu = -std::numeric_limits<R>::infinity();
                Npnu = std::numeric_limits<R>::infinity();
                return;
            }

            const R eps = std::numeric_limits<R>::epsilon();
            //  When the multiplier is N i.e.
            //  fp_min = N * min()
            //  Then J_0 and N_0 tank at x = 8 * N (J_0 = 0 and N_0 = nan)!
            //const R fp_min = R(20) * std::numeric_limits<R>::min();
            const R fp_min = std::sqrt(std::numeric_limits<R>::min());
            const int max_iter = 15000;
            const R x_min = R{2};

            const int nl = (x < x_min
                              ? static_cast<int>(nu + R(0.5L))
                              : std::max(0, static_cast<int>(nu - x + R(1.5L))));

            const R mu = nu - nl;
            const R mu2 = mu * mu;
            const R xi = R{1} / x;
            const R xi2 = R{2} * xi;
            R w = xi2 / 3.1415926535897932384626433832795029L;
            int isign = 1;
            R h = nu * xi;
            if (h < fp_min)
                h = fp_min;
            R b = xi2 * nu;
            R d = R{0};
            R c = h;
            int i;
            for (i = 1; i <= max_iter; ++i)
            {
                b += xi2;
                d = b - d;
                if (std::abs(d) < fp_min)
                    d = fp_min;
                c = b - R{1} / c;
                if (std::abs(c) < fp_min)
                    c = fp_min;
                d = R{1} / d;
                const R del = c * d;
                h *= del;
                if (d < R{0})
                    isign = -isign;
                if (std::abs(del - R{1}) < eps)
                    break;
            }
            if (i > max_iter)
                throw std::runtime_error("Argument x too large in bessel_jn; "
                                               "try asymptotic expansion.");
            R Jnul = isign * fp_min;
            R Jpnul = h * Jnul;
            R Jnul1 = Jnul;
            R Jpnu1 = Jpnul;
            R fact = nu * xi;
            for ( int l = nl; l >= 1; --l )
            {
                const R Jnutemp = fact * Jnul + Jpnul;
                fact -= xi;
                Jpnul = fact * Jnutemp - Jnul;
                Jnul = Jnutemp;
            }
            if (Jnul == R{0})
                Jnul = eps;
            R f= Jpnul / Jnul;
            R Nmu, Nnu1, Npmu, Jmu;
            if (x < x_min)
            {
                const R x2 = x / R{2};
                const R pimu = 3.1415926535897932384626433832795029L * mu;
                R fact = (std::abs(pimu) < eps
                              ? R{1} : pimu / std::sin(pimu));
                R d = -std::log(x2);
                R e = mu * d;
                R fact2 = (std::abs(e) < eps
                               ? R{1} : std::sinh(e) / e);
                R gam1, gam2, gampl, gammi;
                gamma_temme(mu, gam1, gam2, gampl, gammi);
                R ff = (R{2} / 3.1415926535897932384626433832795029L)
                           * fact * (gam1 * std::cosh(e) + gam2 * fact2 * d);
                e = std::exp(e);
                R p = e / (3.1415926535897932384626433832795029L * gampl);
                R q = R{1} / (e * 3.1415926535897932384626433832795029L * gammi);
                const R pimu2 = pimu / R{2};
                R fact3 = (std::abs(pimu2) < eps
                               ? R{1} : std::sin(pimu2) / pimu2 );
                R r = 3.1415926535897932384626433832795029L * pimu2 * fact3 * fact3;
                R c = R{1};
                d = -x2 * x2;
                R sum = ff + r * q;
                R sum1 = p;
                for (i = 1; i <= max_iter; ++i)
                {
                    ff = (i * ff + p + q) / (i * i - mu2);
                    c *= d / R{i};
                    p /= R{i} - mu;
                    q /= R{i} + mu;
                    const R del = c * (ff + r * q);
                    sum += del;
                    const R del1 = c * p - i * del;
                    sum1 += del1;
                    if ( std::abs(del) < eps * (R{1} + std::abs(sum)) )
                        break;
                }
                if ( i > max_iter )
                    throw std::runtime_error("Bessel y series failed to converge "
                                                   "in bessel_jn.");
                Nmu = -sum;
                Nnu1 = -sum1 * xi2;
                Npmu = mu * xi * Nmu - Nnu1;
                Jmu = w / (Npmu - f * Nmu);
            }
            else
            {
                R a = R(0.25L) - mu2;
                R q = R{1};
                R p = -xi / R{2};
                R br = R{2} * x;
                R bi = R{2};
                R fact = a * xi / (p * p + q * q);
                R cr = br + q * fact;
                R ci = bi + p * fact;
                R den = br * br + bi * bi;
                R dr = br / den;
                R di = -bi / den;
                R dlr = cr * dr - ci * di;
                R dli = cr * di + ci * dr;
                R temp = p * dlr - q * dli;
                q = p * dli + q * dlr;
                p = temp;
                int i;
                for (i = 2; i <= max_iter; ++i)
                {
                    a += R(2 * (i - 1));
                    bi += R{2};
                    dr = a * dr + br;
                    di = a * di + bi;
                    if (std::abs(dr) + std::abs(di) < fp_min)
                        dr = fp_min;
                    fact = a / (cr * cr + ci * ci);
                    cr = br + cr * fact;
                    ci = bi - ci * fact;
                    if (std::abs(cr) + std::abs(ci) < fp_min)
                        cr = fp_min;
                    den = dr * dr + di * di;
                    dr /= den;
                    di /= -den;
                    dlr = cr * dr - ci * di;
                    dli = cr * di + ci * dr;
                    temp = p * dlr - q * dli;
                    q = p * dli + q * dlr;
                    p = temp;
                    if (std::abs(dlr - R{1}) + std::abs(dli) < eps)
                        break;
                }
                if (i > max_iter)
                    throw std::runtime_error("Lentz's method failed "
                                                   "in bessel_jn.");
                const R gam = (p - f) / q;
                Jmu = std::sqrt(w / ((p - f) * gam + q));
                Jmu = std::copysign(Jmu, Jnul);
                Nmu = gam * Jmu;
                Npmu = (p + q / gam) * Nmu;
                Nnu1 = mu * xi * Nmu - Npmu;
            }
            fact = Jmu / Jnul;
            Jnu = fact * Jnul1;
            Jpnu = fact * Jpnu1;
            for (i = 1; i <= nl; ++i)
            {
                const R Nnutemp = (mu + i) * xi2 * Nnu1 - Nmu;
                Nmu = Nnu1;
                Nnu1 = Nnutemp;
            }
            Nnu = Nmu;
            Npnu = nu * xi * Nmu - Nnu1;

            return;
        }
        
        template <typename R>
        void cyl_bessel_jn_asymp (R nu, R x, R & Jnu, R & Nnu)
        {
            const R mu      = R(4) * nu * nu;
            const R mum1    = mu - R{1};
            const R mum9    = mu - R(9);
            const R mum25   = mu - R(25);
            const R mum49   = mu - R(49);
            const R xx      = R(64) * x * x;
            const R P       = R{1} - mum1 * mum9 / (R{2} * xx) * (R{1} - mum25 * mum49 / (R(12) * xx));
            const R Q       = mum1 / (R(8) * x) * (R{1} - mum9 * mum25 / (R(6) * xx));

            const R chi = x - (nu + R(0.5L)) * 1.5707963267948966192313216916397514L;
            const R c = std::cos(chi);
            const R s = std::sin(chi);

            const R coef = std::sqrt(R{2} / (3.1415926535897932384626433832795029L * x));
            Jnu = coef * (c * P - s * Q);
            Nnu = coef * (s * P + c * Q);

            return;
        }

        template<typename R>
        R cyl_bessel_i(R nu, R x)
        {
            if (nu < R{0} || x < R{0})
                throw std::runtime_error("Bad argument in cyl_bessel_i.");
            else if (std::isnan(nu) || std::isnan(x))
                return std::numeric_limits<R>::quiet_NaN();
            else if (x * x < R{10} * (nu + R{1}))
                return cyl_bessel_ij_series(nu, x, + R{1}, 200);
            else
            {
                R I_nu, K_nu, Ip_nu, Kp_nu;
                bessel_ik(nu, x, I_nu, K_nu, Ip_nu, Kp_nu);
                return I_nu;
            }
        }

        template<typename R>
        R cyl_bessel_j(R nu, R x)
        {
            if (nu < R{0} || x < R{0})
                throw std::runtime_error("Bad argument in cyl_bessel_j.");
            else if (std::isnan(nu) || std::isnan(x))
                return std::numeric_limits<R>::quiet_NaN();
            else if (x * x < R(10) * (nu + R{1}))
                return cyl_bessel_ij_series(nu, x, -R{1}, 200);
            else if (x > R{1000})
            {
                R J_nu, N_nu;
                cyl_bessel_jn_asymp(nu, x, J_nu, N_nu);
                return J_nu;
            }
            else
            {
                R J_nu, N_nu, Jp_nu, Np_nu;
                bessel_jn(nu, x, J_nu, N_nu, Jp_nu, Np_nu);
                return J_nu;
            }
        }
    }
}

#endif //DLIB_MATH_DETAIL_BESSEL
