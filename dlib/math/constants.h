// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_CONSTANTS
#define DLIB_MATH_CONSTANTS

namespace dlib
{
    template<typename R>
    struct numeric_constants
    {
        static R pi() noexcept
        /*!
            ensures
                - returns pi
        !*/
        { return static_cast<R>(3.1415926535897932384626433832795029L); }

        static R pi_2() noexcept
        /*!
            ensures
                - returns pi / 2
        !*/
        { return static_cast<R>(1.5707963267948966192313216916397514L); }

        static R pi_3() noexcept
        /*!
            ensures
                - returns pi / 3
        !*/
        { return static_cast<R>(1.0471975511965977461542144610931676L); }

        static R pi_4() noexcept
        /*!
            ensures
                - returns pi / 4
        !*/
        { return static_cast<R>(0.7853981633974483096156608458198757L); }

        static R _1_pi() noexcept
        /*!
            ensures
                - returns 1 / pi
        !*/
        { return static_cast<R>(0.3183098861837906715377675267450287L); }

        static R _2_sqrtpi() noexcept
        /*!
            ensures
                - returns 2 / sqrt(pi)
        !*/
        { return static_cast<R>(1.1283791670955125738961589031215452L); }

        static R sqrt2() noexcept
        /*!
            ensures
                - returns sqrt(2)
        !*/
        { return static_cast<R>(1.4142135623730950488016887242096981L); }

        static R sqrt3() noexcept
        /*!
            ensures
                - returns sqrt(3)
        !*/
        { return static_cast<R>(1.7320508075688772935274463415058723L); }

        static R sqrtpio2() noexcept
        /*!
            ensures
                - returns sqrt(pi / 2)
        !*/
        { return static_cast<R>(1.2533141373155002512078826424055226L); }

        static R sqrt1_2() noexcept
        /*!
            ensures
                - returns 1 / sqrt(2)
        !*/
        { return static_cast<R>(0.7071067811865475244008443621048490L); }

        static R lnpi() noexcept
        /*!
            ensures
                - returns log(pi)
        !*/
        { return static_cast<R>(1.1447298858494001741434273513530587L); }

        static R gamma_e() noexcept
        /*!
            ensures
                - returns Euler constant
        !*/
        { return static_cast<R>(0.5772156649015328606065120900824024L); }

        static R euler() noexcept
        /*!
            ensures
                - returns Euler-Mascheroni constant
        !*/
        { return static_cast<R>(2.7182818284590452353602874713526625L); }
    };
}

#endif //DLIB_MATH_CONSTANTS
