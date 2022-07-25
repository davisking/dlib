// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MATRIX_WINDOWS_ABSTRACT_H
#ifdef DLIB_MATRIX_WINDOWS_ABSTRACT_H

// ----------------------------------------------------------------------------------------

    const matrix_exp hann (
        const matrix_exp& m,
        window_symmetry type
    );
    /*!
        requires
            - is_vector(m) == true
              (i.e. m is a row or column matrix)
        ensures
            - returns another vector M with a Hann window pointwise multiplied with m using symmetry type
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp blackman (
        const matrix_exp& m,
        window_symmetry type
    );
    /*!
        requires
            - is_vector(m) == true
              (i.e. m is a row or column matrix)
        ensures
            - returns another vector M with a Blackman window pointwise multiplied with m using symmetry type
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp blackman_nuttall (
        const matrix_exp& m,
        window_symmetry type
    );
    /*!
        requires
            - is_vector(m) == true
              (i.e. m is a row or column matrix)
        ensures
            - returns another vector M with a Blackman-Nuttall window pointwise multiplied with m using symmetry type
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp blackman_harris (
        const matrix_exp& m,
        window_symmetry type
    );
    /*!
        requires
            - is_vector(m) == true
              (i.e. m is a row or column matrix)
        ensures
            - returns another vector M with a Blackman-Harris window pointwise multiplied with m using symmetry type
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp blackman_harris7 (
        const matrix_exp& m,
        window_symmetry type
    );
    /*!
        requires
            - is_vector(m) == true
              (i.e. m is a row or column matrix)
        ensures
            - returns another vector M with a 7-order Blackman-Harris window pointwise multiplied with m using symmetry type
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp kaiser (
        const matrix_exp& m,
        beta_t beta,
        window_symmetry type
    );
    /*!
        requires
            - is_vector(m) == true
              (i.e. m is a row or column matrix)
        ensures
            - returns another vector M with a kaiser window pointwise multiplied with m using symmetry type and beta value.
    !*/

// ----------------------------------------------------------------------------------------

#endif //DLIB_MATRIX_WINDOWS_ABSTRACT_H
