// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_QUANTUM_COMPUTINg_ABSTRACT_
#ifdef DLIB_QUANTUM_COMPUTINg_ABSTRACT_

#include <complex>
#include "../matrix.h"
#include "../rand.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    typedef std::complex<double> qc_scalar_type;

// ----------------------------------------------------------------------------------------

    class quantum_register
    {
        /*!
            INITIAL VALUE
                - num_bits() == 1

            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        quantum_register(
        );

        int num_bits (
        ) const;

        void set_num_bits (
            int num_bits
        );

        void zero_all_bits(
        );

        void append ( 
            const quantum_register& reg
        );

        template <typename rand_type>
        bool measure_bit (
            int bit,
            rand_type& rnd
        );

        template <typename rand_type>
        bool measure_and_remove_bit (
            int bit,
            rand_type& rnd
        );

        double probability_of_bit (
            int bit
        ) const;
        /*!
            requires
                - 0 <= bit < num_bits()
            ensures
                - returns the probability of measuring the given bit and it being true
        !*/

        const matrix<qc_scalar_type,0,1>& state_vector(
        ) const;

        matrix<qc_scalar_type,0,1>& state_vector(
        );

        void swap (
            quantum_register& item
        );

    };

    inline void swap (
        quantum_register& a,
        quantum_register& b
    ) { a.swap(b); }

// ----------------------------------------------------------------------------------------

    template <typename T>
    class gate_exp
    {
        /*!
            REQUIREMENTS ON T
                T must be some object that implements an interface compatible with
                a gate_exp or gate object.

            WHAT THIS OBJECT REPRESENTS
                This object represents an expression that evaluates to a quantum gate 
                that operates on T::num_bits qubits.
        !*/

    public:

        static const long num_bits = T::num_bits;
        static const long dims = T::dims; 

        gate_exp(
            T& exp
        );
        /*!
            ensures
                - #&ref() == &exp
        !*/

        const qc_scalar_type operator() (
            long r, 
            long c
        ) const;
        /*!
            ensures
                - returns ref()(r,c)
        !*/

        void apply_gate_to (
            quantum_register& reg
        ) const;
        /*!
            requires
                - reg.num_bits() == num_bits
            ensures
                - applies this quantum gate to the given quantum register
                - Let M represent the matrix for this quantum gate
                - reg().state_vector() = M*reg().state_vector()
        !*/

        template <typename exp>
        qc_scalar_type compute_state_element (
            const matrix_exp<exp>& reg,
            long row_idx
        ) const;
        /*!
            requires
                - reg.nr() == dims
                - reg.nc() == 1
                - 0 <= row_idx < dims
            ensures
                - Let M represent the matrix for this gate.   
                - returns rowm(M*reg, row_idx)
                  (i.e. returns the row_idx row of what you get when you apply this
                  gate to the given column vector in reg)
                - This function works by calling ref().compute_state_element(reg,row_idx)
        !*/

        const T& ref(
        );
        /*!
            ensures
                - returns a reference to the subexpression contained in this object
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    class composite_gate : public gate_exp<composite_gate<T,U> >
    {
    public:

        composite_gate (
            const composite_gate& g
        );
        /*!
            ensures
                - *this is a copy of g
        !*/

        composite_gate(
            const gate_exp<T>& lhs_, 
            const gate_exp<U>& rhs_
        ): 
        /*!
            ensures
                - #lhs == lhs_.ref()
                - #rhs == rhs_.ref()
                - #num_bits == T::num_bits + U::num_bits
                - #dims == 2^num_bits
                - #&ref() == this
        !*/

        const qc_scalar_type operator() (
            long r, 
            long c
        ) const; 
        /*!
            requires
                - 0 <= r < dims
                - 0 <= c < dims
            ensures
                - Let M denote the tensor product of lhs with rhs
                - returns M(r,c)
                  (i.e. returns lhs(r/U::dims,c/U::dims)*rhs(r%U::dims, c%U::dims))
        !*/

        template <typename exp>
        qc_scalar_type compute_state_element (
            const matrix_exp<exp>& reg,
            long row_idx
        ) const;
        /*!
            requires
                - reg.nr() == dims
                - reg.nc() == 1
                - 0 <= row_idx < dims
            ensures
                - Let M represent the matrix for this gate.   
                - returns rowm(M*reg, row_idx)
                  (i.e. returns the row_idx row of what you get when you apply this
                  gate to the given column vector in reg)
                - This function works by calling rhs.compute_state_element() and using elements
                  of the matrix in lhs.  
        !*/

        static const long num_bits;
        static const long dims;

        const T lhs;
        const U rhs;
    };

// ----------------------------------------------------------------------------------------

    template <long bits>
    class gate : public gate_exp<gate<bits> >
    {
        /*!
            REQUIREMENTS ON bits
                0 < bits <= 30

            WHAT THIS OBJECT REPRESENTS

        !*/

    public:
        gate(
        );
        /*!
            ensures
                - num_bits == bits
                - dims == 2^bits
                - #&ref() == this
                - for all valid r and c:
                    #(*this)(r,c) == 0
        !*/

        gate (
            const gate& g
        );
        /*!
            ensures
                - *this is a copy of g
        !*/

        template <typename T>
        explicit gate(
            const gate_exp<T>& g
        );
        /*!
            requires
                - T::num_bits == num_bits
            ensures
                - num_bits == bits
                - dims == 2^bits
                - #&ref() == this
                - for all valid r and c:
                    #(*this)(r,c) == g(r,c)
        !*/

        const qc_scalar_type& operator() (
            long r, 
            long c
        ) const { return data(r,c); }

        qc_scalar_type& operator() (
            long r, 
            long c
        ) { return data(r,c); }

        template <typename exp>
        qc_scalar_type compute_state_element (
            const matrix_exp<exp>& reg,
            long row_idx
        ) const;
        /*!
            requires
                - reg.nr() == dims
                - reg.nc() == 1
                - 0 <= row_idx < dims
            ensures
                - Let M represent the matrix for this gate.   
                - returns rowm(M*reg, row_idx)
                  (i.e. returns the row_idx row of what you get when you apply this
                  gate to the given column vector in reg)
        !*/

        static const long num_bits;
        static const long dims;

    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    const composite_gate<T,U> operator, ( 
        const gate_exp<T>& lhs,
        const gate_exp<U>& rhs
    ) { return composite_gate<T,U>(lhs,rhs); }
    /*!
    !*/

// ----------------------------------------------------------------------------------------

    namespace quantum_gates
    {

        inline const gate<1> hadamard(
        )
        {
            gate<1> h;
            h(0,0) = std::sqrt(1/2.0);
            h(0,1) = std::sqrt(1/2.0);
            h(1,0) = std::sqrt(1/2.0);
            h(1,1) = -std::sqrt(1/2.0);
            return h;
        }

        inline const gate<1> x(
        )
        {
            gate<1> x;
            x(0,1) = 1;
            x(1,0) = 1;
            return x;
        }

        inline const gate<1> y(
        )
        {
            gate<1> x;
            qc_scalar_type i(0,1);
            x(0,1) = -i;
            x(1,0) = i;
            return x;
        }

        inline const gate<1> z(
        )
        {
            gate<1> z;
            z(0,0) = 1;
            z(1,1) = -1;
            return z;
        }


        inline const gate<1> noop(
        )
        {
            gate<1> i;
            i(0,0) = 1;
            i(1,1) = 1;
            return i;
        }


        template <int control_bit, int target_bit>
        class cnot : public gate_exp<cnot<control_bit, target_bit> >
        {
        public:
            COMPILE_TIME_ASSERT(control_bit != target_bit);
        };


        template <int control_bit1, int control_bit2, int target_bit>
        class taffoli : public gate_exp<taffoli<control_bit1, control_bit2, target_bit> >
        {
        public:
            COMPILE_TIME_ASSERT(control_bit1 != target_bit && control_bit2 != target_bit && control_bit1 != control_bit2);
            COMPILE_TIME_ASSERT((control_bit1 < target_bit && control_bit2 < target_bit) ||(control_bit1 > target_bit && control_bit2 > target_bit) );

        };

    // ------------------------------------------------------------------------------------

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_QUANTUM_COMPUTINg_ABSTRACT_



