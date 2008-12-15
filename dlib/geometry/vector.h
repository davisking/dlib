// Copyright (C) 2003  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_VECTOr_H_
#define DLIB_VECTOr_H_

#include <cmath>
#include "vector_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include <functional>
#include <iostream>
#include "../matrix/matrix.h"

namespace dlib
{

    template <
        typename T,
        long NR = 3
        >
    class vector;

// ----------------------------------------------------------------------------------------

    template <typename T, typename U, typename enabled = void>
    struct vector_assign_helper
    {
        template <long NR>
        static void assign (
            vector<T,2>& dest,
            const vector<U,NR>& src
        )
        {
            dest.x() = static_cast<T>(src.x());
            dest.y() = static_cast<T>(src.y());
        }

        template <long NR>
        static void assign (
            vector<T,3>& dest,
            const vector<U,NR>& src
        )
        {
            dest.x() = static_cast<T>(src.x());
            dest.y() = static_cast<T>(src.y());
            dest.z() = static_cast<T>(src.z());
        }

        template <typename EXP>
        static void assign (
            vector<T,2>& dest,
            const matrix_exp<EXP>& m
        )
        {
            dest.x() = static_cast<T>(m(0));
            dest.y() = static_cast<T>(m(1));
        }

        template <typename EXP>
        static void assign (
            vector<T,3>& dest,
            const matrix_exp<EXP>& m
        )
        {
            dest.x() = static_cast<T>(m(0));
            dest.y() = static_cast<T>(m(1));
            dest.z() = static_cast<T>(m(2));
        }
    };

    // This is an overload for the case where you are converting from a floating point
    // type to an integral type.  These overloads make sure values are rounded to 
    // the nearest integral value.
    template <typename T, typename U>
    struct vector_assign_helper<T,U, typename enable_if_c<std::numeric_limits<T>::is_integer == true && 
                                                          std::numeric_limits<U>::is_integer == false>::type>
    {
        template <long NR>
        static void assign (
            vector<T,2>& dest,
            const vector<U,NR>& src
        )
        {
            dest.x() = static_cast<T>(std::floor(src.x() + 0.5));
            dest.y() = static_cast<T>(std::floor(src.y() + 0.5));
        }

        template <long NR>
        static void assign (
            vector<T,3>& dest,
            const vector<U,NR>& src
        )
        {
            dest.x() = static_cast<T>(std::floor(src.x() + 0.5));
            dest.y() = static_cast<T>(std::floor(src.y() + 0.5));
            dest.z() = static_cast<T>(std::floor(src.z() + 0.5));
        }

        template <typename EXP>
        static void assign (
            vector<T,3>& dest,
            const matrix_exp<EXP>& m
        )
        {
            dest.x() = static_cast<T>(std::floor(m(0) + 0.5));
            dest.y() = static_cast<T>(std::floor(m(1) + 0.5));
            dest.z() = static_cast<T>(std::floor(m(2) + 0.5));
        }

        template <typename EXP>
        static void assign (
            vector<T,2>& dest,
            const matrix_exp<EXP>& m
        )
        {
            dest.x() = static_cast<T>(std::floor(m(0) + 0.5));
            dest.y() = static_cast<T>(std::floor(m(1) + 0.5));
        }

    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    class vector<T,3> 
    {
        /*!
            INITIAL VALUE
                - x_value == 0
                - y_value == 0  
                - z_value == 0 

            CONVENTION
                - x_value == x() 
                - y_value == y() 
                - z_value == z()

        !*/

    public:

        typedef T type;
        
        vector (
        ) 
        {
            x() = 0;
            y() = 0;
            z() = 0;
        }

        // ---------------------------------------

        vector (
            const T _x,
            const T _y,
            const T _z
        ) 
        {
            x() = _x;
            y() = _y;
            z() = _z;
        }

        // ---------------------------------------

        vector (
            const vector& item
        )
        {
            x() = item.x();
            y() = item.y();
            z() = item.z();
        }

        // ---------------------------------------

        template <typename U>
        vector (
            const vector<U,2>& item
        )
        {
            // Do this so that we get the appropriate rounding depending on the relative
            // type of T and U.
            vector<T,2> temp(item);
            x() = temp.x();
            y() = temp.y();
            z() = 0;
        }

        // ---------------------------------------

        vector (
            const vector<T,2>& item
        )
        {
            x() = item.x();
            y() = item.y();
            z() = 0;
        }

        // ---------------------------------------

        template <typename U>
        vector (
            const vector<U,3>& item
        )
        {
            (*this) = item;
        }

        // ---------------------------------------

        template <typename EXP>
        vector ( const matrix_exp<EXP>& m)
        {
            (*this) = m;
        }

        // ---------------------------------------

        template <typename EXP>
        vector& operator = (
            const matrix_exp<EXP>& m
        )
        {
            // you can only assign vectors with 3 elements to a dlib::vector<T,3> object
            COMPILE_TIME_ASSERT(EXP::NR*EXP::NC == 3 || EXP::NR*EXP::NC == 0);

            // make sure requires clause is not broken
            DLIB_ASSERT((m.nr() == 1 || m.nc() == 1) && (m.size() == 3),
                "\t vector(const matrix_exp& m)"
                << "\n\t the given matrix is of the wrong size"
                << "\n\t m.nr():   " << m.nr() 
                << "\n\t m.nc():   " << m.nc() 
                << "\n\t m.size(): " << m.size() 
                << "\n\t this: " << this
                );

            vector_assign_helper<T, typename EXP::type>::assign(*this, m);
            return *this;
        }

        // ---------------------------------------

        template <typename U>
        vector& operator = (
            const vector<U,3>& item
        )
        {
            vector_assign_helper<T,U>::assign(*this, item);
            return *this;
        }

        // ---------------------------------------

        vector& operator= (
            const vector& item
        )
        {
            x() = item.x();
            y() = item.y();
            z() = item.z();
            return *this;
        }

        // ---------------------------------------

        T length(
        ) const 
        { 
            return (T)std::sqrt((double)(x()*x() + y()*y() + z()*z())); 
        }

        // ---------------------------------------

        vector normalize (
        ) const 
        {
            T tmp = (T)std::sqrt((double)(x()*x() + y()*y() + z()*z()));
            return vector ( x()/tmp,
                            y()/tmp,
                            z()/tmp
                            );
        }

        // ---------------------------------------

        T& x (
        ) 
        { 
            return x_value;
        }

        // ---------------------------------------

        T& y (
        ) 
        { 
            return y_value;
        }

        // ---------------------------------------

        T& z (
        ) 
        { 
            return z_value;
        }

        // ---------------------------------------

        const T& x (
        ) const
        { 
            return x_value;
        }

        // ---------------------------------------

        const T& y (
        ) const 
        { 
            return y_value;
        }

        // ---------------------------------------

        const T& z (
        ) const
        { 
            return z_value;
        }

        // ---------------------------------------

        T dot (
            const vector& rhs
        ) const 
        { 
            return x()*rhs.x() + y()*rhs.y() + z()*rhs.z();
        }

        // ---------------------------------------

        vector cross (
            const vector& rhs
        ) const
        {
            return vector (
                y()*rhs.z() - z()*rhs.y(),
                z()*rhs.x() - x()*rhs.z(),
                x()*rhs.y() - y()*rhs.x()
                );
        }

        // ---------------------------------------

        vector& operator += (
            const vector& rhs
        )
        {
            x() += rhs.x();
            y() += rhs.y();
            z() += rhs.z();
            return *this;
        }

        // ---------------------------------------

        vector& operator -= (
            const vector& rhs
        )
        {
            x() -= rhs.x();
            y() -= rhs.y();
            z() -= rhs.z();
            return *this;
        }

        // ---------------------------------------

        vector& operator /= (
            const T& rhs
        )
        {
            x() /= rhs;
            y() /= rhs;
            z() /= rhs;
            return *this;
        }

        // ---------------------------------------

        vector& operator *= (
            const T& rhs
        )
        {
            x() *= rhs;
            y() *= rhs;
            z() *= rhs;
            return *this;
        }

        // ---------------------------------------

        vector operator + (
            const vector& rhs
        ) const
        {
            return vector(x()+rhs.x(), y()+rhs.y(), z()+rhs.z());
        }

        // ---------------------------------------

        vector operator - (
            const vector& rhs
        ) const
        {
            return vector(x()-rhs.x(), y()-rhs.y(), z()-rhs.z());
        }

        // ---------------------------------------

        vector operator / (
            const T& val
        ) const
        {
            return vector(x()/val, y()/val, z()/val);
        }

        // ---------------------------------------

        template <typename U, long NR2>
        bool operator== (
            const vector<U,NR2>& rhs
        ) const
        {
            return x()==rhs.x() && y()==rhs.y() && z()==rhs.z();
        }

        // ---------------------------------------

        template <typename U, long NR2>
        bool operator!= (
            const vector<U,NR2>& rhs
        ) const
        {
            return !(*this == rhs);
        }

        // ---------------------------------------

        template <long NRm, long NC, typename MM, typename l>
        operator matrix<T,NRm, NC, MM,l> (
        ) const
        {
            matrix<T,NRm, NC, MM,l> m(3,1);
            m(0) = x();
            m(1) = y();
            m(2) = z();
            return m;
        }

        // ---------------------------------------

        void swap (
            vector& item
        )
        {
            dlib::exchange(x_value, item.x_value);
            dlib::exchange(y_value, item.y_value);
            dlib::exchange(z_value, item.z_value);
        }

        // ---------------------------------------

    private:
        T x_value;
        T y_value;
        T z_value;
    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    class vector<T,2> 
    {
        /*!
            INITIAL VALUE
                - x_value == 0
                - y_value == 0  

            CONVENTION
                - x_value == x() 
                - y_value == y() 
                - z() == 0
        !*/

    public:

        typedef T type;
        
        vector (
        ) 
        {
            x() = 0;
            y() = 0;
        }

        // ---------------------------------------

        vector (
            const T _x,
            const T _y
        ) 
        {
            x() = _x;
            y() = _y;
        }

        // ---------------------------------------

        template <typename U>
        vector (
            const vector<U,3>& item
        )
        {
            // Do this so that we get the appropriate rounding depending on the relative
            // type of T and U.
            vector<T,3> temp(item);
            x() = temp.x();
            y() = temp.y();
        }

        // ---------------------------------------

        vector (
            const vector& item
        )
        {
            x() = item.x();
            y() = item.y();
        }

        // ---------------------------------------

        vector (
            const vector<T,3>& item
        )
        {
            x() = item.x();
            y() = item.y();
        }

        // ---------------------------------------

        template <typename U>
        vector (
            const vector<U,2>& item
        )
        {
            (*this) = item;
        }

        // ---------------------------------------

        template <typename EXP>
        vector ( const matrix_exp<EXP>& m)
        {
            (*this) = m;
        }

        // ---------------------------------------

        template <typename EXP>
        vector& operator = (
            const matrix_exp<EXP>& m
        )
        {
            // you can only assign vectors with 2 elements to a dlib::vector<T,2> object
            COMPILE_TIME_ASSERT(EXP::NR*EXP::NC == 2 || EXP::NR*EXP::NC == 0);

            // make sure requires clause is not broken
            DLIB_ASSERT((m.nr() == 1 || m.nc() == 1) && (m.size() == 2),
                "\t vector(const matrix_exp& m)"
                << "\n\t the given matrix is of the wrong size"
                << "\n\t m.nr():   " << m.nr() 
                << "\n\t m.nc():   " << m.nc() 
                << "\n\t m.size(): " << m.size() 
                << "\n\t this: " << this
                );

            vector_assign_helper<T, typename EXP::type>::assign(*this, m);
            return *this;
        }

        // ---------------------------------------

        template <typename U>
        vector& operator = (
            const vector<U,2>& item
        )
        {
            vector_assign_helper<T,U>::assign(*this, item);
            return *this;
        }

        // ---------------------------------------

        vector& operator= (
            const vector& item
        )
        {
            x() = item.x();
            y() = item.y();
            return *this;
        }

        // ---------------------------------------

        T length(
        ) const 
        { 
            return (T)std::sqrt((double)(x()*x() + y()*y())); 
        }

        // ---------------------------------------

        vector normalize (
        ) const 
        {
            T tmp = (T)std::sqrt((double)(x()*x() + y()*y()));
            return vector ( x()/tmp,
                            y()/tmp
                            );
        }

        // ---------------------------------------

        T& x (
        ) 
        { 
            return x_value;
        }

        // ---------------------------------------

        T& y (
        ) 
        { 
            return y_value;
        }

        // ---------------------------------------

        const T& x (
        ) const
        { 
            return x_value;
        }

        // ---------------------------------------

        const T& y (
        ) const 
        { 
            return y_value;
        }

        // ---------------------------------------

        const T z (
        ) const
        {
            return 0;
        }

        // ---------------------------------------

        T dot (
            const vector& rhs
        ) const 
        { 
            return x()*rhs.x() + y()*rhs.y();
        }

        // ---------------------------------------

        vector& operator += (
            const vector& rhs
        )
        {
            x() += rhs.x();
            y() += rhs.y();
            return *this;
        }

        // ---------------------------------------

        vector& operator -= (
            const vector& rhs
        )
        {
            x() -= rhs.x();
            y() -= rhs.y();
            return *this;
        }

        // ---------------------------------------

        vector& operator /= (
            const T& rhs
        )
        {
            x() /= rhs;
            y() /= rhs;
            return *this;
        }

        // ---------------------------------------

        vector& operator *= (
            const T& rhs
        )
        {
            x() *= rhs;
            y() *= rhs;
            return *this;
        }

        // ---------------------------------------

        vector operator + (
            const vector& rhs
        ) const
        {
            return vector(x()+rhs.x(), y()+rhs.y());
        }

        // ---------------------------------------

        vector operator - (
            const vector& rhs
        ) const
        {
            return vector(x()-rhs.x(), y()-rhs.y());
        }

        // ---------------------------------------

        vector operator / (
            const T& val
        ) const
        {
            return vector(x()/val, y()/val);
        }

        // ---------------------------------------

        template <typename U, long NR2>
        bool operator== (
            const vector<U,NR2>& rhs
        ) const
        {
            return x()==rhs.x() && y()==rhs.y() && z()==rhs.z();
        }

        // ---------------------------------------

        bool operator== (
            const vector& rhs
        ) const
        {
            return x()==rhs.x() && y()==rhs.y();
        }

        // ---------------------------------------

        template <typename U, long NR2>
        bool operator!= (
            const vector<U,NR2>& rhs
        ) const
        {
            return !(*this == rhs);
        }

        // ---------------------------------------

        bool operator!= (
            const vector& rhs
        ) const
        {
            return !(*this == rhs);
        }

        // ---------------------------------------

        template <long NRm, long NC, typename MM, typename l>
        operator matrix<T,NRm, NC, MM,l> (
        ) const
        {
            matrix<T,NRm, NC, MM,l> m(2,1);
            m(0) = x();
            m(1) = y();
            return m;
        }

        // ---------------------------------------

        void swap (
            vector& item
        )
        {
            dlib::exchange(x_value, item.x_value);
            dlib::exchange(y_value, item.y_value);
        }

        // ---------------------------------------

        vector<T,3> cross (
            const vector<T,3>& rhs
        ) const
        {
            return vector<T,3> (
                y()*rhs.z(),
                - x()*rhs.z(),
                x()*rhs.y() - y()*rhs.x()
                );
        }

        // ---------------------------------------

    private:
        T x_value;
        T y_value;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    inline const vector<T,2> operator* (
        const vector<T,2>& v,
        const U& s
    )
    {
        return vector<T,2>(v.x()*s, v.y()*s);
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    inline const vector<T,2> operator* (
        const U& s,
        const vector<T,2>& v
    )
    {
        return vector<T,2>(v.x()*s, v.y()*s);
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    inline const vector<T,3> operator* (
        const vector<T,3>& v,
        const U& s
    )
    {
        return vector<T,3>(v.x()*s, v.y()*s, v.z()*s);
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    inline const vector<T,3> operator* (
        const U& s,
        const vector<T,3>& v
    )
    {
        return vector<T,3>(v.x()*s, v.y()*s, v.z()*s);
    }

// ----------------------------------------------------------------------------------------

    template<typename T, long NR>
    inline void swap (
        vector<T,NR> & a, 
        vector<T,NR> & b 
    ) { a.swap(b); }   

// ----------------------------------------------------------------------------------------

    template<typename T>
    inline void serialize (
        const vector<T,3>& item,  
        std::ostream& out
    )
    {
        try
        {
            serialize(item.x(),out);
            serialize(item.y(),out);
            serialize(item.z(),out);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while serializing object of type vector"); 
        }
    }

    template<typename T>
    inline void deserialize (
        vector<T,3>& item,  
        std::istream& in
    )
    {
        try
        {
            deserialize(item.x(),in);
            deserialize(item.y(),in);
            deserialize(item.z(),in);
        }
        catch (serialization_error& e)
        { 
            item.x() = 0;
            item.y() = 0;
            item.z() = 0;
            throw serialization_error(e.info + "\n   while deserializing object of type vector"); 
        }
    }

// ----------------------------------------------------------------------------------------

    template<typename T>
    inline void serialize (
        const vector<T,2>& item,  
        std::ostream& out
    )
    {
        try
        {
            serialize(item.x(),out);
            serialize(item.y(),out);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while serializing object of type vector"); 
        }
    }

    template<typename T>
    inline void deserialize (
        vector<T,2>& item,  
        std::istream& in
    )
    {
        try
        {
            deserialize(item.x(),in);
            deserialize(item.y(),in);
        }
        catch (serialization_error& e)
        { 
            item.x() = 0;
            item.y() = 0;
            throw serialization_error(e.info + "\n   while deserializing object of type vector"); 
        }
    }

// ----------------------------------------------------------------------------------------

    template<typename T>
    std::ostream& operator<< (
        std::ostream& out, 
        const vector<T,3>& item 
    )
    {
        out << "(" << item.x() << ", " << item.y() << ", " << item.z() << ")";
        return out;
    }

    template<typename T>
    std::istream& operator>>(
        std::istream& in, 
        vector<T,3>& item 
    )   
    {

        // eat all the crap up to the '(' 
        while (in.peek() == ' ' || in.peek() == '\t' || in.peek() == '\r' || in.peek() == '\n')
            in.get();

        // there should be a '(' if not then this is an error
        if (in.get() != '(')
        {
            in.setstate(in.rdstate() | std::ios::failbit);
            return in;
        }

        // eat all the crap up to the first number 
        while (in.peek() == ' ' || in.peek() == '\t')
            in.get();
        in >> item.x();

        if (!in.good())
            return in;
              
        // eat all the crap up to the next number
        while (in.peek() == ' ' || in.peek() == '\t' || in.peek() == ',')
            in.get();
        in >> item.y();

        if (!in.good())
            return in;
              
        // eat all the crap up to the next number
        while (in.peek() == ' ' || in.peek() == '\t' || in.peek() == ',')
            in.get();
        in >> item.z();

        if (!in.good())
            return in;
              
        // eat all the crap up to the ')'
        while (in.peek() == ' ' || in.peek() == '\t')
            in.get();

        // there should be a ')' if not then this is an error
        if (in.get() != ')')
            in.setstate(in.rdstate() | std::ios::failbit);
        return in;
    }

// ----------------------------------------------------------------------------------------


    template<typename T>
    std::ostream& operator<< (
        std::ostream& out, 
        const vector<T,2>& item 
    )
    {
        out << "(" << item.x() << ", " << item.y() << ")";
        return out;
    }

    template<typename T>
    std::istream& operator>>(
        std::istream& in, 
        vector<T,2>& item 
    )   
    {

        // eat all the crap up to the '(' 
        while (in.peek() == ' ' || in.peek() == '\t' || in.peek() == '\r' || in.peek() == '\n')
            in.get();

        // there should be a '(' if not then this is an error
        if (in.get() != '(')
        {
            in.setstate(in.rdstate() | std::ios::failbit);
            return in;
        }

        // eat all the crap up to the first number 
        while (in.peek() == ' ' || in.peek() == '\t')
            in.get();
        in >> item.x();

        if (!in.good())
            return in;
              
        // eat all the crap up to the next number
        while (in.peek() == ' ' || in.peek() == '\t' || in.peek() == ',')
            in.get();
        in >> item.y();

        if (!in.good())
            return in;
              
        // eat all the crap up to the ')'
        while (in.peek() == ' ' || in.peek() == '\t')
            in.get();

        // there should be a ')' if not then this is an error
        if (in.get() != ')')
            in.setstate(in.rdstate() | std::ios::failbit);
        return in;
    }

// ----------------------------------------------------------------------------------------

    typedef vector<long,2> point;

// ----------------------------------------------------------------------------------------

}

namespace std
{
    /*!
        Define std::less<vector<T,3> > so that you can use vectors in the associative containers.
    !*/
    template<typename T>
    struct less<dlib::vector<T,3> > : public binary_function<dlib::vector<T,3> ,dlib::vector<T,3> ,bool>
    {
        inline bool operator() (const dlib::vector<T,3> & a, const dlib::vector<T,3> & b) const
        { 
            if      (a.x() < b.x()) return true;
            else if (a.x() > b.x()) return false;
            else if (a.y() < b.y()) return true;
            else if (a.y() > b.y()) return false;
            else if (a.z() < b.z()) return true;
            else if (a.z() > b.z()) return false;
            else                    return false;
        }
    };

    /*!
        Define std::less<vector<T,2> > so that you can use vector<T,2>s in the associative containers.
    !*/
    template<typename T>
    struct less<dlib::vector<T,2> > : public binary_function<dlib::vector<T,2> ,dlib::vector<T,2> ,bool>
    {
        inline bool operator() (const dlib::vector<T,2> & a, const dlib::vector<T,2> & b) const
        { 
            if      (a.x() < b.x()) return true;
            else if (a.x() > b.x()) return false;
            else if (a.y() < b.y()) return true;
            else if (a.y() > b.y()) return false;
            else                    return false;
        }
    };
}

#endif // DLIB_VECTOr_H_

