// Copyright (C) 2005  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MEMBER_FUNCTION_POINTER_KERNEl_1_
#define DLIB_MEMBER_FUNCTION_POINTER_KERNEl_1_

#include "../algs.h"
#include "member_function_pointer_kernel_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename PARAM1 = void,
        typename PARAM2 = void,
        typename PARAM3 = void,
        typename PARAM4 = void
        >
    class member_function_pointer_kernel_1;

    template <
        typename PARAM1,
        typename PARAM2,
        typename PARAM3,
        typename PARAM4 
        >
    void swap (
        member_function_pointer_kernel_1<PARAM1,PARAM2,PARAM3,PARAM4>& a,
        member_function_pointer_kernel_1<PARAM1,PARAM2,PARAM3,PARAM4>& b
    ) { a.swap(b); }

// ----------------------------------------------------------------------------------------

    template <>
    class member_function_pointer_kernel_1<void,void,void,void>
    {
        /*!
            INITIAL VALUE
                - mp == 0

            CONVENTION
                - is_set() == (mp != 0)


                Note that I'm using reinterpret_cast rather than dynamic_cast here 
                in the is_same() function.  It would be better if dynamic_cast was used
                but some compilers don't enable RTTI by default so using it would make the
                build process more complicated for users so I'm not using it.  I'm 
                not aware of any platforms/compilers where reinterpret_cast won't end 
                up doing the right thing for us here so it should be ok.  
        !*/

        class mp_base
        {
        public:
            virtual ~mp_base(){}
            virtual void call() const = 0;
            virtual mp_base* clone() const = 0;
            virtual bool is_same (const mp_base* item) const = 0;
        };

        template <typename T>
        class mp_impl : public mp_base 
        {
        public:
            mp_impl (
                T& object,
                void (T::*cb)()
            ) :
                callback(cb),
                o(&object)
            {
            }

            void call (
            ) const
            {
                (o->*callback)();
            }

            mp_base* clone() const { return new mp_impl(*o,callback); }

            bool is_same (const mp_base* item) const 
            {
                const mp_impl* i = reinterpret_cast<const mp_impl*>(item);
                return (i != 0 && i->o == o && i->callback == callback);
            }

        private:
            void (T::*callback)();
            T* o;
        };

    public:
        typedef void param1_type;
        typedef void param2_type;
        typedef void param3_type;
        typedef void param4_type;

        member_function_pointer_kernel_1 (  
        ) : 
            mp(0)
        {}

        member_function_pointer_kernel_1(
            const member_function_pointer_kernel_1& item
        ) :
            mp(0)
        {
            if (item.is_set())
                mp = item.mp->clone();
        }

        member_function_pointer_kernel_1& operator=(
            const member_function_pointer_kernel_1& item
        )
        {
            if (this != &item)
            {
                clear();

                if (item.is_set())
                    mp = item.mp->clone();
            }
            return *this;
        }

        bool operator == (
            const member_function_pointer_kernel_1& item
        ) const
        {
            if (is_set() != item.is_set())
                return false;
            if (is_set() == false)
                return true;
            return mp->is_same(item.mp);
        }

        bool operator != (
            const member_function_pointer_kernel_1& item
        ) const { return !(*this == item); }

        ~member_function_pointer_kernel_1 (
        )
        {
            if (mp)
                delete mp;
        }

        void clear(
        )
        {
            if (mp)
            {
                delete mp;
                mp = 0;
            }
        }

        bool is_set (
        ) const { return mp != 0; } 

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)()
        )
        {
            clear();
            mp = new mp_impl<T>(object,cb);
        }

        void operator () (
        ) const { mp->call(); }

        void swap (
            member_function_pointer_kernel_1& item
        ) { exchange(mp,item.mp); }

    private:

        mp_base* mp;


    };    

// ----------------------------------------------------------------------------------------

    template <
        typename PARAM1
        >
    class member_function_pointer_kernel_1<PARAM1,void,void,void>
    {
        /*!
            INITIAL VALUE
                - mp == 0

            CONVENTION
                - is_set() == (mp != 0)
        !*/

        class mp_base
        {
        public:
            virtual ~mp_base(){}
            virtual void call(PARAM1) const = 0;
            virtual mp_base* clone() const = 0;
            virtual bool is_same (const mp_base* item) const = 0;
        };

        template <typename T>
        class mp_impl : public mp_base 
        {
        public:
            mp_impl (
                T& object,
                void (T::*cb)(PARAM1)
            ) :
                callback(cb),
                o(&object)
            {
            }

            void call (
                PARAM1 param1
            ) const
            {
                (o->*callback)(param1);
            }

            mp_base* clone() const { return new mp_impl(*o,callback); }

            bool is_same (const mp_base* item) const 
            {
                const mp_impl* i = reinterpret_cast<const mp_impl*>(item);
                return (i != 0 && i->o == o && i->callback == callback);
            }

        private:
            void (T::*callback)(PARAM1);
            T* o;
        };

    public:
        typedef PARAM1 param1_type;
        typedef void param2_type;
        typedef void param3_type;
        typedef void param4_type;

        member_function_pointer_kernel_1 (  
        ) : 
            mp(0)
        {}

        member_function_pointer_kernel_1(
            const member_function_pointer_kernel_1& item
        ) :
            mp(0)
        {
            if (item.is_set())
                mp = item.mp->clone();
        }

        member_function_pointer_kernel_1& operator=(
            const member_function_pointer_kernel_1& item
        )
        {
            if (this != &item)
            {
                clear();

                if (item.is_set())
                    mp = item.mp->clone();
            }
            return *this;
        }

        bool operator == (
            const member_function_pointer_kernel_1& item
        ) const
        {
            if (is_set() != item.is_set())
                return false;
            if (is_set() == false)
                return true;
            return mp->is_same(item.mp);
        }

        bool operator != (
            const member_function_pointer_kernel_1& item
        ) const { return !(*this == item); }

        ~member_function_pointer_kernel_1 (
        )
        {
            if (mp)
                delete mp;
        }

        void clear(
        )
        {
            if (mp)
            {
                delete mp;
                mp = 0;
            }
        }

        bool is_set (
        ) const { return mp != 0; } 

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1)
        )
        {
            clear();
            mp = new mp_impl<T>(object,cb);
        }

        void operator () (
            PARAM1 param1
        ) const { mp->call(param1); }

        void swap (
            member_function_pointer_kernel_1& item
        ) { exchange(mp,item.mp); }

    private:

        mp_base* mp;

    };    

// ----------------------------------------------------------------------------------------

    template <
        typename PARAM1,
        typename PARAM2
        >
    class member_function_pointer_kernel_1<PARAM1,PARAM2,void,void>
    {
        /*!
            INITIAL VALUE
                - mp == 0

            CONVENTION
                - is_set() == (mp != 0)
        !*/

        class mp_base
        {
        public:
            virtual ~mp_base(){}
            virtual void call(PARAM1,PARAM2) const = 0;
            virtual mp_base* clone() const = 0;
            virtual bool is_same (const mp_base* item) const = 0;
        };

        template <typename T>
        class mp_impl : public mp_base 
        {
        public:
            mp_impl (
                T& object,
                void (T::*cb)(PARAM1,PARAM2)
            ) :
                callback(cb),
                o(&object)
            {
            }

            void call (
                PARAM1 param1,
                PARAM2 param2
            ) const
            {
                (o->*callback)(param1,param2);
            }

            mp_base* clone() const { return new mp_impl(*o,callback); }

            bool is_same (const mp_base* item) const 
            {
                const mp_impl* i = reinterpret_cast<const mp_impl*>(item);
                return (i != 0 && i->o == o && i->callback == callback);
            }

        private:
            void (T::*callback)(PARAM1,PARAM2);
            T* o;
        };

    public:
        typedef PARAM1 param1_type;
        typedef PARAM2 param2_type;
        typedef void param3_type;
        typedef void param4_type;

        member_function_pointer_kernel_1 (  
        ) : 
            mp(0)
        {}

        member_function_pointer_kernel_1(
            const member_function_pointer_kernel_1& item
        ) :
            mp(0)
        {
            if (item.is_set())
                mp = item.mp->clone();
        }

        member_function_pointer_kernel_1& operator=(
            const member_function_pointer_kernel_1& item
        )
        {
            if (this != &item)
            {
                clear();

                if (item.is_set())
                    mp = item.mp->clone();
            }
            return *this;
        }

        bool operator == (
            const member_function_pointer_kernel_1& item
        ) const
        {
            if (is_set() != item.is_set())
                return false;
            if (is_set() == false)
                return true;
            return mp->is_same(item.mp);
        }

        bool operator != (
            const member_function_pointer_kernel_1& item
        ) const { return !(*this == item); }

        ~member_function_pointer_kernel_1 (
        )
        {
            if (mp)
                delete mp;
        }

        void clear(
        )
        {
            if (mp)
            {
                delete mp;
                mp = 0;
            }
        }

        bool is_set (
        ) const { return mp != 0; } 

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1,PARAM2)
        )
        {
            clear();
            mp = new mp_impl<T>(object,cb);
        }

        void operator () (
            PARAM1 param1,
            PARAM2 param2
        ) const { mp->call(param1,param2); }

        void swap (
            member_function_pointer_kernel_1& item
        ) { exchange(mp,item.mp); }

    private:

        mp_base* mp;

    };    

// ----------------------------------------------------------------------------------------

    template <
        typename PARAM1,
        typename PARAM2,
        typename PARAM3
        >
    class member_function_pointer_kernel_1<PARAM1,PARAM2,PARAM3,void>
    {
        /*!
            INITIAL VALUE
                - mp == 0

            CONVENTION
                - is_set() == (mp != 0)
        !*/

        class mp_base
        {
        public:
            virtual ~mp_base(){}
            virtual void call(PARAM1,PARAM2,PARAM3) const = 0;
            virtual mp_base* clone() const = 0;
            virtual bool is_same (const mp_base* item) const = 0;
        };

        template <typename T>
        class mp_impl : public mp_base 
        {
        public:
            mp_impl (
                T& object,
                void (T::*cb)(PARAM1,PARAM2,PARAM3)
            ) :
                callback(cb),
                o(&object)
            {
            }

            void call (
                PARAM1 param1,
                PARAM2 param2,
                PARAM3 param3
            ) const
            {
                (o->*callback)(param1,param2,param3);
            }

            mp_base* clone() const { return new mp_impl(*o,callback); }

            bool is_same (const mp_base* item) const 
            {
                const mp_impl* i = reinterpret_cast<const mp_impl*>(item);
                return (i != 0 && i->o == o && i->callback == callback);
            }

        private:
            void (T::*callback)(PARAM1,PARAM2,PARAM3);
            T* o;
        };

    public:
        typedef PARAM1 param1_type;
        typedef PARAM2 param2_type;
        typedef PARAM3 param3_type;
        typedef void param4_type;

        member_function_pointer_kernel_1 (  
        ) : 
            mp(0)
        {}

        member_function_pointer_kernel_1(
            const member_function_pointer_kernel_1& item
        ) :
            mp(0)
        {
            if (item.is_set())
                mp = item.mp->clone();
        }

        member_function_pointer_kernel_1& operator=(
            const member_function_pointer_kernel_1& item
        )
        {
            if (this != &item)
            {
                clear();

                if (item.is_set())
                    mp = item.mp->clone();
            }
            return *this;
        }

        bool operator == (
            const member_function_pointer_kernel_1& item
        ) const
        {
            if (is_set() != item.is_set())
                return false;
            if (is_set() == false)
                return true;
            return mp->is_same(item.mp);
        }

        bool operator != (
            const member_function_pointer_kernel_1& item
        ) const { return !(*this == item); }

        ~member_function_pointer_kernel_1 (
        )
        {
            if (mp)
                delete mp;
        }

        void clear(
        )
        {
            if (mp)
            {
                delete mp;
                mp = 0;
            }
        }

        bool is_set (
        ) const { return mp != 0; } 

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1,PARAM2,PARAM3)
        )
        {
            clear();
            mp = new mp_impl<T>(object,cb);
        }

        void operator () (
            PARAM1 param1,
            PARAM2 param2,
            PARAM3 param3
        ) const { mp->call(param1,param2,param3); }

        void swap (
            member_function_pointer_kernel_1& item
        ) { exchange(mp,item.mp); }

    private:

        mp_base* mp;

    };    

// ----------------------------------------------------------------------------------------

    template <
        typename PARAM1,
        typename PARAM2,
        typename PARAM3,
        typename PARAM4
        >
    class member_function_pointer_kernel_1
    {
        /*!
            INITIAL VALUE
                - mp == 0

            CONVENTION
                - is_set() == (mp != 0)
        !*/

        class mp_base
        {
        public:
            virtual ~mp_base(){}
            virtual void call(PARAM1,PARAM2,PARAM3,PARAM4) const = 0;
            virtual mp_base* clone() const = 0;
            virtual bool is_same (const mp_base* item) const = 0;
        };

        template <typename T>
        class mp_impl : public mp_base 
        {
        public:
            mp_impl (
                T& object,
                void (T::*cb)(PARAM1,PARAM2,PARAM3,PARAM4)
            ) :
                callback(cb),
                o(&object)
            {
            }

            void call (
                PARAM1 param1,
                PARAM2 param2,
                PARAM3 param3,
                PARAM4 param4
            ) const
            {
                (o->*callback)(param1,param2,param3,param4);
            }

            mp_base* clone() const { return new mp_impl(*o,callback); }

            bool is_same (const mp_base* item) const 
            {
                const mp_impl* i = reinterpret_cast<const mp_impl*>(item);
                return (i != 0 && i->o == o && i->callback == callback);
            }

        private:
            void (T::*callback)(PARAM1,PARAM2,PARAM3,PARAM4);
            T* o;
        };

    public:
        typedef PARAM1 param1_type;
        typedef PARAM2 param2_type;
        typedef PARAM3 param3_type;
        typedef PARAM4 param4_type;

        member_function_pointer_kernel_1 (  
        ) : 
            mp(0)
        {}

        member_function_pointer_kernel_1(
            const member_function_pointer_kernel_1& item
        ) :
            mp(0)
        {
            if (item.is_set())
                mp = item.mp->clone();
        }

        member_function_pointer_kernel_1& operator=(
            const member_function_pointer_kernel_1& item
        )
        {
            if (this != &item)
            {
                clear();

                if (item.is_set())
                    mp = item.mp->clone();
            }
            return *this;
        }

        bool operator == (
            const member_function_pointer_kernel_1& item
        ) const
        {
            if (is_set() != item.is_set())
                return false;
            if (is_set() == false)
                return true;
            return mp->is_same(item.mp);
        }

        bool operator != (
            const member_function_pointer_kernel_1& item
        ) const { return !(*this == item); }

        ~member_function_pointer_kernel_1 (
        )
        {
            if (mp)
                delete mp;
        }

        void clear(
        )
        {
            if (mp)
            {
                delete mp;
                mp = 0;
            }
        }

        bool is_set (
        ) const { return mp != 0; } 

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1,PARAM2,PARAM3,PARAM4)
        )
        {
            clear();
            mp = new mp_impl<T>(object,cb);
        }

        void operator () (
            PARAM1 param1,
            PARAM2 param2,
            PARAM3 param3,
            PARAM4 param4
        ) const { mp->call(param1,param2,param3,param4); }

        void swap (
            member_function_pointer_kernel_1& item
        ) { exchange(mp,item.mp); }

    private:

        mp_base* mp;

    };    

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MEMBER_FUNCTION_POINTER_KERNEl_1_

