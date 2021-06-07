/////////////////////////////////////////////////////////////////////////////
// Name:        wx/meta/pod.h
// Purpose:     Test if a type is POD
// Author:      Vaclav Slavik, Jaakko Salli
// Created:     2010-06-14
// Copyright:   (c) wxWidgets team
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_META_POD_H_
#define _WX_META_POD_H_

#include "wx/defs.h"

//
// TODO: Use TR1 is_pod<> implementation where available. VC9 SP1 has it
//       in tr1 namespace, VC10 has it in std namespace. GCC 4.2 has it in
//       <tr1/type_traits>, while GCC 4.3 and later have it in <type_traits>.
//

// Helper to decide if an object of type T is POD (Plain Old Data)
template<typename T>
struct wxIsPod
{
    static const bool value = false;
};

// Macro to add wxIsPod<T> specialization for given type that marks it
// as Plain Old Data:
#define WX_DECLARE_TYPE_POD(type)                           \
    template<> struct wxIsPod<type>                         \
    {                                                       \
        static const bool value = true;                     \
    };

WX_DECLARE_TYPE_POD(bool)
WX_DECLARE_TYPE_POD(unsigned char)
WX_DECLARE_TYPE_POD(signed char)
WX_DECLARE_TYPE_POD(unsigned int)
WX_DECLARE_TYPE_POD(signed int)
WX_DECLARE_TYPE_POD(unsigned short int)
WX_DECLARE_TYPE_POD(signed short int)
WX_DECLARE_TYPE_POD(signed long int)
WX_DECLARE_TYPE_POD(unsigned long int)
WX_DECLARE_TYPE_POD(float)
WX_DECLARE_TYPE_POD(double)
WX_DECLARE_TYPE_POD(long double)
#if wxWCHAR_T_IS_REAL_TYPE
WX_DECLARE_TYPE_POD(wchar_t)
#endif
#ifdef wxLongLong_t
WX_DECLARE_TYPE_POD(wxLongLong_t)
WX_DECLARE_TYPE_POD(wxULongLong_t)
#endif

// pointers are Plain Old Data:
template<typename T>
struct wxIsPod<T*>
{
    static const bool value = true;
};

template<typename T>
struct wxIsPod<const T*>
{
    static const bool value = true;
};

#endif // _WX_META_POD_H_
