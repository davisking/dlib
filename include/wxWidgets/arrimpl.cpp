///////////////////////////////////////////////////////////////////////////////
// Name:        wx/arrimpl.cpp
// Purpose:     helper file for implementation of dynamic lists
// Author:      Vadim Zeitlin
// Modified by:
// Created:     16.10.97
// Copyright:   (c) 1997 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 * Purpose: implements helper functions used by the template class used by   *
 *          DECLARE_OBJARRAY macro and which couldn't be implemented inline  *
 *          (because they need the full definition of type T in scope)       *
 *                                                                           *
 * Usage:   1) #include dynarray.h                                           *
 *          2) WX_DECLARE_OBJARRAY                                           *
 *          3) #include arrimpl.cpp                                          *
 *          4) WX_DEFINE_OBJARRAY                                            *
 *****************************************************************************/

#undef WX_DEFINE_OBJARRAY
#define WX_DEFINE_OBJARRAY(name)                                              \
name::value_type*                                                             \
wxObjectArrayTraitsFor##name::Clone(const name::value_type& item)             \
{                                                                             \
    return new name::value_type(item);                                        \
}                                                                             \
                                                                              \
void wxObjectArrayTraitsFor##name::Free(name::value_type* p)                  \
{                                                                             \
    delete p;                                                                 \
}
