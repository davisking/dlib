///////////////////////////////////////////////////////////////////////////////
// Name:        wx/dynarray.h
// Purpose:     auto-resizable (i.e. dynamic) array support
// Author:      Vadim Zeitlin
// Modified by:
// Created:     12.09.97
// Copyright:   (c) 1998 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef   _DYNARRAY_H
#define   _DYNARRAY_H

#include "wx/defs.h"

#include "wx/vector.h"

/*
  This header defines legacy dynamic arrays and object arrays (i.e. arrays
  which own their elements) classes.

  Do *NOT* use them in the new code, these classes exist for compatibility
  only. Simply use standard container, e.g. std::vector<>, in your own code.
 */

#define _WX_ERROR_REMOVE "removing inexistent element in wxArray::Remove"

// ----------------------------------------------------------------------------
// types
// ----------------------------------------------------------------------------

/*
    Callback compare function for quick sort.

    It must return negative value, 0 or positive value if the first item is
    less than, equal to or greater than the second one.
 */
extern "C"
{
typedef int (wxCMPFUNC_CONV *CMPFUNC)(const void* pItem1, const void* pItem2);
}

// ----------------------------------------------------------------------------
// Array class providing legacy dynamic arrays API on top of wxVector<>
// ----------------------------------------------------------------------------

// For some reasons lost in the depths of time, sort functions with different
// signatures are used to sort normal arrays and to keep sorted arrays sorted.
// These two functors can be used as predicates with std::sort() adapting the
// sort function to it, whichever signature it uses.

template<class T>
class wxArray_SortFunction
{
public:
    typedef int (wxCMPFUNC_CONV *CMPFUNC)(T* pItem1, T* pItem2);

    wxArray_SortFunction(CMPFUNC f) : m_f(f) { }
    bool operator()(const T& i1, const T& i2)
      { return m_f(const_cast<T*>(&i1), const_cast<T*>(&i2)) < 0; }
private:
    CMPFUNC m_f;
};

template<class T>
class wxSortedArray_SortFunction
{
public:
    typedef int (wxCMPFUNC_CONV *CMPFUNC)(T, T);

    wxSortedArray_SortFunction(CMPFUNC f) : m_f(f) { }
    bool operator()(const T& i1, const T& i2)
      { return m_f(i1, i2) < 0; }
private:
    CMPFUNC m_f;
};

template <typename T, typename Sorter = wxSortedArray_SortFunction<T> >
class wxBaseArray : public wxVector<T>
{
public:
    typedef typename Sorter::CMPFUNC SCMPFUNC;
    typedef typename wxArray_SortFunction<T>::CMPFUNC CMPFUNC;

    typedef wxVector<T> base_vec;

    typedef typename base_vec::value_type value_type;
    typedef typename base_vec::reference reference;
    typedef typename base_vec::const_reference const_reference;
    typedef typename base_vec::iterator iterator;
    typedef typename base_vec::const_iterator const_iterator;
    typedef typename base_vec::const_reverse_iterator const_reverse_iterator;
    typedef typename base_vec::difference_type difference_type;
    typedef typename base_vec::size_type size_type;

public:
    typedef T base_type;

    wxBaseArray() : base_vec() { }
    explicit wxBaseArray(size_t n) : base_vec(n) { }
    wxBaseArray(size_t n, const_reference v) : base_vec(n, v) { }

    template <class InputIterator>
    wxBaseArray(InputIterator first, InputIterator last)
        : base_vec(first, last)
    { }

    void Empty() { this->clear(); }
    void Clear() { this->clear(); }
    void Alloc(size_t uiSize) { this->reserve(uiSize); }

    void Shrink()
    {
        wxShrinkToFit(*this);
    }

    size_t GetCount() const { return this->size(); }
    void SetCount(size_t n, T v = T()) { this->resize(n, v); }
    bool IsEmpty() const { return this->empty(); }
    size_t Count() const { return this->size(); }

    T& Item(size_t uiIndex) const
    {
        wxASSERT( uiIndex < this->size() );
        return const_cast<T&>((*this)[uiIndex]);
    }

    T& Last() const { return Item(this->size() - 1); }

    int Index(T item, bool bFromEnd = false) const
    {
        if ( bFromEnd )
        {
            const const_reverse_iterator b = this->rbegin(),
                  e = this->rend();
            for ( const_reverse_iterator i = b; i != e; ++i )
                if ( *i == item )
                    return (int)(e - i - 1);
        }
        else
        {
            const const_iterator b = this->begin(),
                  e = this->end();
            for ( const_iterator i = b; i != e; ++i )
                if ( *i == item )
                    return (int)(i - b);
        }

        return wxNOT_FOUND;
    }

    int Index(T lItem, SCMPFUNC fnCompare) const
    {
        Sorter p(fnCompare);
        const_iterator i = std::lower_bound(this->begin(), this->end(), lItem, p);
        return i != this->end() && !p(lItem, *i) ? (int)(i - this->begin())
                                                 : wxNOT_FOUND;
    }

    size_t IndexForInsert(T lItem, SCMPFUNC fnCompare) const
    {
        Sorter p(fnCompare);
        const_iterator i = std::lower_bound(this->begin(), this->end(), lItem, p);
        return i - this->begin();
    }

    void Add(T lItem, size_t nInsert = 1)
    {
        this->insert(this->end(), nInsert, lItem);
    }

    size_t Add(T lItem, SCMPFUNC fnCompare)
    {
        size_t n = IndexForInsert(lItem, fnCompare);
        Insert(lItem, n);
        return n;
    }

    void Insert(T lItem, size_t uiIndex, size_t nInsert = 1)
    {
        this->insert(this->begin() + uiIndex, nInsert, lItem);
    }

    void Remove(T lItem)
    {
        int n = Index(lItem);
        wxCHECK_RET( n != wxNOT_FOUND, _WX_ERROR_REMOVE );
        RemoveAt((size_t)n);
    }

    void RemoveAt(size_t uiIndex, size_t nRemove = 1)
    {
        this->erase(this->begin() + uiIndex, this->begin() + uiIndex + nRemove);
    }

    void Sort(CMPFUNC fCmp)
    {
        wxArray_SortFunction<T> p(fCmp);
        std::sort(this->begin(), this->end(), p);
    }

    void Sort(SCMPFUNC fCmp)
    {
        Sorter p(fCmp);
        std::sort(this->begin(), this->end(), p);
    }
};

// ============================================================================
// The private helper macros containing the core of the array classes
// ============================================================================

// ----------------------------------------------------------------------------
// _WX_DEFINE_SORTED_TYPEARRAY: sorted array for simple data types
//    cannot handle types with size greater than pointer because of sorting
// ----------------------------------------------------------------------------

// Note that "classdecl" here is intentionally not used because this class has
// only inline methods and so never needs to be exported from a DLL.
#define _WX_DEFINE_SORTED_TYPEARRAY_2(T, name, base, defcomp, classdecl)      \
    typedef wxBaseSortedArray<T> wxBaseSortedArrayFor##name;                  \
    class name : public wxBaseSortedArrayFor##name                            \
    {                                                                         \
    public:                                                                   \
        name(wxBaseSortedArrayFor##name::SCMPFUNC fn defcomp)                 \
            : wxBaseSortedArrayFor##name(fn) { }                              \
    }


template <typename T, typename Sorter = wxSortedArray_SortFunction<T> >
class wxBaseSortedArray : public wxBaseArray<T, Sorter>
{
public:
    typedef typename Sorter::CMPFUNC SCMPFUNC;

    explicit wxBaseSortedArray(SCMPFUNC fn) : m_fnCompare(fn) { }

    size_t IndexForInsert(T item) const
    {
        return this->wxBaseArray<T, Sorter>::IndexForInsert(item, m_fnCompare);
    }

    void AddAt(T item, size_t index)
    {
        this->insert(this->begin() + index, item);
    }

    size_t Add(T item)
    {
        return this->wxBaseArray<T, Sorter>::Add(item, m_fnCompare);
    }

    void push_back(T item)
    {
        Add(item);
    }

protected:
    SCMPFUNC GetCompareFunction() const wxNOEXCEPT { return m_fnCompare; }

private:
    SCMPFUNC m_fnCompare;
};


// ----------------------------------------------------------------------------
// _WX_DECLARE_OBJARRAY: an array for pointers to type T with owning semantics
// ----------------------------------------------------------------------------

// This class must be able to be declared with incomplete types, so it doesn't
// actually use type T in its definition, and relies on a helper template
// parameter, which is declared by WX_DECLARE_OBJARRAY() and defined by
// WX_DEFINE_OBJARRAY(), for providing a way to create and destroy objects of
// type T
template <typename T, typename Traits>
class wxBaseObjectArray : private wxBaseArray<T*>
{
    typedef wxBaseArray<T*> base;

public:
    typedef T value_type;

    typedef int (wxCMPFUNC_CONV *CMPFUNC)(T **pItem1, T **pItem2);

    wxBaseObjectArray()
    {
    }

    wxBaseObjectArray(const wxBaseObjectArray& src) : base()
    {
        DoCopy(src);
    }

    wxBaseObjectArray& operator=(const wxBaseObjectArray& src)
    {
        Empty();
        DoCopy(src);

        return *this;
    }

    ~wxBaseObjectArray()
    {
        Empty();
    }

    void Alloc(size_t count) { base::reserve(count); }
    void reserve(size_t count) { base::reserve(count); }
    size_t GetCount() const { return base::size(); }
    size_t size() const { return base::size(); }
    bool IsEmpty() const { return base::empty(); }
    bool empty() const { return base::empty(); }
    size_t Count() const { return base::size(); }
    void Shrink() { base::Shrink(); }

    T& operator[](size_t uiIndex) const
    {
        return *base::operator[](uiIndex);
    }

    T& Item(size_t uiIndex) const
    {
        return *base::operator[](uiIndex);
    }

    T& Last() const
    {
        return *(base::operator[](size() - 1));
    }

    int Index(const T& item, bool bFromEnd = false) const
    {
        if ( bFromEnd )
        {
            if ( size() > 0 )
            {
                size_t ui = size() - 1;
                do
                {
                    if ( base::operator[](ui) == &item )
                        return static_cast<int>(ui);
                    ui--;
                }
                while ( ui != 0 );
            }
        }
        else
        {
            for ( size_t ui = 0; ui < size(); ++ui )
            {
                if( base::operator[](ui) == &item )
                    return static_cast<int>(ui);
            }
        }

        return wxNOT_FOUND;
    }

    void Add(const T& item, size_t nInsert = 1)
    {
        if ( nInsert == 0 )
            return;

        T* const pItem = Traits::Clone(item);

        const size_t nOldSize = size();
        if ( pItem != NULL )
            base::insert(this->end(), nInsert, pItem);

        for ( size_t i = 1; i < nInsert; i++ )
            base::operator[](nOldSize + i) = Traits::Clone(item);
    }

    void Add(const T* pItem)
    {
        base::push_back(const_cast<T*>(pItem));
    }

    void push_back(const T* pItem) { Add(pItem); }
    void push_back(const T& item) { Add(item); }

    void Insert(const T& item,  size_t uiIndex, size_t nInsert = 1)
    {
        if ( nInsert == 0 )
            return;

        T* const pItem = Traits::Clone(item);
        if ( pItem != NULL )
            base::insert(this->begin() + uiIndex, nInsert, pItem);

        for ( size_t i = 1; i < nInsert; ++i )
            base::operator[](uiIndex + i) = Traits::Clone(item);
    }

    void Insert(const T* pItem, size_t uiIndex)
    {
        base::insert(this->begin() + uiIndex, const_cast<T*>(pItem));
    }

    void Empty() { DoEmpty(); base::clear(); }
    void Clear() { DoEmpty(); base::clear(); }

    T* Detach(size_t uiIndex)
    {
        T* const p = base::operator[](uiIndex);

        base::erase(this->begin() + uiIndex);
        return p;
    }

    void RemoveAt(size_t uiIndex, size_t nRemove = 1)
    {
        wxCHECK_RET( uiIndex < size(), "bad index in RemoveAt()" );

        for ( size_t i = 0; i < nRemove; ++i )
            Traits::Free(base::operator[](uiIndex + i));

        base::erase(this->begin() + uiIndex, this->begin() + uiIndex + nRemove);
    }

    void Sort(CMPFUNC fCmp) { base::Sort(fCmp); }

private:
    void DoEmpty()
    {
        for ( size_t n = 0; n < size(); ++n )
            Traits::Free(base::operator[](n));
    }

    void DoCopy(const wxBaseObjectArray& src)
    {
        reserve(src.size());
        for ( size_t n = 0; n < src.size(); ++n )
            Add(src[n]);
    }
};

// ============================================================================
// The public macros for declaration and definition of the dynamic arrays
// ============================================================================

// Please note that for each macro WX_FOO_ARRAY we also have
// WX_FOO_EXPORTED_ARRAY and WX_FOO_USER_EXPORTED_ARRAY which are exactly the
// same except that they use an additional __declspec(dllexport) or equivalent
// under Windows if needed.
//
// The first (just EXPORTED) macros do it if wxWidgets was compiled as a DLL
// and so must be used used inside the library. The second kind (USER_EXPORTED)
// allow the user code to do it when it wants. This is needed if you have a dll
// that wants to export a wxArray daubed with your own import/export goo.
//
// Finally, you can define the macro below as something special to modify the
// arrays defined by a simple WX_FOO_ARRAY as well. By default is empty.
#define wxARRAY_DEFAULT_EXPORT

// ----------------------------------------------------------------------------
// WX_DECLARE_BASEARRAY(T, name): now is the same as WX_DEFINE_TYPEARRAY()
// below, only preserved for compatibility.
// ----------------------------------------------------------------------------

#define wxARRAY_DUMMY_BASE

#define WX_DECLARE_BASEARRAY(T, name)                             \
    WX_DEFINE_TYPEARRAY(T, name)

#define WX_DECLARE_EXPORTED_BASEARRAY(T, name)                    \
    WX_DEFINE_EXPORTED_TYPEARRAY(T, name, WXDLLIMPEXP_CORE)

#define WX_DECLARE_USER_EXPORTED_BASEARRAY(T, name, expmode)      \
    WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, wxARRAY_DUMMY_BASE, class expmode)

// ----------------------------------------------------------------------------
// WX_DEFINE_TYPEARRAY(T, name, base) define an array class named "name"
// containing the elements of type T. Note that the argument "base" is unused
// and is preserved for compatibility only. Also, macros with and without
// "_PTR" suffix are identical, and the latter ones are also kept only for
// compatibility.
// ----------------------------------------------------------------------------

#define WX_DEFINE_TYPEARRAY(T, name, base)                        \
    WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, base, class wxARRAY_DEFAULT_EXPORT)

#define WX_DEFINE_TYPEARRAY_PTR(T, name, base)                        \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, base, class wxARRAY_DEFAULT_EXPORT)

#define WX_DEFINE_EXPORTED_TYPEARRAY(T, name, base)               \
    WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, base, class WXDLLIMPEXP_CORE)

#define WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, base)               \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, base, class WXDLLIMPEXP_CORE)

#define WX_DEFINE_USER_EXPORTED_TYPEARRAY(T, name, base, expdecl) \
    WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, base, class expdecl)

#define WX_DEFINE_USER_EXPORTED_TYPEARRAY_PTR(T, name, base, expdecl) \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, base, class expdecl)

// This is the only non-trivial macro, which actually defines the array class
// with the given name containing the elements of the specified type.
//
// Note that "name" must be a class and not just a typedef because it can be
// (and is) forward declared in the existing code.
//
// As mentioned above, "base" is unused and so is "classdecl" as this class has
// only inline methods and so never needs to be exported from MSW DLLs.
//
// Note about apparently redundant wxBaseArray##name typedef: this is needed to
// avoid clashes between T and symbols defined in wxBaseArray<> scope, e.g. if
// we didn't do this, we would have compilation problems with arrays of type
// "Item" (which is also the name of a method in wxBaseArray<>).
#define WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, base, classdecl)               \
    typedef wxBaseArray<T> wxBaseArrayFor##name;                              \
    class name : public wxBaseArrayFor##name                                  \
    {                                                                         \
        typedef wxBaseArrayFor##name Base;                                    \
    public:                                                                   \
        name() : Base() { }                                                   \
        explicit name(size_t n) : Base(n) { }                                 \
        name(size_t n, Base::const_reference v) : Base(n, v) { }              \
        template <class InputIterator>                                        \
        name(InputIterator first, InputIterator last) : Base(first, last) { } \
    }


#define WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, base, classdecl) \
    WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, base, classdecl)

// ----------------------------------------------------------------------------
// WX_DEFINE_SORTED_TYPEARRAY: this is the same as the previous macro, but it
// defines a sorted array.
//
// Differences:
//  1) it must be given a COMPARE function in ctor which takes 2 items of type
//     T* and should return -1, 0 or +1 if the first one is less/greater
//     than/equal to the second one.
//  2) the Add() method inserts the item in such was that the array is always
//     sorted (it uses the COMPARE function)
//  3) it has no Sort() method because it's always sorted
//  4) Index() method is much faster (the sorted arrays use binary search
//     instead of linear one), but Add() is slower.
//  5) there is no Insert() method because you can't insert an item into the
//     given position in a sorted array but there is IndexForInsert()/AddAt()
//     pair which may be used to optimize a common operation of "insert only if
//     not found"
//
// Note that you have to specify the comparison function when creating the
// objects of this array type. If, as in 99% of cases, the comparison function
// is the same for all objects of a class, WX_DEFINE_SORTED_TYPEARRAY_CMP below
// is more convenient.
//
// Summary: use this class when the speed of Index() function is important, use
// the normal arrays otherwise.
// ----------------------------------------------------------------------------

// we need a macro which expands to nothing to pass correct number of
// parameters to a nested macro invocation even when we don't have anything to
// pass it
#define wxARRAY_EMPTY

#define WX_DEFINE_SORTED_TYPEARRAY(T, name, base)                         \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, base,               \
                                             wxARRAY_DEFAULT_EXPORT)

#define WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, base)                \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, base, WXDLLIMPEXP_CORE)

#define WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, base, expmode)  \
    typedef T _wxArray##name;                                             \
    _WX_DEFINE_SORTED_TYPEARRAY_2(_wxArray##name, name, base,             \
                                  wxARRAY_EMPTY, class expmode)

// ----------------------------------------------------------------------------
// WX_DEFINE_SORTED_TYPEARRAY_CMP: exactly the same as above but the comparison
// function is provided by this macro and the objects of this class have a
// default constructor which just uses it.
//
// The arguments are: the element type, the comparison function and the array
// name
//
// NB: this is, of course, how WX_DEFINE_SORTED_TYPEARRAY() should have worked
//     from the very beginning - unfortunately I didn't think about this earlier
// ----------------------------------------------------------------------------

#define WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, base)               \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, base,     \
                                                 wxARRAY_DEFAULT_EXPORT)

#define WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, base)      \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, base,     \
                                                 WXDLLIMPEXP_CORE)

#define WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, base, \
                                                     expmode)                \
    typedef T _wxArray##name;                                                \
    _WX_DEFINE_SORTED_TYPEARRAY_2(_wxArray##name, name, base, = cmpfunc,     \
                                  class expmode)

// ----------------------------------------------------------------------------
// WX_DECLARE_OBJARRAY(T, name): this macro generates a new array class
// named "name" which owns the objects of type T it contains, i.e. it will
// delete them when it is destroyed.
//
// An element is of type T*, but arguments of type T& are taken (see below!)
// and T& is returned.
//
// Don't use this for simple types such as "int" or "long"!
//
// Note on Add/Insert functions:
//  1) function(T*) gives the object to the array, i.e. it will delete the
//     object when it's removed or in the array's dtor
//  2) function(T&) will create a copy of the object and work with it
//
// Also:
//  1) Remove() will delete the object after removing it from the array
//  2) Detach() just removes the object from the array (returning pointer to it)
//
// NB1: Base type T should have an accessible copy ctor if Add(T&) is used
// NB2: Never ever cast a array to it's base type: as dtor is not virtual
//      and so you risk having at least the memory leaks and probably worse
//
// Some functions of this class are not inline, so it takes some space to
// define new class from this template even if you don't use it - which is not
// the case for the simple (non-object) array classes
//
// To use an objarray class you must
//      #include "dynarray.h"
//      WX_DECLARE_OBJARRAY(element_type, list_class_name)
//      #include "arrimpl.cpp"
//      WX_DEFINE_OBJARRAY(list_class_name) // name must be the same as above!
//
// This is necessary because at the moment of DEFINE_OBJARRAY class parsing the
// element_type must be fully defined (i.e. forward declaration is not
// enough), while WX_DECLARE_OBJARRAY may be done anywhere. The separation of
// two allows to break cicrcular dependencies with classes which have member
// variables of objarray type.
// ----------------------------------------------------------------------------

#define WX_DECLARE_OBJARRAY(T, name)                        \
    WX_DECLARE_USER_EXPORTED_OBJARRAY(T, name, wxARRAY_DEFAULT_EXPORT)

#define WX_DECLARE_EXPORTED_OBJARRAY(T, name)               \
    WX_DECLARE_USER_EXPORTED_OBJARRAY(T, name, WXDLLIMPEXP_CORE)

#define WX_DECLARE_OBJARRAY_WITH_DECL(T, name, classdecl)                     \
    classdecl wxObjectArrayTraitsFor##name                                    \
    {                                                                         \
    public:                                                                   \
        static T* Clone(T const& item);                                       \
        static void Free(T* p);                                               \
    };                                                                        \
    typedef wxBaseObjectArray<T, wxObjectArrayTraitsFor##name>                \
        wxBaseObjectArrayFor##name;                                           \
    classdecl name : public wxBaseObjectArrayFor##name                        \
    {                                                                         \
    }

#define WX_DECLARE_USER_EXPORTED_OBJARRAY(T, name, expmode) \
    WX_DECLARE_OBJARRAY_WITH_DECL(T, name, class expmode)

// WX_DEFINE_OBJARRAY is going to be redefined when arrimpl.cpp is included,
// try to provoke a human-understandable error if it used incorrectly.
//
// there is no real need for 3 different macros in the DEFINE case but do it
// anyhow for consistency
#define WX_DEFINE_OBJARRAY(name) DidYouIncludeArrimplCpp
#define WX_DEFINE_EXPORTED_OBJARRAY(name)   WX_DEFINE_OBJARRAY(name)
#define WX_DEFINE_USER_EXPORTED_OBJARRAY(name)   WX_DEFINE_OBJARRAY(name)

// ----------------------------------------------------------------------------
// Some commonly used predefined base arrays
// ----------------------------------------------------------------------------

WX_DECLARE_USER_EXPORTED_BASEARRAY(const void *, wxBaseArrayPtrVoid,
                                   WXDLLIMPEXP_BASE);
WX_DECLARE_USER_EXPORTED_BASEARRAY(char, wxBaseArrayChar, WXDLLIMPEXP_BASE);
WX_DECLARE_USER_EXPORTED_BASEARRAY(short, wxBaseArrayShort, WXDLLIMPEXP_BASE);
WX_DECLARE_USER_EXPORTED_BASEARRAY(int, wxBaseArrayInt, WXDLLIMPEXP_BASE);
WX_DECLARE_USER_EXPORTED_BASEARRAY(long, wxBaseArrayLong, WXDLLIMPEXP_BASE);
WX_DECLARE_USER_EXPORTED_BASEARRAY(size_t, wxBaseArraySizeT, WXDLLIMPEXP_BASE);
WX_DECLARE_USER_EXPORTED_BASEARRAY(double, wxBaseArrayDouble, WXDLLIMPEXP_BASE);

// ----------------------------------------------------------------------------
// Convenience macros to define arrays from base arrays
// ----------------------------------------------------------------------------

#define WX_DEFINE_ARRAY(T, name)                                       \
    WX_DEFINE_TYPEARRAY(T, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_ARRAY_PTR(T, name)                                \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_EXPORTED_ARRAY(T, name)                              \
    WX_DEFINE_EXPORTED_TYPEARRAY(T, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_EXPORTED_ARRAY_PTR(T, name)                       \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_ARRAY_WITH_DECL_PTR(T, name, decl)                \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayPtrVoid, decl)
#define WX_DEFINE_USER_EXPORTED_ARRAY(T, name, expmode)                \
    WX_DEFINE_TYPEARRAY_WITH_DECL(T, name, wxBaseArrayPtrVoid, wxARRAY_EMPTY expmode)
#define WX_DEFINE_USER_EXPORTED_ARRAY_PTR(T, name, expmode)         \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayPtrVoid, wxARRAY_EMPTY expmode)

#define WX_DEFINE_ARRAY_CHAR(T, name)                                 \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArrayChar)
#define WX_DEFINE_EXPORTED_ARRAY_CHAR(T, name)                        \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArrayChar)
#define WX_DEFINE_USER_EXPORTED_ARRAY_CHAR(T, name, expmode)          \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayChar, wxARRAY_EMPTY expmode)

#define WX_DEFINE_ARRAY_SHORT(T, name)                                 \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArrayShort)
#define WX_DEFINE_EXPORTED_ARRAY_SHORT(T, name)                        \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArrayShort)
#define WX_DEFINE_USER_EXPORTED_ARRAY_SHORT(T, name, expmode)          \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayShort, wxARRAY_EMPTY expmode)

#define WX_DEFINE_ARRAY_INT(T, name)                                   \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArrayInt)
#define WX_DEFINE_EXPORTED_ARRAY_INT(T, name)                          \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArrayInt)
#define WX_DEFINE_USER_EXPORTED_ARRAY_INT(T, name, expmode)            \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayInt, wxARRAY_EMPTY expmode)

#define WX_DEFINE_ARRAY_LONG(T, name)                                  \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArrayLong)
#define WX_DEFINE_EXPORTED_ARRAY_LONG(T, name)                         \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArrayLong)
#define WX_DEFINE_USER_EXPORTED_ARRAY_LONG(T, name, expmode)           \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayLong, wxARRAY_EMPTY expmode)

#define WX_DEFINE_ARRAY_SIZE_T(T, name)                                  \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArraySizeT)
#define WX_DEFINE_EXPORTED_ARRAY_SIZE_T(T, name)                         \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArraySizeT)
#define WX_DEFINE_USER_EXPORTED_ARRAY_SIZE_T(T, name, expmode)           \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArraySizeT, wxARRAY_EMPTY expmode)

#define WX_DEFINE_ARRAY_DOUBLE(T, name)                                \
    WX_DEFINE_TYPEARRAY_PTR(T, name, wxBaseArrayDouble)
#define WX_DEFINE_EXPORTED_ARRAY_DOUBLE(T, name)                       \
    WX_DEFINE_EXPORTED_TYPEARRAY_PTR(T, name, wxBaseArrayDouble)
#define WX_DEFINE_USER_EXPORTED_ARRAY_DOUBLE(T, name, expmode)         \
    WX_DEFINE_TYPEARRAY_WITH_DECL_PTR(T, name, wxBaseArrayDouble, wxARRAY_EMPTY expmode)

// ----------------------------------------------------------------------------
// Convenience macros to define sorted arrays from base arrays
// ----------------------------------------------------------------------------

#define WX_DEFINE_SORTED_ARRAY(T, name)                                \
    WX_DEFINE_SORTED_TYPEARRAY(T, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY(T, name)                       \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY(T, name, expmode)         \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, wxBaseArrayPtrVoid, wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_CHAR(T, name)                          \
    WX_DEFINE_SORTED_TYPEARRAY(T, name, wxBaseArrayChar)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CHAR(T, name)                 \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, wxBaseArrayChar)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CHAR(T, name, expmode)   \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, wxBaseArrayChar, wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_SHORT(T, name)                          \
    WX_DEFINE_SORTED_TYPEARRAY(T, name, wxBaseArrayShort)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_SHORT(T, name)                 \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, wxBaseArrayShort)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_SHORT(T, name, expmode)   \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, wxBaseArrayShort, wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_INT(T, name)                            \
    WX_DEFINE_SORTED_TYPEARRAY(T, name, wxBaseArrayInt)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_INT(T, name)                   \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, wxBaseArrayInt)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_INT(T, name, expmode)     \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, wxBaseArrayInt, expmode)

#define WX_DEFINE_SORTED_ARRAY_LONG(T, name)                           \
    WX_DEFINE_SORTED_TYPEARRAY(T, name, wxBaseArrayLong)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_LONG(T, name)                  \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, wxBaseArrayLong)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_LONG(T, name, expmode)    \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, wxBaseArrayLong, expmode)

#define WX_DEFINE_SORTED_ARRAY_SIZE_T(T, name)                           \
    WX_DEFINE_SORTED_TYPEARRAY(T, name, wxBaseArraySizeT)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_SIZE_T(T, name)                  \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY(T, name, wxBaseArraySizeT)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_SIZE_T(T, name, expmode)    \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY(T, name, wxBaseArraySizeT, wxARRAY_EMPTY expmode)

// ----------------------------------------------------------------------------
// Convenience macros to define sorted arrays from base arrays
// ----------------------------------------------------------------------------

#define WX_DEFINE_SORTED_ARRAY_CMP(T, cmpfunc, name)                   \
    WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CMP(T, cmpfunc, name)          \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayPtrVoid)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CMP(T, cmpfunc,           \
                                                     name, expmode)    \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name,     \
                                                 wxBaseArrayPtrVoid,   \
                                                 wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_CMP_CHAR(T, cmpfunc, name)             \
    WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayChar)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CMP_CHAR(T, cmpfunc, name)    \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayChar)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CMP_CHAR(T, cmpfunc,      \
                                                       name, expmode)  \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name,     \
                                                 wxBaseArrayChar,      \
                                                 wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_CMP_SHORT(T, cmpfunc, name)             \
    WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayShort)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CMP_SHORT(T, cmpfunc, name)    \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayShort)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CMP_SHORT(T, cmpfunc,     \
                                                       name, expmode)  \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name,     \
                                                 wxBaseArrayShort,     \
                                                 wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_CMP_INT(T, cmpfunc, name)               \
    WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayInt)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CMP_INT(T, cmpfunc, name)      \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayInt)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CMP_INT(T, cmpfunc,       \
                                                     name, expmode)    \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name,     \
                                                 wxBaseArrayInt,       \
                                                 wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_CMP_LONG(T, cmpfunc, name)              \
    WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayLong)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CMP_LONG(T, cmpfunc, name)     \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArrayLong)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CMP_LONG(T, cmpfunc,      \
                                                      name, expmode)   \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name,     \
                                                 wxBaseArrayLong,      \
                                                 wxARRAY_EMPTY expmode)

#define WX_DEFINE_SORTED_ARRAY_CMP_SIZE_T(T, cmpfunc, name)              \
    WX_DEFINE_SORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArraySizeT)
#define WX_DEFINE_SORTED_EXPORTED_ARRAY_CMP_SIZE_T(T, cmpfunc, name)     \
    WX_DEFINE_SORTED_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name, wxBaseArraySizeT)
#define WX_DEFINE_SORTED_USER_EXPORTED_ARRAY_CMP_SIZE_T(T, cmpfunc,      \
                                                      name, expmode)   \
    WX_DEFINE_SORTED_USER_EXPORTED_TYPEARRAY_CMP(T, cmpfunc, name,     \
                                                 wxBaseArraySizeT,     \
                                                 wxARRAY_EMPTY expmode)

// ----------------------------------------------------------------------------
// Some commonly used predefined arrays
// ----------------------------------------------------------------------------

WX_DEFINE_USER_EXPORTED_ARRAY_SHORT(short, wxArrayShort, class WXDLLIMPEXP_BASE);
WX_DEFINE_USER_EXPORTED_ARRAY_INT(int, wxArrayInt, class WXDLLIMPEXP_BASE);
WX_DEFINE_USER_EXPORTED_ARRAY_DOUBLE(double, wxArrayDouble, class WXDLLIMPEXP_BASE);
WX_DEFINE_USER_EXPORTED_ARRAY_LONG(long, wxArrayLong, class WXDLLIMPEXP_BASE);
WX_DEFINE_USER_EXPORTED_ARRAY_PTR(void *, wxArrayPtrVoid, class WXDLLIMPEXP_BASE);

// -----------------------------------------------------------------------------
// convenience functions: they used to be macros, hence the naming convention
// -----------------------------------------------------------------------------

// prepend all element of one array to another one; e.g. if first array contains
// elements X,Y,Z and the second contains A,B,C (in those orders), then the
// first array will be result as A,B,C,X,Y,Z
template <typename A1, typename A2>
inline void WX_PREPEND_ARRAY(A1& array, const A2& other)
{
    const size_t size = other.size();
    array.reserve(size);
    for ( size_t n = 0; n < size; n++ )
    {
        array.Insert(other[n], n);
    }
}

// append all element of one array to another one
template <typename A1, typename A2>
inline void WX_APPEND_ARRAY(A1& array, const A2& other)
{
    size_t size = other.size();
    array.reserve(size);
    for ( size_t n = 0; n < size; n++ )
    {
        array.push_back(other[n]);
    }
}

// delete all array elements
//
// NB: the class declaration of the array elements must be visible from the
//     place where you use this macro, otherwise the proper destructor may not
//     be called (a decent compiler should give a warning about it, but don't
//     count on it)!
template <typename A>
inline void WX_CLEAR_ARRAY(A& array)
{
    size_t size = array.size();
    for ( size_t n = 0; n < size; n++ )
    {
        delete array[n];
    }

    array.clear();
}

#endif // _DYNARRAY_H
