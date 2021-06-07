/////////////////////////////////////////////////////////////////////////////
// Name:        wx/memory.h
// Purpose:     Memory operations
// Author:      Arthur Seaton, Julian Smart
// Modified by:
// Created:     29/01/98
// Copyright:   (c) 1998 Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_MEMORY_H_
#define _WX_MEMORY_H_

#include "wx/defs.h"
#include "wx/string.h"
#include "wx/msgout.h"

#if wxUSE_MEMORY_TRACING || wxUSE_DEBUG_CONTEXT

#include <stddef.h>

WXDLLIMPEXP_BASE void * wxDebugAlloc(size_t size, wxChar * fileName, int lineNum, bool isObject, bool isVect = false);
WXDLLIMPEXP_BASE void wxDebugFree(void * buf, bool isVect = false);

//**********************************************************************************
/*
  The global operator new used for everything apart from getting
  dynamic storage within this function itself.
*/

// We'll only do malloc and free for the moment: leave the interesting
// stuff for the wxObject versions.


#if wxUSE_GLOBAL_MEMORY_OPERATORS

// Undefine temporarily (new is #defined in object.h) because we want to
// declare some new operators.
#ifdef new
    #undef new
#endif

#if defined(__SUNCC__)
    #define wxUSE_ARRAY_MEMORY_OPERATORS 0
#elif defined (__SGI_CC_)
    // only supported by -n32 compilers
    #ifndef __EDG_ABI_COMPATIBILITY_VERSION
        #define wxUSE_ARRAY_MEMORY_OPERATORS 0
    #endif
#else
    #define wxUSE_ARRAY_MEMORY_OPERATORS 1
#endif

// devik 2000-8-29: All new/delete ops are now inline because they can't
// be marked as dllexport/dllimport. It then leads to weird bugs when
// used on MSW as DLL
#if defined(__WINDOWS__) && (defined(WXUSINGDLL) || defined(WXMAKINGDLL_BASE))
inline void * operator new (size_t size, wxChar * fileName, int lineNum)
{
    return wxDebugAlloc(size, fileName, lineNum, false, false);
}

inline void * operator new (size_t size)
{
    return wxDebugAlloc(size, NULL, 0, false);
}

inline void operator delete (void * buf)
{
    wxDebugFree(buf, false);
}

#if wxUSE_ARRAY_MEMORY_OPERATORS
inline void * operator new[] (size_t size)
{
    return wxDebugAlloc(size, NULL, 0, false, true);
}

inline void * operator new[] (size_t size, wxChar * fileName, int lineNum)
{
    return wxDebugAlloc(size, fileName, lineNum, false, true);
}

inline void operator delete[] (void * buf)
{
    wxDebugFree(buf, true);
}
#endif // wxUSE_ARRAY_MEMORY_OPERATORS

#else

void * operator new (size_t size, wxChar * fileName, int lineNum);

void * operator new (size_t size);

void operator delete (void * buf);

#if wxUSE_ARRAY_MEMORY_OPERATORS
void * operator new[] (size_t size);

void * operator new[] (size_t size, wxChar * fileName, int lineNum);

void operator delete[] (void * buf);
#endif // wxUSE_ARRAY_MEMORY_OPERATORS
#endif // defined(__WINDOWS__) && (defined(WXUSINGDLL) || defined(WXMAKINGDLL_BASE))

#if defined(__VISUALC__)
inline void operator delete(void* pData, wxChar* /* fileName */, int /* lineNum */)
{
    wxDebugFree(pData, false);
}
inline void operator delete[](void* pData, wxChar* /* fileName */, int /* lineNum */)
{
    wxDebugFree(pData, true);
}
#endif // __VISUALC__
#endif // wxUSE_GLOBAL_MEMORY_OPERATORS

//**********************************************************************************

typedef unsigned int wxMarkerType;

/*
  Define the struct which will be placed at the start of all dynamically
  allocated memory.
*/

class WXDLLIMPEXP_BASE wxMemStruct {

friend class WXDLLIMPEXP_FWD_BASE wxDebugContext; // access to the m_next pointer for list traversal.

public:
public:
    int AssertList ();

    size_t RequestSize () { return m_reqSize; }
    wxMarkerType Marker () { return m_firstMarker; }

    // When an object is deleted we set the id slot to a specific value.
    inline void SetDeleted ();
    inline int IsDeleted ();

    int Append ();
    int Unlink ();

    // Used to determine if the object is really a wxMemStruct.
    // Not a foolproof test by any means, but better than none I hope!
    int AssertIt ();

    // Do all validation on a node.
    int ValidateNode ();

    // Check the integrity of a node and of the list, node by node.
    int CheckBlock ();
    int CheckAllPrevious ();

    // Print a single node.
    void PrintNode ();

    // Called when the memory linking functions get an error.
    void ErrorMsg (const char *);
    void ErrorMsg ();

    inline void *GetActualData(void) const { return m_actualData; }

    void Dump(void);

public:
    // Check for underwriting. There are 2 of these checks. This one
    // inside the struct and another right after the struct.
    wxMarkerType        m_firstMarker;

    // File name and line number are from cpp.
    wxChar*             m_fileName;
    int                 m_lineNum;

    // The amount of memory requested by the caller.
    size_t              m_reqSize;

    // Used to try to verify that we really are dealing with an object
    // of the required class. Can be 1 of 2 values these indicating a valid
    // wxMemStruct object, or a deleted wxMemStruct object.
    wxMarkerType        m_id;

    wxMemStruct *       m_prev;
    wxMemStruct *       m_next;

    void *              m_actualData;
    bool                m_isObject;
};


typedef void (wxMemStruct::*PmSFV) ();

// Type of the app function that can be installed and called at wxWidgets shutdown
// (after all other registered files with global destructors have been closed down).
typedef void (*wxShutdownNotifyFunction)();

/*
  Debugging class. This will only have a single instance, but it's
  a reasonable way to keep everything together and to make this
  available for change if needed by someone else.
  A lot of this stuff would be better off within the wxMemStruct class, but
  it's stuff which we need to access at times when there is no wxMemStruct
  object so we use this class instead. Think of it as a collection of
  globals which have to do with the wxMemStruct class.
*/

class WXDLLIMPEXP_BASE wxDebugContext {

protected:
    // Used to set alignment for markers.
    static size_t CalcAlignment ();

    // Returns the amount of padding needed after something of the given
    // size. This is so that when we cast pointers backwards and forwards
    // the pointer value will be valid for a wxMarkerType.
    static size_t GetPadding (size_t size) ;

    // Traverse the list.
    static void TraverseList (PmSFV, wxMemStruct *from = NULL);

    static int debugLevel;
    static bool debugOn;

    static int m_balign;            // byte alignment
    static int m_balignmask;        // mask for performing byte alignment
public:
    // Set a checkpoint to dump only the memory from
    // a given point
    static wxMemStruct *checkPoint;

    wxDebugContext(void);
    ~wxDebugContext(void);

    static int GetLevel(void) { return debugLevel; }
    static void SetLevel(int level) { debugLevel = level; }

    static bool GetDebugMode(void) { return debugOn; }
    static void SetDebugMode(bool flag) { debugOn = flag; }

    static void SetCheckpoint(bool all = false);
    static wxMemStruct *GetCheckpoint(void) { return checkPoint; }

    // Calculated from the request size and any padding needed
    // before the final marker.
    static size_t PaddedSize (size_t reqSize);

    // Calc the total amount of space we need from the system
    // to satisfy a caller request. This includes all padding.
    static size_t TotSize (size_t reqSize);

    // Return valid pointers to offsets within the allocated memory.
    static char * StructPos (const char * buf);
    static char * MidMarkerPos (const char * buf);
    static char * CallerMemPos (const char * buf);
    static char * EndMarkerPos (const char * buf, size_t size);

    // Given a pointer to the start of the caller requested area
    // return a pointer to the start of the entire alloc\'d buffer.
    static char * StartPos (const char * caller);

    // Access to the list.
    static wxMemStruct * GetHead () { return m_head; }
    static wxMemStruct * GetTail () { return m_tail; }

    // Set the list sentinels.
    static wxMemStruct * SetHead (wxMemStruct * st) { return (m_head = st); }
    static wxMemStruct * SetTail (wxMemStruct * st) { return (m_tail = st); }

    // If this is set then every new operation checks the validity
    // of the all previous nodes in the list.
    static bool GetCheckPrevious () { return m_checkPrevious; }
    static void SetCheckPrevious (bool value) { m_checkPrevious = value; }

    // Checks all nodes, or all nodes if checkAll is true
    static int Check(bool checkAll = false);

    // Print out the list of wxMemStruct nodes.
    static bool PrintList(void);

    // Dump objects
    static bool Dump(void);

    // Print statistics
    static bool PrintStatistics(bool detailed = true);

    // Print out the classes in the application.
    static bool PrintClasses(void);

    // Count the number of non-wxDebugContext-related objects
    // that are outstanding
    static int CountObjectsLeft(bool sinceCheckpoint = false);

    // This function is used to output the dump
    static void OutputDumpLine(const wxChar *szFormat, ...);

    static void SetShutdownNotifyFunction(wxShutdownNotifyFunction shutdownFn);

private:
    // Store these here to allow access to the list without
    // needing to have a wxMemStruct object.
    static wxMemStruct*         m_head;
    static wxMemStruct*         m_tail;

    // Set to false if we're not checking all previous nodes when
    // we do a new. Set to true when we are.
    static bool                 m_checkPrevious;

    // Holds a pointer to an optional application function to call at shutdown.
    static wxShutdownNotifyFunction sm_shutdownFn;

    // Have to access our shutdown hook
    friend class wxDebugContextDumpDelayCounter;
};

// Final cleanup (e.g. deleting the log object and doing memory leak checking)
// will be delayed until all wxDebugContextDumpDelayCounter objects have been
// destructed. Adding one wxDebugContextDumpDelayCounter per file will delay
// memory leak checking until after destructing all global objects.

class WXDLLIMPEXP_BASE wxDebugContextDumpDelayCounter
{
public:
    wxDebugContextDumpDelayCounter();
    ~wxDebugContextDumpDelayCounter();

private:
    void DoDump();
    static int sm_count;
};

// make leak dump after all globals have been destructed
static wxDebugContextDumpDelayCounter wxDebugContextDumpDelayCounter_File;
#define WXDEBUG_DUMPDELAYCOUNTER \
    static wxDebugContextDumpDelayCounter wxDebugContextDumpDelayCounter_Extra;

// Output a debug message, in a system dependent fashion.
void WXDLLIMPEXP_BASE wxTrace(const wxChar *fmt ...) WX_ATTRIBUTE_PRINTF_1;
void WXDLLIMPEXP_BASE wxTraceLevel(int level, const wxChar *fmt ...) WX_ATTRIBUTE_PRINTF_2;

#define WXTRACE wxTrace
#define WXTRACELEVEL wxTraceLevel

#else // wxUSE_MEMORY_TRACING) || wxUSE_DEBUG_CONTEXT

#define WXDEBUG_DUMPDELAYCOUNTER

#define wxTrace(fmt)
#define wxTraceLevel(l, fmt)

#define WXTRACE true ? (void)0 : wxTrace
#define WXTRACELEVEL true ? (void)0 : wxTraceLevel

#endif // wxUSE_MEMORY_TRACING) || wxUSE_DEBUG_CONTEXT

#endif // _WX_MEMORY_H_
