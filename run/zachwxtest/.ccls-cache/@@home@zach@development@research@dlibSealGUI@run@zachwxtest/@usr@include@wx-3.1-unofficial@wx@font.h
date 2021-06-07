/////////////////////////////////////////////////////////////////////////////
// Name:        wx/font.h
// Purpose:     wxFontBase class: the interface of wxFont
// Author:      Vadim Zeitlin
// Modified by:
// Created:     20.09.99
// Copyright:   (c) wxWidgets team
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_FONT_H_BASE_
#define _WX_FONT_H_BASE_

// ----------------------------------------------------------------------------
// headers
// ----------------------------------------------------------------------------

#include "wx/defs.h"        // for wxDEFAULT &c
#include "wx/fontenc.h"     // the font encoding constants
#include "wx/gdiobj.h"      // the base class
#include "wx/gdicmn.h"      // for wxGDIObjListBase
#include "wx/math.h"        // for wxRound()

// ----------------------------------------------------------------------------
// forward declarations
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_FWD_CORE wxFont;

// ----------------------------------------------------------------------------
// font constants
// ----------------------------------------------------------------------------

// standard font families: these may be used only for the font creation, it
// doesn't make sense to query an existing font for its font family as,
// especially if the font had been created from a native font description, it
// may be unknown
enum wxFontFamily
{
    wxFONTFAMILY_DEFAULT = wxDEFAULT,
    wxFONTFAMILY_DECORATIVE = wxDECORATIVE,
    wxFONTFAMILY_ROMAN = wxROMAN,
    wxFONTFAMILY_SCRIPT = wxSCRIPT,
    wxFONTFAMILY_SWISS = wxSWISS,
    wxFONTFAMILY_MODERN = wxMODERN,
    wxFONTFAMILY_TELETYPE = wxTELETYPE,
    wxFONTFAMILY_MAX,
    wxFONTFAMILY_UNKNOWN = wxFONTFAMILY_MAX
};

// font styles
enum wxFontStyle
{
    wxFONTSTYLE_NORMAL = wxNORMAL,
    wxFONTSTYLE_ITALIC = wxITALIC,
    wxFONTSTYLE_SLANT = wxSLANT,
    wxFONTSTYLE_MAX
};

// font weights
enum wxFontWeight
{
    wxFONTWEIGHT_INVALID = 0,
    wxFONTWEIGHT_THIN = 100,
    wxFONTWEIGHT_EXTRALIGHT = 200,
    wxFONTWEIGHT_LIGHT = 300,
    wxFONTWEIGHT_NORMAL = 400,
    wxFONTWEIGHT_MEDIUM = 500,
    wxFONTWEIGHT_SEMIBOLD = 600,
    wxFONTWEIGHT_BOLD = 700,
    wxFONTWEIGHT_EXTRABOLD = 800,
    wxFONTWEIGHT_HEAVY = 900,
    wxFONTWEIGHT_EXTRAHEAVY = 1000,
    wxFONTWEIGHT_MAX = wxFONTWEIGHT_EXTRAHEAVY
};

// Symbolic font sizes as defined in CSS specification.
enum wxFontSymbolicSize
{
    wxFONTSIZE_XX_SMALL = -3,
    wxFONTSIZE_X_SMALL,
    wxFONTSIZE_SMALL,
    wxFONTSIZE_MEDIUM,
    wxFONTSIZE_LARGE,
    wxFONTSIZE_X_LARGE,
    wxFONTSIZE_XX_LARGE
};

// the font flag bits for the new font ctor accepting one combined flags word
enum wxFontFlag
{
    // no special flags: font with default weight/slant/anti-aliasing
    wxFONTFLAG_DEFAULT          = 0,

    // slant flags (default: no slant)
    wxFONTFLAG_ITALIC           = 1 << 0,
    wxFONTFLAG_SLANT            = 1 << 1,

    // weight flags (default: medium):
    wxFONTFLAG_LIGHT            = 1 << 2,
    wxFONTFLAG_BOLD             = 1 << 3,

    // anti-aliasing flag: force on or off (default: the current system default)
    wxFONTFLAG_ANTIALIASED      = 1 << 4,
    wxFONTFLAG_NOT_ANTIALIASED  = 1 << 5,

    // underlined/strikethrough flags (default: no lines)
    wxFONTFLAG_UNDERLINED       = 1 << 6,
    wxFONTFLAG_STRIKETHROUGH    = 1 << 7,

    // the mask of all currently used flags
    wxFONTFLAG_MASK = wxFONTFLAG_ITALIC             |
                      wxFONTFLAG_SLANT              |
                      wxFONTFLAG_LIGHT              |
                      wxFONTFLAG_BOLD               |
                      wxFONTFLAG_ANTIALIASED        |
                      wxFONTFLAG_NOT_ANTIALIASED    |
                      wxFONTFLAG_UNDERLINED         |
                      wxFONTFLAG_STRIKETHROUGH
};

// ----------------------------------------------------------------------------
// wxFontInfo describes a wxFont
// ----------------------------------------------------------------------------

class wxFontInfo
{
public:
    // Default ctor uses the default font size appropriate for the current
    // platform.
    wxFontInfo()
        : m_pointSize(-1.0)
        , m_pixelSize(wxDefaultSize)
    {
        Init();
    }

    // These ctors specify the font size, either in points or in pixels.
    explicit wxFontInfo(double pointSize)
        : m_pointSize(pointSize >= 0.0 ? pointSize : -1.0)
        , m_pixelSize(wxDefaultSize)
    {
        Init();
        if (!wxIsSameDouble(m_pointSize, pointSize))
        {
            wxFAIL_MSG("Invalid font point size");
        }
    }
    explicit wxFontInfo(const wxSize& pixelSize)
        : m_pointSize(-1.0)
        , m_pixelSize(pixelSize)
    {
        Init();
    }
    // Default copy ctor, assignment operator and dtor are OK

    // Setters for the various attributes. All of them return the object itself
    // so that the calls to them could be chained.
    wxFontInfo& Family(wxFontFamily family)
        { m_family = family; return *this; }
    wxFontInfo& FaceName(const wxString& faceName)
        { m_faceName = faceName; return *this; }

    wxFontInfo& Weight(int weight)
        { m_weight = weight; return *this; }
    wxFontInfo& Bold(bool bold = true)
        { return Weight(bold ? wxFONTWEIGHT_BOLD : wxFONTWEIGHT_NORMAL); }
    wxFontInfo& Light(bool light = true)
        { return Weight(light ? wxFONTWEIGHT_LIGHT : wxFONTWEIGHT_NORMAL); }

    wxFontInfo& Italic(bool italic = true)
        { SetFlag(wxFONTFLAG_ITALIC, italic); return *this; }
    wxFontInfo& Slant(bool slant = true)
        { SetFlag(wxFONTFLAG_SLANT, slant); return *this; }
    wxFontInfo& Style(wxFontStyle style)
    {
        if ( style == wxFONTSTYLE_ITALIC )
            return Italic();

        if ( style == wxFONTSTYLE_SLANT )
            return Slant();

        return *this;
    }

    wxFontInfo& AntiAliased(bool antiAliased = true)
        { SetFlag(wxFONTFLAG_ANTIALIASED, antiAliased); return *this; }
    wxFontInfo& Underlined(bool underlined = true)
        { SetFlag(wxFONTFLAG_UNDERLINED, underlined); return *this; }
    wxFontInfo& Strikethrough(bool strikethrough = true)
        { SetFlag(wxFONTFLAG_STRIKETHROUGH, strikethrough); return *this; }

    wxFontInfo& Encoding(wxFontEncoding encoding)
        { m_encoding = encoding; return *this; }

    // Set all flags at once.
    wxFontInfo& AllFlags(int flags)
    {
        m_flags = flags;

        m_weight = m_flags & wxFONTFLAG_BOLD
                        ? wxFONTWEIGHT_BOLD
                        : m_flags & wxFONTFLAG_LIGHT
                            ? wxFONTWEIGHT_LIGHT
                            : wxFONTWEIGHT_NORMAL;

        return *this;
    }

    // Accessors are mostly meant to be used by wxFont itself to extract the
    // various pieces of the font description.

    bool IsUsingSizeInPixels() const { return m_pixelSize != wxDefaultSize; }
    double GetFractionalPointSize() const { return m_pointSize; }
    int GetPointSize() const { return wxRound(m_pointSize); }
    wxSize GetPixelSize() const { return m_pixelSize; }

    // If face name is not empty, it has priority, otherwise use family.
    bool HasFaceName() const { return !m_faceName.empty(); }
    wxFontFamily GetFamily() const { return m_family; }
    const wxString& GetFaceName() const { return m_faceName; }

    wxFontStyle GetStyle() const
    {
        return m_flags & wxFONTFLAG_ITALIC
                        ? wxFONTSTYLE_ITALIC
                        : m_flags & wxFONTFLAG_SLANT
                            ? wxFONTSTYLE_SLANT
                            : wxFONTSTYLE_NORMAL;
    }

    int GetNumericWeight() const
    {
        return m_weight;
    }

    wxFontWeight GetWeight() const
    {
        return GetWeightClosestToNumericValue(m_weight);
    }

    bool IsAntiAliased() const
    {
        return (m_flags & wxFONTFLAG_ANTIALIASED) != 0;
    }

    bool IsUnderlined() const
    {
        return (m_flags & wxFONTFLAG_UNDERLINED) != 0;
    }

    bool IsStrikethrough() const
    {
        return (m_flags & wxFONTFLAG_STRIKETHROUGH) != 0;
    }

    wxFontEncoding GetEncoding() const { return m_encoding; }

    // Another helper for converting arbitrary numeric weight to the closest
    // value of wxFontWeight enum. It should be avoided in the new code (also
    // note that the function for the conversion in the other direction is
    // trivial and so is not provided, we only have GetNumericWeightOf() which
    // contains backwards compatibility hacks, but we don't need it here).
    static wxFontWeight GetWeightClosestToNumericValue(int numWeight)
    {
        wxASSERT(numWeight > 0);
        wxASSERT(numWeight <= 1000);

        // round to nearest hundredth = wxFONTWEIGHT_ constant
        int weight = ((numWeight + 50) / 100) * 100;

        if (weight < wxFONTWEIGHT_THIN)
            weight = wxFONTWEIGHT_THIN;
        if (weight > wxFONTWEIGHT_MAX)
            weight = wxFONTWEIGHT_MAX;

        return static_cast<wxFontWeight>(weight);
    }

private:
    void Init()
    {
        m_family = wxFONTFAMILY_DEFAULT;
        m_flags = wxFONTFLAG_DEFAULT;
        m_weight = wxFONTWEIGHT_NORMAL;
        m_encoding = wxFONTENCODING_DEFAULT;
    }

    // Turn on or off the given bit in m_flags depending on the value of the
    // boolean argument.
    void SetFlag(int flag, bool on)
    {
        if ( on )
            m_flags |= flag;
        else
            m_flags &= ~flag;
    }

    // The size information: if m_pixelSize is valid (!= wxDefaultSize), then
    // it is used. Otherwise m_pointSize is used, except if it is < 0, which
    // means that the platform dependent font size should be used instead.
    double m_pointSize;
    wxSize m_pixelSize;

    wxFontFamily m_family;
    wxString m_faceName;
    int m_flags;
    int m_weight;
    wxFontEncoding m_encoding;
};

// ----------------------------------------------------------------------------
// wxFontBase represents a font object
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_FWD_CORE wxNativeFontInfo;

class WXDLLIMPEXP_CORE wxFontBase : public wxGDIObject
{
public:
    /*
        derived classes should provide the following ctors:

    wxFont();
    wxFont(const wxFontInfo& info);
    wxFont(const wxString& nativeFontInfoString);
    wxFont(const wxNativeFontInfo& info);
    wxFont(int size,
           wxFontFamily family,
           wxFontStyle style,
           wxFontWeight weight,
           bool underlined = false,
           const wxString& face = wxEmptyString,
           wxFontEncoding encoding = wxFONTENCODING_DEFAULT);
    wxFont(const wxSize& pixelSize,
           wxFontFamily family,
           wxFontStyle style,
           wxFontWeight weight,
           bool underlined = false,
           const wxString& face = wxEmptyString,
           wxFontEncoding encoding = wxFONTENCODING_DEFAULT);
    */

    // creator function
    virtual ~wxFontBase();


    // from the font components
    static wxFont *New(
        int pointSize,              // size of the font in points
        wxFontFamily family,        // see wxFontFamily enum
        wxFontStyle style,          // see wxFontStyle enum
        wxFontWeight weight,        // see wxFontWeight enum
        bool underlined = false,    // not underlined by default
        const wxString& face = wxEmptyString,              // facename
        wxFontEncoding encoding = wxFONTENCODING_DEFAULT); // ISO8859-X, ...

    // from the font components
    static wxFont *New(
        const wxSize& pixelSize,    // size of the font in pixels
        wxFontFamily family,        // see wxFontFamily enum
        wxFontStyle style,          // see wxFontStyle enum
        wxFontWeight weight,        // see wxFontWeight enum
        bool underlined = false,    // not underlined by default
        const wxString& face = wxEmptyString,              // facename
        wxFontEncoding encoding = wxFONTENCODING_DEFAULT); // ISO8859-X, ...

    // from the font components but using the font flags instead of separate
    // parameters for each flag
    static wxFont *New(int pointSize,
                       wxFontFamily family,
                       int flags = wxFONTFLAG_DEFAULT,
                       const wxString& face = wxEmptyString,
                       wxFontEncoding encoding = wxFONTENCODING_DEFAULT);


    // from the font components but using the font flags instead of separate
    // parameters for each flag
    static wxFont *New(const wxSize& pixelSize,
                       wxFontFamily family,
                       int flags = wxFONTFLAG_DEFAULT,
                       const wxString& face = wxEmptyString,
                       wxFontEncoding encoding = wxFONTENCODING_DEFAULT);

    // from the (opaque) native font description object
    static wxFont *New(const wxNativeFontInfo& nativeFontDesc);

    // from the string representation of wxNativeFontInfo
    static wxFont *New(const wxString& strNativeFontDesc);

    // Load the font from the given file and return true on success or false on
    // error (an error message will be logged in this case).
#if wxUSE_PRIVATE_FONTS
    static bool AddPrivateFont(const wxString& filename);
#endif // wxUSE_PRIVATE_FONTS

    // comparison
    bool operator==(const wxFont& font) const;
    bool operator!=(const wxFont& font) const { return !(*this == font); }

    // accessors: get the font characteristics
    virtual int GetPointSize() const;
    virtual double GetFractionalPointSize() const = 0;
    virtual wxSize GetPixelSize() const;
    virtual bool IsUsingSizeInPixels() const;
    wxFontFamily GetFamily() const;
    virtual wxFontStyle GetStyle() const = 0;
    virtual int GetNumericWeight() const = 0;
    virtual bool GetUnderlined() const = 0;
    virtual bool GetStrikethrough() const { return false; }
    virtual wxString GetFaceName() const = 0;
    virtual wxFontEncoding GetEncoding() const = 0;
    virtual const wxNativeFontInfo *GetNativeFontInfo() const = 0;

    // Accessors that can be overridden in the platform-specific code but for
    // which we provide a reasonable default implementation in the base class.
    virtual wxFontWeight GetWeight() const;
    virtual bool IsFixedWidth() const;

    wxString GetNativeFontInfoDesc() const;
    wxString GetNativeFontInfoUserDesc() const;

    // change the font characteristics
    virtual void SetPointSize( int pointSize );
    virtual void SetFractionalPointSize( double pointSize ) = 0;
    virtual void SetPixelSize( const wxSize& pixelSize );
    virtual void SetFamily( wxFontFamily family ) = 0;
    virtual void SetStyle( wxFontStyle style ) = 0;
    virtual void SetNumericWeight( int weight ) = 0;

    virtual void SetUnderlined( bool underlined ) = 0;
    virtual void SetStrikethrough( bool WXUNUSED(strikethrough) ) {}
    virtual void SetEncoding(wxFontEncoding encoding) = 0;
    virtual bool SetFaceName( const wxString& faceName );
    void SetNativeFontInfo(const wxNativeFontInfo& info)
        { DoSetNativeFontInfo(info); }

    // Similarly to the accessors above, the functions in this group have a
    // reasonable default implementation in the base class.
    virtual void SetWeight( wxFontWeight weight );

    bool SetNativeFontInfo(const wxString& info);
    bool SetNativeFontInfoUserDesc(const wxString& info);

    // Symbolic font sizes support: set the font size to "large" or "very
    // small" either absolutely (i.e. compared to the default font size) or
    // relatively to the given font size.
    void SetSymbolicSize(wxFontSymbolicSize size);
    void SetSymbolicSizeRelativeTo(wxFontSymbolicSize size, int base)
    {
        SetPointSize(AdjustToSymbolicSize(size, base));
    }

    // Adjust the base size in points according to symbolic size.
    static int AdjustToSymbolicSize(wxFontSymbolicSize size, int base);


    // translate the fonts into human-readable string (i.e. GetStyleString()
    // will return "wxITALIC" for an italic font, ...)
    wxString GetFamilyString() const;
    wxString GetStyleString() const;
    wxString GetWeightString() const;

    // the default encoding is used for creating all fonts with default
    // encoding parameter
    static wxFontEncoding GetDefaultEncoding() { return ms_encodingDefault; }
    static void SetDefaultEncoding(wxFontEncoding encoding);

    // Account for legacy font weight values: if the argument is one of
    // wxNORMAL, wxLIGHT or wxBOLD, return the corresponding wxFONTWEIGHT_XXX
    // enum value. Otherwise just return it unchanged.
    static int ConvertFromLegacyWeightIfNecessary(int weight);

    // Convert between symbolic and numeric font weights. This function uses
    // ConvertFromLegacyWeightIfNecessary(), so takes legacy values into
    // account as well.
    static int GetNumericWeightOf(wxFontWeight weight);

    // this doesn't do anything and is kept for compatibility only
#if WXWIN_COMPATIBILITY_2_8
    wxDEPRECATED_INLINE(void SetNoAntiAliasing(bool no = true), wxUnusedVar(no);)
    wxDEPRECATED_INLINE(bool GetNoAntiAliasing() const, return false;)
#endif // WXWIN_COMPATIBILITY_2_8

    wxDEPRECATED_MSG("use wxFONTWEIGHT_XXX constants instead of raw values")
    void SetWeight(int weight)
        { SetWeight(static_cast<wxFontWeight>(weight)); }

    wxDEPRECATED_MSG("use wxFONTWEIGHT_XXX constants instead of wxLIGHT/wxNORMAL/wxBOLD")
    void SetWeight(wxDeprecatedGUIConstants weight)
        { SetWeight(static_cast<wxFontWeight>(weight)); }

    // from the font components
    wxDEPRECATED_MSG("use wxFONT{FAMILY,STYLE,WEIGHT}_XXX constants")
    static wxFont *New(
        int pointSize,              // size of the font in points
        int family,                 // see wxFontFamily enum
        int style,                  // see wxFontStyle enum
        int weight,                 // see wxFontWeight enum
        bool underlined = false,    // not underlined by default
        const wxString& face = wxEmptyString,              // facename
        wxFontEncoding encoding = wxFONTENCODING_DEFAULT)  // ISO8859-X, ...
        { return New(pointSize, (wxFontFamily)family, (wxFontStyle)style,
                     (wxFontWeight)weight, underlined, face, encoding); }

    // from the font components
    wxDEPRECATED_MSG("use wxFONT{FAMILY,STYLE,WEIGHT}_XXX constants")
    static wxFont *New(
        const wxSize& pixelSize,    // size of the font in pixels
        int family,                 // see wxFontFamily enum
        int style,                  // see wxFontStyle enum
        int weight,                 // see wxFontWeight enum
        bool underlined = false,    // not underlined by default
        const wxString& face = wxEmptyString,              // facename
        wxFontEncoding encoding = wxFONTENCODING_DEFAULT)  // ISO8859-X, ...
        { return New(pixelSize, (wxFontFamily)family, (wxFontStyle)style,
                     (wxFontWeight)weight, underlined, face, encoding); }


protected:
    // the function called by both overloads of SetNativeFontInfo()
    virtual void DoSetNativeFontInfo(const wxNativeFontInfo& info);

    // The function called by public GetFamily(): it can return
    // wxFONTFAMILY_UNKNOWN unlike the public method (see comment there).
    virtual wxFontFamily DoGetFamily() const = 0;


    // Helper functions to recover wxFONTSTYLE/wxFONTWEIGHT and underlined flag
    // values from flags containing a combination of wxFONTFLAG_XXX.
    static wxFontStyle GetStyleFromFlags(int flags)
    {
        return flags & wxFONTFLAG_ITALIC
                        ? wxFONTSTYLE_ITALIC
                        : flags & wxFONTFLAG_SLANT
                            ? wxFONTSTYLE_SLANT
                            : wxFONTSTYLE_NORMAL;
    }

    static wxFontWeight GetWeightFromFlags(int flags)
    {
        return flags & wxFONTFLAG_LIGHT
                        ? wxFONTWEIGHT_LIGHT
                        : flags & wxFONTFLAG_BOLD
                            ? wxFONTWEIGHT_BOLD
                            : wxFONTWEIGHT_NORMAL;
    }

    static bool GetUnderlinedFromFlags(int flags)
    {
        return (flags & wxFONTFLAG_UNDERLINED) != 0;
    }

    static bool GetStrikethroughFromFlags(int flags)
    {
        return (flags & wxFONTFLAG_STRIKETHROUGH) != 0;
    }

    // Create wxFontInfo object from the parameters passed to the legacy wxFont
    // ctor/Create() overload. This function implements the compatibility hack
    // which interprets wxDEFAULT value of size as meaning -1 and also supports
    // specifying wxNORMAL, wxLIGHT and wxBOLD as weight values.
    static wxFontInfo InfoFromLegacyParams(int pointSize,
                                           wxFontFamily family,
                                           wxFontStyle style,
                                           wxFontWeight weight,
                                           bool underlined,
                                           const wxString& face,
                                           wxFontEncoding encoding);

    static wxFontInfo InfoFromLegacyParams(const wxSize& pixelSize,
                                           wxFontFamily family,
                                           wxFontStyle style,
                                           wxFontWeight weight,
                                           bool underlined,
                                           const wxString& face,
                                           wxFontEncoding encoding);

private:
    // the currently default encoding: by default, it's the default system
    // encoding, but may be changed by the application using
    // SetDefaultEncoding() to make all subsequent fonts created without
    // specifying encoding parameter using this encoding
    static wxFontEncoding ms_encodingDefault;
};

// wxFontBase <-> wxString utilities, used by wxConfig
WXDLLIMPEXP_CORE wxString wxToString(const wxFontBase& font);
WXDLLIMPEXP_CORE bool wxFromString(const wxString& str, wxFontBase* font);


// this macro must be used in all derived wxFont classes declarations
#define wxDECLARE_COMMON_FONT_METHODS() \
    wxDEPRECATED_MSG("use wxFONTFAMILY_XXX constants") \
    void SetFamily(int family) \
        { SetFamily((wxFontFamily)family); } \
    wxDEPRECATED_MSG("use wxFONTSTYLE_XXX constants") \
    void SetStyle(int style) \
        { SetStyle((wxFontStyle)style); } \
    wxDEPRECATED_MSG("use wxFONTFAMILY_XXX constants") \
    void SetFamily(wxDeprecatedGUIConstants family) \
        { SetFamily((wxFontFamily)family); } \
    wxDEPRECATED_MSG("use wxFONTSTYLE_XXX constants") \
    void SetStyle(wxDeprecatedGUIConstants style) \
        { SetStyle((wxFontStyle)style); } \
 \
    /* functions for modifying font in place */ \
    wxFont& MakeBold(); \
    wxFont& MakeItalic(); \
    wxFont& MakeUnderlined(); \
    wxFont& MakeStrikethrough(); \
    wxFont& MakeLarger() { return Scale(1.2f); } \
    wxFont& MakeSmaller() { return Scale(1/1.2f); } \
    wxFont& Scale(float x); \
    /* functions for creating fonts based on this one */ \
    wxFont Bold() const; \
    wxFont GetBaseFont() const; \
    wxFont Italic() const; \
    wxFont Underlined() const; \
    wxFont Strikethrough() const; \
    wxFont Larger() const { return Scaled(1.2f); } \
    wxFont Smaller() const { return Scaled(1/1.2f); } \
    wxFont Scaled(float x) const

// include the real class declaration
#if defined(__WXMSW__)
    #include "wx/msw/font.h"
#elif defined(__WXMOTIF__)
    #include "wx/motif/font.h"
#elif defined(__WXGTK20__)
    #include "wx/gtk/font.h"
#elif defined(__WXGTK__)
    #include "wx/gtk1/font.h"
#elif defined(__WXX11__)
    #include "wx/x11/font.h"
#elif defined(__WXDFB__)
    #include "wx/dfb/font.h"
#elif defined(__WXMAC__)
    #include "wx/osx/font.h"
#elif defined(__WXQT__)
    #include "wx/qt/font.h"
#endif

class WXDLLIMPEXP_CORE wxFontList: public wxGDIObjListBase
{
public:
    wxFont *FindOrCreateFont(int pointSize,
                             wxFontFamily family,
                             wxFontStyle style,
                             wxFontWeight weight,
                             bool underline = false,
                             const wxString& face = wxEmptyString,
                             wxFontEncoding encoding = wxFONTENCODING_DEFAULT);

    wxDEPRECATED_MSG("use wxFONT{FAMILY,STYLE,WEIGHT}_XXX constants")
    wxFont *FindOrCreateFont(int pointSize, int family, int style, int weight,
                              bool underline = false,
                              const wxString& face = wxEmptyString,
                              wxFontEncoding encoding = wxFONTENCODING_DEFAULT)
        { return FindOrCreateFont(pointSize, (wxFontFamily)family, (wxFontStyle)style,
                                  (wxFontWeight)weight, underline, face, encoding); }

    wxFont *FindOrCreateFont(const wxFontInfo& fontInfo)
        { return FindOrCreateFont(fontInfo.GetPointSize(), fontInfo.GetFamily(),
                                  fontInfo.GetStyle(), fontInfo.GetWeight(),
                                  fontInfo.IsUnderlined(), fontInfo.GetFaceName(),
                                  fontInfo.GetEncoding()); }
};

extern WXDLLIMPEXP_DATA_CORE(wxFontList*)    wxTheFontList;


// provide comparison operators to allow code such as
//
//      if ( font.GetStyle() == wxFONTSTYLE_SLANT )
//
// to compile without warnings which it would otherwise provoke from some
// compilers as it compares elements of different enums

wxDEPRECATED_MSG("use wxFONTFAMILY_XXX constants") \
inline bool operator==(wxFontFamily s, wxDeprecatedGUIConstants t)
    { return static_cast<int>(s) == static_cast<int>(t); }
wxDEPRECATED_MSG("use wxFONTFAMILY_XXX constants") \
inline bool operator!=(wxFontFamily s, wxDeprecatedGUIConstants t)
    { return static_cast<int>(s) != static_cast<int>(t); }
wxDEPRECATED_MSG("use wxFONTSTYLE_XXX constants") \
inline bool operator==(wxFontStyle s, wxDeprecatedGUIConstants t)
    { return static_cast<int>(s) == static_cast<int>(t); }
wxDEPRECATED_MSG("use wxFONTSTYLE_XXX constants") \
inline bool operator!=(wxFontStyle s, wxDeprecatedGUIConstants t)
    { return static_cast<int>(s) != static_cast<int>(t); }
wxDEPRECATED_MSG("use wxFONTWEIGHT_XXX constants") \
inline bool operator==(wxFontWeight s, wxDeprecatedGUIConstants t)
    { return static_cast<int>(s) == static_cast<int>(t); }
wxDEPRECATED_MSG("use wxFONTWEIGHT_XXX constants") \
inline bool operator!=(wxFontWeight s, wxDeprecatedGUIConstants t)
    { return static_cast<int>(s) != static_cast<int>(t); }

#endif // _WX_FONT_H_BASE_
