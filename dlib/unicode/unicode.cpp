// Copyright (C) 2008 Keita Mochizuki, Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_UNICODe_CPp_
#define DLIB_UNICODe_CPp_
#include "unicode.h"
#include <cwchar>
#include "../string.h"
#include <vector>

namespace dlib
{

    void unichar_to_surrogate_pair(unichar input, unichar &first, unichar &second)
    {
        using namespace unicode_helpers;
        first = ((input - SMP_TOP) >> VALID_BITS) | SURROGATE_FIRST_TOP;
        second = (input & SURROGATE_CLEARING_MASK) | SURROGATE_SECOND_TOP;
    }

    template <int N> struct ustring2wstr
    {
    };

    // for the environment of sizeof(wchar_t) == 2 (i.e. Win32)
    template <> struct ustring2wstr<2>
    {
        wchar_t *wstr;
        size_t wlen;
        ustring2wstr(const ustring &src){
            wlen = 0;
            for (size_t i = 0; i < src.length(); ++i)
            {
                if (src[i] < unicode_helpers::SMP_TOP) wlen++;
                else wlen += 2;
            }
            wstr = new wchar_t[wlen+1];
            wstr[wlen] = L'\0';

            size_t wi = 0;
            for (size_t i = 0; i < src.length(); ++i)
            {
                if (src[i] < unicode_helpers::SMP_TOP)
                {
                    wstr[wi++] = (wchar_t)src[i];
                }
                else
                {
                    unichar high, low;
                    unichar_to_surrogate_pair(src[i], high, low);
                    wstr[wi++] = (wchar_t)high;
                    wstr[wi++] = (wchar_t)low;
                }
            }
        }
        ~ustring2wstr()
        {
            delete[] wstr;
        }
    };

    // for the environment of sizeof(wchar_t) == 4 (i.e. Unix gcc)
    template <> struct ustring2wstr<4>
    {
        const wchar_t *wstr;
        size_t wlen;
        ustring2wstr(const ustring &src){
            wstr = (const wchar_t *)(src.c_str());
            wlen = src.size();
        }
    };

// ----------------------------------------------------------------------------------------

    const ustring convert_utf8_to_utf32(const std::string& str)
    {
        return convert_to_utf32<char>(str);
    }

// ----------------------------------------------------------------------------------------

    const ustring convert_wstring_to_utf32(const std::wstring& str)
    {
        return convert_to_utf32<wchar_t>(str);
    }

// ----------------------------------------------------------------------------------------

    const std::wstring convert_utf32_to_wstring(const ustring &src)
    {
        ustring2wstr<sizeof(wchar_t)> conv(src);
        std::wstring dest(conv.wstr);
        return dest;
    }

// ----------------------------------------------------------------------------------------

    const std::wstring convert_mbstring_to_wstring(const std::string &src)
    {
        // Compute dst length
        std::mbstate_t st{};
        const char* p = src.c_str();
        size_t n = std::mbsrtowcs(nullptr, &p, 0, &st);
        if (n == static_cast<size_t>(-1)) throw std::runtime_error("Invalid multibyte sequence / wrong locale");

        // Convert
        std::wstring out(n, L'\0');
        st = std::mbstate_t{};
        n  = std::mbsrtowcs(&out[0], &p, out.size(), &st);
        if (n == static_cast<size_t>(-1)) throw std::runtime_error("Conversion failed");
        return out;
    }

// ----------------------------------------------------------------------------------------

    const std::string convert_wstring_to_mbstring(const std::wstring &src)
    {
        std::mbstate_t st{};
        const wchar_t* p = src.c_str();

        // Compute length
        size_t n = std::wcsrtombs(nullptr, &p, 0, &st);
        if (n == static_cast<std::size_t>(-1)) throw std::runtime_error("Invalid wide sequence / locale mismatch");

        // Convert
        std::string out(n, '\0');
        st = std::mbstate_t{};
        n  = std::wcsrtombs(out.data(), &p, out.size(), &st);
        if (n == static_cast<std::size_t>(-1)) throw std::runtime_error("Conversion failed");
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_UNICODe_CPp_

