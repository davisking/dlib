// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TEST_FOR_ODR_VIOLATIONS_H_
#define DLIB_TEST_FOR_ODR_VIOLATIONS_H_

#include "../assert.h"
#include "../config.h"

extern "C"
{
// =========================>>> WHY YOU ARE GETTING AN ERROR HERE <<<=========================
// The point of this block of code is to cause a link time error that will prevent a user
// from compiling part of their application with DLIB_ASSERT enabled and part with it
// disabled since doing that would be a violation of C++'s one definition rule.  So if you
// are getting an error here then you are either not enabling DLIB_ASSERT consistently
// (e.g. by compiling part of your program in a debug mode and part in a release mode) or
// you have simply forgotten to compile dlib/all/source.cpp into your application.
// =========================>>> WHY YOU ARE GETTING AN ERROR HERE <<<=========================
#ifdef ENABLE_ASSERTS
    extern int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1;
    inline int dlib_check_consistent_assert_usage() { USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1 = 0; return 0; }
#else
    extern int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_;
    inline int dlib_check_consistent_assert_usage() { USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_ = 0; return 0; }
#endif
    const int DLIB_NO_WARN_UNUSED dlib_check_assert_helper_variable = dlib_check_consistent_assert_usage();



// The point of this block of code is to cause a link time error if someone builds dlib via
// cmake as a separately installable library, and therefore generates a dlib/config.h from
// cmake, but then proceeds to use the default unconfigured dlib/config.h from version
// control.  It should be obvious why this is bad, if it isn't you need to read a book
// about C++.  Moreover, it can only happen if someone manually copies files around and
// messes things up.  If instead they run `make install` or `cmake --build .  --target
// install` things will be setup correctly, which is what they should do.  To summarize: DO
// NOT BUILD A STANDALONE DLIB AND THEN GO CHERRY PICKING FILES FROM THE BUILD FOLDER AND
// MIXING THEM WITH THE SOURCE FROM GITHUB.  USE CMAKE'S INSTALL SCRIPTS TO INSTALL DLIB.
// Or even better, don't install dlib at all and instead build your program as shown in
// examples/CMakeLists.txt
#if defined(DLIB_NOT_CONFIGURED) && !defined(DLIB__CMAKE_GENERATED_A_CONFIG_H_FILE)
    extern int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_2;
    inline int dlib_check_consistent_config_h_usage() { USER_ERROR__inconsistent_build_configuration__see_dlib_faq_2 = 0; return 0; }
#else
    inline int dlib_check_consistent_config_h_usage() { return 0; }
#endif
    const int DLIB_NO_WARN_UNUSED dlib_check_not_configured_helper_variable = dlib_check_consistent_config_h_usage();

}




#endif // DLIB_TEST_FOR_ODR_VIOLATIONS_H_

