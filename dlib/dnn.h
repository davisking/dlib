// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_
#define DLIB_DNn_

// DNN module uses template-based network declaration that leads to very long
// type names. Visual Studio will produce Warning C4503 in such cases
#ifdef _MSC_VER
#   pragma warning( disable: 4503 )
#endif

#include "dnn/tensor.h"
#include "dnn/input.h"
#include "dnn/layers.h"
#include "dnn/loss.h"
#include "dnn/core.h"
#include "dnn/solvers.h"
#include "dnn/trainer.h"
#include "dnn/cpu_dlib.h"
#include "dnn/tensor_tools.h"
#include "dnn/utilities.h"

#endif // DLIB_DNn_


