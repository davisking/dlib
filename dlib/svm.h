// Copyright (C) 2007  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_ALL_SOURCE_END
#include "dlib_basic_cpp_build_tutorial.txt"
#endif

#ifndef DLIB_SVm_HEADER
#define DLIB_SVm_HEADER

#include "svm/svm_rank_trainer.h"
#include "svm/svm.h"
#include "svm/krls.h"
#include "svm/rls.h"
#include "svm/kcentroid.h"
#include "svm/kcentroid_overloads.h"
#include "svm/kkmeans.h"
#include "svm/feature_ranking.h"
#include "svm/rbf_network.h"
#include "svm/linearly_independent_subset_finder.h"
#include "svm/reduced.h"
#include "svm/rvm.h"
#include "svm/pegasos.h"
#include "svm/sparse_kernel.h"
#include "svm/null_trainer.h"
#include "svm/roc_trainer.h"
#include "svm/kernel_matrix.h"
#include "svm/empirical_kernel_map.h"
#include "svm/svm_c_linear_trainer.h"
#include "svm/svm_c_linear_dcd_trainer.h"
#include "svm/svm_c_ekm_trainer.h"
#include "svm/simplify_linear_decision_function.h"
#include "svm/krr_trainer.h"
#include "svm/sort_basis_vectors.h"
#include "svm/svm_c_trainer.h"
#include "svm/svm_one_class_trainer.h"
#include "svm/svr_trainer.h"

#include "svm/one_vs_one_decision_function.h"
#include "svm/multiclass_tools.h"
#include "svm/cross_validate_multiclass_trainer.h"
#include "svm/cross_validate_regression_trainer.h"
#include "svm/cross_validate_object_detection_trainer.h"
#include "svm/cross_validate_sequence_labeler.h"
#include "svm/cross_validate_sequence_segmenter.h"
#include "svm/cross_validate_assignment_trainer.h"

#include "svm/one_vs_all_decision_function.h"

#include "svm/structural_svm_problem.h"
#include "svm/sequence_labeler.h"
#include "svm/assignment_function.h"
#include "svm/track_association_function.h"
#include "svm/active_learning.h"
#include "svm/svr_linear_trainer.h"
#include "svm/sequence_segmenter.h"

#endif // DLIB_SVm_HEADER


