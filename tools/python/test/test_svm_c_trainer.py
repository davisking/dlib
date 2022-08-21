from __future__ import division

import pytest
from random import Random
from dlib import (vectors, vector, sparse_vectors, sparse_vector, pair, array,
                  cross_validate_trainer,
                  svm_c_trainer_radial_basis,
                  svm_c_trainer_sparse_radial_basis,
                  svm_c_trainer_histogram_intersection,
                  svm_c_trainer_sparse_histogram_intersection,
                  svm_c_trainer_linear,
                  svm_c_trainer_sparse_linear,
                  rvm_trainer_radial_basis,
                  rvm_trainer_sparse_radial_basis,
                  rvm_trainer_histogram_intersection,
                  rvm_trainer_sparse_histogram_intersection,
                  rvm_trainer_linear,
                  rvm_trainer_sparse_linear)


@pytest.fixture
def training_data():
    r = Random(0)
    predictors = vectors()
    sparse_predictors = sparse_vectors()
    response = array()
    for i in range(30):
        for c in [-1, 1]:
            response.append(c)
            values = [r.random() + c * 0.5 for _ in range(3)]
            predictors.append(vector(values))
            sp = sparse_vector()
            for i, v in enumerate(values):
                sp.append(pair(i, v))
            sparse_predictors.append(sp)
    return predictors, sparse_predictors, response


@pytest.mark.parametrize('trainer, class1_accuracy, class2_accuracy', [
    (svm_c_trainer_radial_basis, 1.0, 1.0),
    (svm_c_trainer_sparse_radial_basis, 1.0, 1.0),
    (svm_c_trainer_histogram_intersection, 1.0, 1.0),
    (svm_c_trainer_sparse_histogram_intersection, 1.0, 1.0),
    (svm_c_trainer_linear, 1.0, 23 / 30),
    (svm_c_trainer_sparse_linear, 1.0, 23 / 30),
    (rvm_trainer_radial_basis, 1.0, 1.0),
    (rvm_trainer_sparse_radial_basis, 1.0, 1.0),
    (rvm_trainer_histogram_intersection, 1.0, 1.0),
    (rvm_trainer_sparse_histogram_intersection, 1.0, 1.0),
    (rvm_trainer_linear, 1.0, 0.6),
    (rvm_trainer_sparse_linear, 1.0, 0.6)
])
def test_trainers(training_data, trainer, class1_accuracy, class2_accuracy):
    predictors, sparse_predictors, response = training_data
    if 'sparse' in trainer.__name__:
        predictors = sparse_predictors
    cv = cross_validate_trainer(trainer(), predictors, response, folds=10)
    assert cv.class1_accuracy == pytest.approx(class1_accuracy)
    assert cv.class2_accuracy == pytest.approx(class2_accuracy)

    decision_function = trainer().train(predictors, response)
    assert decision_function(predictors[2]) < 0
    assert decision_function(predictors[3]) > 0
    if 'linear' in trainer.__name__:
        assert len(decision_function.weights) == 3
