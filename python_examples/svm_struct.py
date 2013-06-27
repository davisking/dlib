#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# 
#
# COMPILING THE DLIB PYTHON INTERFACE
#   Dlib comes with a compiled python interface for python 2.7 on MS Windows.  If
#   you are using another python version or operating system then you need to
#   compile the dlib python interface before you can use this file.  To do this,
#   run compile_dlib_python_module.bat.  This should work on any operating system
#   so long as you have CMake and boost-python installed.  On Ubuntu, this can be
#   done easily by running the command:  sudo apt-get install libboost-python-dev cmake


import dlib

def dot(a, b):
    return sum(i*j for i,j in zip(a,b))


class three_class_classifier_problem:
    C = 10
    be_verbose = True
    epsilon = 0.0001 

    def __init__(self, samples, labels):
        self.num_samples = len(samples)
        self.num_dimensions = len(samples[0])*3
        self.samples = samples
        self.labels = labels

    def make_psi(self, psi, vector, label):
        psi.resize(self.num_dimensions)
        dims = len(vector)
        if (label == 1):
            for i in range(0,dims):
                psi[i] = vector[i]
        elif (label == 2):
            for i in range(dims,2*dims):
                psi[i] = vector[i-dims]
        else:
            for i in range(2*dims,3*dims):
                psi[i] = vector[i-2*dims]


    def get_truth_joint_feature_vector(self, idx, psi):
        self.make_psi(psi, self.samples[idx], self.labels[idx])

    def separation_oracle(self, idx, current_solution, psi):
        samp = samples[idx]
        dims = len(samp)
        scores = [0,0,0]
        # compute scores for each of the three classifiers
        scores[0] = dot(current_solution[0:dims], samp)
        scores[1] = dot(current_solution[dims:2*dims], samp)
        scores[2] = dot(current_solution[2*dims:3*dims], samp)

        # Add in the loss-augmentation
        if (labels[idx] != 1): 
            scores[0] += 1
        if (labels[idx] != 2): 
            scores[1] += 1
        if (labels[idx] != 3): 
            scores[2] += 1


        # Now figure out which classifier has the largest loss-augmented score.
        max_scoring_label = scores.index(max(scores))+1
        if (max_scoring_label == labels[idx]):
            loss = 0
        else:
            loss = 1

        self.make_psi(psi, samp, max_scoring_label)

        return loss



samples = [ [0,0,1], [0,1,0], [1,0,0]];
labels = [1, 2, 3]

problem = three_class_classifier_problem(samples, labels)
weights = dlib.solve_structural_svm_problem(problem)
print weights

