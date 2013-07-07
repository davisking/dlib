#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This is an example illustrating the use of the structural SVM solver from the dlib C++
# Library.  This example will briefly introduce it and then walk through an example showing
# how to use it to create a simple multi-class classifier.  
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
    "Compute the dot product between the two vectors a and b."
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


    def make_psi(self, vector, label):
        psi = dlib.vector()
        psi.resize(self.num_dimensions)
        dims = len(vector)
        if (label == 0):
            for i in range(0,dims):
                psi[i] = vector[i]
        elif (label == 1):
            for i in range(dims,2*dims):
                psi[i] = vector[i-dims]
        else: # the label must be 2
            for i in range(2*dims,3*dims):
                psi[i] = vector[i-2*dims]
        return psi


    def get_truth_joint_feature_vector(self, idx):
        return self.make_psi(self.samples[idx], self.labels[idx])


    def separation_oracle(self, idx, current_solution):
        samp = samples[idx]
        dims = len(samp)
        scores = [0,0,0]
        # compute scores for each of the three classifiers
        scores[0] = dot(current_solution[0:dims], samp)
        scores[1] = dot(current_solution[dims:2*dims], samp)
        scores[2] = dot(current_solution[2*dims:3*dims], samp)

        # Add in the loss-augmentation
        if (labels[idx] != 0): 
            scores[0] += 1
        if (labels[idx] != 1): 
            scores[1] += 1
        if (labels[idx] != 2): 
            scores[2] += 1

        # Now figure out which classifier has the largest loss-augmented score.
        max_scoring_label = scores.index(max(scores))
        if (max_scoring_label == labels[idx]):
            loss = 0
        else:
            loss = 1

        psi = self.make_psi(samp, max_scoring_label)

        return loss,psi



samples = [[0,0,1], [0,1,0], [1,0,0]];
labels =  [0,1,2]

problem = three_class_classifier_problem(samples, labels)
weights = dlib.solve_structural_svm_problem(problem)
print weights

w1 = weights[0:3]
w2 = weights[3:6]
w3 = weights[6:9]

print "scores for class 1 sample: ", dot(w1, samples[0]), dot(w2,samples[0]), dot(w3, samples[0])
print "scores for class 2 sample: ", dot(w1, samples[1]), dot(w2,samples[1]), dot(w3, samples[1])
print "scores for class 3 sample: ", dot(w1, samples[2]), dot(w2,samples[2]), dot(w3, samples[2])

