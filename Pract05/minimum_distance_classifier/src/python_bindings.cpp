#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "classifier.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mdc_core, m) {
    m.doc() = "Minimum Distance Classifier with C++/CUDA backend";
    
    py::class_<mdc::MinimumDistanceClassifier>(m, "MinimumDistanceClassifier")
        .def(py::init<bool>(), 
             py::arg("use_cuda") = false,
             "Constructor\n\n"
             "Parameters:\n"
             "  use_cuda (bool): Enable CUDA acceleration")
        
        .def("fit", 
             &mdc::MinimumDistanceClassifier::fit,
             py::arg("X"), 
             py::arg("y"),
             "Train the classifier\n\n"
             "Parameters:\n"
             "  X: list of lists, shape (n_samples, n_features)\n"
             "  y: list of ints, shape (n_samples,)")
        
        .def("predict", 
             &mdc::MinimumDistanceClassifier::predict,
             py::arg("X"),
             "Predict class labels\n\n"
             "Parameters:\n"
             "  X: list of lists, shape (n_samples, n_features)\n"
             "Returns:\n"
             "  list of ints, shape (n_samples,)")
        
        .def("get_centroids", 
             &mdc::MinimumDistanceClassifier::get_centroids,
             "Get computed centroids\n\n"
             "Returns:\n"
             "  list of lists, shape (n_classes, n_features)")
        
        .def("is_using_cuda", 
             &mdc::MinimumDistanceClassifier::is_using_cuda,
             "Check if CUDA is being used")
        
        .def("is_fitted", 
             &mdc::MinimumDistanceClassifier::is_fitted,
             "Check if model has been trained")
        
        .def("get_n_classes", 
             &mdc::MinimumDistanceClassifier::get_n_classes,
             "Get number of classes")
        
        .def("get_n_features", 
             &mdc::MinimumDistanceClassifier::get_n_features,
             "Get number of features");
}
