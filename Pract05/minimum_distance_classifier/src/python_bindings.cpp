#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "classifier.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mdc_core, m){
  m.doc() = "Python bindings for Minimum Distance Classifier (C++/CUDA)";

  py::class_<mdc::MinimumDistanceClassifier>(m, "MinimumDistanceClassifier")
    // constructor call
    .def(py::init<bool>(), py::arg("use_cuda") = true)

    // fit method call
    .def("fit", &mdc::MinimumDistanceClassifier::fit,
         "Fit the model using training data X  and labels y",
         py::arg("X"), py::arg("y"))

    // predict method call
    .def("predict", &mdc::MinimumDistanceClassifier::predict,
         "Predict class labels for samples in X (float)",
         py::arg("X"))

    // get centroids metho call
    .def("get_centroids", &mdc::MinimumDistanceClassifier::get_centroids,
         "Get the computed centroids after fitting");
}
