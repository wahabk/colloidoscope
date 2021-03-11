#include "../extern/pybind11/include/pybind11/pybind11.h" // <pybind11/pybind11.h>
#include "funcs.hpp"

namespace py = pybind11;

PYBIND11_PLUGIN(wrap) {
    py::module m("wrap", "pybind11 paddy wrapper");
    m.def("pycrush", &pycrush, "Paddy monte-carlo crusher, returns x,y,z",
          "myFinalEta"_a=0.31);
    return m.ptr();
}

PYBIND11_MODULE(wrap, m) {
    // optional module docstring
    m.doc() = "pybind11 paddy wrapper";

    // define function
    m.def("pycrush", &pycrush, "Paddy monte-carlo crusher, returns x,y,z");


}