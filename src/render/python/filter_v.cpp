#include <mitsuba/render/filter.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(filter) {
    MI_PY_IMPORT_TYPES()
    m.def("depth_filter",
        &depth_filter<UInt32>,
        "depth"_a, "min_depth"_a, "max_depth"_a, "filter_flags"_a, 
        D(depth_filter))
    .def("bsdf_filter",
        &bsdf_filter<Float, Spectrum>,
        "si"_a, "mei"_a, "filter_flags"_a, D(bsdf_filter))
    .def("shape_filter",
        &shape_filter<Float, Spectrum>,
        "si"_a, "mei"_a, "filter_flags"_a, D(bsdf_filter))
    .def("phase_filter",
        &phase_filter<Float, Spectrum>,
        "si"_a, "mei"_a, "filter_flags"_a, D(phase_filter));
}