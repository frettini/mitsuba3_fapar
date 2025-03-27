#include <mitsuba/render/filter.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(filter) {
    auto e = py::enum_<FilterType>(m, "FilterType", D(FilterType))
        .def_value(FilterType, Include)
        .def_value(FilterType, Ignore);
}