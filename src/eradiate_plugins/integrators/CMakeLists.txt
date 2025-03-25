set(MI_PLUGIN_PREFIX "integrators")

add_plugin(piecewise_volpath piecewise_volpath.cpp)
add_plugin(paccumulator paccumulator.cpp)
add_plugin(fpath fpath.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
