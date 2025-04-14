set(MI_PLUGIN_PREFIX "films")

add_plugin(volfilm volfilm.cpp)
add_plugin(tensorfilm tensorfilm.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
