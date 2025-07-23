set(MI_PLUGIN_PREFIX "emitters")

add_plugin(astroobject astroobject.cpp)
add_plugin(directionaltarget directionaltarget.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
