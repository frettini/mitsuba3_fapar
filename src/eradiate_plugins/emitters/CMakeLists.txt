set(MI_PLUGIN_PREFIX "emitters")

add_plugin(astroobject astroobject.cpp)
add_plugin(directionaltarget directionaltarget.cpp)
add_plugin(directionalperiodic directionalperiodic.cpp)
add_plugin(isotropicperiodic isotropicperiodic.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
