message(STATUS "Building Eradiate plugins")

# Plugins
add_subdirectory(bsdfs)
add_subdirectory(emitters)
add_subdirectory(films)
add_subdirectory(integrators)
add_subdirectory(media)
add_subdirectory(phase)
add_subdirectory(sensors)
add_subdirectory(shapes)
add_subdirectory(volumes)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
