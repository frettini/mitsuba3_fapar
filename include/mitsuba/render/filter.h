#pragma once

#include <mitsuba/core/fwd.h>
#include <drjit/vcall.h>

NAMESPACE_BEGIN(mitsuba)

// Note: if Ignore is the type of filter, might be better to use boolean instead
enum class FilterType : uint32_t {
    Include = 0u, // include as usual
    Ignore = 1u // exclude from accumulation
    // Once = 2u, // should only include this once, subsequent bounces are ignored?
    // AtLeast = 3, // The path should interact at least once. 
};

MI_DECLARE_ENUM_OPERATORS(FilterType)

NAMESPACE_END(mitsuba)