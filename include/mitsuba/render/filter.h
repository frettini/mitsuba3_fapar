#pragma once

#include <drjit/vcall.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/interaction.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief This enum is used to specify whether a feature contributes to the
 * path interaction or not.
 */
enum class FilterType : uint32_t {
    Include = 0u,
    Ignore  = 1u
};

MI_DECLARE_ENUM_OPERATORS(FilterType)

/**
 * \brief This list of flags is used to control which filters are used for a
 * given sensor.
 */
enum class SensorFilterFlags : uint32_t {
    // No flags means no filter inclusive, meaning all interactions pass.
    None                = 0x00000,
    Depth               = 0x00001,
    BSDF                = 0x00002,
    Shape               = 0x00004,
    Phase               = 0x00008,
    // Filters not included in the BitFlag will automatically fail.
    Exclusive           = 0x80000,
    Surface = Depth | BSDF | Shape,
    All     = Depth | BSDF | Shape | Phase,
};

MI_DECLARE_ENUM_OPERATORS(SensorFilterFlags)

/**
 * \brief Depth filter
 * Tests that the current depth is within the specific minimum and maximum
 * depth.
 */
template <typename UInt32>
dr::mask_t<UInt32> depth_filter(UInt32 depth, UInt32 min_depth,
                                UInt32 max_depth, UInt32 filter_flags) {

    using Mask = typename dr::mask_t<UInt32>;

    Mask valid = false, pass = true;
    Mask include_filter = has_flag(filter_flags, SensorFilterFlags::Depth);
    Mask exclusive       = has_flag(filter_flags, SensorFilterFlags::Exclusive);

    if (dr::any_or<true>(include_filter)) {
        pass  = depth >= min_depth && depth < max_depth;
        valid = pass && include_filter;
    }

    // Sensors account invalid interactions (e.g. VoxelFlux)
    valid |= !include_filter && !exclusive;
    return valid;
}

/**
 * \brief BSDF filter
 * Test that the interaction is a surface interaction and that the bsdf at
 * interaction has a filter type equal to Include.
 */
template <typename Float, typename Spectrum>
dr::mask_t<Float> bsdf_filter(const SurfaceInteraction<Float, Spectrum> &si,
                              const MediumInteraction<Float, Spectrum> &mei,
                              dr::uint32_array_t<Float> filter_flags) {
    MI_IMPORT_TYPES(BSDFPtr)

    Mask valid = false, pass = true;

    Mask include_filter = has_flag(filter_flags, SensorFilterFlags::BSDF);
    Mask exclusive       = has_flag(filter_flags, SensorFilterFlags::Exclusive);
    Mask is_surface     = si.is_valid() && si.t < mei.t;

    if (dr::any_or<false>(!is_surface))
        return true;

    if (dr::any_or<true>(include_filter)) {
        BSDFPtr bsdf = si.bsdf();
        pass &= dr::eq(bsdf->filter(), +FilterType::Include);
        pass &= !has_flag(bsdf->flags(), BSDFFlags::Null);
        valid = pass && include_filter;
    }
    // Sensors account invalid interactions (e.g. VoxelFlux)
    valid |= !include_filter && !exclusive;
    // Check whether this is a surface interaction, return true if not
    valid |= !is_surface;
    return valid;
}

/**
 * \brief Shape filter
 * Test that the interaction is a surface interaction and that the shape at
 * interaction has a filter type equal to Include.
 */
template <typename Float, typename Spectrum>
dr::mask_t<Float> shape_filter(const SurfaceInteraction<Float, Spectrum> &si,
                               const MediumInteraction<Float, Spectrum> &mei,
                               dr::uint32_array_t<Float> filter_flags) {
    MI_IMPORT_TYPES(ShapePtr)

    Mask valid = false, pass = true;

    Mask include_filter = has_flag(filter_flags, SensorFilterFlags::Shape);
    Mask exclusive       = has_flag(filter_flags, SensorFilterFlags::Exclusive);
    Mask is_surface     = si.is_valid() && si.t < mei.t;

    if (dr::any_or<false>(!is_surface))
        return true;

    if (dr::any_or<true>(include_filter)) {
        ShapePtr shape = si.shape;
        pass &= dr::eq(shape->filter(), +FilterType::Include);
        valid = pass && include_filter;
    }

    // Sensors account invalid interactions (e.g. VoxelFlux)
    valid |= !include_filter && !exclusive;
    // Check whether this is a surface interaction, return true if not
    valid |= !is_surface;
    return valid;
}

/**
 * \brief Phase filter
 * Test that the interaction is a medium interaction and that the phase at
 * interaction has a filter type equal to Include.
 */
template <typename Float, typename Spectrum>
dr::mask_t<Float> phase_filter(const SurfaceInteraction<Float, Spectrum> &si,
                               const MediumInteraction<Float, Spectrum> &mei,
                               dr::uint32_array_t<Float> filter_flags) {
    MI_IMPORT_TYPES(PhaseFunctionPtr)

    Mask valid = false, pass = true;

    Mask include_filter = has_flag(filter_flags, SensorFilterFlags::Phase);
    Mask exclusive       = has_flag(filter_flags, SensorFilterFlags::Exclusive);
    Mask is_medium      = mei.is_valid() && mei.t < si.t;

    if (dr::any_or<false>(!is_medium))
        return true;

    if (dr::any_or<true>(include_filter)) {
        PhaseFunctionPtr phase = mei.medium->phase_function();
        pass &= dr::eq(phase->filter(), +FilterType::Include);
        valid = pass && include_filter;
    }

    // Sensors account invalid interactions (e.g. VoxelFlux)
    valid |= !include_filter && !exclusive;
    // Check whether this is a medium interaction, return true if not
    valid |= !is_medium;
    return valid;
}

NAMESPACE_END(mitsuba)