#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/bsdf.h>
#include <drjit/vcall.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief This enum is used to specify whether a feature contributes to the
 * path interaction or not.
 */
enum class FilterType : uint32_t {
    Include = 0u, // include as usual
    Ignore = 1u // exclude from accumulation
    // Once = 2u, // should only include this once, subsequent bounces are ignored?
    // AtLeast = 3, // The path should interact at least once. 
};

MI_DECLARE_ENUM_OPERATORS(FilterType)

/**
 * \brief This list of flags is used to control which filters are used for a
 * given sensor.
 */
enum class SensorFilterFlags : uint32_t {
    // No flags set (default value)
    None                = 0x00000,
    Depth               = 0x00001,
    BSDF                = 0x00002,
    Shape               = 0x00004,
    Phase               = 0x00008,
    Surface             = Depth | BSDF | Shape,
    All                 = Depth | BSDF | Shape | Phase,
};

MI_DECLARE_ENUM_OPERATORS(SensorFilterFlags)

template<typename UInt32>
dr::mask_t<UInt32> depth_filter(
    UInt32 depth, 
    UInt32 min_depth, 
    UInt32 max_depth, 
    UInt32 filter_flags
){
    // filtered value
    dr::mask_t<UInt32> pass = depth >= min_depth && depth < max_depth;
    return pass || !has_flag(filter_flags, SensorFilterFlags::Depth);
}

template<typename Float, typename Spectrum>
dr::mask_t<Float> bsdf_filter(
    const SurfaceInteraction<Float, Spectrum>& si,
    dr::uint32_array_t<Float> filter_flags
) {
    MI_IMPORT_TYPES(BSDFPtr)

    Mask pass(true);
    if (dr::none_or<false>(si.is_valid()))
        return pass;
    BSDFPtr bsdf = si.bsdf();
    pass &= dr::eq(bsdf->filter(), +FilterType::Include);
    pass &= !has_flag(bsdf->flags(), BSDFFlags::Null);
    return pass || !has_flag(filter_flags, SensorFilterFlags::BSDF);
}

template<typename Float, typename Spectrum>
dr::mask_t<Float> shape_filter(
    const SurfaceInteraction<Float, Spectrum>& si,
    dr::uint32_array_t<Float> filter_flags
) {
    MI_IMPORT_TYPES(ShapePtr)

    Mask pass(true);
    if (dr::none_or<false>(si.is_valid()))
        return pass;
    ShapePtr shape = si.shape();
    pass &= dr::eq(shape->filter(), +FilterType::Include);
    return pass || !has_flag(filter_flags, SensorFilterFlags::Shape);
}

template<typename Float, typename Spectrum>
dr::mask_t<Float> phase_filter(
    const MediumInteraction<Float, Spectrum>& mei,
    dr::uint32_array_t<Float> filter_flags
) {
    MI_IMPORT_TYPES(PhasePtr)

    Mask pass(true);
    if (dr::none_or<false>(mei.is_valid()))
        return pass;
    PhasePtr shape = mei.medium->phase_function();
    pass &= dr::eq(shape->filter(), +FilterType::Include);
    return pass || !has_flag(filter_flags, SensorFilterFlags::Phase);
}


NAMESPACE_END(mitsuba)