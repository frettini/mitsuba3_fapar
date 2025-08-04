
#include <mitsuba/core/properties.h>
#include <mitsuba/render/phase.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT
PhaseFunction<Float, Spectrum>::PhaseFunction(const Properties &props)
    : m_flags(+PhaseFunctionFlags::Empty), m_id(props.id()) {
        m_filter = props.get<uint32_t>("filter", 0);

        dr::set_attr(this, "filter", m_filter);
    }

MI_VARIANT 
void PhaseFunction<Float, Spectrum>::traverse(TraversalCallback *callback) {
    callback->put_parameter("filter", m_filter, +ParamFlags::NonDifferentiable );
}

MI_VARIANT PhaseFunction<Float, Spectrum>::~PhaseFunction() {}

MI_IMPLEMENT_CLASS_VARIANT(PhaseFunction, Object, "phase")
MI_INSTANTIATE_CLASS(PhaseFunction)
NAMESPACE_END(mitsuba)
