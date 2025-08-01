#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>

/**!

.. _phase-rayleigh:

Rayleigh phase function (:monosp:`rayleigh`)
-----------------------------------------------

Scattering by particles that are much smaller than the wavelength
of light (e.g. individual molecules in the atmosphere) is well-approximated
by the Rayleigh phase function. This plugin implements an unpolarized
version of this scattering model (*i.e.* the effects of polarization are
ignored). This plugin is useful for simulating scattering in planetary
atmospheres.

This model has no parameters.

.. tabs::
    .. code-tab:: xml

        <phase type="rayleigh" />

    .. code-tab:: python

        'type': 'rayleigh'

*/

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class RayleighPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume)

    RayleighPhaseFunction(const Properties &props) : Base(props) {
        if constexpr (is_polarized_v<Spectrum>)
            Log(Warn,
                "Polarized version of Rayleigh phase function not implemented, "
                "falling back to scalar version");

        m_depolarization = props.volume<Volume>("depolarization", 0.f);

        if (m_depolarization->max() >= 1.f)
            Log(Error, "Depolarization factor must be in [0, 1[");

        m_flags = +PhaseFunctionFlags::Anisotropic;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("depolarization", m_depolarization.get(),
                             +ParamFlags::Differentiable);
    }

    MI_INLINE Spectrum eval_rayleigh(Float cos_theta, Spectrum rho) const {

        Spectrum r1 = (1.f - rho) / (1.f + rho / 2.f),
                 r2 = (1.f + rho) / (1.f - rho);

        return (3.f / 16.f) * dr::InvPi<Float> * r1 * (r2 + dr::sqr(cos_theta));
    }

    MI_INLINE Float eval_rayleigh_pdf(Float cos_theta) const {
        return (3.f / 16.f) * dr::InvPi<Float> * (1.f + dr::sqr(cos_theta));
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext & /* ctx */,
                                                 const MediumInteraction3f &mi, 
                                                 Float /* sample1 */,
                                                 const Point2f &sample, 
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);
        UnpolarizedSpectrum rho = m_depolarization->eval(mi);

        Float z   = 2.f * (2.f * sample.x() - 1.f);
        Float tmp = dr::sqrt(dr::sqr(z) + 1.f);
        Float A   = dr::cbrt(z + tmp);
        Float B   = dr::cbrt(z - tmp);
        Float cos_theta = A + B;
        Float sin_theta = dr::safe_sqrt(1.0f - dr::sqr(cos_theta));
        auto [sin_phi, cos_phi] = dr::sincos(dr::TwoPi<Float> * sample.y());

        auto wo = Vector3f{ sin_theta * cos_phi, sin_theta * sin_phi, cos_theta };

        wo = mi.to_world(wo);
        Float pdf       = eval_rayleigh_pdf(-cos_theta);
        Spectrum weight = eval_rayleigh(-cos_theta, rho) / pdf;
        return { wo, weight, pdf };
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext & /* ctx */,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        Float cos_theta         = dot(wo, mi.wi);
        UnpolarizedSpectrum rho = m_depolarization->eval(mi);

        Spectrum phase = eval_rayleigh(cos_theta, rho);
        Float pdf      = eval_rayleigh_pdf(cos_theta);
        return { phase, pdf };
    }

    std::string to_string() const override { return "RayleighPhaseFunction[]"; }

    MI_DECLARE_CLASS()
private:
    ref<Volume> m_depolarization;
};

MI_IMPLEMENT_CLASS_VARIANT(RayleighPhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(RayleighPhaseFunction, "Rayleigh phase function")
NAMESPACE_END(mitsuba)
