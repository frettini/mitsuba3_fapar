#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/mueller.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>

/**!

.. _phase-rayleigh_polarized:

Rayleigh phase function (:monosp:`rayleigh_polarized`)
------------------------------------------------------

.. pluginparameters::

 * - depolarization
   - |float|
   - Depolarization factor, the ratio of intensities parallel and perpendicular
     to the plane of scattering for light scattered at 90°.
   - |exposed| |differentiable|


Scattering by particles that are much smaller than the wavelength
of light (e.g. individual molecules in the atmosphere) is well-approximated
by the Rayleigh phase function.

*/

// This implementation is based on the one developed by Kate Salesin.
// Sampling method reference: Frisvad (2011)

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class RayleighPolarizedPhaseFunction final
    : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume)

    RayleighPolarizedPhaseFunction(const Properties &props) : Base(props) {
        m_depolarization = props.volume<Volume>("depolarization",0.f);

        if(m_depolarization->max() >= 1.f)
            Log(Error, "Depolarization factor must be in [0, 1[");

        m_flags = +PhaseFunctionFlags::Anisotropic;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("depolarization", m_depolarization.get(),
                             +ParamFlags::Differentiable);
    }

    MI_INLINE
    MuellerMatrix<UnpolarizedSpectrum> rayleigh_scatter(
                                        const UnpolarizedSpectrum cos_theta,
                                        const UnpolarizedSpectrum rho) const {
        /* Calculates the Mueller matrix of a Rayleigh scatter event given
           the scattering angle (Hansen & Travis (1974), eq. (2.15)).
           The angle θ is defined in the physics convention. */

        UnpolarizedSpectrum r1 = (1.f - rho) / (1.f + rho / 2.f),
                            r2 = (1.f + rho) / (1.f - rho),
                            r3 = (1.f - 2.f * rho) / (1.f - rho);

        UnpolarizedSpectrum a = r2 + dr::sqr(cos_theta),
                            b = dr::sqr(cos_theta) + 1.f,
                            c = dr::sqr(cos_theta) - 1.f,
                            d = 2.f * cos_theta;

        return UnpolarizedSpectrum(3.f / 16.f) * dr::InvPi<UnpolarizedSpectrum> * r1 * MuellerMatrix<UnpolarizedSpectrum>(
            a, c, 0, 0,
            c, b, 0, 0,
            0, 0, d, 0,
            0, 0, 0, d * r3
        );
    }

    MI_INLINE Float eval_rayleigh_pdf(Float cos_theta) const {
        // TODO: Check vs Frisvad (2011)
        return (3.f / 16.f) * dr::InvPi<Float> * (1.f + dr::sqr(cos_theta));
    }

    MI_INLINE Spectrum eval_rayleigh(const PhaseFunctionContext &ctx,
                                     const MediumInteraction3f &mei,
                                     const Vector3f &wo,
                                     Float cos_theta) const {
        Spectrum phase_val;
        UnpolarizedSpectrum rho = m_depolarization->eval(mei);

        if constexpr (is_polarized_v<Spectrum>) {
            // We first evaluate the Rayleigh phase matrix
            phase_val = rayleigh_scatter(UnpolarizedSpectrum(cos_theta), rho);

            /* Due to the coordinate system rotations for polarization-aware
               phase functions, we need to know the propagation direction of
               light. In the following, light arrives along `-wo_hat` and leaves
               along
               `+wi_hat`. */
            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mei.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? mei.wi : wo;

            /* The Stokes reference frame vector of this matrix lies in the
               scattering plane spanned by wi and wo. */
            Vector3f x_hat      = dr::normalize(dr::cross(-wo_hat, wi_hat)),
                     p_axis_in  = dr::normalize(dr::cross(x_hat, -wo_hat)),
                     p_axis_out = dr::normalize(dr::cross(x_hat, wi_hat));

            /* Rotate in/out reference vector of weight s.t. it aligns with the
             * implicit Stokes bases of -wo_hat & wi_hat.
             */
            phase_val = mueller::rotate_mueller_basis(
                phase_val, -wo_hat, p_axis_in, mueller::stokes_basis(-wo_hat),
                wi_hat, p_axis_out, mueller::stokes_basis(wi_hat));

            // If the cross product x_hat is too small, phase_val may be NaN
            dr::masked(phase_val, dr::isnan(phase_val)) = 0.f;

        } else {
            Spectrum r1  = (1.f - rho) / (1.f + rho / 2.f),
                     r2  = (1.f + rho) / (1.f - rho);

            phase_val = (3.f / 16.f) * dr::InvPi<Float> * r1 *
                        (r2 + dr::sqr(cos_theta));
        }

        return phase_val;
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &ctx, const MediumInteraction3f &mei,
           Float /* sample1 */, const Point2f &sample,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float z                 = 2.f * (2.f * sample.x() - 1.f);
        Float tmp               = dr::sqrt(dr::sqr(z) + 1.f);
        Float A                 = dr::cbrt(z + tmp);
        Float B                 = dr::cbrt(z - tmp);
        Float cos_theta         = A + B; /* cos_theta in physics convention */
        Float sin_theta         = dr::safe_sqrt(1.0f - dr::sqr(cos_theta));
        auto [sin_phi, cos_phi] = dr::sincos(dr::TwoPi<Float> * sample.y());

        /* If θ is the scattering angle in physics convention, and θ'
           the scattering angle in graphics convention, then θ' = π - θ
           and cos(θ') = -cos(θ) and sin(θ') = sin(θ). */
        Vector3f wo = { sin_theta * cos_phi, sin_theta * sin_phi, cos_theta };
        wo          = -mei.to_world(wo);

        /* eval_rayleigh_pdf expects cos(θ) in physics convention, but expects
         * wo in graphics convention */
        Float pdf = eval_rayleigh_pdf(cos_theta);
        Spectrum phase_weight =
            eval_rayleigh(ctx, mei, wo, cos_theta) * dr::rcp(pdf);

        return { wo, phase_weight, pdf };
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mei,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        /* If the incident direction is ω in graphics convention, it
         * is -ω in physics convention. The evaluation implementation routines
         * expect cos(θ) in physics convention.
         */

        Float cos_theta    = dot(wo, -mei.wi);
        Spectrum phase_val = eval_rayleigh(ctx, mei, wo, cos_theta);
        Float pdf          = eval_rayleigh_pdf(cos_theta);
        return { phase_val, pdf };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RayleighPolarizedPhaseFunction["
            << "depolarization = " << m_depolarization << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Volume> m_depolarization;
};

MI_IMPLEMENT_CLASS_VARIANT(RayleighPolarizedPhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(RayleighPolarizedPhaseFunction,
                 "Polarized Rayleigh phase function")
NAMESPACE_END(mitsuba)
