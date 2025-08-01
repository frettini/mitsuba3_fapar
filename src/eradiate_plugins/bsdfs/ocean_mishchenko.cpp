#include <array>
#include <drjit/dynamic.h>
#include <drjit/texture.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/random.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/eradiate/oceanprops.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>
#include <tuple>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _plugin-bsdf-ocean_mishchenko:

Oceanic reflection model (:monosp:`ocean-mishchenko`)
-----------------------------------------------------

.. pluginparameters::

 * - wind_speed
   - |float|
   - Specifies the wind speed at which to evaluate the oceanic reflectance. Range: [0, 37.54] m/s. Default: 0.1 m/s
   - |exposed|, |differentiable|

 * - eta, k
   - |spectrum| or |texture|
   - Real and imaginary components of the water's index of refraction.
     Default: 1.33, 0.0
   - |exposed|, |differentiable|

 * - ext_ior
   - |spectrum| or |texture|
   - Exterior index of refraction specified numerically or using a known
     material name. Note that the complex component is assumed to be 0.
     Default: 1.000277
   - |exposed|, |differentiable|

 * - shadowing
   - |bool|
   - Indicates whether evaluation accounts for the shadowing-masking term.
     Default: :monosp:`true`
   - —

This plugin implements the polarized oceanic reflection model originally
implemented by :cite:`Mishchenko1997AerosolRetrievalPolarization`. This model
focuses on the sunglint reflectance, which follows the Cox and Munk surface
slope probability distribution.

Note that this material is one-sided---that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the ``twosided`` BSDF adapter plugin.
The following snippet describes an oceanic surface material with monochromatic
parameters:

.. tab-set-code::

    .. code-block:: python

        "type": "ocean_mishchenko",
        "wind_speed": 10,
        "eta": 1.33,
        "k": 0.,
        "ext_ior": 1.0,

    .. code-block:: xml

        <bsdf type="ocean_mishchenko">
            <float name="wind_speed" value="10"/>
            <float name="eta" value="1.33"/>
            <float name="k" value="0."/>
            <float name="ext_ior" value=1.0/>
        </bsdf>

.. note:: This model only implements the sunglint reflection. See :ref:`ocean
    legacy <plugin-bsdf-ocean_legacy>` for a BSDF that includes whitecap, sunglint,
    and underlight reflectance.

*/

template <typename Float, typename Spectrum>
class MishchenkoOceanBSDF final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)

    using Complex2u = dr::Complex<UnpolarizedSpectrum>;

    /**
     * @brief Construct a new OceanBSDF object.
     *
     * @param props A set of properties to initialize the oceanic BSDF.
     */
    MishchenkoOceanBSDF(const Properties &props) : Base(props) {
        // Retrieve parameters
        m_wind_speed = props.get<ScalarFloat>("wind_speed", 0.1f);
        m_eta        = props.texture<Texture>("eta", 1.33f);
        m_k          = props.texture<Texture>("k", 0.f);
        m_ext_eta    = props.texture<Texture>("ext_ior", 1.000277f);
        m_shadowing  = props.get<bool>("shadowing", true);

        update();

        // => Sun glint reflectance at the water surface is "specular"
        m_components.push_back(BSDFFlags::GlossyReflection |
                               BSDFFlags::FrontSide);

        // Set all the flags
        for (auto c : m_components)
            m_flags |= c;
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("wind_speed", m_wind_speed, +ParamFlags::Differentiable);
        callback->put_object("eta", m_eta, +ParamFlags::Differentiable);
        callback->put_object("k", m_k, +ParamFlags::Differentiable);
        callback->put_object("ext_ior", m_ext_eta, +ParamFlags::Differentiable);
    }

    void
    parameters_changed(const std::vector<std::string> & /*keys*/) override {
        update();
    }

    /**
     * @brief Update the variables that can be preprocessed.
     * This includes the complex index of refraction and mean square slope
     */
    void update() {

        // compute the index of refraction
        m_sigma = dr::SqrtTwo<Float> * dr::sqrt(0.5f * cox_munk_msslope_squared(m_wind_speed));
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /*sample1*/,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        if (unlikely(dr::none_or<false>(active)) || !ctx.is_enabled(BSDFFlags::GlossyReflection))
            return { bs, 0.f };


        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(MicrofacetType::Beckmann,
                                     m_sigma,
                                     true);

        // Sample M, the microfacet normal
        Normal3f m;
        std::tie(m, bs.pdf) = distr.sample(si.wi, sample2);

        /// Perfect specular reflection based on the microfacet normal
        bs.wo = reflect(si.wi, m);
        bs.eta = 1.f;
        bs.sampled_component = 0;
        bs.sampled_type = +BSDFFlags::GlossyReflection;

        // Ensure that this is a valid sample
        active &= dr::neq(bs.pdf, 0.f) && Frame3f::cos_theta(bs.wo) > 0.f;

        // Jacobian of the half-direction mapping
        bs.pdf /= 4.f * dr::dot(bs.wo, m);

        UnpolarizedSpectrum weight;
        weight = distr.G_height_correlated(si.wi, bs.wo, m) / distr.smith_g1(si.wi, m);

        Complex2u n_air(m_ext_eta->eval(si, active), 0.);
        Complex2u n_water(m_eta->eval(si, active),
                          m_k->eval(si, active));

        // `TransportMode` has two states:
        //     - `Radiance`, trace from the sensor to the light sources
        //     - `Importance`, trace from the light sources to the sensor
        Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? bs.wo : si.wi,
                 wi_hat = ctx.mode == TransportMode::Radiance ? si.wi : bs.wo;

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            F = fresnel_sunglint_polarized(n_air, n_water, -wo_hat, wi_hat);

            /* The Stokes reference frame vector of this matrix lies in the
            meridian plane spanned by wi and n. */
            Vector3f n(0, 0, 1);
            Vector3f p_axis_in  =
                dr::normalize(dr::cross(dr::normalize(dr::cross(n,-wo_hat)),-wo_hat));
            Vector3f p_axis_out =
                dr::normalize(dr::cross(dr::normalize(dr::cross(n,wi_hat)),wi_hat));

            dr::masked(p_axis_in, dr::any(dr::isnan(p_axis_in))) =
                Vector3f(0.f, 1.f, 0.f);
            dr::masked(p_axis_out, dr::any(dr::isnan(p_axis_out))) =
                Vector3f(0.f, 1.f, 0.f);

            // Rotate in/out reference vector of `value` s.t. it aligns with the
            // implicit Stokes bases of -wo_hat & wi_hat. */
            F = mueller::rotate_mueller_basis(
                F, -wo_hat, p_axis_in, mueller::stokes_basis(-wo_hat),
                wi_hat, p_axis_out, mueller::stokes_basis(wi_hat));

        } else {
            F =
                UnpolarizedSpectrum(fresnel_sunglint_polarized(n_air, n_water, -wo_hat, wi_hat)[0][0]);
        }

        return { bs, (F*weight) & (active) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_glint = ctx.is_enabled(BSDFFlags::GlossyReflection);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Ensure incoming and outgoing directions are in the upper hemisphere
        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(dr::none_or<false>(active) || !has_glint))
            return 0.f;

        MicrofacetDistribution distr(
            MicrofacetType::Beckmann,
            m_sigma,
            true
        );

        const Vector3f m = dr::normalize(si.wi + wo);
        Float D = distr.eval(m);
        Float G = distr.G_height_correlated(si.wi, wo, m);
        UnpolarizedSpectrum result = D * G / (4 * cos_theta_i);

        active &= dr::neq(D, 0.f);

        Complex2u n_air(m_ext_eta->eval(si, active), 0.);
        Complex2u n_water(m_eta->eval(si, active),
                          m_k->eval(si, active));

        // `TransportMode` has two states:
        //     - `Radiance`, trace from the sensor to the light sources
        //     - `Importance`, trace from the light sources to the sensor
        Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : si.wi,
                 wi_hat = ctx.mode == TransportMode::Radiance ? si.wi : wo;

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            F = fresnel_sunglint_polarized(n_air, n_water, -wo_hat, wi_hat);
            // Compute the Stokes reference frame vectors of this matrix that
            // are perpendicular to the plane of reflection.
            Vector3f n(0.f, 0.f, 1.f);

            /* The Stokes reference frame vector of this matrix lies in the
               scattering plane spanned by wi and wo. */
            Vector3f p_axis_in  =
                dr::normalize(dr::cross(dr::normalize(dr::cross(n,-wo_hat)),-wo_hat));
            Vector3f p_axis_out =
                dr::normalize(dr::cross(dr::normalize(dr::cross(n,wi_hat)),wi_hat));

            dr::masked(p_axis_in, dr::any(dr::isnan(p_axis_in)))   =
                Vector3f(0.f, 1.f, 0.f);
            dr::masked(p_axis_out, dr::any(dr::isnan(p_axis_out))) =
                Vector3f(0.f, 1.f, 0.f);

            // Rotate in/out reference vector of `F` s.t. it aligns with the
            // implicit Stokes bases of -wo_hat & wi_hat. */
            F = mueller::rotate_mueller_basis(
                F,
                -wo_hat, p_axis_in, mueller::stokes_basis(-wo_hat),
                wi_hat, p_axis_out, mueller::stokes_basis(wi_hat));
        } else {
            F =
                UnpolarizedSpectrum(fresnel_sunglint_polarized(n_air, n_water, -wo_hat, wi_hat)[0][0]);
        }

        return result*F & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Vector3f m = dr::normalize(wo + si.wi);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
                  dr::dot(si.wi, m) > 0.f && dr::dot(wo, m) > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                     dr::none_or<false>(active)))
            return 0.f;

        MicrofacetDistribution distr(
            MicrofacetType::Beckmann,
            m_sigma,
            true
        );

        Float pdf =
            distr.eval(m) * distr.smith_g1(si.wi, m) / (4.f * cos_theta_i);

        return dr::select(active, pdf, 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "OceanMishchenko[" << std::endl
            << "  wind_speed = " << string::indent(m_wind_speed) << std::endl
            << "  eta = " << string::indent(m_eta) << std::endl
            << "  k = " << string::indent(m_k) << std::endl
            << "  ext_ior = " << string::indent(m_ext_eta) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    //  User-provided fields
    ScalarFloat m_wind_speed;
    ref<Texture> m_eta;
    ref<Texture> m_k;
    ref<Texture> m_ext_eta;
    bool m_shadowing;

    ScalarFloat m_sigma;
    OceanProperties<Float, Spectrum> m_ocean_props;
};

MI_IMPLEMENT_CLASS_VARIANT(MishchenkoOceanBSDF, BSDF)
MI_EXPORT_PLUGIN(MishchenkoOceanBSDF, "Mishchenko Ocean material")
NAMESPACE_END(mitsuba)
