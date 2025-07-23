#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-directionaltarget:

Distant directional target emitter (:monosp:`directionaltarget`)
---------------------------------------------------

.. pluginparameters::

 * - irradiance
   - |spectrum|
   - Spectral irradiance, which corresponds to the amount of spectral power
     per unit area received by a hypothetical surface normal to the specified
     direction.
   - |exposed|, |differentiable|

 * - target
   - |transform|
   - Emitter-to-world transformation matrix.
   - |exposed|

 * - to_world
   - |transform|
   - emitter_direction-to-world transformation matrix 

 * - direction
   - |vector|
   - Alternative (and exclusive) to `to_world`. Direction towards which the
     emitter is radiating in world coordinates.

This emitter plugin implements a distant directional source which radiates a
specified power per unit area along a fixed direction. By default, the emitter
radiates in the direction of the positive Z axis, i.e. :math:`(0, 0, 1)`.

.. tabs::
    .. code-tab:: xml
        :name: directional-light

        <emitter type="directional">
            <vector name="direction" value="1.0, 0.0, 0.0"/>
            <rgb name="irradiance" value="1.0"/>
        </emitter>

    .. code-tab:: python

        'type': 'directional',
        'direction': [1.0, 0.0, 0.0],
        'irradiance': {
            'type': 'rgb',
            'value': 1.0,
        }

*/

MI_VARIANT class DirectionalTargetEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world, m_needs_sample_3)
    MI_IMPORT_TYPES(Scene, Texture, Shape)

    DirectionalTargetEmitter(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        if (props.has_property("direction")) {
            if (props.has_property("to_world"))
                Throw("Only one of the parameters 'direction' and 'to_world' "
                      "can be specified at the same time!'");

            ScalarVector3f direction(dr::normalize(props.get<ScalarVector3f>("direction")));
            auto [up, unused] = coordinate_system(direction);

            m_to_world = ScalarTransform4f::look_at(0.0f, ScalarPoint3f(direction), up);
            dr::make_opaque(m_to_world);
        }

        // Target properties.
        auto obj = props.object("target");
        m_target_shape = dynamic_cast<Shape *>(obj.get());
        if (!m_target_shape)
            Throw("Invalid parameter target, must be a Point3f or a Shape.");

        // Irradiance properties
        m_irradiance = props.texture_d65<Texture>("irradiance", 1.f);

        if (m_irradiance->is_spatially_varying())
            Throw("Expected a non-spatially varying irradiance spectra!");

        m_needs_sample_3 = false;

        m_flags      = EmitterFlags::Infinite | EmitterFlags::DeltaDirection;
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("irradiance",   m_irradiance.get(), +ParamFlags::Differentiable);
        callback->put_parameter("to_world", *m_to_world.ptr(),   +ParamFlags::NonDifferentiable);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            m_bsphere = scene->bbox().bounding_sphere();
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                        m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
    }

    Spectrum eval(const SurfaceInteraction3f & /*si*/,
                  Mask /*active*/) const override {
        return 0.f;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f & /*direction_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. "Sample" directional component (fixed, no actual sampling required)
        const auto trafo = m_to_world.value();
        Vector3f d_global = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 2. Use area-based sampling of shape
        PositionSample3f ps =
            m_target_shape->sample_position(time, spatial_sample, active);
        Point3f origin = ps.p - 2.f * d_global * m_bsphere.radius;

        // 3. Sample spectral component
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.t    = 0.f;
        si.time = time;
        si.p    = origin;
        si.uv   = spatial_sample;
        auto [wavelengths, wav_weight] =
            sample_wavelengths(si, wavelength_sample, active);

        Spectrum weight = wav_weight * m_target_shape->surface_area();

        return { Ray3f(origin, d_global, time, wavelengths),
                 depolarizer<Spectrum>(weight) };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &/*it*/, const Point2f & /*sample*/,
                     Mask /*active*/) const override {
        return { dr::zeros<DirectionSample3f>(), dr::zeros<Spectrum>() };
    }

    Spectrum eval_direction(const Interaction3f & it,
                        const DirectionSample3f & /*ds*/,
                        Mask active) const override {
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;
        return depolarizer<Spectrum>(m_irradiance->eval(si, active));
    }


    Float pdf_direction(const Interaction3f & /*it*/,
                        const DirectionSample3f & /*ds*/,
                        Mask /*active*/) const override {
        return 0.f;
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        return m_irradiance->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float /*time*/, const Point2f & /*sample*/,
                    Mask /*active*/) const override {
        if constexpr (dr::is_jit_v<Float>) {
            // When vcalls are recorded in symbolic mode, we can't throw an exception,
            // even though this result will be unused.
            return { dr::zeros<PositionSample3f>(),
                     dr::full<Float>(dr::NaN<ScalarFloat>) };
        } else {
            NotImplementedError("sample_position");
        }
    }

    ScalarBoundingBox3f bbox() const override {
        /* This emitter does not occupy any particular region
           of space, return an invalid bounding box */
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DirectionalTargetEmitter[" << std::endl
            << "  irradiance = " << string::indent(m_irradiance) << ","
            << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

protected:
    ref<Texture> m_irradiance;
    ref<Shape> m_target_shape;
    ScalarBoundingSphere3f m_bsphere;
};

MI_IMPLEMENT_CLASS_VARIANT(DirectionalTargetEmitter, Emitter)
MI_EXPORT_PLUGIN(DirectionalTargetEmitter, "Distant directional target emitter")
NAMESPACE_END(mitsuba)
