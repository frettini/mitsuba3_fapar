#include <drjit/tensor.h>
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/filter.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _plugin-sensor-absorbedflux:

Absorbedflux sensor (:monosp:`absorbedflux`)
-------------------------------------------------

.. pluginparameters::

 * - bbox_min, bbox_max
   - |point|
   - Bounding box of the sensor. If not set, the sensor will use the scene's
     bounding box. Film resolution is set to the number of voxels in the bounding box.
   - —

 * - apply_sample_scale
   - |bool|
   - *Debug* Apply the sample scale to the recorded value.
   - —

This sensor plugin measures the flux absorbed by surface interactions in a voxel 
grid. The number of voxels is defined by the film resolution and the bounding 
box by `bbox_min` and `bbox_max`.

.. tabs::
    .. code-tab::  xml

        <sensor type="absorbedflux">
            <string name="bbmox_min" value="-1, -1, -1"/>
            <string name="bbmox_max" value="1, 1, 1"/>
            <integer name="apply_sample_scale" value="true"/>
        </sensor>

    .. code-tab:: python

        'type': 'absorbedflux',
        'bbmox_min':[-1,-1,-1],
        'bbmox_max':[ 1, 1, 1],
        'apply_sample_scale':True,
*/

template <typename Float, typename Spectrum>
class AbsorbedFluxSensor final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_to_world, m_film, m_needs_sample_2,
                   m_needs_sample_3, m_sensor_filter)
    MI_IMPORT_TYPES(Scene, Shape, BSDF, BSDFPtr)

    using Matrix = dr::Matrix<Float, Transform4f::Size>;
    using Index = dr::int32_array_t<Float>;

    AbsorbedFluxSensor(const Properties &props) : Base(props) {
        // Collect directions and set transforms accordingly
        if (props.has_property("to_world")) {
            Throw("This sensor doesn't have a physical location and cannot "
                  "placed in the scene.");
        }

        if(props.has_property("bbox_min") && props.has_property("bbox_min")){
            ScalarPoint3f bbox_min = props.get<ScalarPoint3f>("bbox_min");
            ScalarPoint3f bbox_max = props.get<ScalarPoint3f>("bbox_max");
            m_bbox = ScalarBoundingBox3f(bbox_min, bbox_max);
        }

        m_apply_sample_scale = props.get<bool>("apply_sample_scale", true);

        m_needs_sample_2 = false;
        m_needs_sample_3 = false;

        m_sensor_filter = +SensorFilterFlags::Surface | +SensorFilterFlags::Exclusive;
    }

    void set_scene(const Scene *scene) override {
        
        if( !m_bbox.valid() ) {
            // Initialize the bounding box to the scene's bounding box.
            ScalarBoundingBox3f s_bbox = scene->bbox();

            if(!s_bbox.valid()){
                Throw("Both the sensor and the scene don't have a valid extent "
                      "for the sensor to use.");
            }
            
            // Add offset to avoid collapsed bounding boxes.
            m_bbox = ScalarBoundingBox3f(s_bbox.min - dr::Epsilon<Point3f>, 
                                         s_bbox.max + dr::Epsilon<Point3f> );
        } 

        for(ScalarUInt32 i = 0; i < 3; ++i) {
            m_grid_res[i] = m_film->size(i);
        } 
        m_voxel_size = m_bbox.extents() / m_grid_res;
    }

    // This sensor does not occupy any particular region of space, return an
    // invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    void accumulate(
        const Ray3f &/*ray*/,
        const SurfaceInteraction3f &si,
        const MediumInteraction3f &/*mei*/,
        Float /*tmax*/,
        Spectrum emitted,
        Spectrum throughput,
        ScalarFloat sample_scale,
        Mask filter = true,
        Mask active = true
    ) override {
           
        Mask accumulate = active && filter && si.is_valid();
    
        if (dr::any_or<true>(accumulate)) {
            // Calculate index of current voxel from the interaction point
            Vector3i current_voxel = Vector3i(dr::floor((si.p - m_bbox.min) / m_voxel_size));
            accumulate &= dr::all(current_voxel >= 0) && dr::all(current_voxel < m_grid_res);
            Float current_voxel_flat = current_voxel.x() 
                                    + current_voxel.y() * m_grid_res.x() 
                                    + current_voxel.z() * m_grid_res.x() * m_grid_res.y();

            // Calculate absorption at intersection point
            BSDFPtr bsdf = si.bsdf();
            Spectrum absorption = 1.0f - bsdf->eval_hdrf(si, active);
            Log(Debug, "emitted: %f, throughput: %f, absorption: %f, si.wi: %f", 
                dr::max(unpolarized_spectrum(emitted)), 
                dr::max(unpolarized_spectrum(throughput)), 
                dr::max(unpolarized_spectrum(absorption)),
                Frame3f::cos_theta(si.wi));
            // Calculated the resulting absorbed flux
            Spectrum result = emitted * throughput * absorption;

            result *= m_apply_sample_scale ? sample_scale : Float(1.f); 
            if constexpr (!is_polarized_v<Spectrum>){
                m_film->write_tensor(result[0], current_voxel_flat, accumulate);
            }
        }
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AbsorbedFluxSensor[" << std::endl
            << "  film = " << string::indent(m_film) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "  apply_sample_scale = " << string::indent(m_apply_sample_scale) << "," << std::endl;

        return oss.str();
    }

    MI_DECLARE_CLASS()

protected:
    // Sensor extent bbox
    ScalarBoundingBox3f m_bbox;
    ScalarVector3i m_grid_res;
    ScalarVector3f m_voxel_size;
    bool m_apply_sample_scale;
};

MI_IMPLEMENT_CLASS_VARIANT(AbsorbedFluxSensor, Sensor)
MI_EXPORT_PLUGIN(AbsorbedFluxSensor, "Absorbed Flux Sensor")
NAMESPACE_END(mitsuba)
