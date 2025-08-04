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

.. _plugin-sensor-voxelflux:

VoxelFlux sensor (:monosp:`voxelflux`)
-------------------------------------------------

.. pluginparameters::

 * - bbox_min, bbox_max
   - |point|
   - *Subject to Change* Bounding box of the sensor. If not set, the sensor will use the scene's
     bounding box. Film resolution is set to the number of voxels in the bounding box.
   - —

 * - apply_sample_scale
   - |bool|
   - *Debug* Apply the sample scale to the recorded value.
   - —

This sensor measures the flux that traverses voxel faces. It keeps track of 
the direction and magnitude of the flux that traverses each voxel face.
*/

template <typename Float, typename Spectrum>
class VoxelFluxSensor final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_to_world, m_film, m_needs_sample_2,
                   m_needs_sample_3, m_sensor_filter)
    MI_IMPORT_TYPES(Scene, Shape, BSDF, BSDFPtr)

    using Matrix = dr::Matrix<Float, Transform4f::Size>;
    using Index = dr::int32_array_t<Float>;

    VoxelFluxSensor(const Properties &props) : Base(props) {
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

        m_sensor_filter = +SensorFilterFlags::Depth;
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

        // expecting the film to be of shape 3xNxMxOx2.
        // TODO: add checks
        for(ScalarUInt32 i = 1; i < 4; ++i) {
            m_grid_res[i-1] = m_film->size(i)-1;
        } 
        m_voxel_size = m_bbox.extents() / ScalarVector3f(m_grid_res);

        Log(Debug, "bbox.extent: %f, m_grid_res: %f, m_voxel_size: %f", m_bbox.extents(), m_grid_res, m_voxel_size);
    }

    // This sensor does not occupy any particular region of space, return an
    // invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    void accumulate(
        const Ray3f &ray,
        const SurfaceInteraction3f &si,
        const MediumInteraction3f &mei,
        Float /*tmax*/,
        Spectrum emitted,
        Spectrum throughput,
        ScalarFloat sample_scale,
        Mask filter = true,
        Mask active = true
    ) override {
           
        Log(Debug, "ACCUMULATE");
        // Find intersection with the bounding box of the volume grid
        Vector3f t_bmin = (m_bbox.min - ray.o) / ray.d;
        Vector3f t_bmax = (m_bbox.max - ray.o) / ray.d;

        Float mint_box = dr::maximum(dr::max(dr::minimum(t_bmin, t_bmax)), 0.f);
        Float maxt_box = dr::min(dr::maximum(t_bmin, t_bmax));

        // Offset the start distance so that it we can account the first voxel face.
        Float maxt = si.t;
        Float t_start = mint_box - math::RayEpsilon<Float>;
        Float t_end = dr::minimum(maxt_box, maxt); 
        Float t = t_start;
        // TODO: Bounding by intersection is not correct, need to fix this.

        active &= dr::isfinite(t_start) && dr::isfinite(t_end) && (t_start < t_end);

        Point3f grid_start = ray(t_start);
        Point3f grid_end = ray(t_end);

        Vector3i step_dir = dr::select(ray.d > 0, 1, -1); // Vector3f?

        // voxel 3d index for start and end
        Vector3i start_voxel = dr::clamp((dr::floor((grid_start - m_bbox.min) / m_voxel_size)), -1, m_grid_res);
        Vector3i  end_voxel = dr::clamp(((grid_end - m_bbox.min) / m_voxel_size), 0, m_grid_res - 1);

        // Distance along th ray to the next voxel boundary
        Vector3f next_voxel_pos = m_bbox.min + (start_voxel + step_dir) * m_voxel_size;

        // If the ray has negative directions, we need to hit the left of the current
        // voxel, not the next one /!\ This is slightly different than the reference
        // algorithm, where they jump to the next cell instead of modifying the
        // boundary, but it should be equivalent.
        next_voxel_pos += dr::select(ray.d < 0, m_voxel_size, 0);

        auto is_valid_dir = dr::abs(ray.d) > 1e-8; // Avoid division by zero (horizontal slope)
        Vector3f dtmax = dr::select(is_valid_dir, (next_voxel_pos - grid_start) / ray.d, dr::Infinity<Float>);
        dr::masked(dtmax, dtmax < 0) = dr::Infinity<Float>;
        Vector3f tstep = dr::select(is_valid_dir, m_voxel_size / ray.d * step_dir, dr::Infinity<Float>); 

        Vector3i current_voxel = start_voxel;
        Float remaining_dist = t_end - t_start;
        
        const ScalarVector3i face_index(0,1,2);

        UInt32 stride_z = 2; 
        UInt32 stride_y = stride_z * (m_grid_res.z() + 1);
        UInt32 stride_x = stride_y * (m_grid_res.y() + 1);
        UInt32 stride_f = stride_x * (m_grid_res.x() + 1);

        // UInt32 stride_x = 2;
        // UInt32 stride_y = stride_x * (m_grid_res.x() + 1);
        // UInt32 stride_z = stride_y * (m_grid_res.y() + 1);
        // UInt32 stride_f = stride_z * (m_grid_res.z() + 1);

        Log(Debug, "ray.o: %d, ray.d: %d", ray.o, ray.d);
        Log(Debug, "m_bbox.min: %d, m_bbox.max: %d, active : %d", m_bbox.min, m_bbox.max, active);
        Log(Debug, "t_end: %f, t_start: %f, remaining_dist: %f, maxt: %f", t_end, t_start, remaining_dist, maxt);
        Log(Debug, "start_voxel: %f, end_voxel: %f, grid_res", start_voxel, end_voxel, m_grid_res);
        Log(Debug, "grid_start: %f, next_voxel_pos: %f, is_valid_dir: %f", grid_start, next_voxel_pos, is_valid_dir);
        Log(Debug, "stride_f: %f, stride_x: %f, stride_y: %f, stride_z: %f,", stride_f, stride_x, stride_y, stride_z );

        Spectrum flux = emitted * throughput;
        flux *= m_apply_sample_scale ? sample_scale : Spectrum(1.f); 

        // TODO write loop.
        dr::Loop<Mask> loop("DDA");
        loop.put(active, dtmax, remaining_dist, current_voxel);
        loop.init();
        while (loop(dr::detach(active))) {

            Float dt = dr::minimum(dr::min(dtmax), remaining_dist);
            Log(Debug, "dt: %0.10d, remaining_dist: %0.10d", dtmax.z(), dt, remaining_dist);
            Log(Debug, "dtmax.x: %0.10d, dtmax.y: %0.10d, dtmax.z: %0.10d", dtmax.x(),dtmax.y(),dtmax.z());
            dr::masked(remaining_dist, active) -= dt;
            
            // Check if we are at the end of the ray
            dr::masked(t, active) += dt;
            active &= (maxt - t) > 1e-6;

            auto mask = dr::abs(dtmax - dt) <= 1e-6;

            // Retrieve the face and direction indices used to access the film
            UInt32 f = dr::sum(dr::select(mask, face_index, 0));
            UInt32 d = dr::sum(dr::maximum(dr::select(mask, step_dir, 0),0));
            Log(Debug, "mask: %d, d: %d, f: %d", mask, d, f);

            // Because we are considering faces, we increment the voxel index by one in positive directions.
            Vector3i current_face = dr::select(mask && (step_dir > 0), current_voxel + 1, current_voxel);
            Log(Debug, "current_voxel: %f, current_face: %f", current_voxel, current_face);

            UInt32 current_voxel_flat = d 
                                       + current_face.x() * stride_x
                                       + current_face.y() * stride_y
                                       + current_face.z() * stride_z
                                       + f * stride_f;
            Log(Debug, "flat_idx: %f", current_voxel_flat);
            // ====== Write to film ======
            if constexpr (!is_polarized_v<Spectrum>){
                Float cos_theta = dr::sum(dr::select(mask, dr::abs(ray.d), 0));
                Log(Debug, "flux: %f, cos_theta : %f", flux, cos_theta);
                // m_film->write_tensor(flux[0] * cos_theta, current_voxel_flat, active && filter);
                m_film->write_tensor(flux[0], current_voxel_flat, active);
            }
            // ===========================

            active &= dr::any(dr::neq(end_voxel, current_voxel)) && (remaining_dist > 1e-6);

            // Update the voxel index by stepping in the axis closest to the current point.
            dtmax = dr::select(mask, tstep, dtmax - dt);
            Vector3i voxel_update = dr::select(mask, step_dir, 0);
            dr::masked(current_voxel, active) += voxel_update;

            active &= dr::all(current_voxel >= 0) && dr::all(current_voxel < m_grid_res);
        }
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "VoxelFluxSensor[" << std::endl
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

MI_IMPLEMENT_CLASS_VARIANT(VoxelFluxSensor, Sensor)
MI_EXPORT_PLUGIN(VoxelFluxSensor, "Absorbed Flux Sensor")
NAMESPACE_END(mitsuba)
