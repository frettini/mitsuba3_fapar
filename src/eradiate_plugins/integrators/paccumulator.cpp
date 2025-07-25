#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/filter.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-paccumulator:

Particle tracer (:monosp:`paccumulator`)
-----------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

 * - samples_per_pass
   - |bool|
   - If specified, divides the workload in successive passes with :paramtype:`samples_per_pass`
     samples per pixel.

This integrator traces rays starting from light sources and attempts to connect them
to the sensor at each bounce.
It does not support media (volumes).

Usually, this is a relatively useless rendering technique due to its high variance, but there
are some cases where it excels. In particular, it does a good job on scenes where most scattering
events are directly visible to the camera.

Note that unlike sensor-based integrators such as :ref:`path <integrator-path>`, it is not
possible to divide the workload in image-space tiles. The :paramtype:`samples_per_pass` parameter
allows splitting work in successive passes of the given sample count per pixel. It is particularly
useful in wavefront mode.

.. tabs::
    .. code-tab::  xml

        <integrator type="paccumulator">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'paccumulator',
        'max_depth': 8

 */

 
template <typename Float, typename Spectrum>
class VolumeIntegrator : public Integrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Integrator, should_stop, aov_names, m_stop, m_timeout,
                    m_render_timer, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sensor, Film, BSDF, BSDFPtr, ImageBlock, Sampler,
                     EmitterPtr)

    /**
     * \brief Sample the incident importance and splat the product of
     * importance and radiance to the film.
     *
     * \param scene
     *    The underlying scene
     *
     * \param sensor
     *    A sensor from which rays should be sampled
     *
     * \param sampler
     *    A source of (pseudo-/quasi-) random numbers
     *
     * \param block
     *    An image block that will be updated during the sampling process
     *
     * \param sample_scale
     *    A scale factor that must be applied to each sample to account
     *    for the film resolution and number of samples.
     */
    virtual void sample(const Scene */*scene*/, const Sensor */*sensor*/,
        Sampler */*sampler*/, ImageBlock */*block*/,
        ScalarFloat /*sample_scale*/) const {} ;

    virtual void sample(const Scene *scene, Sensor *sensor,
        Sampler *sampler, ScalarFloat sample_scale) const = 0 ;

    // =========================================================================
    //! @{ \name Integrator interface implementation
    // =========================================================================

    TensorXf render(Scene *scene,
        Sensor *sensor,
        uint32_t seed = 0,
        uint32_t spp = 0,
        bool develop = true,
        bool evaluate = true) override {

        ScopedPhase sp(ProfilerPhase::Render);
        m_stop = false;

        Film *film = sensor->film();
        film->clear();
        ScalarUInt32 film_size_prod = film->size_product(),
                     crop_size_prod = film->crop_size_product();

        // Potentially adjust the number of samples per pixel if spp != 0
        Sampler *sampler = sensor->sampler();
        if (spp)
            sampler->set_sample_count(spp);
        spp = sampler->sample_count();

        // Figure out how to divide up samples into passes, if needed
        uint32_t spp_per_pass = (m_samples_per_pass == (uint32_t) -1)
                                    ? spp
                                    : std::min(m_samples_per_pass, spp);

        if ((spp % spp_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
                spp, spp_per_pass);

        uint32_t n_passes = spp / spp_per_pass;

        
        size_t samples_per_pass = m_film_scale 
                                ? (size_t) film_size_prod * (size_t) spp_per_pass 
                                : (size_t) spp_per_pass;

        std::vector<std::string> aovs = aov_names();
        if (!aovs.empty())
            Throw("AOVs are not supported in the VolumeIntegrator!");
        film->prepare(aovs);

        // Special case: no emitters present in the scene.
        if (unlikely(scene->emitters().empty())) {
            Log(Info, "Rendering finished (no emitters found, returning black image).");
            TensorXf result;
            if (develop) {
                result = film->develop();
                dr::schedule(result);
            } else {
                film->schedule_storage();
            }
            return result;
        }

        ScalarFloat sample_scale = m_film_scale 
                                ? crop_size_prod / ScalarFloat(spp * film_size_prod)
                                : 1 / ScalarFloat(spp);

        TensorXf result;
        if constexpr (!dr::is_jit_v<Float>) {
            size_t n_threads = Thread::thread_count();

            Log(Info, "Starting render job (%u, %u sample%s,%s %u thread%s)",
                crop_size_prod, spp, spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "", n_threads,
                n_threads == 1 ? "" : "s");

            if (m_timeout > 0.f)
                Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

            // Split up all samples between threads
            size_t grain_size =
                std::max(samples_per_pass / (4 * n_threads), (size_t) 1);
            
            std::mutex mutex;
            ref<ProgressReporter> progress = new ProgressReporter("Rendering");

            size_t total_samples = samples_per_pass * n_passes;

            seed *= (uint32_t) total_samples / (uint32_t) grain_size;
            std::atomic<size_t> samples_done(0);

            // Start the render timer (used for timeouts & log messages)
            m_render_timer.reset();
            Log(Info, "grain_size : %d, total_samples : %d, sample_scale: %f", grain_size, total_samples, sample_scale);
            ThreadEnvironment env;
            dr::parallel_for(
                dr::blocked_range<size_t>(0, total_samples, grain_size),
                [&](const dr::blocked_range<size_t> &range) {
                    ScopedSetThreadEnvironment set_env(env);

                    // Fork a non-overlapping sampler for the current worker
                    ref<Sampler> sampler = sensor->sampler()->clone();

                    sampler->seed(seed +
                                (uint32_t) range.begin() / (uint32_t) grain_size);

                    size_t ctr = 0;
                    for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                        Log(Debug, "inner loop : %d", i);
                        sample(scene, sensor, sampler, sample_scale);
                        sampler->advance();

                        ctr++;
                        if (ctr > 10000) {
                            std::lock_guard<std::mutex> lock(mutex);
                            samples_done += ctr;
                            ctr = 0;
                            progress->update(samples_done / (ScalarFloat) total_samples);
                        }
                    }
                    total_samples += ctr;
                }
            );

            if (develop)
                result = film->develop();
        } else {
            if (n_passes > 1 && !evaluate) {
                Log(Warn, "render(): forcing 'evaluate=true' since multi-pass "
                        "rendering was requested.");
                evaluate = true;
            }

            constexpr size_t wavefront_size_limit = 0xffffffffu;
            if (samples_per_pass > wavefront_size_limit) {
                spp_per_pass /=
                    (uint32_t)((samples_per_pass + wavefront_size_limit - 1) /
                            wavefront_size_limit);
                n_passes = spp / spp_per_pass;
                samples_per_pass = (size_t) film_size_prod * (size_t) spp_per_pass;

                Log(Warn,
                    "The requested rendering task involves %zu Monte Carlo "
                    "samples, which exceeds the upper limit of 2^32 = 4294967296 "
                    "for this variant. Mitsuba will instead split the rendering "
                    "task into %zu smaller passes to avoid exceeding the limits.",
                    samples_per_pass, n_passes);
            }

            Log(Info, "Starting render job (%u, %u sample%s%s)",
                crop_size_prod, spp, spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(", %u passes", n_passes) : "");

            // Inform the sampler about the passes (needed in vectorized modes)
            sampler->set_samples_per_wavefront(spp_per_pass);

            // Seed the underlying random number generators, if applicable
            sampler->seed(seed, (uint32_t) samples_per_pass);

            Timer timer;
            for (size_t i = 0; i < n_passes; i++) {
                sample(scene, sensor, sampler, sample_scale);

                if (n_passes > 1) {
                    sampler->advance(); // Will trigger a kernel launch of size 1
                    sampler->schedule_state();
                }
            }

            if (develop) {
                result = film->develop();
                dr::schedule(result);
            } else {
                film->schedule_storage();
            }

            if (evaluate) {
                dr::eval();

                if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                    jit_flag(JitFlag::LoopRecord)) {
                    Log(Info, "Code generation finished. (took %s)",
                        util::time_string((float) timer.value(), true));

                    /* Separate computation graph recording from the actual
                    rendering time in single-pass mode */
                    m_render_timer.reset();
                }

                dr::sync_thread();
            }
        }

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return result;
    }

    //! @}
    // =========================================================================

    protected:
    /// Create an integrator
    VolumeIntegrator(const Properties &props) : Base(props) {

        m_samples_per_pass = props.get<uint32_t>("samples_per_pass", (uint32_t) -1);

        m_film_scale = props.get<bool>("film_scale", true);
    
        int rr_depth = props.get<int>("rr_depth", 5);
        if (rr_depth <= 0)
            Throw("\"rr_depth\" must be set to a value greater than zero!");
        
        m_rr_depth = (uint32_t) rr_depth;

        int max_depth = props.get<int>("max_depth", -1);
        if (max_depth < 0 && max_depth != -1)
            Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
        
        m_max_depth = (uint32_t) max_depth;

        // Minimum recorded value.
        int min_depth = props.get<int>("min_depth", 0);
        if (min_depth < 0 || min_depth > (int) m_max_depth)
            Throw("\"min_depth\" must be set to 0 or a value smaller than max depth");
        m_min_depth = (uint32_t) min_depth;

        if (props.has_property("pbox_min") && props.has_property("pbox_max")){
            ScalarPoint3f bbox_min = props.get<ScalarPoint3f>("pbox_min");
            ScalarPoint3f bbox_max = props.get<ScalarPoint3f>("pbox_max");
            m_pbox = ScalarBoundingBox3f(bbox_min, bbox_max);
        } else if (props.has_property("pbox")) {
            m_pbox = props.get<ScalarBoundingBox3f>("pbox");
        }
    };

    /// Virtual destructor
    virtual ~VolumeIntegrator() {};

    protected:
    /**
    * \brief Number of samples to compute for each pass over the image blocks.
    *
    * Must be a multiple of the total sample count per pixel.
    * If set to (size_t) -1, all the work is done in a single pass (default).
    */
    uint32_t m_samples_per_pass;

    uint32_t m_min_depth;

    /**
    * Longest visualized path depth (\c -1 = infinite).
    * A value of \c 1 will visualize only directly visible light sources.
    * \c 2 will lead to single-bounce (direct-only) illumination, and so on.
    */
    uint32_t m_max_depth;

    /// Depth to begin using russian roulette
    uint32_t m_rr_depth;

    // periodic boundary
    ScalarBoundingBox3f m_pbox;

    bool m_film_scale;

    MI_DECLARE_CLASS()

};



template <typename Float, typename Spectrum>
class ParticleAccumulatorIntegrator final : public VolumeIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(VolumeIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_min_depth, m_max_depth, m_pbox)
    MI_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                     EmitterPtr, BSDF, BSDFPtr)

    ParticleAccumulatorIntegrator(const Properties &props) : Base(props) { }

    void sample(const Scene *scene, Sensor *sensor, Sampler *sampler, ScalarFloat sample_scale) const override {
        // Primary & further bounces illumination
        auto [ray, throughput] = prepare_ray(scene, sensor, sampler);

        Float throughput_max = dr::max(unpolarized_spectrum(throughput));
        Mask active = dr::neq(throughput_max, 0.f);

        trace_light_ray(ray, scene, sensor, sampler, throughput,
                        sample_scale, active);
    }

    /// Samples a ray from a random emitter in the scene.
    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler) const {
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d() * sensor->shutter_open_time();

        // Prepare random samples.
        Float wavelength_sample  = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample  = sampler->next_2d();

        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample);

        return { ray, ray_weight };
    }

    /**
     * Intersects the given ray with the scene and recursively trace using
     * BSDF sampling. The given `throughput` should account for emitted
     * radiance from the sampled light source, wavelengths sampling weights,
     * etc. At each interaction, we attempt to connect to the sensor and add
     * the current radiance to the given `block`.
     *
     * Note: this will *not* account for directly visible emitters, since
     * they require a direct connection from the emitter to the sensor. See
     * \ref sample_visible_emitters.
     *
     * \return The radiance along the ray and an alpha value.
     */
    std::pair<Spectrum, Float>
    trace_light_ray(Ray3f ray, const Scene *scene, Sensor *sensor,
                    Sampler *sampler, Spectrum throughput,
                    ScalarFloat sample_scale, Mask active = true) const {
        // @PONDER: currently the throughput includes both the weighted emitted 
        // radiance further weighted by the propability to sample the ray. We
        // could decide to rename this to Le, and have throughput act like in 
        // backward tracers, starting at one. This would have an impact on the 
        // Russian Roulette.

        // @PONDER: The depth starts at one here because it technically doesn't 
        // account for emitters that are directly visible. Might need to set it 
        // to 0 because we are doing something fundamentally different?
        

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);
        UInt32 depth = 0;
        
        Mask pbounds_valid = m_pbox.valid();
        Vector3f pbound_extent = m_pbox.extents();
        UInt32 max_periodic_iterations = 100;
        UInt32 periodic_count = 0;

        Log(Debug, "trace_light_ray");
        

        if(dr::any(m_pbox.valid() && !m_pbox.contains(ray.o)))
            Throw("Periodic bounds but sampled ray not in contained in them.");


        if (m_max_depth >= 0)
            active &= depth < m_max_depth;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        dr::Loop<Mask> loop("Particle Tracer Integrator", active, depth, ray,
                            throughput, eta, sampler);

        loop.set_max_iterations(m_max_depth);
        
        // Incrementally build light path using BSDF sampling.
        while (loop(active)) {
            Log(Debug, "====== LOOP depth: %d ======", depth);
            Log(Debug, "ray.o: %f", ray.o);
            Log(Debug, "ray.d: %f", ray.d);
            Mask escaped_pbound = false;
            Mask active_surface = active;

            SurfaceInteraction3f si = 
                scene->ray_intersect(ray, 
                                    /* ray_flags = */ +RayFlags::All, 
                                    /* coherent = */ dr::eq(depth, 0u),
                                    active);
            Float t = si.t;

            /* ------------------- Periodic Bound Part 1 -------------------- */
            Vector3f wrapped_ray_origin = dr::zeros<Vector3f>();
            if(dr::any_or<true>(pbounds_valid)){
                auto [no_intersect, tmin, tmax] = m_pbox.ray_intersect(ray);
                Vector3f pbox_si = ray(tmax) + ray.d * math::RayEpsilon<Float>;

                // Escaped boundary if intersection with it is smaller than surface intersection.
                escaped_pbound = pbounds_valid &&  tmax < si.t;
                // Mark any rays that exist pbounds by the top or bottom as inactive.
                active &= !(escaped_pbound && (pbox_si.z() <= m_pbox.min.z() || pbox_si.z() >= m_pbox.max.z()));
                // add safeguards in case we get stuck in an infinite loop
                periodic_count = dr::select(escaped_pbound, periodic_count + 1, 0);
                active &= periodic_count < max_periodic_iterations;

                // invalidate si, and set next t to the pbox intersection distance
                // NOTE: might need to increment t accordingly for volumetric interactions?
                dr::masked(si.t, escaped_pbound) =  dr::Infinity<Float>;
                dr::masked(t, escaped_pbound) = tmax;

                Log(Debug, "pbox_si.x: %.6f, pbox_si.y: %.6f, pbox_si.z: %.6f", pbox_si.x(), pbox_si.y(), pbox_si.z());
                Log(Debug, "tmin: %f, tmax: %f, no_intersect: %d, si.t: %f", tmin, tmax, no_intersect, si.t);
                
                // Log(Debug, "ray.d*eps: %f", ray.d*math::RayEpsilon<Float>);

                // apply modulo of the offset on the ray origin
                Vector3f offset = pbox_si - m_pbox.min;
                Vector3f wrapped_offset = offset - dr::floor(offset / pbound_extent) * pbound_extent;
                wrapped_ray_origin = wrapped_offset + m_pbox.min;
                

                Log(Debug, "offset: %d, wrapped_offset: %d", offset, wrapped_offset);
                Log(Debug, "ray: %d", ray.o);
            }

            /* ------------------------- Accumulate ------------------------- */
            Mask pass = order_filter(depth);
            pass &= bsdf_filter(si);

            // Accumulate the ray contribution, could be NEE or other strategies.
            sensor->accumulate(
                ray, 
                si, 
                /*tmax=*/t,
                /*emitted=*/throughput, 
                /*throughput=*/Spectrum(1.f), 
                /*sample_scale=*/sample_scale,
                /*filter=*/pass, 
                /*active=*/active);

            active_surface &= 
                (depth + 1 < m_max_depth) 
                && si.is_valid() 
                && !escaped_pbound;

            Log(Debug, "escaped_pbound: %d, active_surface: %d, active: %d", escaped_pbound, active_surface, active);

            /* ------------------- Periodic Bound Part 2 -------------------- */
            dr::masked(ray.o, escaped_pbound) = wrapped_ray_origin;

            if (dr::any_or<false>(escaped_pbound)) {
                Log(Debug, "continue to next");
                continue; // early continue for scalar mode
            }

            if (dr::none_or<false>(active_surface)) {
                break; // early exit for scalar mode
            }
            
            /* ----------------------- BSDF sampling ------------------------ */
            BSDFPtr bsdf = si.bsdf(ray);

            // Sample BSDF * cos(theta).
            BSDFContext ctx(TransportMode::Importance);
            auto [bs, bsdf_val] =
                bsdf->sample(ctx, si, sampler->next_1d(active),
                             sampler->next_2d(active), active);

            // Using geometric normals (wo points to the camera)
            Float wi_dot_geo_n = dr::dot(si.n, -ray.d),
                  wo_dot_geo_n = dr::dot(si.n, si.to_world(bs.wo));

            // Prevent light leaks due to shading normals
            active &= (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                      (wo_dot_geo_n * Frame3f::cos_theta(bs.wo) > 0.f);

            // Adjoint BSDF for shading normals -- [Veach, p. 155]
            Float correction = dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                       (Frame3f::cos_theta(bs.wo) * wi_dot_geo_n));

            // ------ Update loop variables based on current interaction ------
            dr::masked(throughput, active_surface) *= bsdf_val * correction;
            dr::masked(eta, active_surface) *= bs.eta;

            // Spawn ray for next iteration
            dr::masked(ray, active_surface) = si.spawn_ray(si.to_world(bs.wo));
            Log(Debug, "spawned ray.o: %d, ray.d: %d", ray.o, ray.d);
            // -------------------- Stopping criterion ---------------------
            
            dr::masked(depth, si.is_valid()) += 1;

            // Russian Roulette
            Mask use_rr = depth > m_rr_depth;
            if (dr::any_or<true>(use_rr)) {
                Float q = dr::minimum(
                    dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), 0.95f);
                dr::masked(active, use_rr) &= sampler->next_1d(active) < q;
                dr::masked(throughput, use_rr) *= dr::rcp(q);
            }


            active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            active &= active_surface || !escaped_pbound;
        }

        return { throughput, 1.f };
    }

    /**
     * \brief accumulate the values that pass the filters
     */
    Mask order_filter(const UInt32& depth) const {
        // filtered value
        Mask pass = depth >= m_min_depth && depth < m_max_depth;
        return pass;
        
    }

    Mask bsdf_filter(const SurfaceInteraction3f& si) const {
        Mask pass(true);
        if (dr::none_or<false>(si.is_valid()))
            return pass;
        BSDFPtr bsdf = si.bsdf();
        pass = dr::eq(bsdf->filter(), +FilterType::Include);
        return pass;
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("ParticleAccumulatorIntegrator[\n"
                           "  min_depth = %u,\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(VolumeIntegrator, Integrator);
MI_IMPLEMENT_CLASS_VARIANT(ParticleAccumulatorIntegrator, VolumeIntegrator);
MI_EXPORT_PLUGIN(ParticleAccumulatorIntegrator, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)
