#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>

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
    
        m_rr_depth = props.get<int>("rr_depth", 5);
        if (m_rr_depth <= 0)
            Throw("\"rr_depth\" must be set to a value greater than zero!");
    
        m_max_depth = props.get<int>("max_depth", -1);
        if (m_max_depth < 0 && m_max_depth != -1)
            Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
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

    /**
    * Longest visualized path depth (\c -1 = infinite).
    * A value of \c 1 will visualize only directly visible light sources.
    * \c 2 will lead to single-bounce (direct-only) illumination, and so on.
    */
    int m_max_depth;

    /// Depth to begin using russian roulette
    int m_rr_depth;

    bool m_film_scale;

    MI_DECLARE_CLASS()

};



template <typename Float, typename Spectrum>
class ParticleAccumulatorIntegrator final : public VolumeIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(VolumeIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_max_depth)
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

    /**
     * Samples an emitter in the scene and connects it directly to the sensor,
     * splatting the emitted radiance to the given image block.
     */
    // void sample_visible_emitters(const Scene *scene, const Sensor *sensor,
    //                              Sampler *sampler, ScalarFloat sample_scale) const {
    //     // 1. Time sampling
    //     Float time = sensor->shutter_open();
    //     if (sensor->shutter_open_time() > 0)
    //         time += sampler->next_1d() * sensor->shutter_open_time();

    //     // 2. Emitter sampling (select one emitter)
    //     auto [emitter_idx, emitter_idx_weight, _] =
    //         scene->sample_emitter(sampler->next_1d());

    //     EmitterPtr emitter =
    //         dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);

    //     // Don't connect delta emitters with sensor (both position and direction)
    //     Mask active = !has_flag(emitter->flags(), EmitterFlags::Delta);

    //     // 3. Emitter position sampling
    //     Spectrum emitter_weight = dr::zeros<Spectrum>();
    //     SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

    //     // 3.a. Infinite emitters
    //     Mask is_infinite = has_flag(emitter->flags(), EmitterFlags::Infinite),
    //          active_e = active && is_infinite;
    //     if (dr::any_or<true>(active_e)) {
    //         /* Sample a direction toward an envmap emitter starting
    //            from the center of the scene (the sensor is not part of the
    //            scene's bounding box, which could otherwise cause issues.) */
    //         Interaction3f ref_it(0.f, time, dr::zeros<Wavelength>(),
    //                              sensor->world_transform().translation());

    //         auto [ds, dir_weight] = emitter->sample_direction(
    //             ref_it, sampler->next_2d(active), active_e);

    //         /* Note: `dir_weight` already includes the emitter radiance, but
    //            that will be accounted for again when sampling the wavelength
    //            below. Instead, we recompute just the factor due to the PDF.
    //            Also, convert to area measure. */
    //         emitter_weight[active_e] =
    //             dr::select(ds.pdf > 0.f, dr::rcp(ds.pdf), 0.f) *
    //             dr::sqr(ds.dist);

    //         si[active_e] = SurfaceInteraction3f(ds, ref_it.wavelengths);
    //     }

    //     // 3.b. Finite emitters
    //     active_e = active && !is_infinite;
    //     if (dr::any_or<true>(active_e)) {
    //         auto [ps, pos_weight] =
    //             emitter->sample_position(time, sampler->next_2d(active), active_e);

    //         emitter_weight[active_e] = pos_weight;
    //         si[active_e] = SurfaceInteraction3f(ps, dr::zeros<Wavelength>());
    //     }

    //     /* 4. Connect to the sensor.
    //        Query sensor for a direction connecting to `si.p`, which also
    //        produces UVs on the sensor (for splatting). The resulting direction
    //        points from si.p (on the emitter) toward the sensor. */
    //     auto [sensor_ds, sensor_weight] = sensor->sample_direction(si, sampler->next_2d(), active);
    //     si.wi = sensor_ds.d;

    //     // 5. Sample spectrum of the emitter (accounts for its radiance)
    //     auto [wavelengths, wav_weight] =
    //         emitter->sample_wavelengths(si, sampler->next_1d(active), active);
    //     si.wavelengths = wavelengths;
    //     si.shape       = emitter->shape();

    //     // Spectrum weight = emitter_idx_weight * emitter_weight * wav_weight * sensor_weight;

    //     // // No BSDF passed (should not evaluate it since there's no scattering)
    //     // connect_sensor(scene, si, sensor_ds, nullptr, weight, block, sample_scale, active);
    // }

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

        Int32 depth = 0;

        Log(Debug, "trace_light_ray");
        /* ---------------------- Path construction ------------------------- */
        // First intersection from the emitter to the scene
        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Log(Debug, "si.is_valid : %d", si.is_valid());

        active &= si.is_valid();
        if (m_max_depth >= 0)
            active &= depth < m_max_depth;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        dr::Loop<Mask> loop("Particle Tracer Integrator", active, depth, ray,
                            throughput, si, eta, sampler);
        
        // Incrementally build light path using BSDF sampling.
        while (loop(active)) {
            BSDFPtr bsdf = si.bsdf(ray);
            /* Connect to sensor and splat if successful. Sample a direction
               from the sensor to the current surface point. */
            // auto [sensor_ds, sensor_weight] =
            //     sensor->sample_direction(si, sampler->next_2d(), active);
            // connect_sensor(scene, si, sensor_ds, bsdf,
            //                throughput * sensor_weight, sample_scale,
            //                active);

            // Accumulate the ray contribution, could be NEE or other strategies.
            sensor->accumulate(
                ray, 
                si, 
                /*emitted=*/throughput, 
                /*throughput=*/Spectrum(1.f), 
                /*sample_scale=*/sample_scale,
                /*filter=*/Mask(true), 
                /*active=*/active);

            /* ----------------------- BSDF sampling ------------------------ */
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
            throughput *= bsdf_val * correction;
            eta *= bs.eta;

            active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            if (dr::none_or<false>(active))
                break;

            // Intersect the BSDF ray against scene geometry (next vertex).
            ray = si.spawn_ray(si.to_world(bs.wo));
            si = scene->ray_intersect(ray, active);

            depth++;
            if (m_max_depth >= 0)
                active &= depth < m_max_depth;
            active &= si.is_valid();

            // Russian Roulette
            Mask use_rr = depth > m_rr_depth;
            if (dr::any_or<true>(use_rr)) {
                Float q = dr::minimum(
                    dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), 0.95f);
                dr::masked(active, use_rr) &= sampler->next_1d(active) < q;
                dr::masked(throughput, use_rr) *= dr::rcp(q);
            }
        }

        return { throughput, 1.f };
    }

    /**
     * Attempt connecting the given point to the sensor.
     *
     * If the point to connect is on the surface (non-null `bsdf` values),
     * evaluate the BSDF in the direction of the sensor.
     *
     * Finally, splat `weight` (with all appropriate factors) to the
     * given image block.
     *
     * \return The quantity that was accumulated to the block.
     */
    // Spectrum connect_sensor(const Scene *scene,
    //                         const SurfaceInteraction3f &si,
    //                         const DirectionSample3f &sensor_ds,
    //                         const BSDFPtr &bsdf, const Spectrum &weight,
    //                         ImageBlock *block, ScalarFloat sample_scale,
    //                         Mask active) const {
    //     active &= (sensor_ds.pdf > 0.f) &&
    //               dr::any(dr::neq(unpolarized_spectrum(weight), 0.f));
    //     if (dr::none_or<false>(active))
    //         return 0.f;

    //     // Check that sensor is visible from current position (shadow ray).
    //     Ray3f sensor_ray = si.spawn_ray_to(sensor_ds.p);
    //     active &= !scene->ray_test(sensor_ray, active);
    //     if (dr::none_or<false>(active))
    //         return 0.f;

    //     // Foreshortening term and BSDF value for that direction (for surface interactions).
    //     Spectrum result = 0.f;
    //     Spectrum surface_weight = 1.f;
    //     Vector3f local_d        = si.to_local(sensor_ray.d);
    //     Mask on_surface         = active && dr::neq(si.shape, nullptr);
    //     if (dr::any_or<true>(on_surface)) {
    //         /* Note that foreshortening is only missing for directly visible
    //            emitters associated with a shape. Otherwise it's included in the
    //            BSDF. Clamp negative cosines (zero value if behind the surface). */

    //         surface_weight[on_surface && dr::eq(bsdf, nullptr)] *=
    //             dr::maximum(0.f, Frame3f::cos_theta(local_d));

    //         on_surface &= dr::neq(bsdf, nullptr);
    //         if (dr::any_or<true>(on_surface)) {
    //             BSDFContext ctx(TransportMode::Importance);
    //             // Using geometric normals
    //             Float wi_dot_geo_n = dr::dot(si.n, si.to_world(si.wi)),
    //                   wo_dot_geo_n = dr::dot(si.n, sensor_ray.d);

    //             // Prevent light leaks due to shading normals
    //             Mask valid = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
    //                          (wo_dot_geo_n * Frame3f::cos_theta(local_d) > 0.f);

    //             // Adjoint BSDF for shading normals -- [Veach, p. 155]
    //             Float correction = dr::select(valid,
    //                 dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
    //                         (Frame3f::cos_theta(local_d) * wi_dot_geo_n)), 0.f);

    //             surface_weight[on_surface] *=
    //                 correction * bsdf->eval(ctx, si, local_d, on_surface);
    //         }
    //     }

    //     /* Even if the ray is not coming from a surface (no foreshortening),
    //        we still don't want light coming from behind the emitter. */
    //     Mask not_on_surface = active && dr::eq(si.shape, nullptr) && dr::eq(bsdf, nullptr);
    //     if (dr::any_or<true>(not_on_surface)) {
    //         Mask invalid_side = Frame3f::cos_theta(local_d) <= 0.f;
    //         surface_weight[not_on_surface && invalid_side] = 0.f;
    //     }

    //     result = weight * surface_weight * sample_scale;

        
    //     /* Splatting, adjusting UVs for sensor's crop window if needed.
    //        The crop window is already accounted for in the UV positions
    //        returned by the sensor, here we just need to compensate for
    //        the block's offset that will be applied in `put`. */
    //     Float alpha = dr::select(dr::neq(bsdf, nullptr), 1.f, 0.f);
    //     Vector2f adjusted_position = sensor_ds.uv + block->offset();

    //     /* Splat RGB value onto the image buffer. The particle tracer
    //        does not use the weight channel at all */
    //     block->put(adjusted_position, si.wavelengths, result, alpha,
    //                /* weight = */ 0.f, active);

    //     return result;
    // }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("ParticleAccumulatorIntegrator[\n"
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
