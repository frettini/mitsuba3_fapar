#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/filter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>

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


    bool m_film_scale;

    MI_DECLARE_CLASS()

};



template <typename Float, typename Spectrum>
class ParticleAccumulatorIntegrator final : public VolumeIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(VolumeIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_min_depth, m_max_depth)
    MI_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                    EmitterPtr, BSDF, BSDFPtr, Shape, ShapePtr, Medium, MediumPtr,
                    PhaseFunctionContext)

    ParticleAccumulatorIntegrator(const Properties &props) : Base(props) { 
        if (props.has_property("periodic_box")){
            auto obj = props.object("periodic_box");
            Shape *periodic_box = dynamic_cast<Shape *>(obj.get());

            m_pbox.reset();
            if(periodic_box){
                m_pbox = periodic_box->bbox();
            }
            m_pbox_extents = m_pbox.extents();
            m_periodic_box = periodic_box;
    
            if (periodic_box){
                if (!dr::all(has_flag(m_periodic_box->bsdf()->flags(), BSDFFlags::Null))) {
                    Throw("Periodic Box bsdf must have a Null flag!");
                }
            }
        }
        
    }

    MI_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, dr::eq(idx, 1u)) = spec[1];
            dr::masked(m, dr::eq(idx, 2u)) = spec[2];
        } else {
            DRJIT_MARK_USED(idx);
        }
        return m;
    }


    void sample(const Scene *scene, Sensor *sensor, Sampler *sampler, ScalarFloat sample_scale) const override {
        // Primary & further bounces illumination
        auto [ray, throughput, medium] = prepare_ray(scene, sensor, sampler);

        Float throughput_max = dr::max(unpolarized_spectrum(throughput));
        Mask active = dr::neq(throughput_max, 0.f);

        trace_light_ray(ray, scene, sensor, sampler, medium, throughput,
                        sample_scale, active);
    }

    /// Samples a ray from a random emitter in the scene.
    std::tuple<Ray3f, Spectrum, MediumPtr> prepare_ray(const Scene *scene,
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
        MediumPtr medium = emitter->medium();
        return { ray, ray_weight, medium };
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
                    Sampler *sampler, MediumPtr initial_medium, 
                    Spectrum throughput, ScalarFloat sample_scale, 
                    Mask active = true) const {
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
        
        // Initialize medium variables
        MediumPtr medium = initial_medium;
        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) dr::array_size_v<Spectrum>;
            channel = (UInt32) dr::minimum(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        // Initialize periodic bound variables
        Mask pbounds_valid = dr::neq(m_periodic_box, nullptr);
        UInt32 max_periodic_iterations = 10; // TODO: include with depth?
        UInt32 periodic_count = 0;

        // Initialize filter variables
        UInt32 sensor_filter = sensor->sensor_filter();

        Log(Debug, "trace_light_ray");

        if(dr::any(pbounds_valid && !m_pbox.contains(ray.o))){
            Log(Debug, "ray.o: %d.", ray.o);
            Log(Debug, "bbox: %d.", m_pbox);
            Throw("Periodic bounds but sampled ray not in contained in them");

        }

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
            Mask escaped_pbound = false, escaped_medium = false;
            Mask act_null_scatter = false, act_medium_scatter = false;

            Mask active_medium = active && dr::neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Log(Debug, "active_medium: %f, active_surface: %f", active_medium, active_surface);

            // If the medium does not have a spectrally varying extinction,
            // we can perform a few optimizations to speed up rendering
            Mask is_spectral = active_medium;
            Mask not_spectral = false;
            if (dr::any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            /* ------------------------ Interactions ------------------------ */
            SurfaceInteraction3f si = 
            scene->ray_intersect(ray, 
                /* ray_flags = */ +RayFlags::All, 
                /* coherent = */ dr::eq(depth, 0u),
                active);
                
            Float t = si.t;

            MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
            if (dr::any_or<true>(active_medium)) {
                mei = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                Log(Debug, "sampled mei.t: %f", mei.t);
                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;
                dr::masked(t, active_medium && mei.t < si.t) = mei.t;
                
              

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();

                Log(Debug, "escaped_medium: %f, active_medium: %f", escaped_medium, active_medium);
            }
            Log(Debug, "si.t: %f, mei.t: %f, t: %f", si.t, mei.t, t);

            

            /* ------------------------- Accumulate ------------------------- */
            
            // Apply filtering
            Mask pass = active;
            // Null interaction are discarded, Periodic bounds have to be Null
            pass &= depth_filter(depth, UInt32(m_min_depth), UInt32(m_max_depth), sensor_filter);
            pass &= bsdf_filter(si, mei, sensor_filter);
            pass &= shape_filter(si, mei, sensor_filter);
            pass &= phase_filter(si, mei, sensor_filter);

            Log(Debug, "active: %d, filter: %d", active, pass);

            // Accumulate the ray contribution, could be NEE or other strategies.
            sensor->accumulate(
                ray, 
                si, 
                mei,
                /*tmax=*/t,
                /*emitted=*/throughput, 
                /*throughput=*/Spectrum(1.f), 
                /*sample_scale=*/sample_scale,
                /*filter=*/pass, 
                /*active=*/active);


            dr::masked(depth, act_medium_scatter) += 1;
            active &= depth < (uint32_t) m_max_depth;
            act_medium_scatter &= active;

            Log(Debug, "act_null_scatter: %d, act_medium_scatter: %d", act_null_scatter, act_medium_scatter);

            // early exit
            if (dr::none_or<false>(active))
                break;
                
            /* ----------------- Scattering Event Selection ----------------- */
            if (dr::any_or<true>(active_medium)) {
                  if (dr::any_or<true>(is_spectral)) {
                    auto [tr, free_flight_pdf] = medium->transmittance_eval_pdf(mei, si, is_spectral);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(throughput, is_spectral) *= dr::select(tr_pdf > 0.f, mei.combined_extinction*tr / tr_pdf, 0.f);
                    Log(Debug, "Spectral | tr: %f, tr_pdf: %f, throughput: %f", tr, tr_pdf, throughput);
                }
                
                // select scattering event
                Mask null_scatter = sampler->next_1d(active_medium) >= index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel);
                Log(Debug, "sigma_t: %f, sigma_maj: %f", mei.sigma_t, mei.combined_extinction);
                
                act_null_scatter |= null_scatter && active_medium;
                act_medium_scatter |= !act_null_scatter && active_medium; 

                // Null scattering: update throughput only for spectral cases
                if (dr::any_or<true>(is_spectral && act_null_scatter)) {
                    dr::masked(throughput, is_spectral && act_null_scatter) *=
                        mei.sigma_n * index_spectrum(mei.combined_extinction, channel) /
                        (index_spectrum(mei.sigma_n, channel)*mei.combined_extinction);
                }
                 
            }
            if (dr::any_or<true>(act_null_scatter)) {
                dr::masked(ray.o, act_null_scatter) = mei.p;
                // dr::masked(si.t, act_null_scatter) = si.t - mei.t; // useful when optmizing the ray intersections
            }

            if (dr::any_or<true>(act_medium_scatter)) {
               
                // Real scattering: update throughput
                if (dr::any_or<true>(is_spectral)) {
                    Spectrum mul = mei.sigma_s * index_spectrum(mei.combined_extinction, channel) / (index_spectrum(mei.sigma_t, channel)*mei.combined_extinction); 
                    dr::masked(throughput, is_spectral && act_medium_scatter) *= mul;
                    Log(Debug, "Spectral Scattering throughput mul: %f", mul);
                }
                if (dr::any_or<true>(not_spectral)) {
                    Spectrum mul = mei.sigma_s / mei.sigma_t; 
                    dr::masked(throughput, not_spectral && act_medium_scatter) *= mul;
                    Log(Debug, "Non Spectral Scattering throughput mul: %f", mul);

                }    
            
            /* ----------------------- Phase sampling ----------------------- */
                PhaseFunctionContext phase_ctx(sampler);
                auto phase = mei.medium->phase_function();

                dr::masked(phase, !act_medium_scatter) = nullptr;
                auto [wo, phase_weight, phase_pdf] = phase->sample(phase_ctx, mei,
                    sampler->next_1d(act_medium_scatter),
                    sampler->next_2d(act_medium_scatter),
                    act_medium_scatter);
                act_medium_scatter &= phase_pdf > 0.f;

            /* ------------------- Update loop variables -------------------- */
                Ray3f new_ray  = mei.spawn_ray(wo);
                dr::masked(ray, act_medium_scatter) = new_ray;
                dr::masked(throughput, act_medium_scatter) *= phase_weight;
            }

            active_surface |= escaped_medium;
            active_surface &= si.is_valid();

            Log(Debug, "active_surface: %d", active_surface);

            if (dr::any_or<true>(active_surface)) {
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

            /* ------------------- Update loop variables -------------------- */
                
                dr::masked(throughput, active_surface) *= bsdf_val * correction;
                dr::masked(eta, active_surface) *= bs.eta;
                
                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                dr::masked(depth, non_null_bsdf) += 1;
                
            /* ------------------------- Ray Update ------------------------- */
                // Spawn ray for next iteration
                escaped_pbound = pbounds_valid && active_surface && dr::eq(m_periodic_box, si.shape);
                
                // In the case of not escaping pbound, update using BSDF sample
                dr::masked(ray, !escaped_pbound) = si.spawn_ray(si.to_world(bs.wo));
                
                // Update ray and active mask 
                if(dr::any_or<true>(escaped_pbound)){
                    // Mark any rays that exist pbounds by the top or bottom as inactive.
                    Point3f pbox_si = ray(t + math::RayEpsilon<Float>);
                    active &= !(escaped_pbound && (pbox_si.z() <= m_pbox.min.z() || pbox_si.z() >= m_pbox.max.z()));

                    // add safeguards in case we get stuck in an infinite loop
                    periodic_count = dr::select(escaped_pbound, periodic_count + 1, 0);
                    active &= periodic_count < max_periodic_iterations;

                    // apply modulo of the offset on the ray origin
                    Vector3f offset = pbox_si - m_pbox.min;
                    Vector3f wrapped_offset = offset - dr::floor(offset / m_pbox_extents) * m_pbox_extents;
                    dr::masked(ray.o, escaped_pbound) = wrapped_offset + m_pbox.min;
                    
                    Log(Debug, "offset: %d, wrapped_offset: %d", offset, wrapped_offset);
                    Log(Debug, "ray: %d", ray.o);
                }
                
                Log(Debug, "spawned ray.o: %d, ray.d: %d", ray.o, ray.d);
                
                Mask has_medium_trans                = active_surface && si.is_medium_transition();
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
                
            }
        
            /* --------------------- Stopping criterion --------------------- */
            // Russian Roulette
            Mask use_rr = depth > m_rr_depth && !escaped_pbound;
            if (dr::any_or<true>(use_rr)) {
                Float q = dr::minimum(dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), 0.95f);
                dr::masked(active, use_rr) &= sampler->next_1d(active) < q;
                dr::masked(throughput, use_rr) *= dr::rcp(q);
            }
            
            active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            active &= (active_surface | active_medium);
            active &= depth < (uint32_t) m_max_depth;
            Log(Debug, "throughput: %d, use_rr: %d, active: %d", throughput, use_rr, active);
        }

        return { throughput, 1.f };
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

protected:
    ShapePtr m_periodic_box = nullptr;
    ScalarBoundingBox3f m_pbox;
    ScalarVector3f m_pbox_extents;

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(VolumeIntegrator, Integrator);
MI_IMPLEMENT_CLASS_VARIANT(ParticleAccumulatorIntegrator, VolumeIntegrator);
MI_EXPORT_PLUGIN(ParticleAccumulatorIntegrator, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)
