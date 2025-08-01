#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!
.. _bsdf-selectbsdf:

Selector material (:monosp:`selectbsdf`)
----------------------------------------

.. pluginparameters::

 * - indices
   - |texture|
   - A texture of integer indices indicating which underlying BSDF is active as
     a function of space coordinates.
   - |exposed|

 * - (Nested plugin)
   - |bsdf|
   - At least two nested BSDF instances that are selected based on selection
     indices.
   - |exposed|, |differentiable|

This plugin implements a *selector* material, which uses a texture to select
one of an arbitrary number of nested BSDF plugins. The index texture should take
discrete integer values (even if it is internally stored using floating-point
numbers).

The index texture should be initialized carefully when using a ``bitmap``
plugin:

* set the ``raw`` parameter to ``True`` to prevent spectral pre-processing;
* use a ``nearest`` filter type to avoid interpolation between pixels;
* it is more efficient to use a single-channel storage.
*/

template <typename Float, typename Spectrum>
class SelectBSDF final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, BSDFPtr)

    SelectBSDF(const Properties &props) : Base(props) {
        // Collect selector texture
        m_indices = props.texture<Texture>("indices");

        // Initialize nested BSDFs
        size_t bsdf_index = 0;
        for (auto &[name, obj] : props.objects(false)) {
            Base *bsdf = dynamic_cast<Base *>(obj.get());
            if (bsdf) {
                m_nested_bsdf.push_back(bsdf);
                bsdf_index++;
                props.mark_queried(name);
            }
        }

        if (bsdf_index < 2)
            Throw("SelectBSDF: At least two child BSDFs must be specified!");

        // Initialize vectorized call dispatch table
        m_nested_bsdf_dr = dr::load<DynamicBuffer<BSDFPtr>>(
            m_nested_bsdf.data(), m_nested_bsdf.size());
        dr::eval(m_nested_bsdf_dr);

        // Set flags
        m_components.clear();
        for (size_t i = 0; i < bsdf_index; ++i)
            for (size_t j = 0; j < m_nested_bsdf[i]->component_count(); ++j)
                m_components.push_back(m_nested_bsdf[i]->flags(j));

        m_flags = 0;
        for (size_t i = 0; i < bsdf_index; ++i)
            m_flags |= m_nested_bsdf[i]->flags();
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("indices", m_indices.get(),
                             +ParamFlags::NonDifferentiable);
        for (size_t i = 0; i < m_nested_bsdf.size(); ++i) {
            callback->put_object("bsdf_" + std::to_string(i),
                                 m_nested_bsdf[i].get(),
                                 +ParamFlags::Differentiable);
        }
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        UInt32 index = m_indices->eval_1(si, active);
        BSDFPtr bsdf = dr::gather<BSDFPtr>(m_nested_bsdf_dr, index, active);
        return bsdf->sample(ctx, si, sample1, sample2, active);
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        UInt32 index = m_indices->eval_1(si, active);
        BSDFPtr bsdf = dr::gather<BSDFPtr>(m_nested_bsdf_dr, index, active);
        return bsdf->eval(ctx, si, wo, active);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        UInt32 index = m_indices->eval_1(si, active);
        BSDFPtr bsdf = dr::gather<BSDFPtr>(m_nested_bsdf_dr, index, active);
        return bsdf->pdf(ctx, si, wo, active);
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        UInt32 index = m_indices->eval_1(si, active);
        BSDFPtr bsdf = dr::gather<BSDFPtr>(m_nested_bsdf_dr, index, active);
        return bsdf->eval_pdf(ctx, si, wo, active);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SelectBSDF[" << std::endl
            << "  indices = " << string::indent(m_indices) << "," << std::endl;

        for (size_t i = 0; i < m_nested_bsdf.size(); ++i)
            oss << "  nested_bsdf[" << i
                << "] = " << string::indent(m_nested_bsdf[i]) << ","
                << std::endl;

        oss << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    ref<Texture> m_indices;
    std::vector<ref<Base>> m_nested_bsdf;
    DynamicBuffer<BSDFPtr> m_nested_bsdf_dr;
};

MI_IMPLEMENT_CLASS_VARIANT(SelectBSDF, BSDF)
MI_EXPORT_PLUGIN(SelectBSDF, "SelectBSDF material")
NAMESPACE_END(mitsuba)
