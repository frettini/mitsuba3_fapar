#include <drjit/dynamic.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/imageblock.h>

#include <mutex>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _film-volfilm:

Volumetric film (:monosp:`volfilm`)
-------------------------------------------

.. pluginparameters::
 :extra-rows: 7

 * - resx, resy, resz
   - |int|
   - resolution of the film on x, y, and z axis respectively

 * - pixel_format
   - |string|
   - Specifies the desired pixel format of output images. The options are :monosp:`luminance`,
     :monosp:`luminance_alpha`, :monosp:`rgb`, :monosp:`rgba`, :monosp:`xyz` and :monosp:`xyza`.
     (Default: :monosp:`rgb`)

Volumetric film.

.. tabs::
    .. code-tab::  xml

        <film type="volfilm">
            <string name="resx" value="1"/>
            <string name="resy" value="1"/>
            <string name="resz" value="1"/>
        </film>

    .. code-tab:: python

        'type': 'volfilm',
        'resx': 1,
        'resy': 1,
        'resz': 1,

 */

template <typename Float, typename Spectrum>
class VolFilm final : public Film<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset, m_sample_border,
                   m_filter, m_flags)
    MI_IMPORT_TYPES(ImageBlock)

    VolFilm(const Properties &props) : Base(props) {
        // Horizontal and vertical film resolution in pixels
        m_res = ScalarVector3u(
            props.get<uint32_t>("resx", 1),
            props.get<uint32_t>("resy", 1),
            props.get<uint32_t>("resz", 1)
        );

        m_n_channels = 1;
        size_t shape[4] = { 
            (size_t) m_res.z(), 
            (size_t) m_res.y(), 
            (size_t) m_res.x(), 
            m_n_channels
        };

        // Initialize a buffer with zeros and pass it to a tensor.
        using FloatX = DynamicBuffer<ScalarFloat>;
        size_t data_size = (size_t) m_res.z() * (size_t) m_res.y() * (size_t) m_res.x() * (size_t) m_n_channels;
        FloatX zeros = dr::zeros<FloatX>(data_size);
        m_data = TensorXf(zeros.data(), 4, shape);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("data", m_data, +ParamFlags::Differentiable);
        Base::traverse(callback);
    }

    size_t base_channels_count() const override {
        return (size_t) m_n_channels;
    }

    size_t prepare(const std::vector<std::string> &/*aovs*/) override {
        return (size_t) m_n_channels;
    }

    ref<ImageBlock> create_block(const ScalarVector2u &/*size*/, bool /*normalize*/,
                                 bool /*border*/) override {
        NotImplementedError("create_block");
    }

    void put_block(const ImageBlock */*block*/) override {
        NotImplementedError("put_block");
    }

    void clear() override {
        using FloatX = DynamicBuffer<ScalarFloat>;
        size_t data_size = (size_t) m_res.z() * (size_t) m_res.y() * (size_t) m_res.x() * (size_t) m_n_channels;
        FloatX zeros = dr::zeros<FloatX>(data_size);
        m_n_channels = 1;
        size_t shape[4] = { 
            (size_t) m_res.z(), 
            (size_t) m_res.y(), 
            (size_t) m_res.x(), 
            m_n_channels
        };
        m_data = TensorXf(zeros.data(), 4, shape);
    }

    TensorXf develop(bool /*raw*/ = false) const override {
        return m_data;
    }

    ref<Bitmap> bitmap(bool /*raw*/ = false) const override {
        NotImplementedError("bitmap");
    }

    void write(const fs::path &/*path*/) const override {
        NotImplementedError("write");
    }

    void write(const Float values, const UInt32 idx, Mask active) {
        dr::scatter_reduce(ReduceOp::Add, m_data.array(), values, idx, active);
    }

    void schedule_storage() override {
        dr::schedule(m_data);
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "VolFilm[" << std::endl
            << "  res = " << m_res << "," << std::endl
            << "  n_channels = " << m_n_channels << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    ScalarVector3u m_res;
    TensorXf m_data;
    ScalarUInt32 m_n_channels;
};

MI_IMPLEMENT_CLASS_VARIANT(VolFilm, Film)
MI_EXPORT_PLUGIN(VolFilm, "Volumetric Film")
NAMESPACE_END(mitsuba)
