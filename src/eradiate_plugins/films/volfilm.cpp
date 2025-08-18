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

 * - res_x, res_y, res_z
   - |int|
   - resolution of the film on x, y, and z axis respectively

This film is a three dimensional voxel film. It allows to write data in a 3D 
array and therefore doesn't use bitmaps. Note that it currently support a single
channel only. 

Write operations are done using `write_tensor`. The film can be read
using `develop`.

.. tabs::
    .. code-tab::  xml

        <film type="volfilm">
            <integer name="resx" value="1"/>
            <integer name="resy" value="1"/>
            <integer name="resz" value="1"/>
        </film>

    .. code-tab:: python

        'type': 'volfilm',
        'res_x': 1,
        'res_y': 1,
        'res_z': 1,

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
            props.get<uint32_t>("res_x", 1),
            props.get<uint32_t>("res_y", 1),
            props.get<uint32_t>("res_z", 1)
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

    const ScalarUInt32 &size(ScalarUInt32 idx) const override {
        return m_res[idx];
    }

    const ScalarUInt32 size_product() const override {
        return m_res.x() * m_res.y() * m_res.z();
    }

    const ScalarUInt32 &crop_size(ScalarUInt32 idx) const override {
        return m_res[idx];
    }
    
    const ScalarUInt32 crop_size_product() const override {
        return m_res.x() * m_res.y() * m_res.z();
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
        Log(Debug,"clear buffer");
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

    bool is_volume_film() const override { return true; }

    void write(const fs::path &/*path*/) const override {
        NotImplementedError("write");
    }

    void write_tensor(const Float values, const UInt32 idx, Mask active) override {
        if constexpr (!dr::is_jit_v<Float>){
            std::lock_guard<std::mutex> lock(m_mutex);
            Log(Debug,"accumulate val: %f, at idx : %d, active: %d", values, idx, active);
            dr::scatter_reduce(ReduceOp::Add, m_data.array(), values, idx, active);
        } else {
            dr::scatter_reduce(ReduceOp::Add, m_data.array(), values, idx, active);
        }
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
    std::mutex m_mutex;
};

MI_IMPLEMENT_CLASS_VARIANT(VolFilm, Film)
MI_EXPORT_PLUGIN(VolFilm, "Volumetric Film")
NAMESPACE_END(mitsuba)
