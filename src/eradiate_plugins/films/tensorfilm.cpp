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

.. _film-tensorfilm:

Tensor film (:monosp:`tensorfilm`)
-------------------------------------------

.. pluginparameters::
 :extra-rows: 7

 * - ndims
   - |int|
   - number of dimensions of the underlying tensor. (Default:monosp:`3`)

 * - sizes
   - |string|
   - size of each dimension of the underlying tensor. (Default:monosp:`1`)

This film allows to write to a tensor with arbitrary number of dimensions 
and sizes. This can be useful to accumulate values in non-standard formats.

Write operations are done using `write_tensor`. The film can be read
using `develop`.

.. tabs::
    .. code-tab::  xml

        <film type="tensorfilm">
            <string name="ndims" value="3"/>
            <string name="sizes" value="3, 10, 15"/>
        </film>

    .. code-tab:: python

        'type': 'tensorfilm',
        'ndims': 3,
        'sizes': '3, 10, 15',

 */

template <typename Float, typename Spectrum>
class TensorFilm final : public Film<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset, m_sample_border,
                   m_filter, m_flags)
    MI_IMPORT_TYPES(ImageBlock)

    TensorFilm(const Properties &props) : Base(props) {
        // Horizontal and vertical film resolution in pixels
        m_ndims = props.get<size_t>("ndims", 3);

        if (props.has_property("sizes")) {
            std::vector<std::string> sizes_str =
                string::tokenize(props.string("sizes"), " ,");

            if( sizes_str.size() != m_ndims) 
                Throw("'sizes' parameter must have the same size as 'ndims'");

            m_sizes.reserve(m_sizes.size());

            for (size_t i = 0; i < sizes_str.size(); ++i) {
                try {
                    ScalarUInt32 size( string::stof<ScalarFloat>(sizes_str[i]) );
                    m_sizes.push_back(size);
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", sizes_str[i]);
                }
            }
        } else {
            m_sizes = std::vector<ScalarUInt32>(m_ndims, ScalarUInt32(1));
        }

        m_data_size = 1;
        std::vector<size_t> sizes(m_ndims,0);

        for (size_t i = 0; i < m_sizes.size(); ++i) {
            m_data_size *= m_sizes[i];
            sizes[i] = (size_t) m_sizes[i];
        }

        // Initialize a buffer with zeros and pass it to a tensor.
        using FloatX = DynamicBuffer<ScalarFloat>;
        FloatX zeros = dr::zeros<FloatX>(m_data_size);
        m_data = TensorXf(zeros.data(), m_ndims, sizes.data());
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("data", m_data, +ParamFlags::Differentiable);
        Base::traverse(callback);
    }

    size_t base_channels_count() const override {
        return 1;
    }

    const ScalarUInt32 &size(ScalarUInt32 idx) const override {
        return m_sizes.at(idx);
    }

    const ScalarUInt32 size_product() const override {
        return m_data_size;
    }

    const ScalarUInt32 &crop_size(ScalarUInt32 idx) const override {
        return m_sizes.at(idx);
    }
    
    const ScalarUInt32 crop_size_product() const override {
        return m_data_size;
    }

    size_t prepare(const std::vector<std::string> &/*aovs*/) override {
        return 1;
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
        
        FloatX zeros = dr::zeros<FloatX>(m_data_size);
        
        std::vector<size_t> sizes(m_ndims,0);
        for (size_t i = 0; i < m_sizes.size(); ++i) {
            sizes[i] = (size_t) m_sizes[i];
        }

        m_data = TensorXf(zeros.data(), m_ndims, sizes.data());
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
        oss << "TensorFilm[" << std::endl
            << "  ndims = " << m_ndims << "," << std::endl
            << "  sizes = " << m_sizes << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    size_t m_ndims;
    std::vector<ScalarUInt32> m_sizes;
    size_t m_data_size;
    TensorXf m_data;

    std::mutex m_mutex;
};

MI_IMPLEMENT_CLASS_VARIANT(TensorFilm, Film)
MI_EXPORT_PLUGIN(TensorFilm, "Tensor Film")
NAMESPACE_END(mitsuba)
