#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/instancelist.h>
#include <mitsuba/render/shapegroup.h>
#include <mitsuba/core/transform.h>
#include <drjit/tensor.h>


NAMESPACE_BEGIN(mitsuba)

MI_VARIANT InstanceList<Float, Spectrum>::InstanceList(const Properties &props) {
    using ShapeGroup_ = ShapeGroup<Float, Spectrum>;
    Log(Debug, "Start Constructor");

    ShapeGroup_ *shapegroup{ nullptr }; 
    for (auto &kv : props.objects()) {
        Shape *shape = dynamic_cast<Shape *>(kv.second.get());
        if (shape && shape->is_shapegroup()) {
            if (shapegroup)
                Throw("Only a single shapegroup can be specified per instance list.");
            shapegroup = (ShapeGroup_ *) shape;
        } else {
            Throw("Only a shapegroup can be specified in an instance list.");
        }
    }

    if (!shapegroup)
        Throw("A reference to a 'shapegroup' must be specified!");

    TensorXf* tensor = props.tensor<TensorXf>("transforms");
    if (tensor->ndim() != 3)
        Throw("Tensor->has %ul dimensions. Expected 3", tensor->ndim());

    m_instance_count = (uint32_t) tensor->shape(0);
    m_bbox = dr::zeros<ScalarBoundingBox3f>();
    
    Log(Debug, "Start Creating Instances");
    auto pmgr = PluginManager::instance();
    for (uint32_t i = 0; i < m_instance_count; ++i) {
        ScalarTransform4f to_world( dr::gather<ScalarMatrix4f>(tensor->array(), i) );
        Properties props_instance("instance");
        props_instance.set_object("shapegroup", shapegroup);
        props_instance.set_transform("to_world", to_world);
        ref<Shape> shape = pmgr->create_object<Shape>(props_instance);
        m_bbox.expand( shape->bbox() );

        m_shapes.push_back(shape);
    }
    Log(Debug, "End Creating Instances");
}

MI_VARIANT InstanceList<Float, Spectrum>::~InstanceList() { }

MI_IMPLEMENT_CLASS_VARIANT(InstanceList, Object, "instancelist")
MI_INSTANTIATE_CLASS(InstanceList)
NAMESPACE_END(mitsuba)