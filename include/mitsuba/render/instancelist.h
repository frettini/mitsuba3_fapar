#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/shape.h>
// #include <mitsuba/render/shapegroup.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB InstanceList : public Object {
public:
    MI_IMPORT_TYPES(Shape)

    const std::vector<ref<Shape>> &shapes() { return m_shapes; }

    ScalarBoundingBox3f bbox() { return m_bbox; }

    MI_DECLARE_CLASS()
protected:
    InstanceList(const Properties &props);
    inline InstanceList() { }
    virtual ~InstanceList();

private:
    uint32_t m_instance_count;
    ScalarBoundingBox3f m_bbox;
    std::vector<ref<Shape>> m_shapes;
};

MI_EXTERN_CLASS(InstanceList)
NAMESPACE_END(mitsuba)