#include <mitsuba/render/instancelist.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class InstanceListPlugin final : public InstanceList<Float, Spectrum> {
public:
    MI_IMPORT_BASE(InstanceList)
    MI_IMPORT_TYPES()

    InstanceListPlugin(const Properties &props) : Base(props) { }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(InstanceListPlugin, InstanceList)
MI_EXPORT_PLUGIN(InstanceListPlugin, "Instance List plugin")

NAMESPACE_END(mitsuba)