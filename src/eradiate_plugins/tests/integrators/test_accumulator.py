import mitsuba as mi
import numpy as np


def test_accumulator_pbox(variant_scalar_rgb):
    def scene_dict(periodic_box : bool):
        if  periodic_box:
            integrator = mi.load_dict({
                "type":"paccumulator",
                "max_depth":10,
                "pbox_min":(-1,-1,-1),
                "pbox_max":( 1, 1, 1),
            })
        else:
            integrator = mi.load_dict({ "type":"paccumulator", "max_depth":10 })

        return mi.load_dict({
            "type": "scene",
            "sensor": {
                "type":"count",
                "film": {
                    "type": "volfilm",
                    "res_x": 1, "res_y": 1, "res_z": 1,
                }
            },
            "integrator": integrator,
            "surface": {
                "type": "rectangle",
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=(0, 0.05, 0),
                    target=(0, 1, 0),
                    up=(0, 0, 1),
                ),
                "bsdf":{
                    "type":"diffuse",
                    "reflectance":0.3,
                },
            },
            "emitter": {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=(0, -0.05, 0),
                    target=(0, -1, 0),
                    up=(0, 0, 1),
                ),
                'area_emitter': {
                    'type': 'directionalarea',
                    'radiance': {'type': 'rgb', 'value': (1.0, 0.5, 0.2)},
                },
            },
        })

    scene = scene_dict(periodic_box=False)
    res = mi.render(scene, spp=10)
    assert np.allclose(res.numpy().squeeze(), np.array(0))

    scene = scene_dict(periodic_box=True)
    res = mi.render(scene, spp=10)
    assert np.allclose(res.numpy().squeeze(), np.array(1))