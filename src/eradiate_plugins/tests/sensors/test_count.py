import mitsuba as mi
import numpy as np


def sensor_dict():
    return {
        "type": "count",
        "film": {
            "type": "volfilm",
            "res_x": 1,
            "res_y": 1,
            "res_z": 1,
        },
        "bbox_min": [-1, -1, -1],
        "bbox_max": [1, 1, 1],
    }


def test_construct(variant_scalar_rgb):
    sensor = mi.load_dict({"type": "count"})
    assert sensor is not None

    sensor = mi.load_dict(sensor_dict())
    assert sensor is not None


def test_count(variant_scalar_rgb):
    scene = mi.load_dict(
        {
            "type": "scene",
            "sensor": sensor_dict(),
            "integrator": {"type": "paccumulator", "max_depth": 10},
            "surface": {"type": "rectangle"},
            "light": {
                "type": "directionalperiodic",
                "pbox_min": [-0.5, -0.5, 0],
                "pbox_max": [0.5, 0.5, 1],
                "irradiance": 1.0,
                "direction": [0, 0, -1],
            },
        }
    )

    res = mi.render(scene, spp=10)
    assert np.allclose(res.numpy().squeeze(), np.array(1))


def test_count_bbox(variant_scalar_rgb):
    
    def count_dict(in_emitter):
        return mi.load_dict({
            "type": "count",
            "film": {
                "type": "volfilm",
                "res_x": 1, "res_y": 1, "res_z": 1,
            },
            "bbox_min": [0, -0.5, -0.5] if in_emitter else [-0.5, -0.5, -0.5],
            "bbox_max": [0.5, 0.5, 0.5] if in_emitter else [0, 0.5, 0.5],
        })
    
    count_out = count_dict(False)
    count_in = count_dict(True)

    scene = mi.load_dict({
        "type": "scene",
        "count_out": count_out,
        "count_in": count_in,
        "integrator": {"type": "paccumulator", "max_depth": 10},
        "surface": {"type": "rectangle"},
        "light": {
            "type": "directionalperiodic",
            "pbox_min": [0, -0.5, -0.5],
            "pbox_max": [0.5, 0.5, 0.5],
            "irradiance": 1.0,
            "direction": [0, 0, -1],
        },
    })

    res = mi.render(scene, sensor=count_out, spp=10)
    assert np.allclose(res.numpy().squeeze(), np.array(0))

    res = mi.render(scene, sensor=count_in, spp=10)
    assert np.allclose(res.numpy().squeeze(), np.array(1.))