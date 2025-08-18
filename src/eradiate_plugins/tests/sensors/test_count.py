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
