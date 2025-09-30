import mitsuba as mi
import numpy as np


def sensor_dict():
    return {
        "type": "absorbedflux",
        # "surface_flux":False, #to remove
        "film": {
            "type": "volfilm",
            "res_x": 1,
            "res_y": 1,
            "res_z": 1,
        },
        "bbox_min": [-2, -2, -2],
        "bbox_max": [2, 2, 2],
    }


def test_construct(variant_scalar_rgb):
    sensor = mi.load_dict({"type": "absorbedflux"})
    assert sensor is not None

    sensor = mi.load_dict(sensor_dict())
    assert sensor is not None


def test_absorbedflux(variant_scalar_rgb):
    scene = mi.load_dict(
        {
            "type": "scene",
            "sensor": sensor_dict(),
            "integrator": {"type": "paccumulator", "max_depth": 10},
            "surface": {
                "type": "rectangle",
                "bsdf":{
                    "type":"diffuse",
                    "reflectance":0.3,
                },
            },
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
    assert np.allclose(res.numpy().squeeze(), 0.7)


def test_closed_box(variant_scalar_rgb):
    def closed_box_scene(reflectance, direction):
        return mi.load_dict(
            {
                "type": "scene",
                "sensor": sensor_dict(),
                "integrator": {
                    "type": "paccumulator", 
                    "max_depth": 1000,
                    "rr_depth":1000,
                },
                "surface": {
                    "type": "cube",
                    "flip_normals":True,
                    "bsdf":{
                        "type":"diffuse",
                        "reflectance":reflectance,
                    },
                },
                "light": {
                    "type": "directionalperiodic",
                    "pbox_min": [-0.5, -0.5, 0],
                    "pbox_max": [0.5, 0.5, 0.1],
                    "irradiance": 1.0,
                    "direction": direction,
                },
            }
        )

    scene = closed_box_scene(0.3, [0.,0.,-1])
    res = mi.render(scene, spp=1)
    print(res.numpy().squeeze())
    assert np.allclose(res.numpy().squeeze(), 1.)

    scene = closed_box_scene(0., [-0.5,0.,-0.5])
    res = mi.render(scene, spp=1)
    print(res.numpy().squeeze(), np.cos(np.deg2rad(45)))
    # assert np.allclose(res.numpy().squeeze(), 1.)
    assert np.allclose(res.numpy().squeeze(), np.cos(np.deg2rad(45)))

    scene = closed_box_scene(0.3, [-0.5,0.,-0.5])
    res = mi.render(scene, spp=1)
    print(res.numpy().squeeze(), np.cos(np.deg2rad(45)))
    assert np.allclose(res.numpy().squeeze(), np.cos(np.deg2rad(45)))