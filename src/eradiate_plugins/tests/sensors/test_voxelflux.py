import mitsuba as mi
import numpy as np


def sensor_dict(with_film: bool):
    if with_film:
        return {
            "type": "voxelflux",
            "surface_flux":False,
            "res_x":1,
            "res_y":1,
            "res_z":1,
            "bbox_min": [-0.49, -0.49, -0.5],
            "bbox_max": [0.49, 0.49, 0.5],
        }
    else:
        return {
            "type": "voxelflux",
            "surface_flux":False,
            "film": {
                "type": "tensorfilm",
                "ndims": 6,
                "sizes": "2, 3, 2, 2, 2, 1",
            },
            "bbox_min": [-0.49, -0.49, -0.49],
            "bbox_max": [0.49, 0.49, 0.49],
        }

def test_construct(variant_scalar_rgb):
    sensor = mi.load_dict({"type": "voxelflux"})
    assert sensor is not None

    sensor = mi.load_dict(sensor_dict(False))
    assert sensor is not None

    sensor = mi.load_dict(sensor_dict(True))
    assert sensor is not None


def test_voxelflux(variant_scalar_rgb):

    def voxel_flux_scene(surface:bool, light_dir:list=[0,0,-1]):
        scene = {
            "type": "scene",
            "sensor": sensor_dict(False),
            "integrator": {"type": "paccumulator", "max_depth": 10},
            "light": {
                "type": "directionalperiodic",
                "pbox_min": [-0.5, -0.5, 0],
                "pbox_max": [0.5, 0.5, 1],
                "irradiance": 1.0,
                "direction": [0, 0, -1],
            },
        }

        if surface:
            scene["surface"] = {
                "type":"rectangle",
                "bsdf":{
                    "type":"conductor", # perfect mirror
                },
                "to_world":mi.ScalarTransform4f().scale([2,2,1])
            }
        print(scene)
        return mi.load_dict(scene)

    scene = voxel_flux_scene(False)
    res = mi.render(scene, spp=1)
    gt = np.zeros((2,3,2,2,2,1))
    gt[0,2,0,0,:] = 1.
    print(res.numpy())
    assert np.allclose(res.numpy(), gt)

    scene = voxel_flux_scene(True)
    res = mi.render(scene, spp=1)
    gt = np.zeros((2,3,2,2,2,1))
    gt[:,2,0,0,1] = 1.
    print(res.numpy())
    assert np.allclose(res.numpy(), gt)


def test_energy_conservation(variant_scalar_rgb):
    scene = mi.load_dict({
        "type": "scene",
        "sensor": sensor_dict(False),
        "integrator": {"type": "paccumulator", "max_depth": 10},
        "light": {
            "type": "directionalperiodic",
            "pbox_min": [-0.5, -0.5, 0],
            "pbox_max": [0.5, 0.5, 1],
            "irradiance": 1.0,
            "direction": [0, 0, -1],
        },
        "surface":{
            "type":"rectangle",
            "bsdf":{
                "type":"diffuse",
                "reflectance":1.,
            },
            "to_world":mi.ScalarTransform4f().scale([2,2,1])
        }
    })
     
    res = mi.render(scene, spp=100)
    res = res.numpy()
    flux_in = res[1, :, 0, 0, 0, 0].sum() + res[0, 0, 1, 0, 0] + res[0, 1, 0, 1, 0] + res[0, 2, 0, 0, 1]
    flux_out = res[0, :, 0, 0, 0, 0].sum() + res[1, 0, 1, 0, 0] + res[1, 1, 0, 1, 0] + res[1, 2, 0, 0, 1]
    assert np.allclose(flux_in, flux_out)