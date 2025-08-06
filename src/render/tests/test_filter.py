import pytest
import drjit as dr
import mitsuba as mi

def test01_depth_filter(variant_scalar_rgb):
    depth_flag = mi.SensorFilterFlags.Depth

    # depth tests
    assert mi.depth_filter(2, 2, 10, depth_flag)
    assert not mi.depth_filter(1, 2, 10, depth_flag)
    assert not mi.depth_filter(11, 2, 10, depth_flag)
    
    # flag tests
    assert mi.depth_filter(3, 2, 10, mi.SensorFilterFlags.BSDF)
    assert not mi.depth_filter(3, 2, 10, mi.SensorFilterFlags.BSDF | mi.SensorFilterFlags.Exclusif)
    assert mi.depth_filter(3, 2, 10, mi.SensorFilterFlags.Depth | mi.SensorFilterFlags.Exclusif)


def test02_bsdf_filter(variant_scalar_rgb):
    bsdf_flag = mi.SensorFilterFlags.BSDF
    si = dr.zeros(mi.SurfaceInteraction3f)
    mei = dr.zeros(mi.MediumInteraction3f)

    shape_include = mi.load_dict({
        "type":"rectangle",
        "bsdf":{ "type":"diffuse", "filter":0 },
    })

    shape_exclude = mi.load_dict({
        "type":"rectangle",
        "bsdf":{ "type":"diffuse", "filter":1 },
    })

    null_shape = mi.load_dict({
        "type":"rectangle",
        "bsdf":{"type":"null", "filter":0},
    })

    si.t = 1.
    mei.t = dr.inf

    # test filter property
    si.shape = shape_include
    assert mi.bsdf_filter(si, mei, bsdf_flag)
    
    si.shape = shape_exclude
    assert not mi.bsdf_filter(si, mei, bsdf_flag)
    
    # test null bsdf
    si.shape = null_shape
    assert not mi.bsdf_filter(si, mei, bsdf_flag)

    # test invalid surface 
    si.t = dr.inf
    mei.t = 0.5
    si.shape = shape_include
    assert mi.bsdf_filter(si, mei, bsdf_flag)

    mei.t = dr.inf
    assert mi.bsdf_filter(si, mei, bsdf_flag)

    # test flags
    si.t = 1.
    mei.t = dr.inf
    si.shape = shape_include
    assert mi.bsdf_filter(si, mei, mi.SensorFilterFlags.Depth)
    assert not mi.bsdf_filter(si, mei, mi.SensorFilterFlags.Depth | mi.SensorFilterFlags.Exclusif)
    assert mi.bsdf_filter(si, mei, mi.SensorFilterFlags.BSDF | mi.SensorFilterFlags.Exclusif)


def test02_shape_filter(variant_scalar_rgb):
    shape_flag = mi.SensorFilterFlags.Shape
    si = dr.zeros(mi.SurfaceInteraction3f)
    mei = dr.zeros(mi.MediumInteraction3f)

    shape_include = mi.load_dict({
        "type":"rectangle",
        "filter":0,
    })

    shape_exclude = mi.load_dict({
        "type":"rectangle",
        "filter":1,
    })

    si.t = 1.
    mei.t = dr.inf

    # test filter property
    si.shape = shape_include
    assert mi.shape_filter(si, mei, shape_flag)
    
    si.shape = shape_exclude
    assert not mi.shape_filter(si, mei, shape_flag)
    
    # test invalid surface
    si.t = dr.inf
    mei.t = 0.5
    si.shape = shape_include
    assert mi.shape_filter(si, mei, shape_flag)

    mei.t = dr.inf
    assert mi.shape_filter(si, mei, shape_flag)

    # test flags
    si.t = 1.
    mei.t = dr.inf
    si.shape = shape_include
    assert mi.shape_filter(si, mei, mi.SensorFilterFlags.Depth)
    assert not mi.shape_filter(si, mei, mi.SensorFilterFlags.Depth | mi.SensorFilterFlags.Exclusif)
    assert mi.shape_filter(si, mei, mi.SensorFilterFlags.Shape | mi.SensorFilterFlags.Exclusif)


def test03_phase_filter(variant_scalar_rgb):
    phase_flag = mi.SensorFilterFlags.Phase
    si = dr.zeros(mi.SurfaceInteraction3f)
    mei = dr.zeros(mi.MediumInteraction3f)

    medium_include = mi.load_dict({
        "type":"homogeneous",
        "phase":{
            "type":"rayleigh",
            "filter":0
        },
    })

    medium_exclude = mi.load_dict({
        "type":"homogeneous",
        "phase":{
            "type":"rayleigh",
            "filter":1
        },
    })

    si.t = dr.inf
    mei.t = 1.

    # test filter property
    mei.medium = medium_include
    assert mi.phase_filter(si, mei, phase_flag)
    
    mei.medium = medium_exclude
    assert not mi.phase_filter(si, mei, phase_flag)
    
    # test invalid medium
    si.t = 0.5
    mei.t = dr.inf
    mei.medium = medium_include
    assert mi.phase_filter(si, mei, phase_flag)

    mei.t = dr.inf
    assert mi.phase_filter(si, mei, phase_flag)

    # test flags
    si.t = dr.inf
    mei.t = 1.
    mei.medium = medium_include
    assert mi.phase_filter(si, mei, mi.SensorFilterFlags.Depth)
    assert not mi.phase_filter(si, mei, mi.SensorFilterFlags.Depth | mi.SensorFilterFlags.Exclusif)
    assert mi.phase_filter(si, mei, mi.SensorFilterFlags.Phase | mi.SensorFilterFlags.Exclusif)

