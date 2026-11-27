"""Native camera segmentation shares camera geometry and restores RGB rendering."""

from types import SimpleNamespace

import numpy as np
import pytest

from abc_sim.rendering.replay.camera_providers import MujocoCameraProvider


def test_segmentation_uses_actual_occlusion_and_restores_rgb():
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <visual><quality offsamples="0"/><global offwidth="32" offheight="32"/></visual>
      <worldbody>
        <light pos="0 0 2"/>
        <camera name="top" pos="0 0 2"/>
        <body name="target"><geom name="target_geom" type="box" size=".15 .15 .15" rgba="1 0 0 1"/></body>
        <body name="occluder" pos="2 0 .5"><geom type="box" size=".3 .3 .1"/></body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    provider = MujocoCameraProvider(model, data, width=32, height=32, camera_names=("top",))
    try:
        ids = provider.segmentation_for_step()["top"]
        target = model.geom("target_geom").id
        assert ids.shape == (32, 32, 2)
        assert np.any((ids[..., 0] == target) & (ids[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        model.body_pos[model.body("occluder").id, 0] = 0
        mujoco.mj_forward(model, data)
        hidden = provider.segmentation_for_step()["top"]
        assert not np.any((hidden[..., 0] == target) & (hidden[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        rgb = provider.frames_for_step(1, 0.0)["top"]
        assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8
    finally:
        provider.close()


def test_failed_segmentation_render_restores_rgb_mode():
    changes = []

    def fail():
        raise RuntimeError("render failed")

    provider = MujocoCameraProvider.__new__(MujocoCameraProvider)
    provider.camera_names = ("top",)
    provider._data = object()
    provider._renderer = SimpleNamespace(
        enable_segmentation_rendering=lambda: changes.append("segmentation"),
        disable_segmentation_rendering=lambda: changes.append("rgb"),
        update_scene=lambda *_args, **_kwargs: None,
        render=fail,
    )
    with pytest.raises(RuntimeError, match="render failed"):
        provider.segmentation_for_step()
    assert changes == ["segmentation", "rgb"]
