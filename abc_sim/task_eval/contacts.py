"""Contact helpers shared by tasks that require physical placement."""

import mujoco


def subtree_geom_ids(model: mujoco.MjModel, root: int) -> set[int]:
    bodies = {root}
    for body in range(model.nbody):
        ancestor = body
        while ancestor > 0:
            if ancestor == root:
                bodies.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    return {
        geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) in bodies
    }


def gripper_geom_ids(model: mujoco.MjModel) -> set[int]:
    names = (
        "link_left_finger",
        "link_right_finger",
        "_lf_rot",
        "_lf_down",
        "_rf_rot",
        "_rf_down",
    )
    result: set[int] = set()
    for body in range(model.nbody):
        if any(part in (model.body(body).name or "") for part in names):
            result.update(subtree_geom_ids(model, body))
    return result


def has_contact(data: mujoco.MjData, first: set[int], second: set[int]) -> bool:
    return any(
        (int(contact.geom1) in first and int(contact.geom2) in second)
        or (int(contact.geom2) in first and int(contact.geom1) in second)
        for contact in data.contact[: data.ncon]
    )
