"""Pure initial-episode definitions shared by drawer scoring and restoration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrawerTarget:
    """One ordered retrieval goal."""

    body: str
    joint: str
    label: str
    drawer: str | None


@dataclass(frozen=True)
class DrawerEpisode:
    """Validated target order and source-drawer associations."""

    targets: tuple[DrawerTarget, ...]
    drawers: tuple[str, ...]
    drawer_joints: tuple[str, ...]

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> DrawerEpisode:
        """Parse a sequence without dropping malformed entries or changing count."""
        if "target_sequence" in metadata:
            sequence = metadata["target_sequence"]
            if not isinstance(sequence, (list, tuple)) or not sequence:
                raise ValueError("A drawer target sequence must be nonempty.")
        else:
            # Native single-target recordings predate ordered sequences.
            sequence = [
                {
                    "body": metadata.get("target_body", "drawer_search_target"),
                    "joint": metadata.get("target_joint", "drawer_search_target_joint"),
                    "label": metadata.get("target_label", "target object"),
                    "drawer": metadata.get("target_drawer"),
                }
            ]
        targets = []
        for target in sequence:
            if not isinstance(target, Mapping) or any(
                not isinstance(target.get(key), str) or not target[key]
                for key in ("body", "joint", "label")
            ):
                raise ValueError("Each drawer target needs a body, joint, and label.")
            drawer = target.get("drawer")
            if drawer is not None and (not isinstance(drawer, str) or not drawer):
                raise ValueError("A target's source drawer must be a nonempty name.")
            targets.append(
                DrawerTarget(
                    body=target["body"],
                    joint=target["joint"],
                    label=target["label"],
                    drawer=drawer,
                )
            )
        count = metadata.get("target_count", len(targets))
        if type(count) is not int or count != len(targets):
            raise ValueError(
                "Declared target_count must equal the target sequence length."
            )
        if (
            len({target.body for target in targets}) != count
            or len({target.joint for target in targets}) != count
        ):
            raise ValueError("A drawer sequence must identify each target only once.")
        drawers = tuple(metadata.get("active_drawers", ()))
        joints = tuple(metadata.get("drawer_joints", ()))
        if (
            len(drawers) != len(joints)
            or len(set(drawers)) != len(drawers)
            or len(set(joints)) != len(joints)
            or any(
                not isinstance(name, str) or not name for name in (*drawers, *joints)
            )
        ):
            raise ValueError(
                "Drawer names and joints must identify each active drawer once."
            )
        if drawers and any(target.drawer not in drawers for target in targets):
            raise ValueError("Every target must belong to an active drawer.")
        return cls(tuple(targets), drawers, joints)
