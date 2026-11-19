"""Task randomizer registry."""

from __future__ import annotations

from .core import SceneRandomizer
from .tasks.ball_sorting import BallSortingRandomizer
from .tasks.ball_tray import BallTrayBalancingRandomizer
from .tasks.blocks import BlocksRandomizer
from .tasks.bottles import BottlesRandomizer, WaterBottleRandomizer
from .tasks.bottles_upright import BottlesUprightRandomizer
from .tasks.chess import ChessRandomizer
from .tasks.conveyor_pick import ConveyorPickObjectRandomizer
from .tasks.count_box import CountIntoOpaqueBoxRandomizer
from .tasks.dishrack import DishRackRandomizer
from .tasks.grab_clutter import GrabClutterRandomizer
from .tasks.drawer import DrawerRandomizer
from .tasks.mugs import MugFlipRandomizer, MugTreeRandomizer
from .tasks.marker import MarkerRandomizer
from .tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from .tasks.pack_and_pour_fruit import PackAndPourFruitRandomizer
from .tasks.pack_suitcase import PackSuitcaseRandomizer
from .tasks.pour import PourRandomizer
from .tasks.put_relative import PutRelativeRandomizer
from .tasks.shape_matching import ShapeMatchingRandomizer
from .tasks.sorting import LegoBlocksSortingRandomizer, NutsBoltsSortingRandomizer
from .tasks.sweep import SweepRandomizer

_WATER_BOTTLE_RANDOMIZER = WaterBottleRandomizer()

TASK_RANDOMIZERS: dict[str, SceneRandomizer] = {
    "shape_matching": ShapeMatchingRandomizer(),
    "pack_and_pour_fruit": PackAndPourFruitRandomizer(),
    "pack_suitcase": PackSuitcaseRandomizer(),
    "lift_bottles_upright": BottlesUprightRandomizer(),
    "bottles":      BottlesRandomizer(),
    "put_bottles":  _WATER_BOTTLE_RANDOMIZER,
    "water_bottles": _WATER_BOTTLE_RANDOMIZER,
    "marker":       MarkerRandomizer(),
    "pour":         PourRandomizer(),
    "drawer":       DrawerRandomizer(),
    "dishrack":     DishRackRandomizer(),
    "blocks":       BlocksRandomizer(),
    "mug_tree":     MugTreeRandomizer(),
    "mug_flip":     MugFlipRandomizer(),
    "ball_sorting": BallSortingRandomizer(),
    "chess":        ChessRandomizer(),
    "sweep":        SweepRandomizer(),
    "count_into_opaque_box": CountIntoOpaqueBoxRandomizer(),
    "put_relative": PutRelativeRandomizer(),
    "conveyor_pick": ConveyorPickObjectRandomizer(),
    "ball_tray_balancing": BallTrayBalancingRandomizer(),
    "grab_clutter":  GrabClutterRandomizer(),
    "multi_drawer_search": MultiDrawerSearchRandomizer(),
    "nuts_bolts_sorting": NutsBoltsSortingRandomizer(),
    "lego_blocks_sorting": LegoBlocksSortingRandomizer(),
    # "empty": no free-jointed objects to randomize
}


__all__ = ['TASK_RANDOMIZERS']
