"""Task evaluation helpers for automatic reward/success scoring."""

from abc_sim.task_eval.base import (
    EpisodeTaskEvaluator,
    PhysicsObservedTaskEvaluator,
    TaskEvalResult,
    TaskEvaluator,
)
from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator
from abc_sim.task_eval.bottles import BottlesInBinEvaluator, PutBottlesInBinEvaluator
from abc_sim.task_eval.chess import ChessBoardSetupEvaluator
from abc_sim.task_eval.conveyor import ConveyorPickObjectsInBinEvaluator
from abc_sim.task_eval.count_box import CountIntoOpaqueBoxEvaluator
from abc_sim.task_eval.dishrack import DishrackEvaluator
from abc_sim.task_eval.grab_clutter import GrabClutterTargetInBoxEvaluator
from abc_sim.task_eval.inhand_transfer import InhandTransferOtherSideEvaluator
from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval.mug_tree import MugOnRackEvaluator
from abc_sim.task_eval.multi_drawer import MultiDrawerSearchTargetInBinEvaluator
from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
from abc_sim.task_eval.pour import PourBeadsInContainerEvaluator
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_eval.spell import SpellWordEvaluator
from abc_sim.task_eval.spelling_placement import FreePlacementSpellingEvaluator
from abc_sim.task_eval.sweep import SweepAwayEvaluator
from abc_sim.task_eval.registry import make_task_evaluator

__all__ = [
    "TaskEvalResult",
    "TaskEvaluator",
    "EpisodeTaskEvaluator",
    "PhysicsObservedTaskEvaluator",
    "BottlesInBinEvaluator",
    "PutBottlesInBinEvaluator",
    "ChessBoardSetupEvaluator",
    "ConveyorPickObjectsInBinEvaluator",
    "CountIntoOpaqueBoxEvaluator",
    "DishrackEvaluator",
    "MugFlipUprightEvaluator",
    "SweepAwayEvaluator",
    "BallTrayBalancingEvaluator",
    "GrabClutterTargetInBoxEvaluator",
    "InhandTransferOtherSideEvaluator",
    "LegoBlocksSortingEvaluator",
    "MugOnRackEvaluator",
    "MultiDrawerSearchTargetInBinEvaluator",
    "NutsBoltsSortingEvaluator",
    "PourBeadsInContainerEvaluator",
    "PutRelativeEvaluator",
    "SpellWordEvaluator",
    "FreePlacementSpellingEvaluator",
    "make_task_evaluator",
]
