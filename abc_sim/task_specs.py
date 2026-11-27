"""Task registry for policy deployment and delivered replay."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

# Every task in this catalogue is assembled from the one bimanual scene family, so the
# scene name exported alongside an episode is a constant rather than a per-spec field.
# The internal sim datasets record it, so exports have to carry it too.
DEFAULT_SCENE = "hybrid"

SHAPE_MATCHING_PROMPTS = (
    "Place the eight colored blocks shown in the picture into the matching board slots. Leave the gray slots empty.",
    "Match the reference card by filling its eight colored positions and leaving the other board slots empty.",
    "Arrange the eight colored shapes shown on the card in the board. Leave the gray positions empty.",
    "Fill only the eight colored positions shown in the reference image, matching each block's shape and color.",
    "Choose the eight blocks pictured in color and put them into their matching openings in the board.",
    "Put the eight shapes shown in color into the board, matching the picture and leaving the gray slots empty.",
)

FRUIT_BOWL_COLOR_NAMES = ("blue", "white", "red", "yellow", "green", "purple", "orange", "pink")
PACK_AND_POUR_FRUIT_TEMPLATES = (
    "Place all the fruits into the {source_color} bowl, then pour the fruits from the "
    "{source_color} bowl into the large {target_color} bowl. "
    "Finally, put the empty {source_color} bowl back on the table.",
    "Put the three fruits in the {source_color} bowl, then lift it by the rim and pour them "
    "into the large {target_color} bowl. Finish by setting the empty {source_color} bowl upright on the table.",
    "Gather all three fruits in the small {source_color} bowl, then tip them into the large {target_color} bowl. "
    "Return the empty {source_color} bowl to the table.",
    "Fill the {source_color} bowl with the fruits, then hold its rim and pour everything "
    "into the large {target_color} bowl. "
    "Set the empty {source_color} bowl back down on the table.",
    "First collect the fruits in the {source_color} bowl. Then pick up that bowl and empty it "
    "into the large {target_color} bowl. Put the empty {source_color} bowl upright on the table when done.",
    "Pack all the fruit into the small {source_color} bowl and pour its contents into the large {target_color} bowl. "
    "Then place the empty {source_color} bowl upright on the table and let go.",
)
PACK_AND_POUR_FRUIT_PROMPTS = tuple(
    template.format(source_color=source, target_color=target)
    for source in FRUIT_BOWL_COLOR_NAMES
    for target in FRUIT_BOWL_COLOR_NAMES
    if source != target
    for template in PACK_AND_POUR_FRUIT_TEMPLATES
)

PACK_SUITCASE_PROMPTS = (
    "put all objects on the table into the suitcase and close it",
    "pack all three objects into the suitcase, then close the lid",
    "put everything on the table inside the suitcase and shut it",
    "place the three loose items in the suitcase and close it",
    "pack the scattered objects into the open suitcase and shut the lid",
    "put all three items in the suitcase, then shut it",
    "move the objects from the table into the suitcase and close the lid",
    "fill the suitcase with all three objects and close it",
    "pack away the three tabletop objects and shut the suitcase",
    "place all the loose objects inside the suitcase before closing it",
    "put the three objects into the open suitcase, then close it",
    "pack every object on the table into the suitcase and shut the lid",
)

LIFT_BOTTLES_UPRIGHT_PROMPTS = (
    "lift the bottles upright",
    "stand the bottles upright",
    "stand all three water bottles upright",
    "set the bottles upright on the table",
    "turn the fallen bottles upright",
    "pick up the bottles and stand them on their bases",
    "make all three bottles stand upright",
    "place the water bottles standing up on the table",
    "stand up the three bottles lying on their sides",
    "put all the bottles right side up",
    "lift each bottle and set it down upright",
    "set all three water bottles on their bases",
    "bring the lying bottles into an upright position",
    "get the three water bottles standing on the table",
    "turn each water bottle right side up",
    "stand the fallen water bottles back up",
)


@dataclass(frozen=True)
class SimTaskSpec:
    """Canonical definition of a deployable sim task."""

    name: str
    env_task: str
    prompt: str
    max_chunks: int = 10
    randomize: bool = True
    description: str = ""
    aliases: tuple[str, ...] = ()
    evaluator_name: str | None = None
    evaluator_options: tuple[tuple[str, object], ...] = ()
    reset_request: tuple[tuple[str, object], ...] = ()
    reset_constraints: tuple[tuple[str, object], ...] = ()

    def evaluator_kwargs(self) -> dict[str, object]:
        """Return evaluator options as a plain kwargs dict."""

        return dict(self.evaluator_options)

    def reset_request_options(self) -> dict[str, object] | None:
        """Return task defaults, including fixed randomization constraints."""

        return {**dict(self.reset_request), **dict(self.reset_constraints)} or None

    def resolve_reset_request(self, request: Any | None) -> Any:
        """Resolve caller options and reject changes to the task's fixed condition."""
        if not self.reset_constraints:
            return self.reset_request_options() if request is None else request
        if request is None:
            supplied = {}
        elif isinstance(request, Mapping):
            supplied = dict(request)
        elif is_dataclass(request) and not isinstance(request, type):
            supplied = asdict(request)
        else:
            raise TypeError("Constrained task reset options must be a mapping or dataclass.")
        resolved = {**dict(self.reset_request), **supplied}
        for key, required in self.reset_constraints:
            value = resolved.get(key)
            if value is not None and (type(value) is not type(required) or value != required):
                raise ValueError(f"Task {self.name!r} requires {key}={required!r}; got {value!r}.")
            resolved[key] = required
        return resolved

    def validate_replay_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Require recordings to identify every fixed reset condition."""
        values = {}
        for key, _ in self.reset_constraints:
            if key not in metadata or metadata[key] is None:
                raise ValueError(f"Recording for task {self.name!r} must identify {key!r}.")
            values[key] = metadata[key]
        self.resolve_reset_request(values)


def _normalize_key(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


_TASK_SPECS: tuple[SimTaskSpec, ...] = (
    SimTaskSpec(
        name="shape_matching",
        env_task="shape_matching",
        prompt=SHAPE_MATCHING_PROMPTS[0],
        max_chunks=240,
        description="Select eight of sixteen scattered pieces to fill the colored positions on a reference card.",
        evaluator_name="shape_matching",
        aliases=("shape matching", *SHAPE_MATCHING_PROMPTS[1:]),
    ),
    SimTaskSpec(
        name="pack_and_pour_fruit",
        env_task="pack_and_pour_fruit",
        prompt=PACK_AND_POUR_FRUIT_PROMPTS[0],
        max_chunks=120,
        description="Pack three fruits, pour them into a large bowl, and return the empty small bowl to the table.",
        evaluator_name="pack_and_pour_fruit",
        aliases=("pack and pour fruit", *PACK_AND_POUR_FRUIT_PROMPTS[1:]),
    ),
    SimTaskSpec(
        name="pack_suitcase",
        env_task="pack_suitcase",
        prompt=PACK_SUITCASE_PROMPTS[0],
        max_chunks=120,
        description="Pack three scattered household objects into an open suitcase and close its lid.",
        evaluator_name="pack_suitcase",
        aliases=PACK_SUITCASE_PROMPTS[1:],
    ),
    SimTaskSpec(
        name="lift_bottles_upright",
        env_task="lift_bottles_upright",
        prompt=LIFT_BOTTLES_UPRIGHT_PROMPTS[0],
        description="Stand three side-lying water bottles upright among household distractors.",
        evaluator_name="bottles_upright",
        aliases=LIFT_BOTTLES_UPRIGHT_PROMPTS[1:],
    ),
    SimTaskSpec(
        name="throw_plastic_bottles_in_bin",
        env_task="bottles",
        prompt="throw plastic bottles in bin",
        description="Throw plastic bottles into the bin.",
        evaluator_name="bottles_in_bin",
        aliases=(
            "bottles",
            "sim_throw_plastic_bottles_in_bin",
        ),
    ),
    SimTaskSpec(
        name="put_plastic_bottles_in_bin",
        env_task="put_bottles",
        prompt="put the plastic bottles in the bin",
        description="Put plastic bottles into the bin.",
        evaluator_name="put_bottles_in_bin",
        aliases=(
            "put_bottles",
            "water_bottles",
            "put plastic bottles in bin",
            "put the plastic bottles in the bin",
            "sim_put the plastic bottles in the bin",
            "sim_put_the_plastic_bottles_in_the_bin",
            "sim_put_the_plastic_bottles_in_bin",
        ),
    ),
    SimTaskSpec(
        name="conveyor_pick",
        env_task="conveyor_pick",
        prompt="pick objects from the conveyor belt and put them in the bin",
        description="Pick objects from a moving conveyor belt before they reach the end and place them into the bin.",
        evaluator_name="conveyor_pick_objects_in_bin",
        aliases=(
            "conveyor_pick",
            "sim_conveyor_pick",
            "pick objects from the conveyor",
            "pick objects off a belt before they fall off the end",
        ),
    ),
    SimTaskSpec(
        name="rl_conveyor_pick",
        env_task="rl_conveyor_pick",
        prompt="Pick the requested objects from the conveyor and gently place them in the bin.",
        max_chunks=120,
        description="RL conveyor: seeded all-three or semantic-category collection with normal distractor exits.",
        evaluator_name="rl_conveyor_pick",
    ),
    SimTaskSpec(
        name="count_into_opaque_box",
        env_task="count_into_opaque_box",
        prompt="put the prompted number of objects in the opaque box",
        description="Follow a sampled counting directive into the opaque box; the scene randomizer "
        "writes the live prompt each episode.",
        evaluator_name="count_into_opaque_box",
        aliases=("sim_count_into_opaque_box",),
    ),
    SimTaskSpec(
        name="rl_count_into_opaque_box",
        env_task="count_into_opaque_box",
        prompt="Follow the counting instruction and put the requested objects in the box.",
        evaluator_name="count_into_opaque_box",
        evaluator_options=(("partial_overfill_penalty", True),),
        reset_constraints=(("family_first", True),),
        description="Six counting families sampled before their attributes, including exact one.",
    ),
    SimTaskSpec(
        name="count_one_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        reset_request=(("target_count", 1),),
        prompt="put exactly one object in the opaque box",
        description="Put exactly one countable object into the opaque box.",
        aliases=(
            "sim_count_one_into_opaque_box",
            "put exactly 1 object in the opaque box",
            "put exactly one item in the opaque box",
            "put 1 item in opaque box",
        ),
    ),
    SimTaskSpec(
        name="count_two_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        reset_request=(("target_count", 2),),
        prompt="put exactly two objects in the opaque box",
        description="Put exactly two countable objects into the opaque box.",
        aliases=(
            "sim_count_two_into_opaque_box",
            "put exactly 2 objects in the opaque box",
            "put exactly two items in the opaque box",
            "put 2 items in opaque box",
        ),
    ),
    SimTaskSpec(
        name="count_three_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        reset_request=(("target_count", 3),),
        prompt="put exactly three objects in the opaque box",
        description="Put exactly three countable objects into the opaque box.",
        aliases=(
            "sim_count_three_into_opaque_box",
            "put exactly 3 objects in the opaque box",
            "put exactly three items in the opaque box",
            "put 3 items in opaque box",
        ),
    ),
    SimTaskSpec(
        name="count_four_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        reset_request=(("target_count", 4),),
        prompt="put exactly four objects in the opaque box",
        description="Put exactly four countable objects into the opaque box.",
        aliases=(
            "sim_count_four_into_opaque_box",
            "put exactly 4 objects in the opaque box",
            "put exactly four items in the opaque box",
            "put 4 items in opaque box",
        ),
    ),
    SimTaskSpec(
        name="put_relative",
        env_task="put_relative",
        prompt="put one object in a relative position to another object",
        description="Place one sampled object in a prompted relative direction around another sampled object.",
        evaluator_name="put_relative",
        aliases=(
            "put_relative",
            "sim_put_relative",
            "relative_put",
            "put object relative to object",
            "put one object relative to another",
        ),
    ),
    SimTaskSpec(
        name="rl_put_relative",
        env_task="put_relative",
        prompt="Follow the placement instruction, including a change of plan.",
        evaluator_name="rl_put_relative",
        reset_constraints=(("interrupted", True),),
    ),
    SimTaskSpec(
        name="rl_grab_specific_object_from_clutter",
        env_task="grab_clutter",
        prompt="Pick up the requested object.",
        evaluator_name="rl_grab_clutter",
        reset_constraints=(("interrupted", True),),
    ),
    SimTaskSpec(
        name="grab_specific_object_from_clutter",
        env_task="grab_clutter",
        prompt="grab the target object from the cluttered table and place it in the box",
        description="Grab the prompted target object from a cluttered tabletop full of distractors "
        "and place it in the target box.",
        evaluator_name="grab_clutter_target_in_box",
        aliases=(
            "grab_clutter",
            "grab object from clutter",
            "grab specific object from clutter",
            "grab specific object from cluttered table",
            "pick specific object from clutter",
            "sim_grab_specific_object_from_clutter",
            "sim_grab_specific_object_from_cluttered_table",
        ),
    ),
    SimTaskSpec(
        name="ball_tray_balancing",
        env_task="ball_tray_balancing",
        prompt="keep the ball balanced on the tray for 10 to 15 seconds, then gently place the tray on the table",
        description="Hold the lifted tray with the ball balanced for 10–15 seconds, then gently "
        "place it on the table with the ball still on it.",
        evaluator_name="ball_tray_balancing",
        aliases=(
            "ball tray balancing",
            "ball-on-tray balancing",
            "ball on tray balancing",
            "balance ball on tray",
            "keep ball on tray",
            "sim_ball_tray_balancing",
            "sim_ball_on_tray_balancing",
        ),
    ),
    SimTaskSpec(
        name="inhand_transfer_item_to_other_side",
        env_task="inhand_transfer",
        prompt="inhand transfer the item to other side",
        description="Pick the sampled kitchen tool with the near arm, pass it to the other hand, "
        "and set it down on the far side of the table.",
        evaluator_name="inhand_transfer_other_side",
        aliases=(
            "inhand_transfer",
            "inhand transfer",
            "sim_inhand_transfer",
            "sim_inhand_transfer_the_item_to_other_side",
            "transfer the item to the other side",
        ),
    ),
    SimTaskSpec(
        name="multi_drawer_search",
        env_task="multi_drawer_search",
        prompt="find the target object hidden in one of the drawers and place it in the bin",
        description="Remember objects seen across several drawers, then find and place a prompted "
        "sequence of them into the bin.",
        evaluator_name="multi_drawer_target_in_bin",
        aliases=(
            "drawer search",
            "multi drawer search",
            "multi-drawer search",
            "find object in drawer",
            "find block in drawer",
            "find target in drawer",
            "find target hidden in drawer",
            "sim_multi_drawer_search",
            "sim_find_target_hidden_in_drawer",
        ),
    ),
    SimTaskSpec(
        name="multi_drawer_search_2_shaped",
        env_task="multi_drawer_search",
        prompt="find two target objects in the drawers and place them in the bin one at a time",
        description="Training variant with two ordered targets and episode-unique drawer-opening "
        "credit; the caller controls the horizon.",
        evaluator_name="multi_drawer_search_shaped_v2",
        reset_constraints=(("target_count", 2),),
    ),
    SimTaskSpec(
        name="rl_put_markers_in_drawer",
        env_task="drawer",
        prompt="Put all the markers in the requested drawer, then close it.",
        evaluator_name="marker_drawer",
        reset_constraints=(("random_target_drawer", True),),
    ),
    SimTaskSpec(
        name="put_markers_in_top_drawer",
        env_task="drawer",
        prompt="put the markers in the top drawer",
        aliases=("sim_put_the_markers_in_the_top_drawer",),
    ),
    SimTaskSpec(
        name="put_markers_in_middle_drawer",
        env_task="drawer",
        prompt="put the markers in the middle drawer",
        aliases=("sim_put_the_markers_in_the_middle_drawer",),
    ),
    SimTaskSpec(
        name="put_markers_in_bottom_drawer",
        env_task="drawer",
        prompt="put the markers in the bottom drawer",
        aliases=("sim_put_the_markers_in_the_bottom_drawer",),
    ),
    SimTaskSpec(
        name="pouring",
        env_task="pour",
        prompt="pour from one container into another",
        evaluator_name="pour_beads_in_container",
        aliases=("sim_pouring", "sim_pouring_beads", "pour"),
    ),
    SimTaskSpec(
        name="hang_mug_on_mug_rack",
        env_task="mug_tree",
        prompt="hang mug on mug rack",
        evaluator_name="mug_on_rack_v2",
        # These three handles cannot pass over a rack branch at the sampled
        # scales. The remaining sixteen variants have collision-free fits.
        reset_request=(
            (
                "mug_variant_pool",
                tuple(f"mug_{i}" for i in range(19) if i not in (1, 7, 10)),
            ),
        ),
        aliases=(
            "sim_hang_mug_on_mug_rack",
            "sim_hang_the_mug_on_the_mug_rack",
            "mug_tree",
        ),
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up",
        env_task="mug_flip",
        prompt="turn mug right side up",
        evaluator_name="mug_flip_upright",
        aliases=(
            "sim_turn_mug_right_side_up",
            "sim_turn_the_mug_right_side_up",
            "mug_flip",
        ),
    ),
    SimTaskSpec(
        name="sweep_away_paper_scraps_from_table",
        env_task="sweep",
        prompt="sweep away paper scraps from the table",
        evaluator_name="sweep_away",
        aliases=("sim_sweep_away_paper_scraps_from_the_table", "sweep"),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts",
        env_task="nuts_bolts_sorting",
        prompt="sort the nuts and bolts into their matching bins",
        description="Sort loose nuts into the teal bin and loose bolts into the orange bin.",
        evaluator_name="nuts_bolts_sorting",
        aliases=(
            "nuts_bolts_sorting",
            "nuts and bolts sorting",
            "sort nuts and bolts",
            "sim_sort_nuts_and_bolts",
            "sim_sort_the_nuts_and_bolts",
        ),
    ),
    SimTaskSpec(
        name="sort_lego_blocks",
        env_task="lego_blocks_sorting",
        prompt="sort the red, yellow, and blue lego blocks into their matching bins",
        description="Sort red, yellow, and blue LEGO-style blocks into matching color bins.",
        evaluator_name="lego_blocks_sorting",
        aliases=(
            "lego_blocks_sorting",
            "lego sorting",
            "lego block sorting",
            "sort lego blocks",
            "sort red yellow blue lego blocks",
            "sim_sort_lego_blocks",
            "sim_sort_the_lego_blocks",
        ),
    ),
    SimTaskSpec(
        name="load_plates_into_dish_rack",
        env_task="dishrack",
        prompt="load plates into tabletop dish rack",
        evaluator_name="dishrack_plates_in_rack",
        evaluator_options=(
            ("margin_m", 0.05),
            ("height_min_m", -0.06),
            ("height_max_m", 0.45),
            ("upright_normal_z_max", 2**-0.5),  # At most 45 degrees from vertical.
            ("horizontal_normal_z_min", 0.75),
        ),
        aliases=(
            "sim_load_plates_into_tabletop_dish_rack",
            "sim_load_the_plates_into_the_dish_rack",
            "load the plates into the dish rack",
            "dishrack",
        ),
    ),
    SimTaskSpec(
        name="rl_load_plates_into_dish_rack",
        env_task="dishrack",
        prompt="Load all four plates upright into the tabletop dish rack.",
        evaluator_name="dishrack_plates_in_rack",
        evaluator_options=(
            ("margin_m", 0.05),
            ("height_min_m", -0.06),
            ("height_max_m", 0.45),
            ("upright_normal_z_max", 2**-0.5),
            ("horizontal_normal_z_min", 0.75),
        ),
        reset_constraints=(("plate_count", 4),),
    ),
    SimTaskSpec(
        name="set_up_chess_pieces_on_board",
        env_task="chess",
        prompt="set up chess pieces on the board",
        evaluator_name="chess_board_setup",
        aliases=("sim_set_up_chess_pieces_on_the_board", "chess"),
    ),
    SimTaskSpec(
        name="build_wood_block_tower",
        env_task="building_blocks",
        prompt="build a wood block tower",
        aliases=("sim_build_wood_block_tower", "building_blocks"),
    ),
    SimTaskSpec(
        name="spell_cat",
        env_task="blocks",
        prompt="spell cat",
        evaluator_name="spell_word",
        evaluator_options=(("word", "cat"),),
        aliases=("sim_spell_cat",),
    ),
    SimTaskSpec(
        name="spell_dog",
        env_task="blocks",
        prompt="spell dog",
        evaluator_name="spell_word",
        evaluator_options=(("word", "dog"),),
        aliases=("sim_spell_dog",),
    ),
    SimTaskSpec(
        name="spell_fish",
        env_task="blocks",
        prompt="spell fish",
        evaluator_name="spell_word",
        evaluator_options=(("word", "fish"),),
        aliases=("sim_spell_fish",),
    ),
    SimTaskSpec(
        name="spell_bair",
        env_task="blocks",
        prompt="spell bair",
        evaluator_name="spell_word",
        evaluator_options=(("word", "bair"),),
        aliases=("sim_spell_bair",),
    ),
    SimTaskSpec(
        name="spell_xdof",
        env_task="blocks",
        prompt="spell xdof",
        evaluator_name="spell_word",
        evaluator_options=(("word", "xdof"),),
        aliases=("sim_spell_xdof",),
    ),
    SimTaskSpec(
        name="spell_abc",
        env_task="blocks",
        prompt="spell abc",
        evaluator_name="spell_word",
        evaluator_options=(("word", "abc"),),
        aliases=("sim_spell_abc",),
    ),
    SimTaskSpec(
        name="spell_yam",
        env_task="blocks",
        prompt="spell yam",
        evaluator_name="spell_word",
        evaluator_options=(("word", "yam"),),
        aliases=("sim_spell_yam",),
    ),
    SimTaskSpec(
        name="spell_agi",
        env_task="blocks",
        prompt="spell agi",
        evaluator_name="spell_word",
        evaluator_options=(("word", "agi"),),
        aliases=("sim_spell_agi",),
    ),
    *(
        SimTaskSpec(
            name=f"spell_random_{length}_appearance",
            env_task="blocks",
            prompt=f"spell the sampled {length}-letter sequence with varied block appearances",
            evaluator_name="abc-spelling-placement-v2",
            evaluator_options=(("sequence_length", length),),
            reset_request=(
                ("sequence_length", length),
                ("randomize_appearance", True),
                ("appearance_profile", "native_wood" if length == 3 else "varied"),
            ),
            description=(
                "Spell three seeded distinct letters on the original wood blocks with modest wood tint and "
                "legible glyph-color variation."
                if length == 3
                else "Spell a seeded sequence of distinct letters with independently sampled "
                "cube colors, wood/plastic finishes, face panels and glyph decoration."
            ),
        )
        for length in (2, 3, 4)
    ),
)


_LOOKUP: dict[str, SimTaskSpec] = {}
for _spec in _TASK_SPECS:
    for _alias in {_spec.name, _spec.prompt, *_spec.aliases}:
        _LOOKUP[_normalize_key(_alias)] = _spec


def list_task_specs() -> tuple[SimTaskSpec, ...]:
    """Return all known sim task specs."""

    return _TASK_SPECS


def get_task_spec(name: str) -> SimTaskSpec:
    """Resolve a task spec from a canonical name, alias, or prompt text."""

    key = _normalize_key(name)
    try:
        return _LOOKUP[key]
    except KeyError as exc:
        available = ", ".join(spec.name for spec in _TASK_SPECS)
        raise KeyError(f"Unknown sim task '{name}'. Available: {available}") from exc


def maybe_get_task_spec(name: str | None) -> SimTaskSpec | None:
    """Resolve a task spec if present, otherwise return ``None``."""

    if not name:
        return None
    return _LOOKUP.get(_normalize_key(name))
