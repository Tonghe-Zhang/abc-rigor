"""Write a CPU-only contact sheet of the actual procedural material tiles."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from abc_sim.randomization.assets.block_appearance import (
    FAMILIES,
    LETTERS,
    REVISION,
    face_tile_view,
    glyph_masks,
    sample_appearance,
    texture_tiles,
    validate_appearance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sheet = Image.new("RGB", (1240, 1050), "white")
    draw = ImageDraw.Draw(sheet)
    heading = ImageFont.load_default()
    small = heading
    draw.text(
        (20, 12),
        "Seeded spelling materials: actual generated RGB tiles",
        fill="black",
        font=heading,
    )
    draw.text(
        (20, 46),
        "Face tiles shown upright; adjacent strips show body grain/color. No 3D lighting or GPU rendering.",
        fill="black",
        font=small,
    )
    records = [sample_appearance(seed) for seed in range(40)]
    for record in records:
        validate_appearance(record)
    tiles = []
    for row, family in enumerate(FAMILIES):
        top = 85 + row * 214
        draw.text((20, top), family.replace("_", " "), fill="black", font=heading)
        for column, letter in enumerate("ABJMQW"):
            record = next(
                record
                for record in records
                if record["blocks"][letter]["family"] == family
            )
            style = record["blocks"][letter]
            body, face = texture_tiles(letter, style)
            left = 20 + column * 202
            face_image = Image.fromarray(face_tile_view(letter, face))
            sheet.paste(face_image, (left, top + 31))
            sheet.paste(Image.fromarray(body[:, :48]), (left + 132, top + 31))
            draw.text(
                (left, top + 164),
                f"{letter} · seed {record['seed']} · {style['finish']}",
                fill="black",
                font=small,
            )
            draw.text(
                (left, top + 183),
                f"{style['panel']} / {style['border']}",
                fill="black",
                font=small,
            )
            tiles.append({"letter": letter, "seed": record["seed"], "style": style})
    draw.text(
        (20, 746),
        "All 26 cleaned glyph masks: original shapes, no legacy wood or border",
        fill="black",
        font=heading,
    )
    for index, letter in enumerate(LETTERS):
        mask = face_tile_view(letter, glyph_masks()[letter])
        tile = Image.fromarray(
            np.where(mask[..., None], 20, 245).repeat(3, axis=2).astype(np.uint8)
        )
        tile = tile.resize((82, 82))
        left, top = 20 + (index % 13) * 94, 785 + (index // 13) * 119
        sheet.paste(tile, (left, top))
        draw.text((left + 33, top + 86), letter, fill="black", font=small)
    output = args.output_dir / "contact-sheet.png"
    sheet.save(output)
    receipt = {
        "revision": REVISION,
        "preview": str(output.resolve()),
        "kind": "actual generated face/body tiles; no 3D renderer",
        "sampled_seeds": [0, 39],
        "tiles": tiles,
        "glyphs": LETTERS,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output_dir / "preview-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: receipt[key] for key in ("preview", "elapsed_seconds", "max_rss_kib")}
        )
    )


if __name__ == "__main__":
    main()
