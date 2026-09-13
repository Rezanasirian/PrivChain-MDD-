"""Extract eGeMAPSv02 functionals from each participant's own speech turns.

The raw DAIC-WOZ WAV contains the participant, the interviewer, and silence.
This script uses transcript timings to concatenate only participant turns before
running the standard openSMILE eGeMAPSv02 functional set. Extraction is fixed
and label-free; corpus normalization is fitted later on the train split only.

Usage:
    python scripts/extract_egemaps.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from privchain.data.daic_woz import SpeechWindow, _read_participant_turns


def concatenate_intervals(
    signal: NDArray[np.float32],
    sample_rate: int,
    intervals: tuple[tuple[float, float], ...],
) -> NDArray[np.float32]:
    """Concatenate non-empty, clipped time intervals from a mono waveform."""
    pieces: list[NDArray[np.float32]] = []
    size = signal.shape[0]
    for start, stop in intervals:
        first = max(0, min(size, math.floor(start * sample_rate)))
        last = max(first, min(size, math.ceil(stop * sample_rate)))
        if last > first:
            pieces.append(signal[first:last])
    if not pieces:
        raise ValueError("participant transcript selects no audio samples")
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _participant_ids(root: Path, template: str, excluded: set[int]) -> list[int]:
    """Return participant IDs that have both a directory and a WAV file."""
    marker = template.replace("{pid}", "*")
    ids: list[int] = []
    for directory in root.glob(marker):
        try:
            pid = int(directory.name.split("_", maxsplit=1)[0])
        except ValueError:
            continue
        if pid not in excluded and (directory / f"{pid}_AUDIO.wav").is_file():
            ids.append(pid)
    return sorted(set(ids))


def _extract_one(
    pid: int,
    root: Path,
    directory_template: str,
    text_cfg: dict[str, Any],
    smile: Any,
    *,
    pad_seconds: float,
) -> NDArray[np.float32]:
    """Extract one 88-dimensional eGeMAPSv02 vector."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("eGeMAPS extraction needs the `audio` extra") from exc

    directory = root / directory_template.format(pid=pid)
    turns = _read_participant_turns(
        directory / text_cfg["file_template"].format(pid=pid),
        delimiter=text_cfg.get("delimiter", "\t"),
        speaker_column=text_cfg["speaker_column"],
        value_column=text_cfg["value_column"],
        participant_speaker=text_cfg["participant_speaker"],
        start_column=text_cfg.get("start_column"),
        stop_column=text_cfg.get("stop_column"),
    )
    window = SpeechWindow.from_turns(turns, pad_seconds=pad_seconds)
    waveform, sample_rate = sf.read(directory / f"{pid}_AUDIO.wav", dtype="float32", always_2d=True)
    mono = np.asarray(waveform.mean(axis=1), dtype=np.float32)
    participant = concatenate_intervals(mono, int(sample_rate), window.intervals)
    frame = smile.process_signal(participant, int(sample_rate))
    values = np.asarray(frame.to_numpy(), dtype=np.float32)
    if values.shape != (1, 88):
        raise ValueError(
            f"participant {pid}: expected eGeMAPSv02 shape (1, 88), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"participant {pid}: eGeMAPSv02 produced NaN/Inf")
    return values


def main() -> None:
    """Extract every available participant and write an auditable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daic-config", type=Path, default=Path("configs/daic_woz.yaml"))
    parser.add_argument("--pad-seconds", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: stop after N IDs")
    args = parser.parse_args()
    if args.pad_seconds < 0.0:
        raise ValueError("--pad-seconds must be non-negative")

    try:
        import opensmile
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("eGeMAPS extraction needs the `audio` extra") from exc

    raw = yaml.safe_load(args.daic_config.read_text(encoding="utf-8"))
    cfg: dict[str, Any] = raw["daic_woz"]
    root = Path(cfg["root"])
    directory_template = str(cfg.get("participant_dir_template", "{pid}_P"))
    excluded = {int(pid) for pid in cfg.get("exclude_participants", [])}
    pids = _participant_ids(root, directory_template, excluded)
    if args.limit is not None:
        pids = pids[: args.limit]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    written = skipped = 0
    for index, pid in enumerate(pids, start=1):
        directory = root / directory_template.format(pid=pid)
        output = directory / f"{pid}_eGeMAPSv02.csv"
        if output.is_file() and not args.overwrite:
            skipped += 1
        else:
            values = _extract_one(
                pid,
                root,
                directory_template,
                cfg["text"],
                smile,
                pad_seconds=args.pad_seconds,
            )
            temporary = output.with_suffix(".csv.tmp")
            np.savetxt(temporary, values, delimiter=",", fmt="%.9g")
            temporary.replace(output)
            written += 1
        print(
            f"[{index}/{len(pids)}] participant={pid} written={written} skipped={skipped}",
            flush=True,
        )

    manifest = {
        "feature_set": "eGeMAPSv02",
        "feature_level": "Functionals",
        "width": 88,
        "speech_source": "participant transcript turns",
        "pad_seconds": args.pad_seconds,
        "participants": len(pids),
        "written": written,
        "skipped_existing": skipped,
        "opensmile_version": getattr(opensmile, "__version__", "unknown"),
    }
    (root / "_egemaps_v02_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
