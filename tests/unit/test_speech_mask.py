"""Tests for the participant-speech frame mask (Phase 1, ADR-0031).

The session-level view used to hand the acoustic branch every frame of the
recording — the interviewer's questions and the silence between turns included.
These tests pin the repaired behaviour on a tiny on-disk fixture whose feature
values encode their own row index, so an assertion can name exactly which frames
survived rather than only counting them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from privchain.data.daic_woz import DaicWozDataset, FeatureConfigError, SpeechWindow
from privchain.data.segment_alignment import TimedTurn
from privchain.data.text_vectorizers import HashingTextVectorizer

#: Frames per file in the fixture; one per second, so row index == seconds.
NUM_FRAMES = 10


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def daic_root(tmp_path: Path) -> Path:
    """A one-participant corpus at 1 Hz, with two participant turns.

    Ellie holds 0-3 s and 5-7 s; the participant holds 3-5 s and 7-9 s. Every
    feature row is ``[r, 100 + r]`` for source row ``r``, so the first channel of
    a retained frame is its own timestamp in seconds.
    """
    root = tmp_path / "daic"
    _write(
        root / "train.csv",
        "Participant_ID,PHQ8_Binary,PHQ8_Score\n300,1,15\n",
    )
    covarep = "\n".join(f"{r},{100 + r}" for r in range(NUM_FRAMES))
    _write(root / "300_P" / "300_COVAREP.csv", covarep + "\n")

    aus = ["frame, timestamp, confidence, success, AU01, AU02"]
    aus += [f"{r}, {r}, 0.99, 1, {r}, {100 + r}" for r in range(NUM_FRAMES)]
    _write(root / "300_P" / "300_CLNF_AUs.txt", "\n".join(aus) + "\n")

    transcript = "start_time\tstop_time\tspeaker\tvalue\n"
    transcript += "0.0\t3.0\tEllie\thow are you\n"
    transcript += "3.0\t5.0\tParticipant\ti feel tired most days\n"
    transcript += "5.0\t7.0\tEllie\tand your sleep\n"
    transcript += "7.0\t9.0\tParticipant\tnot good at all\n"
    _write(root / "300_P" / "300_TRANSCRIPT.csv", transcript)
    return root


def _config(root: Path, **audio: Any) -> dict[str, Any]:
    """The fixture's config, with ``audio`` keys layered over the defaults."""
    return {
        "root": str(root),
        "phq8_max": 24,
        "participant_dir_template": "{pid}_P",
        "feature_cache_dir": None,
        "splits": {"train": "train.csv"},
        "label_columns": {
            "participant_id": "Participant_ID",
            "phq_binary": "PHQ8_Binary",
            "phq_score": "PHQ8_Score",
        },
        "audio": {
            "file_template": "{pid}_COVAREP.csv",
            "has_header": False,
            "delimiter": ",",
            "max_frames": 100,
            "frame_stride": 1,
            "normalization": "none",
            # COVAREP carries no clock, so a row's time is its index at this rate.
            "sample_rate_hz": 1.0,
            **audio,
        },
        "video": {
            "file_template": "{pid}_CLNF_AUs.txt",
            "has_header": True,
            "delimiter": ",",
            "drop_columns": ["frame", "timestamp", "confidence", "success"],
            "max_frames": 100,
            "frame_stride": 1,
            "normalization": "none",
            "timestamp_column": "timestamp",
            "quality_columns": ["timestamp", "confidence", "success"],
        },
        "text": {
            "file_template": "{pid}_TRANSCRIPT.csv",
            "delimiter": "\t",
            "speaker_column": "speaker",
            "value_column": "value",
            "participant_speaker": "Participant",
            "start_column": "start_time",
            "stop_column": "stop_time",
            "dim": 8,
        },
    }


def _dataset(config: dict[str, Any]) -> DaicWozDataset:
    return DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))


def _seconds(config: dict[str, Any], modality: str) -> list[float]:
    """First-channel values of the retained frames, i.e. their times in seconds."""
    sample = _dataset(config)[0]
    return [float(v) for v in sample[modality][:, 0].tolist()]  # type: ignore[literal-required]


def test_without_the_mask_every_frame_survives(daic_root: Path) -> None:
    """The committed behaviour: the whole recording reaches the encoder."""
    assert _seconds(_config(daic_root), "audio") == list(range(NUM_FRAMES))


def test_the_mask_keeps_only_the_participants_own_frames(daic_root: Path) -> None:
    """Ellie's turns and the silence between them are dropped."""
    config = _config(
        daic_root, speech_mask={"enabled": True, "source": "transcript", "pad_seconds": 0.0}
    )
    # Participant spans are 3-5 s and 7-9 s, endpoints included.
    assert _seconds(config, "audio") == [3.0, 4.0, 5.0, 7.0, 8.0, 9.0]


def test_padding_widens_each_turn_and_merges_the_overlap(daic_root: Path) -> None:
    """One second of padding bridges the 5-7 s gap into a single span."""
    config = _config(
        daic_root, speech_mask={"enabled": True, "source": "transcript", "pad_seconds": 1.0}
    )
    assert _seconds(config, "audio") == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_padding_never_reaches_before_the_recording_starts(daic_root: Path) -> None:
    """A turn near zero is clamped rather than given a negative start."""
    window = SpeechWindow.from_turns(
        [TimedTurn(start=0.2, stop=1.0, text="hi")], pad_seconds=1.0, sample_rate_hz=1.0
    )
    assert window.intervals == ((0.0, 2.0),)


def test_video_is_placed_by_its_own_timestamp_column(daic_root: Path) -> None:
    """OpenFace records a clock, so the mask reads it instead of the row index."""
    config = _config(daic_root)
    config["video"]["speech_mask"] = {"enabled": True, "pad_seconds": 0.0}
    assert _seconds(config, "video") == [3.0, 4.0, 5.0, 7.0, 8.0, 9.0]


def test_the_stride_counts_speech_frames_not_file_rows(daic_root: Path) -> None:
    """Striding after the mask keeps one of every N frames the participant spoke.

    Striding on the file's row index instead would make ``decimate`` and the
    aggregating modes cover different frames whenever a mask is on.
    """
    config = _config(
        daic_root,
        frame_stride=2,
        speech_mask={"enabled": True, "source": "transcript", "pad_seconds": 0.0},
    )
    # Speech frames are 3,4,5,7,8,9; every second one of those is 3,5,8.
    assert _seconds(config, "audio") == [3.0, 5.0, 8.0]


def test_a_transcript_without_usable_timings_fails_loudly(daic_root: Path) -> None:
    """Masking a session away must raise, not fall back to the whole recording.

    This is a *data* fault, so it surfaces when that session is actually read —
    one damaged transcript must not stop a run at construction time (ADR-0010).
    """
    _write(daic_root / "train.csv", "Participant_ID,PHQ8_Binary,PHQ8_Score\n300,1,15\n301,0,3\n")
    covarep = "\n".join(f"{r},{100 + r}" for r in range(NUM_FRAMES))
    _write(daic_root / "301_P" / "301_COVAREP.csv", covarep + "\n")
    aus = ["frame, timestamp, confidence, success, AU01, AU02"]
    aus += [f"{r}, {r}, 0.99, 1, {r}, {100 + r}" for r in range(NUM_FRAMES)]
    _write(daic_root / "301_P" / "301_CLNF_AUs.txt", "\n".join(aus) + "\n")
    broken = "start_time\tstop_time\tspeaker\tvalue\n"
    broken += "n/a\tn/a\tParticipant\ti feel tired\n"
    _write(daic_root / "301_P" / "301_TRANSCRIPT.csv", broken)

    config = _config(
        daic_root, speech_mask={"enabled": True, "source": "transcript", "pad_seconds": 0.0}
    )
    dataset = _dataset(config)
    assert int(dataset[0]["audio"].shape[0]) == 6  # the healthy session still loads
    with pytest.raises(ValueError, match="no usable turn timings"):
        dataset[1]


def test_a_mask_without_a_clock_is_rejected(daic_root: Path) -> None:
    """A headerless file with no sample rate cannot place its rows in time.

    A wrong setting is wrong for every session, so it is reported as such rather
    than being skipped into "no readable participant found".
    """
    config = _config(
        daic_root,
        sample_rate_hz=0.0,
        speech_mask={"enabled": True, "source": "transcript", "pad_seconds": 0.0},
    )
    with pytest.raises(FeatureConfigError, match="needs a clock"):
        _dataset(config)


def test_an_unknown_mask_source_is_rejected(daic_root: Path) -> None:
    """Only transcript timings are implemented; a typo must not disable the mask."""
    config = _config(daic_root, speech_mask={"enabled": True, "source": "vad"})
    with pytest.raises(FeatureConfigError, match="speech_mask.source"):
        _dataset(config)


def test_a_missing_timestamp_column_is_a_config_fault(daic_root: Path) -> None:
    """Naming a clock column the file lacks must say so, not skip the session."""
    config = _config(daic_root)
    config["video"]["timestamp_column"] = "clock"
    config["video"]["speech_mask"] = {"enabled": True, "pad_seconds": 0.0}
    with pytest.raises(FeatureConfigError, match="timestamp column"):
        _dataset(config)


def test_enabling_the_mask_keys_to_a_different_cache_entry(daic_root: Path) -> None:
    """A cached matrix parsed without the mask must not be reused with it on."""
    config = _config(daic_root)
    config["feature_cache_dir"] = "_feature_cache"
    cache_dir = daic_root / "_feature_cache"

    _dataset(config)[0]
    unmasked = {p.name for p in cache_dir.glob("*.npz")}
    assert unmasked

    config["audio"]["speech_mask"] = {"enabled": True, "pad_seconds": 0.0}
    masked_sample = _dataset(config)[0]
    assert {p.name for p in cache_dir.glob("*.npz")} > unmasked
    assert int(masked_sample["audio"].shape[0]) == 6


def test_corpus_statistics_are_fitted_under_the_same_mask(daic_root: Path) -> None:
    """Otherwise the stats describe silence while the sessions describe speech."""
    masked = _config(
        daic_root,
        normalization="corpus",
        speech_mask={"enabled": True, "source": "transcript", "pad_seconds": 0.0},
    )
    unmasked = _config(daic_root, normalization="corpus")

    # Channel 0 is the frame's own time. Fitted on the six speech frames its mean
    # is 6.0, so the normalized frames are centred on that; fitted on all ten it
    # is 4.5. A stats path that ignored the mask would give both arms the latter.
    masked_first = float(_dataset(masked)[0]["audio"][0, 0].item())
    unmasked_first = float(_dataset(unmasked)[0]["audio"][0, 0].item())
    assert masked_first == pytest.approx((3.0 - 6.0) / np.std([3, 4, 5, 7, 8, 9]), rel=1e-3)
    assert unmasked_first == pytest.approx((0.0 - 4.5) / np.std(range(NUM_FRAMES)), rel=1e-3)


def test_from_turns_drops_turns_whose_timings_were_unreadable() -> None:
    """A zero-length turn contributes text and no frames (see ``_timed_turn``)."""
    window = SpeechWindow.from_turns(
        [
            TimedTurn(start=4.0, stop=4.0, text="unreadable timings"),
            TimedTurn(start=1.0, stop=2.0, text="real turn"),
        ],
        sample_rate_hz=1.0,
    )
    assert window.intervals == ((1.0, 2.0),)


def test_from_turns_merges_overlapping_spans() -> None:
    """Overlaps are merged so a frame cannot be counted twice."""
    window = SpeechWindow.from_turns(
        [
            TimedTurn(start=0.0, stop=2.0, text="a"),
            TimedTurn(start=1.5, stop=3.0, text="b"),
            TimedTurn(start=9.0, stop=10.0, text="c"),
        ],
        sample_rate_hz=1.0,
    )
    assert window.intervals == ((0.0, 3.0), (9.0, 10.0))


def test_negative_padding_is_rejected() -> None:
    """Shrinking a turn is not a supported reading of `pad_seconds`."""
    with pytest.raises(ValueError, match="pad_seconds"):
        SpeechWindow.from_turns([TimedTurn(0.0, 1.0, "a")], pad_seconds=-0.5)
