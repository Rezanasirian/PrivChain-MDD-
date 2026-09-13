from __future__ import annotations

import numpy as np
from scripts.extract_egemaps import concatenate_intervals


def test_concatenate_intervals_clips_and_preserves_order() -> None:
    signal = np.arange(10, dtype=np.float32)

    selected = concatenate_intervals(signal, 2, ((-1.0, 1.0), (3.0, 8.0)))

    np.testing.assert_array_equal(selected, np.asarray([0, 1, 6, 7, 8, 9], dtype=np.float32))


def test_concatenate_intervals_rejects_empty_selection() -> None:
    signal = np.arange(4, dtype=np.float32)

    try:
        concatenate_intervals(signal, 2, ((3.0, 4.0),))
    except ValueError as exc:
        assert "no audio samples" in str(exc)
    else:
        raise AssertionError("an entirely out-of-range selection must fail")
