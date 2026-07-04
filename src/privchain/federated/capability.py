"""Capability-declared subgraph grouping (Phase 4, objective H2).

The core of the proposed aggregation protocol: each client declares a modality
capability vector ``[audio, video, text]`` and the server aggregates **each
modality's encoder only across the clients that actually possess that modality**
(its *subgraph*), while the shared fusion + prediction heads are aggregated
across every participating client.

This is what prevents the failure mode plain FedAvg suffers (Phase 2): a
text-only client zero-imputes audio/video and, under FedAvg, drags those
encoders toward the zero-signal solution. Subgraph aggregation keeps each
encoder learning only from clients with real signal for that modality.

Parameter→group mapping reuses the same name-prefix convention as the DP-SGD
grouping (:mod:`privchain.privacy.dp_sgd`): ``encoders.<modality>`` → that
modality; everything else (fusion + heads) → the shared group.
"""

from __future__ import annotations

from privchain.config import CAPABILITY_MODALITIES

# Group name for parameters shared across all modalities (fusion + heads).
SHARED_GROUP = "shared"


def param_group(param_name: str) -> str:
    """Map a ``state_dict`` key to its aggregation group.

    Args:
        param_name: A parameter key, e.g. ``encoders.audio.gru.weight_ih_l0``.

    Returns:
        The owning modality (``"audio"``/``"video"``/``"text"``) when the
        parameter belongs to a modality encoder, else :data:`SHARED_GROUP`.
    """
    for modality in CAPABILITY_MODALITIES:
        if param_name.startswith(f"encoders.{modality}"):
            return modality
    return SHARED_GROUP


def has_modality(capability: tuple[int, int, int], modality: str) -> bool:
    """Return whether a capability vector declares ``modality``.

    Args:
        capability: ``[audio, video, text]`` 0/1 availability flags.
        modality: One of :data:`~privchain.config.CAPABILITY_MODALITIES`.

    Returns:
        ``True`` if the client possesses ``modality``.

    Raises:
        ValueError: If ``modality`` is not a known modality name.
    """
    if modality not in CAPABILITY_MODALITIES:
        raise ValueError(f"unknown modality: {modality!r}")
    return capability[CAPABILITY_MODALITIES.index(modality)] == 1


def is_missing_any(capability: tuple[int, int, int]) -> bool:
    """Return whether a client lacks at least one modality.

    Args:
        capability: ``[audio, video, text]`` 0/1 availability flags.

    Returns:
        ``True`` if any modality flag is 0 (a missing-modality client).
    """
    return any(flag == 0 for flag in capability)


def modality_subgraphs(
    capabilities: list[tuple[int, int, int]],
) -> dict[str, list[int]]:
    """Group client indices into one subgraph per modality.

    Args:
        capabilities: Per-client capability vectors (indexed as passed in).

    Returns:
        Mapping ``{modality: [client index, ...]}`` listing, for each modality,
        the positions of clients that declare it (possibly empty).
    """
    subgraphs: dict[str, list[int]] = {modality: [] for modality in CAPABILITY_MODALITIES}
    for idx, capability in enumerate(capabilities):
        for modality in CAPABILITY_MODALITIES:
            if has_modality(capability, modality):
                subgraphs[modality].append(idx)
    return subgraphs
