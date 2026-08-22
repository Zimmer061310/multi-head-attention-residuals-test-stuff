"""Partition utilities for the MHAR residual-group compatibility experiment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


Pair = tuple[int, int]
Partition = tuple[Pair, ...]

REFERENCE_PARTITION_H4: Partition = ((0, 1), (2, 3), (4, 5), (6, 7))


def canonicalize_partition(
    partition: Iterable[Sequence[int]],
    *,
    num_primitive_blocks: int | None = None,
) -> Partition:
    """Return an unordered pairing in a deterministic canonical form.

    Members inside each pair are sorted, then the pairs are sorted
    lexicographically.  A valid partition uses every primitive-block index
    exactly once.
    """

    pairs: list[Pair] = []
    for raw_pair in partition:
        if len(raw_pair) != 2:
            raise ValueError(f"each routing group must contain exactly two blocks: {raw_pair!r}")
        a, b = (int(raw_pair[0]), int(raw_pair[1]))
        if a == b:
            raise ValueError(f"a primitive block cannot be paired with itself: {(a, b)!r}")
        pairs.append((min(a, b), max(a, b)))

    canonical = tuple(sorted(pairs))
    flat = [index for pair in canonical for index in pair]

    if num_primitive_blocks is None:
        num_primitive_blocks = len(flat)
    if num_primitive_blocks < 2 or num_primitive_blocks % 2:
        raise ValueError("num_primitive_blocks must be a positive even integer")
    if len(flat) != num_primitive_blocks:
        raise ValueError(
            f"partition contains {len(flat)} entries, expected {num_primitive_blocks}")

    expected = list(range(num_primitive_blocks))
    if sorted(flat) != expected:
        raise ValueError(
            "partition must use every primitive-block index exactly once; "
            f"got {sorted(flat)!r}, expected {expected!r}")
    return canonical


def generate_pair_partitions(num_primitive_blocks: int = 8) -> tuple[Partition, ...]:
    """Enumerate every unordered perfect matching in deterministic order."""

    if num_primitive_blocks < 2 or num_primitive_blocks % 2:
        raise ValueError("num_primitive_blocks must be a positive even integer")

    def _generate(remaining: tuple[int, ...]):
        if not remaining:
            yield ()
            return
        first = remaining[0]
        for partner_offset in range(1, len(remaining)):
            partner = remaining[partner_offset]
            rest = remaining[1:partner_offset] + remaining[partner_offset + 1 :]
            for suffix in _generate(rest):
                yield ((first, partner),) + suffix

    partitions = tuple(
        canonicalize_partition(p, num_primitive_blocks=num_primitive_blocks)
        for p in _generate(tuple(range(num_primitive_blocks)))
    )
    if len(partitions) != len(set(partitions)):
        raise AssertionError("partition generator emitted duplicates")
    return partitions


def original_pair_retention(
    partition: Iterable[Sequence[int]],
    reference: Iterable[Sequence[int]] = REFERENCE_PARTITION_H4,
) -> tuple[int, float]:
    """Return the retained-reference-pair count and fraction."""

    reference_partition = canonicalize_partition(reference)
    candidate = canonicalize_partition(
        partition, num_primitive_blocks=2 * len(reference_partition))
    retained = len(set(candidate).intersection(reference_partition))
    return retained, retained / len(reference_partition)


def coordinate_distance(partition: Iterable[Sequence[int]]) -> tuple[int, float]:
    """Return total and mean absolute primitive-block distance."""

    candidate = canonicalize_partition(partition)
    total = sum(abs(a - b) for a, b in candidate)
    return total, total / len(candidate)


def partition_id(partition: Iterable[Sequence[int]]) -> str:
    """Return a compact stable identifier such as ``0-1__2-3__4-5__6-7``."""

    candidate = canonicalize_partition(partition)
    return "__".join(f"{a}-{b}" for a, b in candidate)


def parse_partition_id(value: str) -> Partition:
    """Parse the representation produced by :func:`partition_id`."""

    try:
        pairs = [tuple(int(index) for index in pair.split("-")) for pair in value.split("__")]
    except ValueError as exc:
        raise ValueError(f"invalid partition id: {value!r}") from exc
    return canonicalize_partition(pairs)


def arbitrary_group_mhar_eager(
    V: torch.Tensor,
    query: torch.Tensor,
    norm,
    partition: Iterable[Sequence[int]],
    num_heads: int,
) -> torch.Tensor:
    """Route arbitrary pairs of half-head blocks through shared depth softmaxes.

    ``V`` has shape ``[N, B, T, D]`` and ``query`` contains the D learned
    query coefficients attached to their original residual coordinates.
    Normalization happens in the original full-D basis.  The operation gathers
    paired blocks for routing and scatters them back before returning.
    """

    n, b, t, d = V.shape
    if num_heads < 1 or d % (2 * num_heads):
        raise ValueError(
            f"arbitrary-group MHAR requires hidden size {d} to be divisible by "
            f"2 * num_heads ({2 * num_heads})")
    canonical = canonicalize_partition(
        partition, num_primitive_blocks=2 * num_heads)

    primitive_width = d // (2 * num_heads)
    flat_indices = [index for pair in canonical for index in pair]
    index = torch.tensor(flat_indices, device=V.device, dtype=torch.long)

    K = norm(V)
    K_blocks = K.view(n, b, t, 2 * num_heads, primitive_width)
    V_blocks = V.view(n, b, t, 2 * num_heads, primitive_width)
    q_blocks = query.view(2 * num_heads, primitive_width)

    grouped_width = 2 * primitive_width
    K_grouped = K_blocks.index_select(-2, index).reshape(n, b, t, num_heads, grouped_width)
    V_grouped = V_blocks.index_select(-2, index).reshape(n, b, t, num_heads, grouped_width)
    q_grouped = q_blocks.index_select(0, index).reshape(num_heads, grouped_width)

    logits = torch.einsum("h k, n b t h k -> n b t h", q_grouped, K_grouped)
    weights = logits.softmax(dim=0)
    routed = torch.einsum("n b t h, n b t h k -> b t h k", weights, V_grouped)

    routed_in_gather_order = routed.reshape(b, t, 2 * num_heads, primitive_width)
    output_blocks = torch.empty_like(routed_in_gather_order)
    output_blocks.index_copy_(-2, index, routed_in_gather_order)
    return output_blocks.reshape(b, t, d)
