"""Partition utilities for the MHAR residual-group compatibility experiment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


Pair = tuple[int, int]
Partition = tuple[Pair, ...]
AtomicGroup = tuple[int, ...]
MixedPartition = tuple[AtomicGroup, ...]
ContiguousSegment = tuple[int, int]
ContiguousPartition = tuple[ContiguousSegment, ...]

REFERENCE_PARTITION_H4: Partition = ((0, 1), (2, 3), (4, 5), (6, 7))


def canonicalize_contiguous_partition(
    partition: Iterable[Sequence[int]],
    *,
    hidden_size: int,
    num_groups: int | None = None,
    min_width: int = 1,
) -> ContiguousPartition:
    """Validate ordered half-open coordinate segments covering ``[0, D)``.

    This representation is used by Experiment 3E, where a routing boundary can
    move by less than the native equal-head width. Coordinates are never
    reordered and every residual coordinate must occur exactly once.
    """

    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    if min_width < 1:
        raise ValueError("min_width must be positive")
    segments = tuple(tuple(int(value) for value in segment) for segment in partition)
    if not segments:
        raise ValueError("contiguous partition cannot be empty")
    if num_groups is not None and len(segments) != num_groups:
        raise ValueError(
            f"contiguous partition has {len(segments)} groups, expected {num_groups}")
    cursor = 0
    for segment in segments:
        if len(segment) != 2:
            raise ValueError(f"segments must be (start, end) pairs: {segment!r}")
        start, end = segment
        if start != cursor:
            raise ValueError(
                "contiguous partition must be ordered with no gaps or overlap; "
                f"expected start {cursor}, got {start}")
        if end - start < min_width:
            raise ValueError(
                f"segment {segment!r} is narrower than min_width={min_width}")
        cursor = end
    if cursor != hidden_size:
        raise ValueError(
            f"contiguous partition ends at {cursor}, expected hidden_size={hidden_size}")
    return segments


def contiguous_partition_from_boundaries(
    boundaries: Iterable[int],
    *,
    hidden_size: int,
    num_groups: int | None = None,
    min_width: int = 1,
) -> ContiguousPartition:
    """Construct coordinate segments from strictly increasing inner boundaries."""

    inner = tuple(int(value) for value in boundaries)
    if tuple(sorted(inner)) != inner or len(set(inner)) != len(inner):
        raise ValueError("boundaries must be strictly increasing")
    if any(value <= 0 or value >= hidden_size for value in inner):
        raise ValueError(f"boundaries must lie strictly inside [0, {hidden_size}]")
    endpoints = (0,) + inner + (hidden_size,)
    segments = tuple(zip(endpoints[:-1], endpoints[1:]))
    return canonicalize_contiguous_partition(
        segments,
        hidden_size=hidden_size,
        num_groups=num_groups,
        min_width=min_width,
    )


def contiguous_partition_id(partition: Iterable[Sequence[int]], *, hidden_size: int) -> str:
    canonical = canonicalize_contiguous_partition(partition, hidden_size=hidden_size)
    return "__".join(f"{start}-{end}" for start, end in canonical)


def parse_contiguous_partition_id(
    value: str,
    *,
    hidden_size: int,
    num_groups: int | None = None,
    min_width: int = 1,
) -> ContiguousPartition:
    try:
        segments = tuple(
            tuple(int(coordinate) for coordinate in segment.split("-"))
            for segment in value.split("__")
        )
    except ValueError as exc:
        raise ValueError(f"invalid contiguous partition id: {value!r}") from exc
    return canonicalize_contiguous_partition(
        segments,
        hidden_size=hidden_size,
        num_groups=num_groups,
        min_width=min_width,
    )


def canonicalize_mixed_partition(
    partition: Iterable[Sequence[int]],
    *,
    num_atomic_blocks: int = 16,
) -> MixedPartition:
    """Validate an ordered contiguous partition of singleton/doubleton atoms."""

    if num_atomic_blocks < 1:
        raise ValueError("num_atomic_blocks must be positive")
    groups = tuple(tuple(int(index) for index in group) for group in partition)
    if not groups:
        raise ValueError("mixed partition cannot be empty")
    for group in groups:
        if len(group) not in (1, 2):
            raise ValueError(
                f"mixed routing groups must contain one or two atoms: {group!r}")
        if len(group) == 2 and group[1] != group[0] + 1:
            raise ValueError(f"merged atoms must be adjacent and ordered: {group!r}")
    flat = tuple(index for group in groups for index in group)
    expected = tuple(range(num_atomic_blocks))
    if flat != expected:
        raise ValueError(
            "mixed partition must cover every atom exactly once in coordinate order; "
            f"got {flat!r}, expected {expected!r}")
    return groups


def mixed_partition_from_merges(
    merged_boundaries: Iterable[int],
    *,
    num_atomic_blocks: int = 16,
) -> MixedPartition:
    """Build an ordered partition from left-edge indices of adjacent merges."""

    merges = tuple(sorted(int(index) for index in merged_boundaries))
    if len(merges) != len(set(merges)):
        raise ValueError("merged boundaries must be unique")
    if any(index < 0 or index >= num_atomic_blocks - 1 for index in merges):
        raise ValueError(
            f"merged boundaries must be in [0, {num_atomic_blocks - 2}]")
    if any(right == left + 1 for left, right in zip(merges, merges[1:])):
        raise ValueError("merged boundaries cannot overlap")

    groups: list[AtomicGroup] = []
    atom = 0
    merge_set = set(merges)
    while atom < num_atomic_blocks:
        if atom in merge_set:
            groups.append((atom, atom + 1))
            atom += 2
        else:
            groups.append((atom,))
            atom += 1
    return canonicalize_mixed_partition(groups, num_atomic_blocks=num_atomic_blocks)


def generate_adjacent_merge_partitions(
    num_atomic_blocks: int = 16,
    num_merges: int = 4,
) -> tuple[MixedPartition, ...]:
    """Enumerate every ordered partition with ``num_merges`` disjoint merges."""

    if num_merges < 0 or num_merges > num_atomic_blocks // 2:
        raise ValueError("num_merges must be between zero and floor(num_atomic_blocks / 2)")

    def _choose(next_edge: int, remaining: int, chosen: tuple[int, ...]):
        if remaining == 0:
            yield chosen
            return
        last_edge = num_atomic_blocks - 2
        for edge in range(next_edge, last_edge + 1):
            # After choosing this edge, each remaining merge needs two atoms.
            if last_edge - edge < 2 * (remaining - 1):
                break
            yield from _choose(edge + 2, remaining - 1, chosen + (edge,))

    partitions = tuple(
        mixed_partition_from_merges(edges, num_atomic_blocks=num_atomic_blocks)
        for edges in _choose(0, num_merges, ())
    )
    if len(partitions) != len(set(partitions)):
        raise AssertionError("mixed partition generator emitted duplicates")
    return partitions


def mixed_partition_id(partition: Iterable[Sequence[int]]) -> str:
    canonical = canonicalize_mixed_partition(partition)
    return "__".join("-".join(str(index) for index in group) for group in canonical)


def parse_mixed_partition_id(value: str, *, num_atomic_blocks: int = 16) -> MixedPartition:
    try:
        groups = tuple(
            tuple(int(index) for index in group.split("-"))
            for group in value.split("__")
        )
    except ValueError as exc:
        raise ValueError(f"invalid mixed partition id: {value!r}") from exc
    return canonicalize_mixed_partition(groups, num_atomic_blocks=num_atomic_blocks)


def merged_boundaries(partition: Iterable[Sequence[int]]) -> tuple[int, ...]:
    canonical = canonicalize_mixed_partition(partition)
    return tuple(group[0] for group in canonical if len(group) == 2)


def mixed_segment_widths(
    partition: Iterable[Sequence[int]],
    *,
    primitive_width: int = 80,
) -> tuple[int, ...]:
    if primitive_width < 1:
        raise ValueError("primitive_width must be positive")
    return tuple(
        len(group) * primitive_width
        for group in canonicalize_mixed_partition(partition)
    )


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


def mixed_width_mhar_eager(
    V: torch.Tensor,
    query: torch.Tensor,
    norm,
    partition: Iterable[Sequence[int]],
    *,
    num_atomic_blocks: int = 16,
) -> torch.Tensor:
    """Route ordered singleton/doubleton atomic regions with separate softmaxes.

    RMS normalization is performed once in the original full-dimensional
    basis.  Every query coefficient stays attached to its coordinate and the
    output segments are concatenated in their original coordinate order.
    """

    if V.ndim != 4:
        raise ValueError(f"V must have shape [N,B,T,D], got {tuple(V.shape)}")
    _, _, _, hidden_size = V.shape
    if hidden_size % num_atomic_blocks:
        raise ValueError(
            f"hidden size {hidden_size} must be divisible by {num_atomic_blocks}")
    if query.numel() != hidden_size:
        raise ValueError(
            f"query contains {query.numel()} coefficients, expected {hidden_size}")

    canonical = canonicalize_mixed_partition(
        partition, num_atomic_blocks=num_atomic_blocks)
    primitive_width = hidden_size // num_atomic_blocks
    K = norm(V)
    flat_query = query.reshape(hidden_size)

    # Use the identical vectorized contraction as ordinary equal-width MHAR at
    # both endpoints.  Besides being faster, this avoids bf16 reduction-order
    # drift in the parity gate before a frozen-checkpoint experiment.
    if all(len(group) == 1 for group in canonical):
        num_groups = num_atomic_blocks
        grouped_width = primitive_width
        logits = torch.einsum(
            "h k, n b t h k -> n b t h",
            flat_query.view(num_groups, grouped_width),
            K.view(*K.shape[:-1], num_groups, grouped_width),
        )
        weights = logits.softmax(dim=0)
        routed = torch.einsum(
            "n b t h, n b t h k -> b t h k",
            weights,
            V.view(*V.shape[:-1], num_groups, grouped_width),
        )
        return routed.reshape(*V.shape[1:-1], hidden_size)

    if all(len(group) == 2 for group in canonical):
        num_groups = num_atomic_blocks // 2
        grouped_width = 2 * primitive_width
        logits = torch.einsum(
            "h k, n b t h k -> n b t h",
            flat_query.view(num_groups, grouped_width),
            K.view(*K.shape[:-1], num_groups, grouped_width),
        )
        weights = logits.softmax(dim=0)
        routed = torch.einsum(
            "n b t h, n b t h k -> b t h k",
            weights,
            V.view(*V.shape[:-1], num_groups, grouped_width),
        )
        return routed.reshape(*V.shape[1:-1], hidden_size)

    outputs = []
    for group in canonical:
        start = group[0] * primitive_width
        end = (group[-1] + 1) * primitive_width
        group_keys = K[..., start:end]
        group_values = V[..., start:end]
        group_query = flat_query[start:end]
        logits = torch.einsum("k, n b t k -> n b t", group_query, group_keys)
        weights = logits.softmax(dim=0)
        outputs.append(torch.einsum("n b t, n b t k -> b t k", weights, group_values))
    return torch.cat(outputs, dim=-1)


def contiguous_mhar_eager(
    V: torch.Tensor,
    query: torch.Tensor,
    norm,
    partition: Iterable[Sequence[int]],
    *,
    num_groups: int | None = None,
    min_width: int = 1,
) -> torch.Tensor:
    """Route arbitrary-width ordered coordinate segments with one softmax each.

    Full-width RMS normalization happens before slicing. Query coefficients and
    routed values remain attached to their original coordinates. This eager
    reference is intentionally separate from the equal-width Triton kernel.
    """

    if V.ndim != 4:
        raise ValueError(f"V must have shape [N,B,T,D], got {tuple(V.shape)}")
    hidden_size = V.shape[-1]
    if query.numel() != hidden_size:
        raise ValueError(
            f"query contains {query.numel()} coefficients, expected {hidden_size}")
    canonical = canonicalize_contiguous_partition(
        partition,
        hidden_size=hidden_size,
        num_groups=num_groups,
        min_width=min_width,
    )
    K = norm(V)
    flat_query = query.reshape(hidden_size)

    widths = tuple(end - start for start, end in canonical)
    if len(set(widths)) == 1:
        group_count = len(canonical)
        group_width = widths[0]
        logits = torch.einsum(
            "h k, n b t h k -> n b t h",
            flat_query.view(group_count, group_width),
            K.view(*K.shape[:-1], group_count, group_width),
        )
        weights = logits.softmax(dim=0)
        routed = torch.einsum(
            "n b t h, n b t h k -> b t h k",
            weights,
            V.view(*V.shape[:-1], group_count, group_width),
        )
        return routed.reshape(*V.shape[1:-1], hidden_size)

    outputs = []
    for start, end in canonical:
        group_keys = K[..., start:end]
        group_values = V[..., start:end]
        group_query = flat_query[start:end]
        logits = torch.einsum("k, n b t k -> n b t", group_query, group_keys)
        weights = logits.softmax(dim=0)
        outputs.append(torch.einsum("n b t, n b t k -> b t k", weights, group_values))
    return torch.cat(outputs, dim=-1)
