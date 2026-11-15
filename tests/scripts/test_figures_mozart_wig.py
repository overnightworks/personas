import math
from typing import TYPE_CHECKING

from figures.mozart_wig import QUEUE, mozart_wig
from tests.support.surface import seen_from_side

if TYPE_CHECKING:
    from figures.figure_parts import Part

# Steps across the queue's top, from in front of it to behind it, at which the side view is sampled.
STEPS = 60
# Shares of the way down the queue's top segment at which the wig's edge crosses the queue's side; lower, it runs
# along the queue's front, where the side view sees both surfaces edge on.
CROSSING = (0.2, 0.3, 0.4)
# The largest angle between the wig's and the queue's shading on either side of the edge where the wig ends over
# the queue, below which the light runs on across it instead of breaking into a hard edge.
LARGEST_TURN = math.radians(15.0)


def seen(part: Part, height: float, depth: float) -> tuple[float, list[float]] | None:
    return seen_from_side(part.shape.positions, part.shape.normals, part.shape.indices, height, depth)


def angle(first: list[float], second: list[float]) -> float:
    cosine = sum(a * b for a, b in zip(first, second, strict=True)) / (math.hypot(*first) * math.hypot(*second))
    return math.acos(max(-1.0, min(1.0, cosine)))


def turns_where_the_wig_ends_over_the_queue() -> list[float]:
    """Where the wig's edge crosses the queue's side, seen from the side: the turn between the shading of the last
    point that shows the wig and the first that shows the queue."""
    parts = mozart_wig()
    cap = parts[0]
    queue = min(parts, key=lambda part: min(position[1] for position in part.shape.positions))
    (top, (_, top_depth)), (below, _) = QUEUE[0], QUEUE[1]
    turns: list[float] = []
    for height in (top[1] - share * (top[1] - below[1]) for share in CROSSING):
        last_wig: list[float] | None = None
        for step in range(STEPS + 1):
            depth = top[2] + top_depth * (1.0 - 2.0 * step / STEPS)
            over_wig, over_queue = seen(cap, height, depth), seen(queue, height, depth)
            if over_wig is not None and (over_queue is None or over_wig[0] >= over_queue[0]):
                last_wig = over_wig[1]
            elif over_queue is not None and last_wig is not None:
                turns.append(angle(last_wig, over_queue[1]))
                break
    return turns


def test_the_wig_shades_like_the_queue_where_it_ends_over_it():
    turns = turns_where_the_wig_ends_over_the_queue()

    assert turns
    assert max(turns) < LARGEST_TURN
