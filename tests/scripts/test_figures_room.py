from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.support.glb import Glb

# Half the width of the open lane between the figure and a camera in front of it, the T-posed arms' reach and more.
LANE_HALF_WIDTH = 1.0
# The lane's height: above the figure's head and a camera at eye height, below the ceiling.
LANE_HEIGHT = 2.5


def positions_and_triangles(room: Glb) -> tuple[list[tuple[float, ...]], list[tuple[int, int, int]]]:
    positions: list[tuple[float, ...]] = []
    triangles: list[tuple[int, int, int]] = []
    for primitive in room.primitives("room"):
        base = len(positions)
        positions += room.accessor(primitive["attributes"]["POSITION"])
        indices = [int(index) for (index,) in room.accessor(primitive["indices"])]
        triangles += [
            (base + indices[start], base + indices[start + 1], base + indices[start + 2])
            for start in range(0, len(indices), 3)
        ]
    return positions, triangles


def test_nothing_of_the_room_reaches_below_the_floor(mozart_room):
    positions, _ = positions_and_triangles(mozart_room)

    assert min(position[1] for position in positions) >= 0.0


def test_nothing_stands_between_the_figure_and_a_camera_in_front_of_him(mozart_room):
    positions, triangles = positions_and_triangles(mozart_room)

    def stands_in_the_lane(triangle: tuple[int, int, int]) -> bool:
        corners = [positions[index] for index in triangle]
        in_front = all(corner[2] > 0.0 and abs(corner[0]) < LANE_HALF_WIDTH for corner in corners)
        return in_front and any(0.0 < corner[1] < LANE_HEIGHT for corner in corners)

    assert [triangle for triangle in triangles if stands_in_the_lane(triangle)] == []
