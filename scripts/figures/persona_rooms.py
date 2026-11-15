"""The rooms the personas stand in, each designed by hand for its persona."""

from figures.leonardo_room import leonardo_room
from figures.mozart_room import mozart_room
from figures.room_glb import Room

MOZART = Room(name="Mozart", furnish=mozart_room)
LEONARDO = Room(name="Leonardo", furnish=leonardo_room)
