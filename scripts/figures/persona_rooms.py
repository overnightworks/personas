"""The rooms the personas stand in, each designed by hand for its persona."""

from figures.mozart_room import mozart_room
from figures.room_glb import Room

MOZART = Room(name="Mozart", furnish=mozart_room)
