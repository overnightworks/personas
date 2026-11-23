"""The arguments of the accept and decline tools the runner builds from its decisions."""

from __future__ import annotations

import secrets

import pytest
from agents.member import DeclineInvitationReason, FigureRef, HttpsFileRef, PlaceRef, VoiceName

from personas.decision import Accept, Decline, accept_arguments, decline_arguments

PLACE = PlaceRef(kind="call", id="room-7")


def _figure() -> FigureRef:
    return FigureRef(url="https://figures.test/figure.vrm", sha256=secrets.token_hex(32))


def _room() -> HttpsFileRef:
    return HttpsFileRef(url="https://figures.test/figure.room.glb", sha256=secrets.token_hex(32))


def test_an_accept_names_its_place_invitation_and_epoch():
    arguments = accept_arguments(Accept(PLACE, "inv-3", 2))

    assert (arguments.place, arguments.invitation, arguments.epoch) == (PLACE, "inv-3", 2)


def test_an_accept_without_a_figure_or_voices_carries_neither_look_nor_voices():
    arguments = accept_arguments(Accept(PLACE, "inv-3", 2))

    assert (arguments.figure, arguments.looks, arguments.voices) == (None, None, None)


def test_a_room_without_a_figure_carries_no_look():
    assert accept_arguments(Accept(PLACE, "inv-3", 2, room=_room())).looks is None


def test_a_figure_without_a_room_carries_no_look_and_travels_beside_it():
    figure = _figure()

    arguments = accept_arguments(Accept(PLACE, "inv-3", 2, figure))

    assert (arguments.figure, arguments.looks) == (figure, None)


def test_a_figure_with_a_room_carries_one_look_of_both():
    figure, room = _figure(), _room()

    arguments = accept_arguments(Accept(PLACE, "inv-3", 2, figure, room=room))

    assert arguments.looks is not None
    [look] = arguments.looks
    assert (look.url, look.sha256, look.room) == (figure.url, figure.sha256, room)


def test_voices_travel_as_listed_and_none_as_no_voices():
    voices = (VoiceName("amy"),)

    assert accept_arguments(Accept(PLACE, "inv-3", 2, voices=voices)).voices == voices
    assert accept_arguments(Accept(PLACE, "inv-3", 2)).voices is None


@pytest.mark.parametrize("reason", list(DeclineInvitationReason))
def test_a_decline_names_its_place_invitation_epoch_and_reason(reason):
    arguments = decline_arguments(Decline(PLACE, "inv-3", 2, reason))

    assert (arguments.place, arguments.invitation, arguments.epoch, arguments.reason) == (PLACE, "inv-3", 2, reason)
