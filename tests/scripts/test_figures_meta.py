from pathlib import Path

import pytest

from tests.support.glb import read_glb

LICENSE = Path(__file__).parents[2] / "LICENSE"
FIGURE_PROFILE = {"allowRedistribution": True, "avatarPermission": "everyone", "creditNotation": "unnecessary"}
MIT_PERMISSIONS = {
    "commercialUsage": "corporation",
    "modification": "allowModificationRedistribution",
    "allowExcessivelyViolentUsage": True,
    "allowExcessivelySexualUsage": True,
    "allowPoliticalOrReligiousUsage": True,
    "allowAntisocialOrHateUsage": True,
}
ABSENT = ["version", "contactInformation", "references", "thirdPartyLicenses", "thumbnailImage"]


def test_the_meta_holds_the_values_figures_profile_requires(figure):
    meta = figure.vrm()["meta"]

    assert {key: meta[key] for key in FIGURE_PROFILE} == FIGURE_PROFILE


def test_the_meta_grants_what_mit_grants(figure):
    meta = figure.vrm()["meta"]

    assert {key: meta[key] for key in MIT_PERMISSIONS} == MIT_PERMISSIONS


def test_the_meta_carries_the_copyright_line_of_the_repositorys_licence(figure):
    copyright_line = next(line for line in LICENSE.read_text().splitlines() if line.startswith("Copyright"))

    assert copyright_line in figure.vrm()["meta"]["copyrightInformation"]


def test_the_meta_names_the_licence_and_the_author(figure):
    meta = figure.vrm()["meta"]

    assert meta["authors"] == ["overnightworks/personas"]
    assert meta["licenseUrl"].startswith("https://")
    assert meta["otherLicenseUrl"].startswith("https://")
    assert meta["otherLicenseUrl"].endswith("/LICENSE")


@pytest.mark.parametrize("field", ABSENT)
def test_the_meta_states_nothing_it_has_nothing_to_state(figure, field):
    assert field not in figure.vrm()["meta"]


def test_the_meta_names_the_persona(figure_bytes):
    assert {read_glb(data).vrm()["meta"]["name"] for data in figure_bytes.values()} == set(figure_bytes)
