"""The wardrobes of the two personas: the colours and the parts that tell Mozart and Leonardo apart."""

from figures.figure_shapes import ellipsoid, tapered_tube
from figures.humanoid_figure import Part, Wardrobe

MOZART = Wardrobe(
    name="Mozart",
    palette={
        "skin": (244, 220, 204),
        "coat": (178, 34, 46),
        "stocking": (240, 238, 232),
        "shoe": (30, 26, 26),
        "wig": (236, 234, 228),
        "bow": (25, 25, 30),
        "lace": (250, 250, 246),
        "gold": (212, 170, 60),
        "brow": (150, 120, 95),
        "eyeWhite": (250, 250, 250),
        "iris": (70, 90, 120),
        "mouth": (120, 40, 45),
    },
    garment="coat",
    legwear="stocking",
    long_robe=False,
    brow_size=(0.02, 0.0045, 0.005),
    brow_material="brow",
    extras=(
        # The powdered wig, its queue and bow, the neck stock, the jabot and the coat's skirt.
        Part(ellipsoid((0, 1.632, -0.025), (0.102, 0.1, 0.1)), "wig", "head"),
        Part(ellipsoid((0, 1.49, -0.11), (0.028, 0.06, 0.028)), "wig", "head"),
        Part(ellipsoid((0, 1.545, -0.112), (0.05, 0.02, 0.018)), "bow", "head"),
        Part(tapered_tube((0, 1.37, 0), (0, 1.47, 0), 0.056, 0.05), "lace", "neck"),
        Part(ellipsoid((0, 1.35, 0.095), (0.04, 0.06, 0.03)), "lace", "chest"),
        Part(tapered_tube((0, 0.62, 0), (0, 0.98, 0), 0.25, 0.165, 0.7), "coat", "hips"),
        # Cuffs, and two rolls on each side of the wig, front to back over the ear.
        *(
            Part(tapered_tube((side * 0.6, 1.38, 0), (side * 0.66, 1.38, 0), 0.06, 0.065), "lace", prefix + "LowerArm")
            for side, prefix in ((1, "left"), (-1, "right"))
        ),
        *(
            part
            for side in (1, -1)
            for height, radius in ((1.618, 0.024), (1.572, 0.026))
            for part in (
                Part(
                    tapered_tube((side * 0.104, height, -0.055), (side * 0.104, height, 0.03), radius, radius),
                    "wig",
                    "head",
                ),
                Part(ellipsoid((side * 0.104, height, -0.055), (radius, radius, radius)), "wig", "head"),
                Part(ellipsoid((side * 0.104, height, 0.03), (radius, radius, radius)), "wig", "head"),
            )
        ),
        *(
            Part(
                ellipsoid((0.0, 1.3 - button * 0.075, 0.106), (0.011, 0.011, 0.006), 6, 8),
                "gold",
                "chest" if button < 2 else "spine",
            )
            for button in range(5)
        ),
    ),
)

LEONARDO = Wardrobe(
    name="Leonardo",
    palette={
        "skin": (222, 186, 156),
        "robe": (92, 62, 44),
        "robeTrim": (60, 40, 30),
        "shoe": (40, 30, 24),
        "hair": (206, 204, 198),
        "cap": (34, 30, 32),
        "eyeWhite": (246, 244, 238),
        "iris": (85, 70, 50),
        "mouth": (120, 55, 50),
    },
    garment="robe",
    legwear="robe",
    long_robe=True,
    brow_size=(0.024, 0.0075, 0.007),
    brow_material="hair",
    extras=(
        # Long hair, a soft cap, one tapered beard from the jaw to a point on the chest, a moustache, the robe.
        Part(ellipsoid((0, 1.6, -0.03), (0.108, 0.125, 0.103)), "hair", "head"),
        Part(ellipsoid((0, 1.47, -0.06), (0.115, 0.15, 0.065)), "hair", "head"),
        Part(ellipsoid((0, 1.715, -0.01), (0.112, 0.038, 0.112)), "cap", "head"),
        Part(tapered_tube((0, 1.6, 0.0), (0, 1.29, 0.07), 0.098, 0.015, 0.75), "hair", "head"),
        Part(ellipsoid((0, 1.562, 0.093), (0.038, 0.008, 0.012)), "hair", "head"),
        Part(tapered_tube((0, 0.1, 0), (0, 0.98, 0), 0.29, 0.165, 0.75), "robe", "hips"),
        Part(ellipsoid((0, 1.395, 0.0), (0.17, 0.035, 0.115)), "robeTrim", "chest"),
    ),
)
