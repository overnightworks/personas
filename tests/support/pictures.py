"""The pictures an app answers a call with, as a test makes them and reads them back."""

from __future__ import annotations

import base64
import secrets

from agents.tools import ToolImage


def a_picture(size: int = 64) -> ToolImage:
    """A still picture as the library's wire takes it: random bytes behind the JPEG start marker."""
    return ToolImage(content=b"\xff\xd8\xff" + secrets.token_bytes(size))


_REAL_JPEG = (
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYyLjExLjEwMAD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4Nw"
    "cHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABMAAEBAAAAAAAAAAAAAAAAAAAABgEBAQAAAAAAAAAAAAAAAAAABgcQ"
    "AQAAAAAAAAAAAAAAAAAAAAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAAIAAgDASIAAhEAAxEA/9oADAMBAAIRAxEAPwCLAE1/f//Z"
)


def a_real_picture() -> ToolImage:
    """A complete 8 by 8 JPEG file, the shape of a picture a real app answers with."""
    return ToolImage(content=base64.b64decode(_REAL_JPEG))


def base64_of(picture: ToolImage) -> str:
    """The picture as an image block of a tool result and a data URL of a model request write it."""
    return base64.b64encode(picture.content).decode("ascii")
