"""
Functions module for Jupyter notebook utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional

# External packages
import pandas as pd

# Internal packages
from mftsci.utils.notebook import defaults


# =====================================
# CLASSES
# =====================================


class FigCaps():

    def __init__(
                    self,
                    ):
        self.reset_caps()

    def reset_caps(
                    self,
                    ):
        self.counter = defaults.counter()
        self.caps = pd.DataFrame(columns=["number", "id", "cap_str"])

    def id_to_number(
                        self,
                        id: Optional[str] = None,
                        ):

        if id is None:
            return self.caps.iloc[-1]["number"]
        else:
            return self.caps[self.caps["id"] == id].iloc[-1]["number"]

    def calc_xref(
                    self,
                    id: str,
                    ):

        return f"Fig. [{self.id_to_number(id=id)}](#{id})"

    def calc_cap(
                    self,
                    caption: Optional[str] = None,
                    id: Optional[str] = None,
                    number: Optional[int] = None,
                    ) -> pd.core.frame.DataFrame:

        caption = defaults.caption(caption)
        id = defaults.id(id)

        if number is None:
            number = self.counter
            self.counter += 1

        fig_str = f"**Figure {number}"

        if id == "":
            id_str = ""
        else:
            id_str = f"<a id='{id}'></a>"

        if caption == "":
            caption_str = "**"
        else:
            caption_str = f": {caption}**"

        cap_str = f"{id_str}{fig_str}{caption_str}"

        cap = {
                    "number": number,
                    "id": id,
                    "cap_str": cap_str
            }

        self.caps = pd.concat([self.caps, pd.DataFrame(cap, index=[0])],
                              ignore_index=True)

        return cap
