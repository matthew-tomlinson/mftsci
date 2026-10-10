
"""
Functions module for html utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union

# External packages


# Internal packages
from mftsci.utils.html import defaults
from mftsci.utils.file import WPath as _WPath, write as _write


# =====================================
# CLASSES
# =====================================


class Path(_WPath):

    _default_file = "file"


# =====================================
# FUNCTIONS
# =====================================


def head_latex_mathmode(
                            html: str,
                            ) -> str:

    return latex_mathmode() + html


def latex_mathmode() -> str:

    post = """
    <script type="text/javascript">
    MathJax = {
        tex: {
        inlineMath: [['$', '$']]
        },
        svg: {
        fontCache: 'global'
        }
    };
    </script>
    <script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    """

    return post


def margin_paddinng_wrap(
                html: Optional[str] = None,
                margin: Optional[Union[float, int, str,]] = None,
                padding: Optional[Union[float, int, str,]] = None,
                ) -> str:

    if html is None:
        html = ""

    margin = defaults.margin(margin)
    padding = defaults.padding(padding)

    wrapped = f"""<!DOCTYPE html>
                    <html>
                    <head>
                    <style>
                    body {{
                        margin: {margin};
                        padding: {padding};
                    }}
                    </style>
                    </head>
                    <body>
                    {html}
                    </body>
                    </html>"""

    return wrapped


def write(
            s: str,
            file: Optional[str] = None,
            path: Optional[str] = None,
            ext: Optional[str] = None,
            app_ext: Optional[bool] = None,
            app_write: Optional[bool] = None,
            app_mkdir: Optional[bool] = None,
            app_print_path: Optional[bool] = None,
            dPath: Optional[Path] = None,
            ):

    if dPath is None:
        dPath = Path

    if ext is None:
        ext = ".html"

    return _write(
            s=s, file=file, path=path,
            ext=ext, app_ext=app_ext,
            app_write=app_write,
            app_mkdir=app_mkdir, app_print_path=app_print_path,
            dPath=dPath
            )
