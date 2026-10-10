"""
Functions module for file utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Literal
import os

# External packages
import numpy as np


# Internal packages
from mftsci.utils.file import defaults


# =====================================
# CLASSES
# =====================================

class Path():

    _default_root = None
    _default_subdir = None
    _default_file = None

    @classmethod
    def set_default_root(
            cls,
            default_root: Optional[str] = None,
            reset: Optional[bool] = None,
            ):

        if default_root is not None:
            cls._default_root = os.path.normpath(default_root)
        elif defaults.reset(reset):
            cls._default_root = None

    @classmethod
    def set_default_subdir(
            cls,
            default_subdir: Optional[str] = None,
            reset: Optional[bool] = None,
            ):

        if default_subdir is not None:
            cls._default_subdir = os.path.normpath(default_subdir)
        elif defaults.reset(reset):
            cls._default_subdir = None

    @classmethod
    def set_default_file(
            cls,
            default_file: Optional[str] = None,
            reset: Optional[bool] = None,
            ):

        if default_file is not None:
            cls._default_file = os.path.normpath(default_file)
        elif defaults.reset(reset):
            cls._default_file = None

    @classmethod
    def default_root(
            cls,
            root: Optional[str] = None,
            ):
        return defaults.set_default(root, defaults.root(cls._default_root))

    @classmethod
    def default_subdir(
            cls,
            subdir: Optional[str] = None,
            ):
        return defaults.set_default(subdir, defaults.subdir(cls._default_subdir))

    @classmethod
    def default_file(
            cls,
            file: Optional[str] = None,
            ):
        return defaults.set_default(file, defaults.file(cls._default_file))

    @classmethod
    def from_path(
            cls,
            path: Optional[None] = None,
            root_levels: Optional[int] = None,
            ):

        kw = {}
        if path is not None:
            if isinstance(path, Path):
                path = path.path(app_mkdir=False, app_print=False, ret_dir=False)

            s = splitall(p=path, group_dirs=True)
            if root_levels is None:
                root_levels = calc_prerel_levels(path=s[1])

            dirs = s[1].split(div())
            drive_list = [s[0]] if s[0] != "" else []

            kw["root"] = joinlist([*drive_list, *dirs[:root_levels]])
            kw["subdir"] = joinlist(dirs[root_levels:])
            kw["file"] = s[2] + s[3]

        return cls(**kw)

    def __init__(
                self,
                root: Optional[str] = None,
                subdir: Optional[str] = None,
                file: Optional[str] = None,
                ):

        self.set_root(root=root, reset=True)
        self.set_subdir(subdir=subdir, reset=True)
        self.set_file(file=file, reset=True)

    def __repr__(self):

        s = f" root: {self.root()}\nsubdir: {self.subdir()}\n  file: {self.file()}\n  path: {self.path(ret_dir=False)}"  # noqa: E501

        return s

    def set_root(
                self,
                root: Optional[str] = None,
                reset: Optional[bool] = None,
                ):

        if root is not None:
            self._root = os.path.normpath(root)
        elif defaults.reset(reset):
            self._root = None

    def root(
            self,
            ):

        return self.default_root(root=self._root)

    def set_subdir(
                self,
                subdir: Optional[str] = None,
                reset: Optional[bool] = None,
                ):

        if subdir is not None:
            self._subdir = os.path.normpath(subdir)
        elif defaults.reset(reset):
            self._subdir = None

    def subdir(
            self,
            ):

        return self.default_subdir(subdir=self._subdir)

    def set_file(
                self,
                file: Optional[str] = None,
                reset: Optional[bool] = None,
                ):

        if file is not None:
            self._file = os.path.normpath(file)
        elif defaults.reset(reset):
            self._file = None

    def file(
            self,
            ):

        return self.default_file(file=self._file)

    def path(
                self,
                app_mkdir: Optional[bool] = None,
                app_print: Optional[bool] = None,
                ret_dir: Optional[bool] = None,
                ):

        app_mkdir = defaults.set_default(app_mkdir, False)
        app_print = defaults.set_default(app_print, False)
        file = self.file()
        path = joinlist([self.root(), self.subdir(), file])

        group_dirs = np.isin(file, ["", "."])

        path = mkdir(
                path=path,
                app_mkdir=app_mkdir,
                app_print=app_print,
                ret_dir=ret_dir,
                group_dirs=group_dirs,
                )

        return path

    def new_path(
                    self,
                    add: Optional[str] = None,
                    app_mkdir: Optional[bool] = None,
                    app_print: Optional[bool] = None,
                    ret_dir: Optional[bool] = None,
                    ):

        file = None
        subdir = self._subdir
        if add is not None:
            fsplit = os.path.split(p=add)
            is_dir = is_path_dir(fsplit[1])
            if fsplit[0] != "" or is_dir:
                subdir = os.path.join(
                            *[self.subdir(), *fsplit[:1+int(is_dir)]
                              ])
            if not is_dir:
                file = fsplit[1]

        p = type(self)(
                    root=self._root,
                    subdir=subdir, file=file,
                    )

        p.path(
            app_mkdir=app_mkdir, app_print=app_print, ret_dir=ret_dir,
            )

        return p


class RWPath(Path):
    pass


class RPath(RWPath):
    pass


class WPath(RWPath):
    pass


class SPath(Path):
    pass


# =====================================
# FUNCTIONS
# =====================================

def div():

    return os.path.join("a", "")[1:]


def is_path_dir(
                path: str,
                ) -> bool:

    return os.path.splitext(path)[1] == ""


def calc_prerel_levels(
                        path: str,
                        ) -> int:

    dirs = np.array(os.path.normpath(path).split(div()))

    is_level = dirs == ".."

    if np.all(is_level):
        nlevels = is_level.size
    elif is_level[0]:
        nlevels = 1 + np.where(np.diff(is_level.astype(int)) == -1)[0][0]
    else:
        nlevels = 0

    return nlevels


def capp_dirdiv(
                path: str,
                app_dirdiv: Optional[bool] = None,
                ):

    app_dirdiv = defaults.set_default(app_dirdiv, True)
    if app_dirdiv:
        _div = div()
        if is_path_dir(path=path) and path[-len(_div):] != _div:
            path = os.path.join(path, "")

    return path


def capp_dot(
            path: str,
            app_dot: Optional[bool] = None,
            ):

    app_dot = defaults.set_default(app_dot, True)

    if app_dot and not os.path.isabs(s=path):
        if path.split(div())[0] != ".":
            path = os.path.join(".", path)

    return path


def relpath(
            path: str,
            start: Optional[str] = None,
            app_dirdiv: Optional[bool] = None,
            app_dot: Optional[bool] = None,
            ):

    if path == "":
        path = "."

    path = os.path.relpath(path=path, start=start)
    path = capp_dirdiv(path=path, app_dirdiv=app_dirdiv)
    path = capp_dot(path=path, app_dot=app_dot)

    return path


def realpath(
            path: str,
            strict: bool = False,
            app_dirdiv: Optional[bool] = None,
            ):

    path = os.path.realpath(filename=path, strict=strict)
    path = capp_dirdiv(path=path, app_dirdiv=app_dirdiv)

    return path


def cleanpath(
                path: str,
                clean: Optional[bool] = None,
                clean_as: Optional[Literal["real", "rel"]] = None,
                app_dirdiv: Optional[bool] = None,
                app_dot: Optional[bool] = None,
                ):

    clean = defaults.set_default(clean, True)

    if clean:
        if clean_as is None:
            if os.path.isabs(s=path):
                clean_as = "real"
            else:
                clean_as = "rel"
        clean_as = clean_as[:3].lower()

        if clean_as == "rea":
            path = realpath(path=path, app_dirdiv=app_dirdiv)
        else:
            path = relpath(path=path, app_dirdiv=app_dirdiv, app_dot=app_dot)

    return path


def splitall(
        p: str,
        group_dirs: Optional[bool] = None,
        ):

    p = os.path.normpath(path=p)
    split = os.path.split(p=p)
    s = [*os.path.splitdrive(split[0]), *os.path.splitext(split[1])]

    group_dirs = defaults.set_default(group_dirs, False)

    if group_dirs:
        if is_path_dir(path=p):
            s[1] = os.path.join(s[1], s[2])
            s[2] = ""

    return tuple(s)


def joinlist(
                path_list: list,
                clean: Optional[bool] = None,
                clean_as: Optional[Literal["real", "rel"]] = None,
                app_dirdiv: Optional[bool] = None,
                app_dot: Optional[bool] = None,
                ):

    if len(path_list) == 0:
        path = ""
    else:
        path = os.path.join(*path_list)

    path = cleanpath(
                path=path,
                clean=clean,
                clean_as=clean_as,
                app_dirdiv=app_dirdiv,
                app_dot=app_dot,
                )

    return path


def mkdir(
                path: str,
                app_mkdir: Optional[bool] = None,
                app_print: Optional[bool] = None,
                ret_dir: Optional[bool] = None,
                app_dirdiv: Optional[bool] = None,
                app_dot: Optional[bool] = None,
                group_dirs: Optional[bool] = None,
                ) -> str:

    app_mkdir = defaults.set_default(app_mkdir, True)
    app_print = defaults.set_default(app_print, False)
    ret_dir = defaults.set_default(ret_dir, False)

    group_dirs = defaults.set_default(group_dirs, True)

    s = splitall(p=path, group_dirs=group_dirs)
    if app_mkdir:
        dirs = s[1].split(div())
        for i in range(len(dirs)):
            _path = joinlist(path_list=[s[0], *dirs[:(1+i)]], clean=False)
            if not np.isin(dirs[i], ["", ".", ".."]):
                if not os.path.exists(_path):
                    os.mkdir(_path)

    app_dirdiv = defaults.set_default(app_dirdiv, True)
    app_dot = defaults.set_default(app_dot, True)

    path = joinlist(
                path_list=[*s[:2]],
                clean=True,
                app_dirdiv=app_dirdiv,
                app_dot=app_dot,
                )
    if not ret_dir:
        filename = s[2] + s[3]
        path = os.path.join(path, filename)

    if app_print:
        print(path)

    return path


def capp_ext(
            path: str,
            ext: Optional[str] = None,
            app_ext: Optional[bool] = None,
            ):

    ext = str(defaults.set_default(ext, ""))
    app_ext = defaults.set_default(app_ext, True)
    if ext == "":
        app_ext = False

    if app_ext:
        if ext[0] != ".":
            ext = "." + ext

    if os.path.splitext(path)[1] != ext:
        path = path + ext

    return path


def calc_path(
            file: Optional[str] = None,
            path: Optional[str] = None,
            ext: Optional[str] = None,
            app_ext: Optional[bool] = None,
            app_mkdir: Optional[bool] = None,
            app_print: Optional[bool] = None,
            dPath: Optional[Path] = None,
            ):

    app_mkdir = defaults.set_default(app_mkdir, False)
    app_print = defaults.set_default(app_print, False)

    if dPath is None:
        dPath = Path

    p = dPath.from_path(path=path)
    file = defaults.set_default(file, p.file())

    if np.isin(file, ["", ".", ".."]):
        file = None

    path = (
            p
            .new_path(add=file)
            .path(app_mkdir=False, app_print=False, ret_dir=False)
            )

    path = capp_ext(path=path, ext=ext, app_ext=app_ext)
    path = mkdir(path=path, app_mkdir=app_mkdir, app_print=app_print, ret_dir=False)

    return path


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

    app_write = defaults.set_default(app_write, True)

    if app_write:
        if dPath is None:
            dPath = WPath

        app_mkdir = defaults.set_default(app_mkdir, True)
        app_print_path = defaults.set_default(app_print_path, True)

        path = calc_path(
                    path=path, file=file,
                    ext=ext, app_ext=app_ext,
                    app_mkdir=app_mkdir, app_print=app_print_path,
                    dPath=dPath,
                    )
        with open(path, "w") as f:
            f.write(s)

    return s


def mkdir_new(
                path: str,
                mkdir: Optional[bool] = None,
                ) -> str:

    mkdir = defaults.mkdir(mkdir)
    if mkdir:
        if not os.path.exists(path):
            os.mkdir(path)
    return path


def calc_root(
                root: Optional[str] = None,
                mkdir: Optional[bool] = None,
                ) -> str:

    mkdir = defaults.mkdir(mkdir)
    root = defaults.root(root)

    return mkdir_new(path=root)
