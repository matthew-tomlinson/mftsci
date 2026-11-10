
"""
Functions module for sklearn.base.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal, List

# External packages
import numpy as np
import pandas as pd
from sklearn.base import (  # noqa: F401
                            TransformerMixin as _TransformerMixin,
                            OneToOneFeatureMixin as _OneToOneFeatureMixin,
                            ClassNamePrefixFeaturesOutMixin,
                            )


# Internal packages
from . import defaults   # noqa: F401
from ...pd import unitquant, to_subdtype


# =====================================
# CLASSES
# =====================================

class BaseLogDetMixin:
    """
    Adds `calc_logdet(X)` to a transformer.
    Must be combined with a fitted sklearn transformer.
    """
    def calc_scale(
            self,
            X: Optional[np.typing.ArrayLike],
            ) -> Union[float, np.typing.ArrayLike]:
        """
        Return scale.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement calc_scale()."
        )

    def calc_logdet(
            self,
            X: Optional[np.typing.ArrayLike],
            ) -> Union[float, np.typing.ArrayLike]:
        """
        Return per-sample log|det J_T(x)| for y = T(x) at inputs X.
        Default raises; subclasses must implement or call super() to fall back.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement calc_logdet()."
        )


class UnitsMixin(BaseLogDetMixin):
    """
    """

    def __init__(self, *args, **kwargs):
        self.dtypes_in_ = None
        super().__init__(*args, **kwargs)

    def deunit_X(self, X):
        if self.dtypes_in_ is not None and isinstance(X, pd.DataFrame):
            return to_subdtype(df=X, dtypes=self.dtypes_in_, app=True)
        else:
            return X

    def reunit_X(self, X):
        if self.dtypes_in_ is not None and isinstance(X, pd.DataFrame):
            return X.astype(self.dtypes_in_)
        else:
            return X

    def unitquant(
                self,
                prod=None,
                dt_to_pint=None,
                ):
        if self.dtypes_in_ is not None:
            return unitquant(self.dtypes_in_, prod=prod, dt_to_pint=dt_to_pint)
        else:
            return 1

    def calc_det_unitquant(
            self,
            X: Optional[np.typing.ArrayLike] = None,
            ):

        return np.exp(self.calc_logdet(X)) / self.unitquant(prod=True)


class TransformerMixin(UnitsMixin, _OneToOneFeatureMixin, _TransformerMixin):
    """
    Mixin that adds:
      - MultiIndex unit-level dropping/restoring
      - output-follow-fit logic
      - set_output(transform="pandas"|"default"|None)

    Assumes the subclass is *shape-preserving* (e.g., StandardScaler).
    """
    _drop_unit = None
    _sklearn_output_config_alt = {
                                    s: "default"
                                    for s in ["transform", "inverse_transform"]
                                    }

    def __init__(
                    self,
                    *,
                    raise_on_duplicates=False,
                    **kwargs,
                    ):
        super().__init__(**kwargs)
        self.raise_on_duplicates = raise_on_duplicates

    def set_output(self, **kwargs):
        super().set_output(transform="default")
        for s in kwargs:
            self._sklearn_output_config_alt[s] = kwargs[s]

        return self

    def _wrap_transform(
            self,
            X,
            output_format: Optional[Literal["default", "pandas", "polars"]] = None,
            fname: Optional[str] = None,
            ftype: Optional[Literal["o", "i"]] = None,
            index: Optional[pd.Index] = None,
            reunit: Optional[bool] = None,
            ):

        output_format = self.calc_output_format(
                        output_format=output_format,
                        fname=fname,
                        ftype=ftype,
                        default_format="pandas",
                        )
        if output_format == "default":
            return X

        if ftype is None:
            if fname is None:
                ftype = defaults.ftype(ftype)
            else:
                if fname == "inverse_transform":
                    ftype = "i"
                else:
                    ftype = "o"
        ftype = defaults.ftype(ftype)
        try:
            columns = self.make_columns(ftype=ftype)
        except Exception:
            return X

        if output_format == "pandas":
            if isinstance(X, pd.DataFrame):
                X = X.set_axis(columns, axis=1)
            else:
                X = pd.DataFrame(X, columns=columns, index=index)

            if reunit is None:
                reunit = ftype == "i"

            if reunit:
                X = self.reunit_X(X)

        return X

    def _wrap_output_transform(
                    self,
                    X,
                    index: Optional[pd.Index] = None,
                    reunit: Optional[bool] = None,
                    ):

        reunit = False

        return self._wrap_transform(X=X, ftype="o", index=index, reunit=reunit)

    def _wrap_input_transform(
                    self,
                    X,
                    index: Optional[pd.Index] = None,
                    reunit: Optional[bool] = None,
                    ):

        reunit = True

        return self._wrap_transform(X=X, ftype="i", index=index, reunit=reunit)

    def _fit_colums_unit_levels(
            self,
            unit_levels=None,
            ):

        self.unit_levels_ = clean_sublevels(
                                    names=self.columns_names_in_,
                                    sublevels=unit_levels,
                                    to_names=False,
                                    to_FrozenList=True,
                                    )
        if self.raise_on_duplicates and len(self.columns_names_in_) > 1:
            cols_out = self.make_columns(ftype="out")
            duplicated = cols_out.duplicated()
            if duplicated.any():
                d = cols_out[duplicated].unique()
                d_ = list(map(str, d))
                raise ValueError(
                    f"Dropping unit levels created duplicate columns: {d_}"
                    )

    def _fit_columns(
                self,
                X,
                y=None,
                unit_levels=None,
                ):
        self.dtypes_in_ = getattr(X, "dtypes", None)

        if isinstance(X, pd.DataFrame):
            self._fit_type = "pandas"
        else:
            self._fit_type = "default"

        self.set_output(**{
                            s: self._fit_type
                            for s in ["transform", "inverse_transform"]
                            })

        if self._fit_type != "default":

            if self._fit_type == "pandas":
                self.columns_in_ = X.columns.values
                self.columns_names_in_ = X.columns.names
                self.index_type_ = type(X.index)
            else:
                self.columns_in_ = getattr(X, "columns", None)
                self.columns_names_in_ = pd.core.indexes.frozen.FrozenList([None])
                self.index_type_ = None

            self._fit_colums_unit_levels(unit_levels=unit_levels)

        else:
            self.columns_in_ = None
            self.columns_names_in_ = None
            self.index_type_ = None
            self.unit_levels_ = None

    def fit(self, X, y=None, unit_levels=None):
        self._fit_columns(X=X, y=y, unit_levels=unit_levels)
        X_deunit = self.deunit_X(X)
        return super().fit(X_deunit, y)

    def transform(self, X):

        Xt = super().transform(self.deunit_X(X))
        index = None if not isinstance(X, pd.DataFrame) else X.index
        return self._wrap_output_transform(X=Xt, index=index, reunit=False)

    def fit_transform(self, X, y=None, **fit_params):
        Xt = super().fit_transform(X, y, **fit_params)
        if isinstance(Xt, np.ndarray):
            index = None if not isinstance(X, pd.DataFrame) else X.index
            return self._wrap_output_transform(X=Xt, index=index, reunit=False)
        else:
            return Xt

    def inverse_transform(self, X):
        Xinv = super().inverse_transform(X)
        index = None if not isinstance(X, pd.DataFrame) else X.index
        return self._wrap_input_transform(X=Xinv, index=index, reunit=True)

    def make_columns(
                self,
                ftype: Optional[Literal["in", "out"]] = None,
                fname: Optional[str] = None,
                output_format: Optional[Literal["pandas", "polars"]] = None,
                unit_levels: Optional[Union[object, int, list]] = None,
                drop_unit: Optional[bool] = None,
                ):

        ftype = defaults.ftype(ftype)

        if unit_levels is None:
            unit_levels = getattr(self, "unit_levels_", None)

        drop_unit = defaults.set_default(drop_unit, (ftype == "o") and self._drop_unit)
        output_format = self.calc_output_format(
                                output_format=output_format,
                                fname=fname,
                                ftype=ftype,
                                default_format="pandas",
                                )

        if ftype == "o" and isinstance(self, ClassNamePrefixFeaturesOutMixin):
            return make_index(
                        tuples=self.get_feature_names_out(),
                        output_format=output_format,
                        )
        else:
            return make_index(
                        tuples=self.columns_in_,
                        names=self.columns_names_in_,
                        unit_levels=unit_levels,
                        drop_unit=drop_unit,
                        output_format=output_format,
                        )

    def calc_output_format(
            self,
            output_format: Optional[Literal["default", "pandas", "polars"]] = None,
            fname: Optional[str] = None,
            ftype: Optional[Literal["o", "i"]] = None,
            default_format: Optional[str] = None,
            ):

        if output_format is None:
            if default_format is None:
                default_format = "default"
            fname = calc_output_fname(fname=fname, ftype=ftype)
            output_format = self._sklearn_output_config_alt.get(fname, default_format)

        return output_format


# =====================================
# FUNCTIONS
# =====================================


def make_index(
            tuples: List[tuple | object],
            names: Optional[list] = None,
            output_format: Optional[Literal["pandas", "polars"]] = None,
            unit_levels: Optional[Union[object, int, list]] = None,
            drop_unit: Optional[bool] = None,
            ):

    if not isinstance(tuples[0], tuple):
        tuples = [(i,) for i in tuples]

    index = pd.MultiIndex.from_tuples(tuples, names=names)

    if unit_levels and drop_unit:
        unit_levels = clean_sublevels(
                            names=index.names,
                            sublevels=unit_levels,
                            to_names=False,
                            to_FrozenList=False,
                            )
        index = index.droplevel(unit_levels)

    if index.nlevels == 1:
        index = index.get_level_values(0)

    if output_format == "polars":
        index = [str(s) for s in index.get_level_values(0).values]

    return index


def clean_sublevels(
        names: Optional[list] = None,
        sublevels: Optional[Union[list, str]] = None,
        to_names: Optional[bool] = None,
        to_FrozenList: Optional[bool] = None
        ):

    if names is None:
        return None
    else:
        if to_names is None:
            to_names = False

        if to_FrozenList is None:
            to_FrozenList = False

        if sublevels is None:
            sublevs = []
        else:
            idx_isin = np.where(np.isin(names, sublevels))[0]
            ints = np.array([
                        i for i in sublevels
                        if isinstance(i, int) or isinstance(i, np.integer)
                        ])
            idx_ints = ints[ints < len(names)]
            sublevs = sorted([*idx_isin, *idx_ints])

        if to_names:
            sublevs = [names[int(i)] for i in sublevs]

        if to_FrozenList:
            sublevs = pd.core.indexes.frozen.FrozenList(sublevs)

        return sublevs


def calc_output_fname(
        fname: Optional[str] = None,
        ftype: Optional[Literal["o", "i"]] = None,
        ):

    if fname is None:
        ftype = defaults.ftype(ftype)
        if ftype == "i":
            fname = "inverse_transform"
        else:
            fname = "transform"

    return fname
