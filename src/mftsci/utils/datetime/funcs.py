"""
Functions module for datetime utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal
import calendar
import datetime

# External packages
import numpy as np
import pandas as pd


# Internal packages
from mftsci.utils.meta import defaults_kwargs, vectorize, vectorize_pd
from mftsci.utils.arr import iterarr as _iterarr, pad
from mftsci.utils.pd import arr_to_srs
from mftsci.utils.datetime import defaults


# =====================================
# FUNCTIONS
# =====================================


@defaults_kwargs(_defaults_call={"year": defaults.year})
@vectorize_pd()
def isleap(
                    year: Optional[int] = None,
                    ) -> bool:

    return calendar.isleap(year=year)


@vectorize_pd()
def iso_year(
                year: Optional[int] = None,
                ) -> str:

    return f"{year:04d}"


@vectorize_pd()
def iso_month(
                month: Optional[int] = None,
                ) -> str:

    return f"{month:02d}"


@vectorize()
def iso_month_1(
                month: Optional[int] = None,
                ) -> np.ndarray:

    return f"{month:02d}"


@vectorize_pd()
def iso_day(
                day: Optional[Union[
                                    int,
                                    np.ndarray,
                                    list,
                                    pd.core.series.Series,
                                    ]] = None,
                ) -> Union[
                            str,
                            np.ndarray,
                            pd.core.series.Series,
                            ]:

    return iso_month(month=day)


@defaults_kwargs(_defaults={"to_datetime": False})
def iso_date(
                year: Optional[int] = None,
                month: Optional[int] = None,
                day: Optional[int] = None,
                mode: Optional[str] = None,
                iterarr: Optional[bool] = None,
                to_datetime: Optional[bool] = None,
                ) -> np.ndarray:

    year = iso_year(year=year)
    month = iso_month(month=month)
    day = iso_day(day=day)

    if mode is not None:
        year, month, day = pad([year, month, day], mode=mode)

    date = year + "-" + month + "-" + day

    if iterarr is None:
        if to_datetime and mode == "outer":
            iterarr = False
        else:
            iterarr = False

    if iterarr:
        date = _iterarr(date)

    if to_datetime:
        if len(np.shape(date)) > 1:
            date = date.astype(np.datetime64)
        else:
            date = pd.to_datetime(date)

    return date


@defaults_kwargs(
        _defaults_call={
                        **{
                            param: getattr(defaults, param)
                            for param in
                            [
                                "day",
                                ]
                            },
                        },
        )
def _day_of_year_native(
                            day: Optional[Union[str, int]] = None,
                        ) -> int:

    doy_leap = 60

    if isinstance(day, str):
        if day == "02-29":
            day = doy_leap
        else:
            day = pd.Timestamp(f"1970-{day}").day_of_year

    return int(day)


@defaults_kwargs(
        _defaults_call={
                        **{
                            param: getattr(defaults, param)
                            for param in
                            [
                                "leapday",
                                "day_0",
                                ]
                            },
                        },
        )
def day_of_year_shift(
                        dt: Union[
                                    datetime.datetime,
                                    pd._libs.tslibs.timestamps.Timestamp,
                                    pd.core.indexes.accessors.DatetimeProperties,
                                    ],
                        day_0: Optional[Union[str, int]] = None,
                        day_mod: Optional[Union[str, int]] = None,
                        leapday: Optional[Literal[
                                                    "unique", "splice",
                                                    "bmerge", "fmerge",
                                                    ]] = None,
                    ) -> Union[pd.core.series.Series, int,]:

    doy_leap = 60 - 1
    doy_0 = _day_of_year_native(day=day_0) - 1
    doy = dt.day_of_year - 1

    if leapday == "splice":
        days_in_year = 365 + dt.is_leap_year.astype(int)
        doy_correct = 0
    elif leapday[-5:] == "merge":
        days_in_year = 365
        if leapday[0] == "f":
            doy_correct = - np.logical_and(dt.is_leap_year, doy > doy_leap).astype(int)
        else:
            doy_correct = - np.logical_and(dt.is_leap_year, doy >= doy_leap).astype(int)
    else:
        days_in_year = 366
        doy_correct = np.logical_and(~dt.is_leap_year, doy >= doy_leap).astype(int)

    doy += doy_correct

    if day_mod is not None:
        doy_mod = _day_of_year_native(day=day_mod)
        doy_mod_rem = days_in_year - doy_mod
        doy = ((doy + doy_mod_rem) % days_in_year) - doy_mod_rem

    doy -= doy_0

    if False:
        if isinstance(doy, pd.Series):
            doy.name = "day_of_year"

    return doy


def day_of_year_season(
                        dt: Union[
                                    datetime.datetime,
                                    pd._libs.tslibs.timestamps.Timestamp,
                                    pd.core.indexes.accessors.DatetimeProperties,
                                    ],
                    ) -> Union[pd.core.series.Series, int,]:

    return day_of_year_shift(dt=dt, leapday="unique")


@defaults_kwargs(_defaults_call={"kind": defaults.season_kind})
def calc_season_begins(
                        kind: Optional[Literal["all", "ast", "met", "quarter"]] = None,
                        begins: Optional[Union[np.ndarray, list,]] = None,
                        ) -> np.ndarray:

    if begins is None:
        kind = kind.lower()
        if kind == "all":
            begins = day_of_year_season(iso_date(
                                            year=1972,
                                            month=1,
                                            day=1,
                                            to_datetime=True,
                                            ))
        elif kind == "ast":
            begins = day_of_year_season(iso_date(
                                            year=1972,
                                            month=[12, 3, 6, 9],
                                            # day=21,
                                            day=[21, 20, 21, 22],
                                            to_datetime=True,
                                            ))
        elif kind == "met":
            begins = day_of_year_season(iso_date(
                                            year=1972,
                                            month=[12, 3, 6, 9],
                                            day=1,
                                            to_datetime=True,
                                            ))
        elif kind == "quarter":
            begins = day_of_year_season(iso_date(
                                            year=1972,
                                            month=[1, 4, 7, 10],
                                            day=1,
                                            to_datetime=True,
                                            ))

    begins = _iterarr(begins)

    return begins


@defaults_kwargs(_defaults_call={"kind": defaults.season_kind})
def calc_season_names(
                        kind: Optional[Literal["all", "ast", "met", "quarter"]] = None,
                        names: Optional[Union[np.ndarray, list,]] = None,
                        N_seasons: Optional[int] = None,
                        ):

    if names is None:
        kind = kind.lower()
        if kind == "all":
            names = ["All"]
        elif kind == "ast":
            names = ["Winter", "Spring", "Summer", "Autumn"]
        elif kind == "met":
            names = ["DJF", "MAM", "JJA", "SON"]
        elif kind == "quarter":
            names = "Q" + (1 + np.arange(4))
        elif N_seasons is not None:
            names = np.arange(N_seasons)

    names = _iterarr(names)

    return arr_to_srs(names)


@defaults_kwargs(
    _defaults_call={"kind": defaults.season_kind},
    )
def season(
            dt: Union[
                        pd._libs.tslibs.timestamps.Timestamp,
                        pd.core.indexes.accessors.DatetimeProperties,
                        ],
            kind: Optional[Literal["all", "ast", "met", "quarter"]] = None,
            begins: Optional[Union[np.ndarray, list,]] = None,
            names: Optional[Union[np.ndarray, list,]] = None,
            ret_names: Optional[bool] = None,
            ):

    doy = day_of_year_season(dt=dt)
    begins = calc_season_begins(kind=kind, begins=begins)

    N_seasons = np.size(begins)
    i_reset = np.argmax(begins)

    if isinstance(doy, pd.Series):
        s_geq = pd.Series(np.sum(
                        np.greater_equal.outer(doy.values, begins),
                        axis=-1),
                        index=doy.index)
    else:
        s_geq = np.sum(np.greater_equal.outer(doy, begins), axis=-1)

    season = ((s_geq % N_seasons) + i_reset) % N_seasons

    ret_names = defaults.set_default(ret_names, default=names is not None)

    if ret_names:
        names = calc_season_names(kind=kind, names=names, N_seasons=N_seasons)
        if np.isscalar(season) or isinstance(season, np.ndarray):
            season = names[season]
            if isinstance(season, pd.Series):
                season.index = dt
                # season = season.values
        else:
            season = season.replace(names)

    return season
