"""
Functions module for signal utilities.
"""
# =====================================
# IMPORT
# =====================================

# External packages
import numpy as np
import pandas as pd
import scipy

# Internal packages
from ..meta import require_optional, set_default_kwargs
from ..pd import to_subdtype
from ..sklearn.preprocessing import StandardScaler


# =====================================
# FUNCTIONS
# =====================================


def to_int(a, mode=None, **kwargs):

    if a is None:
        pass
    else:
        if mode is None:
            f = "trunc"
        elif not mode:
            f = ""
        else:
            mode = str(mode)[:2].lower()
            N = len(mode)
            fncs = ["round", "ceil", "fix", "floor", "rint", "trunc"]
            f = dict(zip([f[:N] for f in fncs], fncs)).get(mode, "")
        if f:
            _is_scalar = np.isscalar(a)
            a = getattr(np, f)(a).astype(int)
            if _is_scalar:
                a = int(a)
    return a


def polyfit(
        a,
        polyorder=None,
        axis=None,
        ret_polyval=None
        ):

    if ret_polyval is None:
        ret_polyval = False

    if polyorder is None:
        polyorder = 1

    if isinstance(a, pd.DataFrame):
        if axis is None:
            axis = 0
        pf = a.apply(
                lambda v:
                polyfit(v, polyorder, ret_polyval=False),
                axis=axis)
    else:
        is_series = isinstance(a, pd.Series)
        a = np.atleast_1d(a)

        if axis is not None:
            a = np.moveaxis(a, axis, 0)

        x = np.linspace(0, 1, len(a))
        pf = np.polyfit(x, a, polyorder)

        if is_series:
            pf = pd.Series(pf)

    if ret_polyval:

        if isinstance(a, pd.DataFrame):
            x = np.linspace(0, 1, a.shape[axis])
            if np.abs(axis) == 1:
                _idx = a.columns
            else:
                _idx = a.index

            pv = pf.apply(
                    lambda v: np.polyval(v, x), axis=axis
                    ).set_axis(_idx, axis=axis)
        else:
            pv = np.polyval(pf, x)

        return pf, pv
    else:
        return pf


def _filt_w(
        _filter,
        _x_name,
        _w_name,
        x,
        w=None,
        **fkwargs,
        ):

    fkwargs[_x_name] = x
    if w is None or np.isscalar(w):
        fkwargs[_w_name] = w
        res = _filter(**fkwargs)
    else:
        res = np.zeros_like(x)
        unique_ws = np.unique(w)

        for _w in unique_ws:
            idx = _w == w
            fkwargs[_w_name] = _w
            res[idx] = _filter(**fkwargs)[idx]

    _is_srs = isinstance(x, pd.Series)
    _is_df = isinstance(x, pd.DataFrame)
    if _is_srs or _is_df:
        index = x.index
        if _is_srs:
            res = pd.Series(res, index=index)
        else:
            res = pd.DataFrame(res, index=index, columns=x.columns)

    return res


def w_clean(
        w,
        w_min=None,
        w_max=None,
        w_mode=None,
        w_min_mode=None,
        w_max_mode=None,
        force_odd=None,
        ):

    if force_odd is None:
        force_odd = False

    if w_mode is None:
        w_mode = "round"

    w_prime = to_int(w, mode=w_mode)
    if force_odd:
        even = (w_prime % 2) == 0
        if np.any(even):
            if isinstance(w, list):
                w = np.array(w)
            s = np.sign(w - w_prime).astype(int)
            s = np.where(s == 0, 1, s)
            w_prime += np.where(even, s, 0)

        if w_min is not None:
            w_min += to_int((w_min % 2) == 0)

        if w_max is not None:
            w_max -= to_int((w_min % 2) == 0)

    if w_min_mode is None:
        w_min_mode = "ceil"

    if w_max_mode is None:
        w_max_mode = "floor"

    w_min = to_int(w_min, mode=w_min_mode)
    w_max = to_int(w_max, mode=w_max_mode)

    return np.clip(w_prime, a_min=w_min, a_max=w_max)


def filter_w(
        x,
        w,
        filt=None,
        **fkwargs,
        ):

    if filt is None:
        filt = "savgoy"
    filt = str(filt).lower()

    if filt[:6] == "savgoy":
        return savgoy_filter(x=x, w=w, **fkwargs)
    elif filt[:6] == "median":
        return median_filter(x=x, w=w, **fkwargs)
    else:
        return filter_w(x=x, w=w, **fkwargs)


def savgoy_filter(
        x, w, polyorder,
        **fkwargs,
        ):

    return _filt_w(
            _filter=scipy.signal.savgol_filter,
            _x_name="x",
            _w_name="window_length",
            x=x,
            w=w,
            polyorder=polyorder,
            **fkwargs,
            )


def median_filter(
        x, w,
        **fkwargs,
        ):

    return _filt_w(
            _filter=scipy.ndimage.median_filter,
            _x_name="input",
            _w_name="size",
            x=x,
            w=w,
            **fkwargs,
            )


def calc_r(
        r=None,
        d_r=None,
        ):

    if isinstance(r, bool):
        d_r = d_r or np.nan
        r = d_r if r else np.nan
    else:
        r = r or np.nan
    return r


def wavelet_ridge_freq(
        signal,
        sample_rate=None,
        ):
    """
    Returns smooth_freq (dominant frequency over time)
    """
    if sample_rate is None:
        sample_rate = 1.0

    # Define scales (log-spaced is best for chirps)
    scales = np.geomspace(2, 128, num=2**7)

    cwt = require_optional(module="pywt", attr="cwt")
    coeffs, freqs = cwt(
                        signal,
                        scales,
                        'morl',
                        sampling_period=1.0/sample_rate,
                        )

    power = np.abs(coeffs)

    # ridge: frequency of max power at each time
    ridge_idx = np.argmax(power, axis=0)
    ridge_freq = freqs[ridge_idx]

    return ridge_freq


def wavelets_decomp(
        signal,
        sample_rate=None,
        steplike=None,
        eps=None,
        reindex=None,
        ):

    if isinstance(signal, pd.DataFrame):

        if sample_rate is None:
            sample_rate = 1.0 / np.median(np.diff(signal.index))

        comps = []
        for c in signal.columns:
            comps.append(
                    wavelets_decomp(
                        signal=signal[c],
                        sample_rate=sample_rate,
                        steplike=steplike,
                        eps=eps,
                        )
                        )
        comps = pd.concat(comps, axis=1, keys=signal.columns)

        return comps

    if isinstance(signal, pd.Series):
        if sample_rate is None:
            sample_rate = 1.0 / np.median(np.diff(signal.index))

        comps = wavelets_decomp(
                    signal=signal.values,
                    sample_rate=sample_rate,
                    steplike=steplike,
                    eps=eps,
                    reindex=False,
                    )

        comps = comps.set_axis(signal.index, axis=0)

        return comps

    if eps is None:
        eps = 1e-12

    if sample_rate is None:
        sample_rate = 1.0
        reindex = False

    if reindex is None:
        reindex = sample_rate != 1.0

    if hasattr(signal, "values"):
        signal = signal.values

    N = len(signal)

    inst_freq = wavelet_ridge_freq(
                        signal=signal,
                        sample_rate=sample_rate,
                        )

    trend = 0.0 * signal
    noise = 0.0 * signal
    osc = signal - trend
    analytic = scipy.signal.hilbert(osc)
    inst_amp = np.abs(analytic)

    smooth_freq = scipy.signal.medfilt(inst_freq, 11)

    local_cycle_durations = 1.0 / np.where(smooth_freq <= 0, 1e-5, smooth_freq)
    dynamic_w = (local_cycle_durations * 1.5)
    ia_w = w_clean(dynamic_w, w_min=5, w_max=201)
    smooth_amp = savgoy_filter(inst_amp, w=ia_w, polyorder=2, axis=0)

    comps = pd.DataFrame({
                "trend": trend,
                "osc": osc,
                "noise": noise,
                "smooth_amp": smooth_amp,
                "smooth_freq": smooth_freq,
                "inst_amp": inst_amp,
                "inst_freq": inst_freq,
                }).rename_axis("comp", axis=1)

    if reindex:
        comps.index = pd.Index(np.arange(N) / sample_rate)

    return comps


def emd_decomp(
        signal,
        prior_trend=None,
        force_prior_trend=None,
        incl_prior_trend=None,
        sample_rate=None,
        steplike=None,
        low_wmean_freq=None,
        eps=None,
        reindex=None,
        module=None,
        emd_kwargs=None,
        wn_alpha=None,
        min_if_window=None,
        r_cycle=None,
        w_min=None,
        w_max=None,
        ):

    if prior_trend is None:
        prior_trend = 0

    if force_prior_trend is None:
        force_prior_trend = False

    dsignal = signal - prior_trend

    comps, imfs, imf_info, var, eparams = emd_decomp_(
                                    dsignal,
                                    sample_rate=sample_rate,
                                    steplike=steplike,
                                    low_wmean_freq=low_wmean_freq,
                                    wn_alpha=wn_alpha,
                                    eps=eps,
                                    ret_imf_info_only=False,
                                    reindex=reindex,
                                    module=module,
                                    emd_kwargs=emd_kwargs,
                                    min_if_window=min_if_window,
                                    r_cycle=r_cycle,
                                    w_min=w_min,
                                    w_max=w_max,
                                    )

    is_df = isinstance(dsignal, pd.DataFrame)

    if is_df:
        trend_c = comps.columns[
                    comps.columns.to_frame().iloc[:, -1] == "trend"
                    ]
        if not isinstance(force_prior_trend, pd.Series):
            if isinstance(force_prior_trend, dict):
                idx = None
            else:
                idx = trend_c
            force_prior_trend = pd.Series(force_prior_trend, index=idx)

        fpt = force_prior_trend[force_prior_trend].index
        trend_c_fpt = trend_c.to_frame().loc[fpt].index
    else:
        trend_c = ["trend"]
        force_prior_trend = bool(force_prior_trend)
        if force_prior_trend:
            trend_c_fpt = trend_c
        else:
            trend_c_fpt = []

    comps.loc[:, trend_c_fpt] = 0.0
    comps.loc[:, trend_c] += prior_trend

    if incl_prior_trend is None:
        incl_prior_trend = False

    if incl_prior_trend:
        ptrend = (0.0 * comps.loc[:, trend_c]).rename(
                columns={"trend": "prior_trend"}, level=-1
                )
        ptrend += prior_trend
        comps = pd.concat([comps, ptrend], axis=1)
        if is_df:
            comps = comps.loc[:, dsignal.columns]

    var.loc[trend_c] = comps.loc[:, trend_c].var(axis=0)

    return comps, imfs, imf_info, var, eparams


def emd_decomp_(
        signal,
        sample_rate=None,
        steplike=None,
        low_wmean_freq=None,
        eps=None,
        ret_imf_info_only=None,
        reindex=None,
        module=None,
        emd_kwargs=None,
        wn_alpha=None,
        min_if_window=None,
        r_cycle=None,
        w_min=None,
        w_max=None,
        ):

    if ret_imf_info_only is None:
        ret_imf_info_only = False

    if isinstance(signal, pd.DataFrame):

        if sample_rate is None:
            sample_rate = 1.0 / np.median(np.diff(signal.index))

        c_axis = pd.Series({
                    "comps": 1,
                    "imfs": 1,
                    "imf_info": 0,
                    "var": 0,
                    "eparams": 1,
                    })

        if ret_imf_info_only:
            c_axis = c_axis.loc[["imf_info"]]

        columns = signal.columns.to_frame()
        if not isinstance(signal.columns, pd.MultiIndex):
            columns = columns.iloc[:, 0].set_axis(
                            columns.index.get_level_values(0), axis=0,
                            )
        else:
            columns = columns.apply(tuple, axis=1)

        kwargs = {
                    "sample_rate": sample_rate,
                    "steplike": steplike,
                    "low_wmean_freq": low_wmean_freq,
                    "eps": eps,
                    "wn_alpha": wn_alpha,
                    "emd_kwargs": emd_kwargs,
                    "min_if_window": min_if_window,
                    "r_cycle": r_cycle,
                    "w_min": w_min,
                    "w_max": w_max,
                    }

        for k, v in kwargs.items():
            if v is not None:
                if isinstance(v, dict) and k == "emd_kwargs":
                    kwargs[k] = columns.index.to_frame().iloc[:, 0].apply(
                                        lambda w: v,
                                        )
                elif not isinstance(v, pd.Series):
                    kwargs[k] = pd.Series(
                                    v,
                                    index=columns.index
                                    )
            else:
                kwargs[k] = pd.Series()

        outp = columns.apply(
                    lambda c:
                    pd.Series(
                        emd_decomp_(
                            signal=signal[c],
                            ret_imf_info_only=ret_imf_info_only,
                            module=module,
                            **{
                                k: v.get(c, None)
                                for k, v in kwargs.items()
                                },
                            )[:c_axis.size],
                        index=c_axis.index
                    ),
                    )

        is_na = outp.isna().all(axis=0)
        outp = [
                    None if is_na.loc[o]
                    else
                    pd.concat(
                        outp.loc[:, o].values,
                        keys=signal.columns,
                        axis=c_axis.loc[o],
                        )
                    for o in is_na.index
                    ]
        if ret_imf_info_only:
            return outp[0]
        else:
            return tuple(outp)

    if isinstance(signal, pd.Series):
        if sample_rate is None:
            sample_rate = 1.0 / np.median(np.diff(signal.index))

        outp = emd_decomp_(
                    signal=signal.values,
                    sample_rate=sample_rate,
                    steplike=steplike,
                    low_wmean_freq=low_wmean_freq,
                    eps=eps,
                    ret_imf_info_only=ret_imf_info_only,
                    reindex=False,
                    module=module,
                    emd_kwargs=emd_kwargs,
                    wn_alpha=wn_alpha,
                    min_if_window=min_if_window,
                    r_cycle=r_cycle,
                    w_min=w_min,
                    w_max=w_max,
                    )

        if ret_imf_info_only:
            return outp

        comps = outp[0].set_axis(signal.index, axis=0)
        imfs = outp[1].set_axis(signal.index, axis=0)

        return tuple([comps, imfs, *outp[2:],])

    if steplike is None:
        steplike = False

    if eps is None:
        eps = 1e-12

    if sample_rate is None:
        sample_rate = 1.0
        reindex = False

    if reindex is None:
        reindex = sample_rate != 1.0

    if hasattr(signal, "values"):
        signal = signal.values

    if module is None:
        module = "pyemd"
    module = str(module)[:1].lower()

    if emd_kwargs is None:
        emd_kwargs = {}

    if steplike:
        # Sharp steps may require more iterations to stabilize
        if module == "p":
            emd_kwargs.setdefault("spline_kind", "pchip")
        # emd_kwargs.setdefault("max_imfs", 100)
    else:
        # emd_kwargs.setdefault("sift_thresh", 1e-3)
        # emd_kwargs.setdefault("max_imfs", 50)
        pass

    _EMD = None
    try:
        if module == "p":
            if "max_imf" in emd_kwargs:
                mimf = "max_imf"
            elif "max_imfs" in emd_kwargs:
                mimf = "max_imfs"
            else:
                mimf = None

            if mimf is not None:
                fkwargs = {"max_imf": emd_kwargs.pop(mimf)}
            else:
                fkwargs = {}

            EMD = require_optional(module="PyEMD", attr="EMD")
            _EMD = EMD(**emd_kwargs)
            imfs = _EMD(signal, **fkwargs).T
        else:
            sift = require_optional(module="emd.sift", attr="sift")
            imfs = sift(signal, **emd_kwargs)
    except Exception:
        imfs = signal.reshape(-1, 1)

    N = len(signal)
    N_imfs = imfs.shape[1]
    if N_imfs == 0:
        imfs = signal.reshape(-1, 1)
        N_imfs = 1

    if wn_alpha is None:
        wn_alpha = 0.95

    # imf_class
    # 0: trend
    # 1: osc
    # 2: noise
    imf_class = np.repeat(1, N_imfs)
    imf_class[-1] = 0   # Last column is always trend
    if np.any(imf_class != 0):
        # 2. Compute individual frequency stats
        # 'nht' (Normalized Hilbert Transform) optimal for non-stationary data

        frequency_transform = require_optional(
                                module="emd.spectra",
                                attr="frequency_transform",
                                )
        IP, IF, IA = frequency_transform(
                        imfs[:, :-1], sample_rate=sample_rate, method='nht'
                        )
        IA = np.where(np.isnan(IA), 0.0, IA)
        IE = IA ** 2
        WIF = IF * IE

        energy = np.nansum(IE, axis=0)
        tot_energy = np.nansum(energy)

        denom_energy = np.where(energy == 0, 1, energy)
        denom_tot_energy = np.where(tot_energy == 0, 1, tot_energy)

        # Identify IMFs as trend or noise
        cycle_freq = sample_rate / N
        low_wmean_freq = calc_r(r=low_wmean_freq, d_r=1.0 * cycle_freq)

        inst_bw = np.abs(np.gradient(
                    np.log(
                        np.clip(np.abs(IA), eps, None)
                        ),
                    axis=0,
                    ))
        mean_bw = np.nanmean(inst_bw, axis=0)

        energy_frac = energy / denom_tot_energy

        var = np.empty(N_imfs)
        var[:-1] = np.nanvar(IA, axis=0)
        var[-1] = np.nanvar(imfs[:, -1])
        tot_var = np.nansum(var)
        var_frac = var / tot_var

        bw_denom = np.clip(mean_bw, eps, None)

        mean_freq = np.nanmean(IF, axis=0)
        wmean_freq = np.nansum(WIF, axis=0) / denom_energy
        median_freq = np.nanmedian(IF, axis=0)
        centre_freq = (wmean_freq + median_freq) / 2

        Q_t = IF / np.clip(inst_bw, eps, None)

        Q_mean = mean_freq / bw_denom
        Q_wmean = wmean_freq / bw_denom
        Q_median = median_freq / bw_denom
        Q_centre = centre_freq / bw_denom

        Q_t_mean = np.nanmean(Q_t, axis=0)
        Q_t_wmean = np.nansum(IE * Q_t, axis=0) / denom_energy
        Q_t_median = np.nanmedian(Q_t, axis=0)
        Q_t_centre = (Q_t_wmean + Q_t_median) / 2
        Q_t_cv = np.std(Q_t, axis=0) / np.clip(Q_t_mean, eps, None)
        Q_t_wcv = np.sqrt(
                    np.nansum(IE * (Q_t - Q_t_wmean)**2, axis=0) / denom_energy
                    ) / np.clip(Q_t_wmean, eps, None)

        psd = IE / np.clip(
                        np.nansum(IE, axis=0, keepdims=True),
                        eps,
                        None,
                        )
        entropy = -np.nansum(psd * np.log(np.clip(psd, eps, None)), axis=0)
        entropy /= np.log(N)

        is_low_wmean_freq = (wmean_freq < low_wmean_freq)

        imf_class[:-1] = np.where(
                            is_low_wmean_freq,
                            0,
                            imf_class[:-1],
                            )

        if wn_alpha < 1.0:
            whitenoise_check = require_optional(
                                    module="PyEMD.checks",
                                    attr="whitenoise_check",
                                    )
            test = np.array([*whitenoise_check(
                                imfs.T, alpha=wn_alpha
                                ).values()])
            imf_class = np.where(
                            test == 0,
                            2,
                            imf_class,
                            )
        else:
            test = np.full(N_imfs, -1)

        imf_info = pd.DataFrame({
                "class": imf_class[:-1],
                "test": test[:-1],
                "mean_freq": mean_freq,
                "wmean_freq": wmean_freq,
                "median_freq": median_freq,
                "energy": energy,
                "energy_frac": energy_frac,
                "var_frac": var_frac[:-1],
                "mean_bw": mean_bw,
                "Q_mean": Q_mean,
                "Q_wmean": Q_wmean,
                "Q_median": Q_median,
                "Q_centre": Q_centre,
                "Q_t_mean": Q_t_mean,
                "Q_t_wmean": Q_t_wmean,
                "Q_t_median": Q_t_median,
                "Q_t_centre": Q_t_centre,
                "Q_t_cv": Q_t_cv,
                "Q_t_wcv": Q_t_wcv,
                "entropy": entropy,
                "is_low_wmean_freq": is_low_wmean_freq,
                }).rename_axis("imf", axis=0)

        # Calculate energy weighted instantaneous amp and freq of osc component
        inst_e = np.sum(IE[:, imf_class[:-1] == 1], axis=1)
        inst_amp = np.sqrt(inst_e)
        inst_freq = (
                np.nansum(WIF[:, imf_class[:-1] == 1], axis=1)
                / np.where(inst_e == 0, 1, inst_e)
                )
    else:
        imf_info = None

    if ret_imf_info_only:
        return imf_info

    # Sum IMFs aggregated by class
    trend = np.sum(imfs[:, imf_class == 0], axis=1)
    osc = np.sum(imfs[:, imf_class == 1], axis=1)
    noise = np.sum(imfs[:, imf_class == 2], axis=1)

    if min_if_window is None:
        min_if_window = 11
    min_if_window = int(min_if_window)
    if min_if_window % 2 == 0:
        min_if_window += 1  # Make it odd

    if r_cycle is None:
        r_cycle = 1.5

    if w_min is None:
        w_min = 5

    if w_max is None:
        w_max = 201

    if_window = None

    min_denom_smooth_freq = max(1/N, eps)

    if np.any(imf_class == 1):

        # Count the total number of peaks and troughs in osc
        if _EMD is not None:
            extrema = _EMD.find_extrema(np.arange(N), osc)[:2]
        else:
            _find_extrema = require_optional(
                                module="emd.sift",
                                attr="_find_extrema",
                                )
            extrema = _find_extrema(osc)[:2]
        N_extrema = np.concat(extrema[:2]).size

        # Calculate the average number of data rows per full osc cycle
        # (Each cycle contains 1 peak and 1 trough, so we divide extrema by 2)
        avg_cycle_length = N / (N_extrema / 2)
        if_window = int(avg_cycle_length / 2)
        if if_window % 2 == 0:
            if_window += 1  # Make it odd
        if_window = max(min_if_window, if_window)  # Minimum fallback

        smooth_freq = scipy.signal.medfilt(
                        inst_freq, kernel_size=if_window,
                        )

        denom_smooth_freq = np.clip(smooth_freq,
                                    min_denom_smooth_freq,
                                    None)

        local_cycle_durations = (
                        1.0 / denom_smooth_freq
                        )

        dynamic_w = (local_cycle_durations * r_cycle)
        ia_w = w_clean(dynamic_w, w_min=w_min, w_max=w_max)

        smooth_amp = savgoy_filter(inst_amp, w=ia_w, polyorder=2, axis=0)
    else:
        smooth_amp = smooth_freq = inst_amp = inst_freq = np.full(N, np.nan)

    comps = pd.DataFrame({
                "trend": trend,
                "osc": osc,
                "noise": noise,
                "smooth_amp": smooth_amp,
                "smooth_freq": smooth_freq,
                "inst_amp": inst_amp,
                "inst_freq": inst_freq,
                }).rename_axis("comp", axis=1)

    if reindex:
        comps.index = pd.Index(np.arange(N) / sample_rate)

    imfs = pd.DataFrame(imfs, index=comps.index)

    var = comps[["trend", "osc", "noise"]].var(axis=0)

    eparams = pd.Series({
                "sample_rate": sample_rate,
                "steplike": steplike,
                "low_wmean_freq": low_wmean_freq,
                "wn_alpha": wn_alpha,
                "eps": eps,
                "min_if_window": min_if_window,
                "if_window": if_window,
                "min_denom_smooth_freq": min_denom_smooth_freq,
                "r_cycle": r_cycle,
                "w_min": w_min,
                "w_max": w_max,
                "module": module,
                }, dtype=object)

    return comps, imfs, imf_info, var, eparams


def df_interpolate(
        df,
        interp_kwargs=None,
        extrap_kwargs=None,
        app_interp=None,
        app_extrap=None,
        ):

    if app_interp is None:
        app_interp = True

    if app_extrap is None:
        app_extrap = True

    interp_kwargs = set_default_kwargs(
                        interp_kwargs,
                        dkwargs={"method": "index"},
                        )
    extrap_kwargs = set_default_kwargs(
                        extrap_kwargs,
                        dkwargs={
                                    "method": "pchip",
                                    "limit_direction": "both",
                                    "limit_area": "outside",
                                    },
                        )

    if app_interp:
        df = df.interpolate(**interp_kwargs)
    if app_extrap:
        df = df.interpolate(**extrap_kwargs)

    return df


def infer_idx(
        signal,
        column,
        val,
        savgoy_kwargs=None,
        interp_kwargs=None,
        extrap_kwargs=None,
        ):

    if savgoy_kwargs is None:
        savgoy_kwargs = {}

    savgoy_kwargs.setdefault("polyorder", 2)
    savgoy_kwargs.setdefault("w", 3)
    savgoy_kwargs["axis"] = 0

    df_idx = savgoy_filter(
                signal[column],
                **savgoy_kwargs,
                ).to_frame(column).reset_index().set_index(column)

    df_idx = df_idx.reindex(df_idx.index.union([val]))

    return df_interpolate(
                df=df_idx,
                interp_kwargs=interp_kwargs, extrap_kwargs=extrap_kwargs,
                ).loc[val]


def infer_idx_row(
        signal,
        column,
        val,
        savgoy_kwargs=None,
        interp_kwargs=None,
        extrap_kwargs=None,
        idx_interp_kwargs=None,
        idx_extrap_kwargs=None,
        ):

    interp_kwargs = set_default_kwargs(
                        interp_kwargs,
                        dkwargs={"method": "index"},
                        )
    extrap_kwargs = set_default_kwargs(
                        extrap_kwargs,
                        dkwargs={
                                    "method": "pchip",
                                    "limit_direction": "both",
                                    "limit_area": "outside",
                                    },
                        )

    idx_interp_kwargs = set_default_kwargs(
                        idx_interp_kwargs,
                        dkwargs=interp_kwargs,
                        )
    idx_extrap_kwargs = set_default_kwargs(
                        idx_extrap_kwargs,
                        dkwargs=extrap_kwargs,
                        )

    idx = infer_idx(
                signal=signal,
                column=column,
                val=val,
                savgoy_kwargs=savgoy_kwargs,
                interp_kwargs=idx_interp_kwargs,
                extrap_kwargs=idx_extrap_kwargs,
                )

    row = df_interpolate(
                df=signal.reindex(signal.index.union(idx)),
                interp_kwargs=interp_kwargs, extrap_kwargs=extrap_kwargs,
                ).loc[idx]

    return row


def calc_t_prime(
        signal,
        column=None,
        val=None,
        savgoy_kwargs=None,
        interp_kwargs=None,
        extrap_kwargs=None,
        idx_interp_kwargs=None,
        idx_extrap_kwargs=None,
        ):

    if column is None:
        column = "Alt"

    if val is None:
        val = 0.0

    row = infer_idx_row(
            signal=signal,
            column=column,
            val=val,
            savgoy_kwargs=savgoy_kwargs,
            interp_kwargs=interp_kwargs,
            extrap_kwargs=extrap_kwargs,
            idx_interp_kwargs=idx_interp_kwargs,
            idx_extrap_kwargs=idx_extrap_kwargs,
            )

    t_f = row["Time"].iloc[0]

    return (signal["Time"] / t_f), t_f


def arange_index(
        ref_index=None,
        start=None,
        stop=None,
        step=None,
        N=None,
        origin=None,
        ):

    if ref_index is None:
        ref_index = pd.Index([0, 1])

    if start is None:
        start = ref_index[0]

    if stop is None:
        stop = ref_index[-1]

    rdiff = stop - start
    if step is None:
        if N is None:
            N = 101
        step = rdiff / (int(N) - 1)

    steps = step * np.arange(1 + (stop - start) // step)
    if origin:
        nindex = stop - steps[::-1]
    else:
        nindex = start + steps

    index = pd.Index(nindex, name=ref_index.name)

    return index


def hht_features(
        signal,
        incl_feature=None,
        scaler=None,
        polyorder_rtrend=None,
        w_rtrend=None,
        w_min_rtrend=None,
        app_reindex=None,
        reindex_col=None,
        reindex_start=None,
        reindex_stop=None,
        reindex_step=None,
        reindex_N=None,
        reindex_origin=None,
        polyorder_reindex=None,
        w_reindex=None,
        w_min_reindex=None,
        interp_kwargs=None,
        ):

    N = len(signal)

    if incl_feature is None:
        incl_feature = pd.DataFrame(
                            True,
                            index=signal.columns,
                            columns=["trend", "smooth_amp", "smooth_freq"]
                            )

    if scaler is None:
        scaler = True

    if app_reindex is None:
        app_reindex = False

    kwargs = {
                "incl_feature": incl_feature,
                "scaler": None,
                "app_reindex": app_reindex,
                "reindex_col": reindex_col,
                }

    index_dtype = None
    if app_reindex:
        if reindex_col in signal.columns:
            ref_index = signal[reindex_col]
            index_dtype = ref_index.dtype
            ref_index = pd.Index(to_subdtype(
                            ref_index
                            ))
        else:
            ref_index = signal.index.get_level_values(0)
            index_dtype = ref_index.dtype

        new_index = arange_index(
                    ref_index=ref_index,
                    start=reindex_start,
                    stop=reindex_stop,
                    step=reindex_step,
                    N=reindex_N,
                    origin=reindex_origin,
                    )

        merge_index = ref_index.union(new_index)

        reindex_origin = np.sign(np.diff(new_index[:2])[0]).astype(int)

        kwargs["reindex_start"] = reindex_start
        kwargs["reindex_stop"] = reindex_stop
        kwargs["reindex_step"] = reindex_step
        kwargs["reindex_N"] = reindex_N
        kwargs["reindex_origin"] = reindex_origin

    kwargs["scaler"] = scaler

    if isinstance(scaler, bool):
        if scaler:
            scaler = StandardScaler()

    if scaler:
        if not hasattr(scaler, "n_features_in_"):
            asignal = scaler.fit_transform(signal)
        else:
            asignal = scaler.transform(signal)

    svar = np.var(asignal, axis=0)
    active = svar > 0

    if polyorder_rtrend is None:
        polyorder_rtrend = 2

    if w_min_rtrend is None:
        w_min_rtrend = polyorder_rtrend + 1

    if w_rtrend is None:
        w_rtrend = N/4

    w_rtrend = w_clean(w_rtrend, w_min=3, force_odd=True)

    kwargs["polyorder_rtrend"] = polyorder_rtrend
    kwargs["w_rtrend"] = w_rtrend
    kwargs["w_min_rtrend"] = w_min_rtrend

    osc_only = (
                    ~incl_feature["trend"]
                    & incl_feature.drop(columns="trend").any(axis=1)
                    )
    trend_only = (
                    incl_feature["trend"]
                    & ~incl_feature.drop(columns="trend").any(axis=1)
                    )

    decomp = active & ~trend_only
    osc_only_ = osc_only[decomp]

    rtrend = savgoy_filter(
                asignal.loc[:, decomp],
                w=w_rtrend,
                polyorder=polyorder_rtrend,
                axis=0,
                )

    prior_trend = emd_decomp_(
                rtrend, steplike=False, module="e",
                )[0].loc[:, pd.IndexSlice[:, "trend"]].droplevel(
                                                    "comp", axis=1,
                                                    )

    prior_trend.loc[:, osc_only_] = scaler.transform(
                                        pd.DataFrame(0.0, index=[0],
                                                     columns=signal.columns)
                                        ).loc[:, osc_only].values

    comps, imfs, imf_info, var, eparams = emd_decomp(
                                                signal=asignal.loc[:, decomp],
                                                prior_trend=prior_trend,
                                                incl_prior_trend=True,
                                                force_prior_trend=osc_only_,
                                                min_if_window=51,
                                                )

    _trend = pd.Index(["trend"], name="comp")

    features = asignal.loc[:, ~decomp]
    features = features.set_axis(
                    pd.MultiIndex.from_product([
                        features.columns,
                        _trend
                    ]),
                    axis=1,
                    )
    var = pd.concat([
            np.var(features, axis=0), var
            ], axis=0).loc[asignal.columns]

    _icf = incl_feature.loc[decomp].stack()

    c_f = signal.columns[active & ~decomp]

    features = pd.concat([
                features.loc[:, c_f],
                comps.loc[:, _icf[_icf].index],
                ],
                axis=1)

    c = asignal.columns[asignal.columns.isin(
            features.columns.droplevel(-1).unique()
            )]
    features = features.loc[:, c]

    if app_reindex:

        if polyorder_reindex is None:
            polyorder_reindex = 2

        if w_min_reindex is None:
            w_min_reindex = polyorder_rtrend + 1

        if w_reindex is None:
            w_reindex = "median"

        kwargs["polyorder_reindex"] = polyorder_reindex
        kwargs["w_min_reindex"] = w_min_reindex

        if isinstance(w_reindex, str):
            kwargs["w_reindex"] = w_reindex

            edges = (new_index[:-1] + new_index[1:]) / 2
            digits = pd.Series(np.digitize(ref_index, edges), index=ref_index)

            digits_vc = digits.value_counts().sort_index(ascending=False)
            digits_vc.iloc[[0, -1]] *= 2

            width = digits_vc.reindex(merge_index).interpolate().loc[ref_index]

            med_width = np.nanmedian(width)
            if w_reindex == "median":
                w_reindex = med_width
            else:
                w_reindex = width.fillna(med_width)

        w_reindex = w_clean(w_reindex, force_odd=True, w_min=w_min_reindex)

        if np.isscalar(w_reindex):
            kwargs["w_reindex"] = w_reindex

        features = savgoy_filter(
                                features.set_axis(ref_index, axis=0),
                                w=w_reindex,
                                polyorder=polyorder_reindex,
                                axis=0,
                                )

        kwargs["interp_kwargs"] = interp_kwargs

        features = df_interpolate(
                            df=features.reindex(merge_index),
                            app_extrap=False,
                            interp_kwargs=interp_kwargs,
                            ).loc[new_index]

    if scaler:

        nan_df = pd.DataFrame(np.nan, index=[], columns=asignal.columns)
        fcols = features.columns.to_frame()

        for f in ["trend", "smooth_amp"]:
            idx = fcols.index[fcols["comp"] == f]
            idx_ = idx.droplevel("comp")

            features.loc[:, idx] = scaler.inverse_transform(
                        (
                            features.loc[:, idx]
                            .droplevel("comp", axis=1)
                            .combine_first(
                                nan_df).loc[:, nan_df.columns])
                            ).loc[:, idx_].set_axis(
                        idx, axis=1,
                        )
            if f == "smooth_amp":
                loc = scaler.inverse_transform(
                            pd.DataFrame(
                                0.0, index=[0], columns=asignal.columns,
                                )
                            ).loc[:, idx_].set_axis(idx, axis=1)
                features.loc[:, idx] -= loc.loc[0]

        if index_dtype is not None:
            str_dtype = str(index_dtype)
            if str_dtype[:5] == "pint[":
                unit, ftype = str_dtype[5:-1].split("][")[:2]
                # new_dtype = f"pint[1 / {unit}][{ftype}]"
                new_dtype = "pint[1/s]"
                f = "smooth_freq"
                idx = fcols.index[fcols["comp"] == f]
                features.loc[:, idx] = features.loc[:, idx].astype(new_dtype)

    return features, var, eparams, scaler, kwargs,
