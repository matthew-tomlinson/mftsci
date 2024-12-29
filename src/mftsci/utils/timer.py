"""
Module for timer utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Literal
import time
import datetime

# External packages
import pandas as pd


# Internal packages


# =====================================
# CLASSES
# =====================================


class Timer():

    def __init__(
                    self,
                    print_timer: Optional[bool] = None,
                    print_tab: Optional[bool] = None,
                    print_head: Optional[bool] = None,
                    write_timer: Optional[bool] = None,
                    write_dir: Optional[str] = None,
                    write_fn: Optional[str] = None,
                    write_preamble: Optional[str] = None,
                    reset_start: Optional[bool] = None,
                    ):

        self.set_print_params(print_timer=print_timer,
                              print_tab=print_tab,
                              print_head=print_head)
        self.set_write_params(write_timer=write_timer, write_dir=write_dir,
                              write_fn=write_fn, preamble=write_preamble)

        if reset_start is None:
            reset_start = False
        if reset_start:
            self.reset()
        else:
            self.start()

    def set_print_params(
                            self,
                            print_timer: Optional[bool] = None,
                            print_tab: Optional[bool] = None,
                            print_head: Optional[bool] = None,
                            ):

        self.set_print_timer(print_timer=print_timer)
        self.set_print_tab(print_tab=print_tab)
        self.set_print_head(print_head=print_head)

    def set_print_timer(
                        self,
                        print_timer: Optional[bool] = None,
                        ):

        if print_timer is None:
            print_timer = True
        self._print_timer = print_timer

    def set_print_tab(
                        self,
                        print_tab: Optional[bool] = None,
                        ):

        if print_tab is None:
            print_tab = ""
        self._print_tab = print_tab

    def set_print_head(
                        self,
                        print_head: Optional[bool] = None,
                        ):

        self._print_head = print_head

    def set_write_params(
                            self,
                            write_timer: Optional[bool] = None,
                            write_dir: Optional[str] = None,
                            write_fn: Optional[str] = None,
                            preamble: Optional[str] = None,
                            ):

        self.set_write_timer(write_timer=write_timer)
        self.set_write_dir(write_dir=write_dir)
        self.set_write_fn(write_fn=write_fn)
        self.write_preamble(preamble=preamble)

    def write_preamble(
                        self,
                        preamble: Optional[str] = None,
                        ):

        if self._write_timer and preamble is not None:
            self._file = open(self.get_write_dir_fn(), "a")
            self._file.write(preamble)
            self._file.write("\n")
            self._file.close()

    def set_write_timer(
                        self,
                        write_timer: Optional[bool] = None,
                        ):

        if write_timer is None:
            write_timer = False
        self._write_timer = write_timer

    def set_write_dir(
                        self,
                        write_dir: Optional[str] = None,
                        ):

        if write_dir is None:
            write_dir = "./"
        self._write_dir = write_dir

    def set_write_fn(
                        self,
                        write_fn: Optional[str] = None,
                        ):

        if write_fn is None:
            write_fn = "write_timer.txt"
        self._write_fn = write_fn

    def get_write_dir_fn(
                            self,
                            ) -> str:

        return f"{self._write_dir}{self._write_fn}"

    def get_utc_time(
                        self,
                        ) -> pd._libs.tslibs.timestamps.Timestamp:

        return pd.Timestamp(datetime.datetime.utcnow(), tz="UTC")

    def get_cpu_time(
                        self,
                        ) -> float:

        return time.perf_counter()

    def set_lap(
                self,
                reset: Optional[bool] = None,
                utc_time: Optional[pd._libs.tslibs.timestamps.Timestamp] = None,
                cpu_time: Optional[float] = None,
                ):

        if reset is None:
            reset = False
        if utc_time is None:
            utc_time = self.get_utc_time()
        if cpu_time is None:
            cpu_time = self.get_cpu_time()

        if reset:
            self._utc_lap = [utc_time for i in range(2)]
            self._cpu_lap = [cpu_time for i in range(2)]
        else:
            self._utc_lap[0] = self._utc_lap[1]
            self._utc_lap[1] = utc_time

            self._cpu_lap[0] = self._cpu_lap[1]
            self._cpu_lap[1] = cpu_time

        self._utc_lap_delta = self._utc_lap[1] - self._utc_lap[0]
        self._cpu_lap_delta = self._cpu_lap[1] - self._cpu_lap[0]

    def get_lap_delta(
                        self,
                        val: Optional[Literal["utc", "cpu"]] = None,
                        ) -> pd._libs.tslibs.timedeltas.Timedelta:

        if val is None:
            val = "cpu"

        _lap_delta = {
                        "utc": self._utc_lap_delta,
                        "cpu": self._cpu_lap_delta*pd.Timedelta(seconds=1),
                    }

        return _lap_delta[val]

    def reset(
                self,
                ):

        self._utc_start = self.get_utc_time()
        self._cpu_start = self.get_cpu_time()

        self._utc_stop = self._utc_start
        self._cpu_stop = self._cpu_start

        self.set_delta()

        self.set_lap(reset=True, utc_time=self._utc_start, cpu_time=self._cpu_start)

    def start(
                self,
                ):

        self.reset()
        if self._print_timer:
            if self._print_head is not None:
                print(f"{self._print_tab}{self._print_head}")

            print(f"{self._print_tab}start: {self._utc_start}")

        if self._write_timer:
            self._file = open(self.get_write_dir_fn(), "a")
            if self._print_head is not None:
                self._file.write(f"{self._print_tab}{self._print_head}")
                self._file.write("\n")
            self._file.write(f"{self._print_tab}start: {self._utc_start}")
            self._file.write("\n")
            self._file.close()

    def stop(
                self,
                print_timer: Optional[bool] = None,
                ):

        self._cpu_stop = self.get_cpu_time()
        self._utc_stop = self.get_utc_time()
        self.set_delta()
        self.set_print_timer(print_timer=print_timer)
        if self._print_timer:
            print(f"{self._print_tab}  end: {self._utc_stop} | delta: {self._cpu_delta*pd.Timedelta(seconds=1)}")   # noqa: E501
        if self._write_timer:
            self._file = open(self.get_write_dir_fn(), "a")
            self._file.write(f"{self._print_tab}  end: {self._utc_stop} | delta: {self._cpu_delta*pd.Timedelta(seconds=1)}")    # noqa: E501
            self._file.write("\n")
            self._file.close()

    def set_delta(
                    self,
                    ):

        self._cpu_delta = self._cpu_stop - self._cpu_start
        self._utc_delta = self._utc_stop - self._utc_start

    def get_timer_dict(
                        self,
                        pref: Optional[str] = None,
                        ) -> dict:

        if pref is None:
            pref = ""

        timer = {
                    f"{pref}start": self._utc_start,
                    f"{pref}end": self._utc_stop,
                    f"{pref}delta": self._cpu_delta*pd.Timedelta(seconds=1),
                }

        return timer
