"""Raspberry Pi / host metric readers.

Each function is a cheap, mostly-pure read against /proc, /sys, psutil, or a
cached vcgencmd subprocess. No state is stored here -- callers (see
backend/metrics.py) decide the sampling cadence and do the diffing for
rate-based metrics (disk/net I/O, CPU %).

Everything is wrapped defensively so a missing file or a non-Pi dev machine
(Windows, macOS) returns None / 0 instead of raising. The analytics page
tolerates None and renders a dash.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
def prime_cpu_percent() -> None:
    """Call once at startup so the first real cpu_percent() reading is valid.

    psutil.cpu_percent() returns the delta between the current call and the
    previous one. The very first call after import always returns 0.0, so we
    prime the counters here.
    """
    try:
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
    except Exception:
        logger.exception("prime_cpu_percent failed")


def read_cpu() -> dict:
    """Return total + per-core CPU %. Call at a fixed cadence from the sampler."""
    try:
        total = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        freq = psutil.cpu_freq()
        return {
            "total_percent": float(total),
            "per_core_percent": [float(x) for x in per_core],
            "core_count": psutil.cpu_count(logical=True) or len(per_core),
            "freq_current_mhz": float(freq.current) if freq else None,
            "freq_max_mhz": float(freq.max) if freq and freq.max else None,
        }
    except Exception:
        logger.exception("read_cpu failed")
        return {
            "total_percent": None,
            "per_core_percent": [],
            "core_count": 0,
            "freq_current_mhz": None,
            "freq_max_mhz": None,
        }


def read_loadavg() -> Optional[tuple[float, float, float]]:
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def read_memory() -> dict:
    try:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "cached_bytes": int(getattr(vm, "cached", 0) or 0),
            "percent": float(vm.percent),
            "swap_total_bytes": int(sm.total),
            "swap_used_bytes": int(sm.used),
            "swap_percent": float(sm.percent),
        }
    except Exception:
        logger.exception("read_memory failed")
        return {}


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------
def read_disk_usage(path: str = "/") -> dict:
    try:
        du = shutil.disk_usage(path)
        return {
            "path": path,
            "total_bytes": int(du.total),
            "used_bytes": int(du.used),
            "free_bytes": int(du.free),
            "percent": 100.0 * du.used / du.total if du.total else 0.0,
        }
    except Exception:
        logger.exception("read_disk_usage failed")
        return {}


def read_disk_io_raw() -> Optional[dict]:
    """Raw cumulative disk I/O counters. Caller diffs to get a rate."""
    try:
        io = psutil.disk_io_counters()
        if io is None:
            return None
        return {
            "read_bytes": int(io.read_bytes),
            "write_bytes": int(io.write_bytes),
            "read_count": int(io.read_count),
            "write_count": int(io.write_count),
        }
    except Exception:
        logger.exception("read_disk_io_raw failed")
        return None


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def read_net_io_raw() -> dict:
    """Per-NIC cumulative bytes/packets. Caller diffs to get a rate."""
    try:
        per_nic = psutil.net_io_counters(pernic=True)
        return {
            nic: {
                "bytes_sent": int(counters.bytes_sent),
                "bytes_recv": int(counters.bytes_recv),
                "packets_sent": int(counters.packets_sent),
                "packets_recv": int(counters.packets_recv),
                "errin": int(counters.errin),
                "errout": int(counters.errout),
                "dropin": int(counters.dropin),
                "dropout": int(counters.dropout),
            }
            for nic, counters in per_nic.items()
        }
    except Exception:
        logger.exception("read_net_io_raw failed")
        return {}


_WIRELESS_RE = re.compile(
    r"^\s*(?P<iface>\w+):\s+\d+\s+(?P<qual>[-\d.]+)\.?\s+(?P<signal>-?\d+)\.?"
)


def read_wifi() -> Optional[dict]:
    """Parse /proc/net/wireless. Linux-only; returns None elsewhere."""
    path = Path("/proc/net/wireless")
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        m = _WIRELESS_RE.match(line)
        if m:
            return {
                "iface": m.group("iface"),
                "link_quality": float(m.group("qual")),
                "signal_dbm": float(m.group("signal")),
            }
    return None


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
def read_cpu_temp_c() -> Optional[float]:
    """Pi CPU temperature in Celsius. Reads /sys/class/thermal/thermal_zone0/temp."""
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.exists():
        try:
            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
        except Exception:
            return None
        for entries in temps.values():
            for entry in entries:
                if entry.current:
                    return float(entry.current)
        return None
    try:
        raw = path.read_text().strip()
        return float(raw) / 1000.0
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Uptime
# ---------------------------------------------------------------------------
def read_uptime_seconds() -> Optional[float]:
    path = Path("/proc/uptime")
    if not path.exists():
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            return None
    try:
        first = path.read_text().split()[0]
        return float(first)
    except (OSError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# vcgencmd (Pi-only, subprocess -- cached)
# ---------------------------------------------------------------------------
#
# `vcgencmd get_throttled` returns a bitfield where:
#   bit 0  -> under-voltage detected now
#   bit 1  -> ARM frequency capped now
#   bit 2  -> currently throttled
#   bit 3  -> soft temperature limit active now
#   bit 16 -> under-voltage has occurred since boot
#   bit 17 -> ARM frequency capping has occurred since boot
#   bit 18 -> throttling has occurred since boot
#   bit 19 -> soft temperature limit has occurred since boot
#
# This is the single most useful Pi-specific signal for a mobile robot
# sharing power between motors and the Pi -- an under-voltage event here
# explains otherwise-inexplicable crashes.

_THROTTLED_BITS = [
    (0, "under_voltage_now"),
    (1, "freq_capped_now"),
    (2, "throttled_now"),
    (3, "soft_temp_limit_now"),
    (16, "under_voltage_past"),
    (17, "freq_capped_past"),
    (18, "throttled_past"),
    (19, "soft_temp_limit_past"),
]

_VCGENCMD_CACHE_SECONDS = 2.0
_vcgencmd_cache: dict[str, tuple[float, Optional[str]]] = {}


def _vcgencmd(*args: str) -> Optional[str]:
    """Run vcgencmd with the given args, caching the string result for 2 s."""
    key = " ".join(args)
    now = time.monotonic()
    cached = _vcgencmd_cache.get(key)
    if cached and now - cached[0] < _VCGENCMD_CACHE_SECONDS:
        return cached[1]
    result: Optional[str]
    try:
        proc = subprocess.run(
            ["vcgencmd", *args],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if proc.returncode == 0:
            result = proc.stdout.strip()
        else:
            result = None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        result = None
    except Exception:
        logger.exception("vcgencmd %s failed", key)
        result = None
    _vcgencmd_cache[key] = (now, result)
    return result


def read_throttled() -> Optional[dict]:
    """Decoded output of `vcgencmd get_throttled`."""
    raw = _vcgencmd("get_throttled")
    if not raw or "=" not in raw:
        return None
    try:
        hex_str = raw.split("=", 1)[1].strip()
        value = int(hex_str, 16)
    except ValueError:
        return None
    flags = {name: bool(value & (1 << bit)) for bit, name in _THROTTLED_BITS}
    any_now = any(v for k, v in flags.items() if k.endswith("_now"))
    any_past = any(v for k, v in flags.items() if k.endswith("_past"))
    if any_now:
        severity = "critical"
    elif any_past:
        severity = "warning"
    else:
        severity = "ok"
    return {
        "raw": raw,
        "value": value,
        "flags": flags,
        "severity": severity,
    }


def read_core_voltage() -> Optional[float]:
    raw = _vcgencmd("measure_volts", "core")
    if not raw or "=" not in raw:
        return None
    try:
        return float(raw.split("=", 1)[1].rstrip("V"))
    except ValueError:
        return None


def read_arm_clock_hz() -> Optional[int]:
    raw = _vcgencmd("measure_clock", "arm")
    if not raw or "=" not in raw:
        return None
    try:
        return int(raw.split("=", 1)[1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Processes (self + rpicam-vid)
# ---------------------------------------------------------------------------
_self_proc: Optional[psutil.Process] = None
_camera_proc: Optional[psutil.Process] = None


def _proc_self() -> Optional[psutil.Process]:
    global _self_proc
    if _self_proc is None:
        try:
            _self_proc = psutil.Process(os.getpid())
            _self_proc.cpu_percent(interval=None)  # prime
        except Exception:
            logger.exception("_proc_self init failed")
            _self_proc = None
    return _self_proc


def _proc_camera() -> Optional[psutil.Process]:
    global _camera_proc
    if _camera_proc is not None:
        try:
            if _camera_proc.is_running():
                return _camera_proc
        except psutil.Error:
            pass
        _camera_proc = None
    try:
        for p in psutil.process_iter(["name"]):
            name = p.info.get("name") or ""
            if "rpicam-vid" in name or name == "rpicam-vid":
                _camera_proc = p
                try:
                    _camera_proc.cpu_percent(interval=None)
                except psutil.Error:
                    pass
                return _camera_proc
    except Exception:
        logger.exception("_proc_camera scan failed")
    return None


def read_process_info(proc: Optional[psutil.Process]) -> Optional[dict]:
    if proc is None:
        return None
    try:
        with proc.oneshot():
            mem = proc.memory_info()
            cpu = proc.cpu_percent(interval=None)
            try:
                num_fds = proc.num_fds()  # type: ignore[attr-defined]
            except (AttributeError, psutil.Error):
                num_fds = None
            return {
                "pid": proc.pid,
                "name": proc.name(),
                "cpu_percent": float(cpu),
                "rss_bytes": int(mem.rss),
                "num_threads": int(proc.num_threads()),
                "num_fds": num_fds,
            }
    except psutil.NoSuchProcess:
        return None
    except Exception:
        logger.exception("read_process_info failed")
        return None


def read_self_process() -> Optional[dict]:
    return read_process_info(_proc_self())


def read_camera_process() -> Optional[dict]:
    return read_process_info(_proc_camera())
