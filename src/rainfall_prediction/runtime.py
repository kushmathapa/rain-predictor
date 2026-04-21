from __future__ import annotations

import ctypes
import os
import site
import sysconfig
from functools import lru_cache
from pathlib import Path


def _candidate_libomp_paths() -> list[Path]:
    roots: list[Path] = []

    for location in site.getsitepackages():
        roots.append(Path(location))

    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        roots.append(Path(purelib))

    candidates: list[Path] = []
    seen: set[Path] = set()

    relative_paths = [
        Path("cmeel.prefix/lib/libomp.dylib"),
        Path("sklearn/.dylibs/libomp.dylib"),
    ]
    absolute_paths = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    ]

    for root in roots:
        for relative in relative_paths:
            candidate = root / relative
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    for candidate in absolute_paths:
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    return candidates


def _prepend_env_path(variable_name: str, value: str) -> None:
    current = os.environ.get(variable_name)
    if not current:
        os.environ[variable_name] = value
        return

    parts = current.split(":")
    if value in parts:
        return
    os.environ[variable_name] = f"{value}:{current}"


@lru_cache(maxsize=1)
def bootstrap_openmp_runtime() -> str | None:
    for candidate in _candidate_libomp_paths():
        if not candidate.exists():
            continue

        lib_dir = str(candidate.parent)
        _prepend_env_path("DYLD_LIBRARY_PATH", lib_dir)
        _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", lib_dir)

        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return str(candidate)
        except OSError:
            continue

    return None
