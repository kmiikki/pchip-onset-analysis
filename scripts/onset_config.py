#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onset_config.py
---------------

Human-editable per-experiment onset-count configuration for the
PCHIP/IA/FBRM onset workflow.

The truth-source file is:

    analysis/onset-config.ini

This module is intentionally small and dependency-free so it can be imported by
the workflow runner and by the standalone MGI/FBRM analysis scripts.
"""

from __future__ import annotations

import configparser
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


CONFIG_FILENAME = "onset-config.ini"
DEFAULT_MGI_ONSETS = 2
DEFAULT_FBRM_ONSETS = 2
DEFAULT_COMMENT = (
    "Initial default onset counts. Edit after review if this experiment needs "
    "different accepted onset counts."
)


@dataclass(frozen=True)
class OnsetConfig:
    """Resolved onset-count configuration for one analysis directory."""

    analysis_dir: Path
    path: Path
    mgi_onsets: int
    fbrm_onsets: int
    status: str
    message: str
    backup_path: Path | None = None
    invalid_reason: str | None = None
    notes_comment: str = ""


class OnsetConfigInvalid(ValueError):
    """Raised when onset-config.ini exists but cannot be trusted."""


def onset_config_path(analysis_dir: Path) -> Path:
    """Return the canonical onset-config.ini path for an analysis directory."""
    return Path(analysis_dir) / CONFIG_FILENAME


def default_config_text(comment: str = DEFAULT_COMMENT) -> str:
    """Return the default human-editable INI file content."""
    return (
        "# analysis/onset-config.ini\n"
        "#\n"
        "# Human-editable per-experiment onset configuration.\n"
        "# This file stores reviewed analysis decisions.\n"
        "#\n"
        "# Defaults used when this file is missing:\n"
        "#   MGI onsets  = 2\n"
        "#   FBRM onsets = 2\n"
        "#\n"
        "# Edit only the values after \"=\" unless you know what you are doing.\n"
        "\n"
        "[MGI]\n"
        "onsets = 2\n"
        "\n"
        "[FBRM]\n"
        "onsets = 2\n"
        "\n"
        "[Notes]\n"
        f"comment = {comment}\n"
    )


def write_default_config(path: Path, *, comment: str = DEFAULT_COMMENT) -> None:
    """Write a new default onset-config.ini without touching existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_config_text(comment), encoding="utf-8")


def _parse_nonnegative_int(parser: configparser.ConfigParser, section: str) -> int:
    """Read and validate one nonnegative integer onset count."""
    if not parser.has_section(section):
        raise OnsetConfigInvalid(f"missing section [{section}]")
    if not parser.has_option(section, "onsets"):
        raise OnsetConfigInvalid(f"missing [{section}] onsets value")

    raw = parser.get(section, "onsets", fallback=None)
    if raw is None or raw.strip() == "":
        raise OnsetConfigInvalid(f"[{section}] onsets value is empty")

    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise OnsetConfigInvalid(
            f"[{section}] onsets must be an integer >= 0, got: {raw}"
        ) from exc

    if value < 0:
        raise OnsetConfigInvalid(
            f"[{section}] onsets must be an integer >= 0, got: {value}"
        )

    return value


def read_onset_config(analysis_dir: Path) -> OnsetConfig:
    """Read and validate an existing onset-config.ini.

    Valid onset counts are integers >= 0. The value 0 is a real analysis
    decision and must not be treated as a missing/default value.
    """
    analysis_dir = Path(analysis_dir)
    path = onset_config_path(analysis_dir)

    parser = configparser.ConfigParser()
    try:
        with path.open("r", encoding="utf-8") as f:
            parser.read_file(f)
    except configparser.Error as exc:
        raise OnsetConfigInvalid(f"invalid INI syntax: {exc}") from exc
    except OSError as exc:
        raise OnsetConfigInvalid(f"cannot read file: {exc}") from exc

    mgi_onsets = _parse_nonnegative_int(parser, "MGI")
    fbrm_onsets = _parse_nonnegative_int(parser, "FBRM")
    notes_comment = ""
    if parser.has_section("Notes") and parser.has_option("Notes", "comment"):
        notes_comment = parser.get("Notes", "comment", fallback="").strip()

    return OnsetConfig(
        analysis_dir=analysis_dir,
        path=path,
        mgi_onsets=mgi_onsets,
        fbrm_onsets=fbrm_onsets,
        status="existing",
        message="valid existing onset-config.ini",
        notes_comment=notes_comment,
    )


def invalid_backup_path(analysis_dir: Path, *, now: datetime | None = None) -> Path:
    """Return the invalid-file backup path using the agreed filename pattern."""
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(analysis_dir) / f"onset-config.invalid_file-{ts}.ini"


def _warning_banner(text: str) -> str:
    line = "!" * 72
    return f"{line}\n{text.rstrip()}\n{line}"


def format_onset_config_warning(
    *,
    experiment_name: str,
    invalid_path: Path,
    reason: str,
    backup_path: Path,
    dry_run: bool,
) -> str:
    """Return the highly visible terminal warning for invalid config repair."""
    if dry_run:
        action = (
            "Dry-run action:\n"
            f"  Would move invalid file to:\n"
            f"  {backup_path}\n\n"
            "  Would create new default onset-config.ini:\n"
            "    MGI  onsets = 2\n"
            "    FBRM onsets = 2\n\n"
            "  In --run mode this experiment would then be analyzed using default onset counts."
        )
    else:
        action = (
            "Action:\n"
            f"  Moved invalid file to:\n"
            f"  {backup_path}\n\n"
            "  Created new default onset-config.ini:\n"
            "    MGI  onsets = 2\n"
            "    FBRM onsets = 2\n\n"
            "  This experiment will now be analyzed using default onset counts."
        )

    body = (
        f"[WARNING:onset-config] {experiment_name}\n\n"
        "Invalid onset-config.ini was found.\n\n"
        f"File:\n  {invalid_path}\n\n"
        f"Reason:\n  {reason}\n\n"
        f"{action}"
    )
    return _warning_banner(body)


def ensure_onset_config(
    analysis_dir: Path,
    *,
    dry_run: bool = False,
    experiment_name: str | None = None,
    stream: TextIO | None = None,
) -> OnsetConfig:
    """Ensure a valid onset-config.ini exists and return resolved values.

    Missing file:
        initial/default state -> create default config in --run mode.

    Valid file:
        truth source -> use values exactly, including onsets = 0.

    Invalid file:
        in --run mode, move to
        onset-config.invalid_file-YYYYMMDD_HHMMSS.ini, create a fresh default
        config, and continue using defaults. In dry-run mode, report what would
        happen without modifying the filesystem.
    """
    analysis_dir = Path(analysis_dir)
    path = onset_config_path(analysis_dir)
    exp_name = experiment_name or (
        analysis_dir.parent.parent.parent.parent.name
        if len(analysis_dir.parts) >= 4
        else str(analysis_dir)
    )
    out = stream if stream is not None else sys.stdout

    if not path.exists():
        if dry_run:
            result = OnsetConfig(
                analysis_dir=analysis_dir,
                path=path,
                mgi_onsets=DEFAULT_MGI_ONSETS,
                fbrm_onsets=DEFAULT_FBRM_ONSETS,
                status="missing_would_create",
                message="missing onset-config.ini; would create defaults in --run mode",
                notes_comment=DEFAULT_COMMENT,
            )
            print(
                f"[onset-config] {exp_name}: missing; would create default "
                f"MGI={result.mgi_onsets}, FBRM={result.fbrm_onsets}: {path}",
                file=out,
            )
            return result

        write_default_config(path)
        result = OnsetConfig(
            analysis_dir=analysis_dir,
            path=path,
            mgi_onsets=DEFAULT_MGI_ONSETS,
            fbrm_onsets=DEFAULT_FBRM_ONSETS,
            status="created",
            message="created default onset-config.ini",
            notes_comment=DEFAULT_COMMENT,
        )
        print(
            f"[onset-config] {exp_name}: created default "
            f"MGI={result.mgi_onsets}, FBRM={result.fbrm_onsets}: {path}",
            file=out,
        )
        return result

    try:
        result = read_onset_config(analysis_dir)
        print(
            f"[onset-config] {exp_name}: existing "
            f"MGI={result.mgi_onsets}, FBRM={result.fbrm_onsets}: {path}",
            file=out,
        )
        return result
    except OnsetConfigInvalid as exc:
        reason = str(exc)
        backup = invalid_backup_path(analysis_dir)
        warning = format_onset_config_warning(
            experiment_name=exp_name,
            invalid_path=path,
            reason=reason,
            backup_path=backup,
            dry_run=dry_run,
        )
        print(warning, file=out)

        if dry_run:
            return OnsetConfig(
                analysis_dir=analysis_dir,
                path=path,
                mgi_onsets=DEFAULT_MGI_ONSETS,
                fbrm_onsets=DEFAULT_FBRM_ONSETS,
                status="invalid_would_repair",
                message=(
                    "invalid onset-config.ini; would move to "
                    f"{backup.name} and use defaults in --run mode"
                ),
                backup_path=backup,
                invalid_reason=reason,
                notes_comment=DEFAULT_COMMENT,
            )

        shutil.move(str(path), str(backup))
        write_default_config(
            path,
            comment="Default onset counts. Previous invalid config was backed up and replaced.",
        )
        return OnsetConfig(
            analysis_dir=analysis_dir,
            path=path,
            mgi_onsets=DEFAULT_MGI_ONSETS,
            fbrm_onsets=DEFAULT_FBRM_ONSETS,
            status="repaired_invalid",
            message=(
                "invalid onset-config.ini moved to "
                f"{backup.name}; created fresh defaults"
            ),
            backup_path=backup,
            invalid_reason=reason,
            notes_comment="Default onset counts. Previous invalid config was backed up and replaced.",
        )


def infer_analysis_dir_from_paths(
    *,
    analysis_dir: Path | None = None,
    input_csv: Path | None = None,
    output_dir: Path | None = None,
    fallback: Path | None = None,
) -> Path:
    """Infer the analysis directory for direct script runs."""
    if analysis_dir is not None:
        return Path(analysis_dir).resolve()

    if output_dir is not None:
        out = Path(output_dir)
        if out.name in {"pchip", "fbrm_onsets", "combo"}:
            return out.parent.resolve()

    if input_csv is not None:
        p = Path(input_csv)
        if p.parent.name in {"analysis", "pchip", "fbrm_onsets", "combo"}:
            return (p.parent if p.parent.name == "analysis" else p.parent.parent).resolve()

    return Path(fallback or Path.cwd()).resolve()
