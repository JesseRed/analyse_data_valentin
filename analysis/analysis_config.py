"""
Central configuration for the SRTT analysis pipeline.

Edit:
- `EXCLUDED_PIDS` to exclude participants from *all* analyses/outputs.
- `PRIMARY_COVARS` to change the minimal, confirmatory covariate set.
"""

from __future__ import annotations

# Participants to be excluded from *all* analyses and outputs.
# Keep this list at the top-level so it can be manually edited.
EXCLUDED_PIDS: list[int] = [2, 7, 13, 26, 27, 32, 38]

# Primary, publication-focused minimal adjustment set.
# Change here to switch the confirmatory covariate set across the pipeline.
PRIMARY_COVARS: list[str] = ["Age", "fuglmayrshort_sum", "MoCa_sum"]


def is_excluded_pid(pid: int) -> bool:
    return int(pid) in set(EXCLUDED_PIDS)


def filter_excluded(df, pid_col: str = "PID"):
    """Return a copy with excluded PIDs removed (no-op if pid_col missing)."""
    if pid_col not in df.columns:
        return df.copy()
    return df.loc[~df[pid_col].isin(EXCLUDED_PIDS)].copy()

