"""Bounded workspace layout for native gradient statistics; no tensor computation."""
from __future__ import annotations
from dataclasses import dataclass
FAN_IN = 1024
MAX_ROWS = 4096 * 1024

@dataclass(frozen=True)
class StatsStage:
    input_rows: int
    output_rows: int
    output_offset: int  # FP32 elements, not bytes

@dataclass(frozen=True)
class StatsPlan:
    stages: tuple[StatsStage, ...]
    scratch_elements: int
    final_rows: int


def statistics_plan(rows: int) -> StatsPlan:
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("gradient statistics rows must be in 1..4194304")
    stages = []
    offset = 0
    while rows > FAN_IN:
        output_rows = (rows + FAN_IN - 1) // FAN_IN
        stages.append(StatsStage(rows, output_rows, offset))
        offset += output_rows * 3
        rows = output_rows
    return StatsPlan(tuple(stages), offset, rows)
