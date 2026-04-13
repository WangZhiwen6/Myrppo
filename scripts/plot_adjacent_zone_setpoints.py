"""Plot one-day setpoint curves for a reproducible adjacent-zone sample."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zhiwen")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_zone_graph import THERMAL_ZONES, adjacency_list


DEFAULT_MONITOR = (
    REPO_ROOT
    / "myrppo"
    / "Eplus-env-RPPO-Eplus-1-mixed-continuous-stochastic-v1-episodes-200_2026-04-13-15_03-res1"
    / "Eplus-env-sub_run1"
    / "monitor.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, default=DEFAULT_MONITOR)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "adjacent_zone_cooling_setpoints_day15.svg")
    parser.add_argument("--month", type=float, default=7.0)
    parser.add_argument("--day", type=float, default=15.0)
    parser.add_argument("--action-type", choices=("Cooling", "Heating"), default="Cooling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-neighbors", type=int, default=3)
    return parser.parse_args()


def compact_zone_name(zone_name: str) -> str:
    return zone_name.replace("THERMAL ZONE: ", "")


def select_adjacent_zones(seed: int, num_neighbors: int) -> list[int]:
    rng = random.Random(seed)
    neighbors_by_zone = adjacency_list()
    candidate_centers = [
        zone_idx
        for zone_idx, neighbors in neighbors_by_zone.items()
        if len(neighbors) >= num_neighbors
    ]
    center = rng.choice(candidate_centers)
    neighbors = rng.sample(neighbors_by_zone[center], num_neighbors)
    return [center, *neighbors]


def load_rows(monitor_path: Path, month: float, day: float) -> list[dict[str, str]]:
    with monitor_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            row
            for row in reader
            if float(row["month"]) == month and float(row["day_of_month"]) == day
        ]


def row_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def plot_setpoints(args: argparse.Namespace) -> None:
    rows = load_rows(args.monitor, args.month, args.day)
    if not rows:
        raise ValueError(f"No rows found for month={args.month:g}, day={args.day:g}.")

    selected_zones = select_adjacent_zones(args.seed, args.num_neighbors)
    time_hours = [float(row["hour"]) + (float(row["time (hours)"]) % 1.0) for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.8, 6.4), constrained_layout=True)

    for zone_idx in selected_zones:
        zone = THERMAL_ZONES[zone_idx]
        column = f"{zone}-{args.action_type}"
        if column not in rows[0]:
            raise KeyError(f"Column not found: {column}")
        values = [row_float(row, column) for row in rows]
        ax.plot(
            time_hours,
            values,
            linewidth=2.2,
            label=f"{zone_idx:02d}. {compact_zone_name(zone)}",
        )

    ax.set_title(
        f"One-Day {args.action_type} Setpoint Curves for Adjacent Thermal Zones",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Time of day (h)", fontsize=12)
    ax.set_ylabel(f"{args.action_type} setpoint (degC)", fontsize=12)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=9)
    ax.text(
        0.01,
        0.02,
        f"Month={args.month:g}, day={args.day:g}; random seed={args.seed}",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)

    png_output = args.output.with_suffix(".png")
    fig.savefig(png_output, dpi=300)
    print(args.output)
    print(png_output)
    print("selected_zones=" + ", ".join(f"{idx}:{compact_zone_name(THERMAL_ZONES[idx])}" for idx in selected_zones))


def main() -> None:
    plot_setpoints(parse_args())


if __name__ == "__main__":
    main()
