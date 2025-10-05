"""Generate training statistics plots and save them as JPEG images.

This script reads the statistics.csv file produced during training and
creates line plots for each numeric column. The resulting images are
saved in a sibling ``plot`` directory so that they can be reviewed later
without requiring an interactive display environment.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create plots from a training statistics CSV file and save them as JPEG images."
        )
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("statistics.csv"),
        help="Path to the statistics CSV file produced during training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the generated plots will be stored. "
            "Defaults to a 'plot' directory next to the CSV file."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[Sequence[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"The statistics file '{path}' does not exist.")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("The statistics CSV file does not contain a header row.")
        rows = [row for row in reader]
    if not rows:
        raise ValueError("The statistics CSV file is empty.")
    return reader.fieldnames, rows


def _convert_column(values: Iterable[str]) -> list[float] | None:
    converted: list[float] = []
    for value in values:
        text = value.strip() if value is not None else ""
        if not text:
            converted.append(float("nan"))
            continue
        try:
            converted.append(float(text))
        except ValueError:
            return None
    return converted


def _select_x_axis(headers: Sequence[str], columns: dict[str, list[float]]) -> tuple[list[float], str]:
    preferred_names = {"epoch", "step", "iteration", "batch"}
    for name in headers:
        if name.lower() in preferred_names and name in columns:
            column = columns[name]
            if any(not math.isnan(value) for value in column):
                return column, name

    first_header = headers[0]
    column = columns.get(first_header, [])
    if column and any(not math.isnan(value) for value in column):
        return column, first_header

    length = len(next(iter(columns.values()))) if columns else 0
    return [float(index) for index in range(1, length + 1)], "index"


def _sanitize_filename(name: str) -> str:
    sanitized = [character if character.isalnum() else "_" for character in name]
    result = "".join(sanitized).strip("_")
    return result or "metric"


def _plot_metric(
    x_values: Sequence[float],
    x_label: str,
    y_values: Sequence[float],
    metric_name: str,
    output_directory: Path,
) -> Path | None:
    filtered_points = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if not math.isnan(x) and not math.isnan(y)
    ]
    if not filtered_points:
        return None

    xs, ys = zip(*filtered_points)
    plt.figure()
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel(x_label.capitalize())
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.grid(True, linestyle="--", alpha=0.4)

    file_name = f"{_sanitize_filename(metric_name)}.jpg"
    output_path = output_directory / file_name
    plt.tight_layout()
    plt.savefig(output_path, format="jpg", dpi=200)
    plt.close()
    return output_path


def generate_plots(stats_path: Path, output_directory: Path) -> list[Path]:
    headers, rows = _read_csv(stats_path)

    raw_columns: dict[str, list[str]] = {header: [] for header in headers}
    for row in rows:
        for header in headers:
            raw_columns[header].append(row.get(header, ""))

    numeric_columns: dict[str, list[float]] = {}
    for header, values in raw_columns.items():
        converted = _convert_column(values)
        if converted is not None:
            numeric_columns[header] = converted

    if not numeric_columns:
        raise ValueError("No numeric columns were found in the statistics CSV file.")

    x_values, x_label = _select_x_axis(headers, numeric_columns)

    generated_files: list[Path] = []
    for metric_name, values in numeric_columns.items():
        if metric_name == x_label:
            continue
        output_path = _plot_metric(
            x_values,
            x_label,
            values,
            metric_name,
            output_directory,
        )
        if output_path is not None:
            generated_files.append(output_path)

    return generated_files


def main() -> None:
    args = _parse_arguments()
    stats_path: Path = args.stats
    output_directory: Path = args.output_dir or stats_path.parent / "plot"
    output_directory.mkdir(parents=True, exist_ok=True)

    generated_files = generate_plots(stats_path, output_directory)
    if not generated_files:
        print("No plots were generated. Ensure the CSV contains numeric metrics.")
    else:
        print("Generated the following plots:")
        for path in generated_files:
            print(f" - {path}")


if __name__ == "__main__":
    main()