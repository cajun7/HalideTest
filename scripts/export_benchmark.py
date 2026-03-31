#!/usr/bin/env python3
"""
Pull benchmark CSV from Android device and generate a polished Excel report.

Usage:
    python3 scripts/export_benchmark.py [output.xlsx]
    python3 scripts/export_benchmark.py --csv /path/to/local.csv [output.xlsx]

Requirements:
    pip3 install openpyxl
"""
import subprocess
import csv
import sys
import os
from collections import defaultdict, OrderedDict

try:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("ERROR: openpyxl is required. Install with: pip3 install openpyxl")
    sys.exit(1)


# -----------------------------------------------------------------------
# Styling constants
# -----------------------------------------------------------------------
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HALIDE_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
OPENCV_FILL = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
SPEEDUP_FILL = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
CENTER = Alignment(horizontal="center", vertical="center")
NUM_FMT = "#,##0"


def pull_csv_from_device():
    """Pull CSV from Android device via adb."""
    local_path = "/tmp/benchmark_results.csv"
    result = subprocess.run(
        [
            "adb", "pull",
            "/sdcard/Android/data/com.example.halidetest/files/benchmark_results.csv",
            local_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"adb pull failed: {result.stderr.strip()}")
        print("Make sure a device is connected and the app has run benchmarks.")
        sys.exit(1)
    print(f"Pulled CSV to {local_path}")
    return local_path


def read_csv(csv_path):
    """Read CSV into list of dicts. Auto-detect header or add one."""
    rows = []
    with open(csv_path, newline="") as f:
        sample = f.read(512)
        f.seek(0)
        # Check if first line looks like a header
        if sample.startswith("operation"):
            reader = csv.DictReader(f)
        else:
            reader = csv.DictReader(
                f,
                fieldnames=[
                    "operation", "framework", "resolution",
                    "median_us", "mean_us", "min_us", "max_us", "timestamp",
                ],
            )
        for row in reader:
            rows.append(row)
    return rows


def style_header_row(ws, ncols):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = CENTER
        cell.border = THIN_BORDER


def auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)


def generate_excel(rows, output_path):
    wb = Workbook()

    # ------------------------------------------------------------------
    # Sheet 1: Raw Data
    # ------------------------------------------------------------------
    ws_raw = wb.active
    ws_raw.title = "Raw Data"
    headers_raw = ["Operation", "Framework", "Resolution",
                   "Median (us)", "Mean (us)", "Min (us)", "Max (us)", "Timestamp"]
    ws_raw.append(headers_raw)

    for row in rows:
        ws_raw.append([
            row.get("operation", ""),
            row.get("framework", ""),
            row.get("resolution", ""),
            int(row.get("median_us", 0)),
            int(row.get("mean_us", 0)),
            int(row.get("min_us", 0)),
            int(row.get("max_us", 0)),
            row.get("timestamp", ""),
        ])

    style_header_row(ws_raw, len(headers_raw))

    # Apply number format and alternating row fill
    for r_idx in range(2, ws_raw.max_row + 1):
        fw = ws_raw.cell(row=r_idx, column=2).value
        fill = HALIDE_FILL if fw == "Halide" else OPENCV_FILL
        for c_idx in range(1, len(headers_raw) + 1):
            cell = ws_raw.cell(row=r_idx, column=c_idx)
            cell.border = THIN_BORDER
            cell.fill = fill
            if 4 <= c_idx <= 7:
                cell.number_format = NUM_FMT
                cell.alignment = CENTER

    auto_width(ws_raw)

    # ------------------------------------------------------------------
    # Sheet 2: Summary (pivot table)
    # ------------------------------------------------------------------
    ws_summary = wb.create_sheet("Summary")
    headers_sum = ["Operation", "Resolution",
                   "Halide Median (us)", "OpenCV Median (us)",
                   "Halide Mean (us)", "OpenCV Mean (us)",
                   "Speedup (median)"]
    ws_summary.append(headers_sum)

    # Group by (operation, resolution), latest entry wins
    data = OrderedDict()
    for row in rows:
        key = (row.get("operation", ""), row.get("resolution", ""))
        fw = row.get("framework", "")
        if key not in data:
            data[key] = {}
        data[key][fw] = row

    summary_rows = []
    for (op, res), frameworks in data.items():
        h = frameworks.get("Halide", {})
        o = frameworks.get("OpenCV", {})
        h_med = int(h.get("median_us", 0))
        o_med = int(o.get("median_us", 0))
        h_mean = int(h.get("mean_us", 0))
        o_mean = int(o.get("mean_us", 0))
        speedup = o_med / h_med if h_med > 0 else 0
        summary_rows.append([op, res, h_med, o_med, h_mean, o_mean, speedup])
        ws_summary.append([op, res, h_med, o_med, h_mean, o_mean, f"{speedup:.2f}x"])

    style_header_row(ws_summary, len(headers_sum))

    for r_idx in range(2, ws_summary.max_row + 1):
        for c_idx in range(1, len(headers_sum) + 1):
            cell = ws_summary.cell(row=r_idx, column=c_idx)
            cell.border = THIN_BORDER
            if 3 <= c_idx <= 6:
                cell.number_format = NUM_FMT
                cell.alignment = CENTER
            if c_idx == 3:
                cell.fill = HALIDE_FILL
            elif c_idx == 4:
                cell.fill = OPENCV_FILL
            elif c_idx == 5:
                cell.fill = HALIDE_FILL
            elif c_idx == 6:
                cell.fill = OPENCV_FILL
            elif c_idx == 7:
                cell.fill = SPEEDUP_FILL
                cell.alignment = CENTER

    auto_width(ws_summary)

    # ------------------------------------------------------------------
    # Sheet 3: Charts
    # ------------------------------------------------------------------
    ws_chart = wb.create_sheet("Charts")

    if len(summary_rows) > 0:
        # Chart 1: Median comparison (grouped bar)
        chart1 = BarChart()
        chart1.type = "col"
        chart1.grouping = "clustered"
        chart1.title = "Halide vs OpenCV - Median Execution Time"
        chart1.y_axis.title = "Time (microseconds)"
        chart1.x_axis.title = "Operation"
        chart1.style = 10
        chart1.width = 25
        chart1.height = 15

        nrows = len(summary_rows) + 1  # +1 for header

        # Halide median = column 3, OpenCV median = column 4
        halide_data = Reference(ws_summary, min_col=3, min_row=1, max_row=nrows)
        opencv_data = Reference(ws_summary, min_col=4, min_row=1, max_row=nrows)
        cats = Reference(ws_summary, min_col=1, min_row=2, max_row=nrows)

        chart1.add_data(halide_data, titles_from_data=True)
        chart1.add_data(opencv_data, titles_from_data=True)
        chart1.set_categories(cats)

        # Color the series
        chart1.series[0].graphicalProperties.solidFill = "70AD47"  # Green for Halide
        chart1.series[1].graphicalProperties.solidFill = "ED7D31"  # Orange for OpenCV

        ws_chart.add_chart(chart1, "A1")

        # Chart 2: Speedup factor
        chart2 = BarChart()
        chart2.type = "col"
        chart2.title = "Speedup Factor (OpenCV / Halide)"
        chart2.y_axis.title = "Speedup (x)"
        chart2.style = 10
        chart2.width = 25
        chart2.height = 15

        # Write speedup data to a helper area in the chart sheet
        ws_chart.cell(row=1, column=10, value="Operation")
        ws_chart.cell(row=1, column=11, value="Speedup")
        for i, sr in enumerate(summary_rows):
            ws_chart.cell(row=i + 2, column=10, value=sr[0])
            ws_chart.cell(row=i + 2, column=11, value=sr[6])

        speedup_data = Reference(ws_chart, min_col=11, min_row=1, max_row=len(summary_rows) + 1)
        speedup_cats = Reference(ws_chart, min_col=10, min_row=2, max_row=len(summary_rows) + 1)
        chart2.add_data(speedup_data, titles_from_data=True)
        chart2.set_categories(speedup_cats)
        chart2.series[0].graphicalProperties.solidFill = "4472C4"

        ws_chart.add_chart(chart2, "A18")

    wb.save(output_path)
    print(f"Report saved to: {output_path}")


def main():
    args = sys.argv[1:]
    csv_path = None
    output_path = "benchmark_report.xlsx"

    i = 0
    while i < len(args):
        if args[i] == "--csv" and i + 1 < len(args):
            csv_path = args[i + 1]
            i += 2
        else:
            output_path = args[i]
            i += 1

    if csv_path is None:
        csv_path = pull_csv_from_device()

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("ERROR: CSV file is empty or has no data rows.")
        sys.exit(1)

    print(f"Read {len(rows)} benchmark entries.")
    generate_excel(rows, output_path)


if __name__ == "__main__":
    main()
