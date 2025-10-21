import os
import re
import ast
import inspect
import datetime
import numpy as np
import pandas as pd
from io import StringIO
from ..h5analysis.beamlines.REIXS import *

# =========================================================
# Global Timestamp
# =========================================================
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


# =========================================================
# Loader Command Executor
# =========================================================
def run_command_and_export(command, stand_ver, start_index, executed_commands_dict):
    """
    Executes a loader command string, exports results into the given version folder,
    and records executed commands for later comparison.

    Parameters
    ----------
    command : str
        Command string to execute (e.g., "XASLoader.load(RIXS, path, 'TEY', 8, 9)").
    stand_ver : str
        Output directory path for exported files.
    start_index : int
        Numerical suffix index for file naming.
    executed_commands_dict : dict
        Dictionary mapping file index → executed command string(s).

    Returns
    -------
    next_index : int
        Incremented file index for next operation.
    executed_commands : list
        List of successfully executed sub-commands.
    file_name : str
        Path to exported file.
    """
    start_index = int(start_index)
    loader_var_name = command.split(".")[0].strip()
    caller_globals = inspect.currentframe().f_back.f_globals

    # Ensure loader instance is properly initialized
    loader_class_or_instance = caller_globals.get(loader_var_name)
    if isinstance(loader_class_or_instance, type):
        loader_instance = loader_class_or_instance()
        caller_globals[loader_var_name] = loader_instance
    else:
        loader_instance = type(loader_class_or_instance)()
        caller_globals[loader_var_name] = loader_instance

    executed_commands = []
    file_name = None

    try:
        # Handle multiple subcommands separated by semicolons
        subcommands = [cmd.strip() for cmd in command.split(';') if cmd.strip()]
        for subcmd in subcommands:
            exec(subcmd, caller_globals)
            executed_commands.append(subcmd)

        # Export result to versioned folder
        file_name = os.path.join(stand_ver, loader_var_name, f"file{start_index}")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        loader_instance.export(file_name)

        executed_commands_dict[start_index] = " ; ".join(executed_commands)

    except Exception as e:
        print(f"[Error] Failed to execute '{command}': {e}")

    return start_index + 1, executed_commands, file_name


# =========================================================
# Spectrum Comparator
# =========================================================
class SpectrumComparator:
    """
    Compares two ASCII spectrum files (csv/asc/txt) and records differences.

    Handles:
    - Header normalization
    - Dimension classification (1D, 2D, 3D)
    - Column-wise relative error analysis
    """

    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.comparison_results = []

    # ------------------------------
    # File Helpers
    # ------------------------------
    def _normalize_header_line(self, header_line):
        return ",".join([p.strip() for p in header_line.split(",")])

    def _load_ascii(self, file, skip_header=False):
        if not (file.endswith('.csv') or file.endswith('.asc') or file.endswith('.txt')):
            file += '.asc'

        with open(file, 'r') as f:
            lines = [ln for ln in f.readlines() if ln.strip() != ""]

        if len(lines) < 3:
            raise ValueError(f"File {file} must have at least three lines.")

        file_info = lines[0].strip()
        header_line_raw = lines[1].strip()
        header_line_norm = self._normalize_header_line(header_line_raw)
        header_cols = [p.strip() for p in header_line_norm.split(",")]

        data_str = "".join(lines[2:])
        df = pd.read_csv(StringIO(data_str), names=header_cols)

        if df.shape[1] < 1:
            raise ValueError(f"File {file} contains no valid data columns.")

        return df, header_cols, file_info, header_line_raw, header_line_norm

    def _headcount(self, file):
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            header_count = 0
            header_lines = []
            for i in range(len(lines) - 1):
                if lines[i].startswith("#") and re.match(r"^[A-Za-z]", lines[i + 1]):
                    header_count += 1
                    header_lines.append(lines[i + 1])
        return header_count, header_lines

    # ------------------------------
    # Core Comparison
    # ------------------------------
    def compare(self, filename, desc, file1, file2, subfolder="",
                skip_header1=False, skip_header2=False, executed_commands=None):
        """
        Compare two files and store results in comparison_results.
        """
        df1, hdr_cols1, info1, hdr1_raw, hdr1_norm = self._load_ascii(file1, skip_header1)
        df2, hdr_cols2, info2, hdr2_raw, hdr2_norm = self._load_ascii(file2, skip_header2)

        # Determine dimension
        if len(hdr_cols1) == 1 and df1.shape[0] > 1 and self._headcount(file1)[0] >= 2:
            dim = "3D"
        elif len(hdr_cols1) == 1 and df1.shape[0] > 1 and self._headcount(file1)[0] == 1:
            dim = "2D"
        elif len(hdr_cols1) == df1.shape[1]:
            dim = "1D"
        else:
            dim = "Unknown"

        # Align lengths
        len1, len2 = len(df1), len(df2)
        min_len = min(len1, len2)
        df1, df2 = df1.iloc[:min_len].reset_index(drop=True), df2.iloc[:min_len].reset_index(drop=True)

        headers_match = (hdr1_norm == hdr2_norm) if dim != "3D" else (self._headcount(file1)[1] == self._headcount(file2)[1])

        col_issues = []
        max_relative_error = 0.0

        for col_idx in range(min(df1.shape[1], df2.shape[1])):
            colname = hdr_cols1[col_idx] if col_idx < len(hdr_cols1) else f"Col{col_idx+1}"
            col1_raw, col2_raw = df1.iloc[:, col_idx], df2.iloc[:, col_idx]

            def parse_cell(cell):
                if isinstance(cell, str):
                    parts = cell.strip().split()
                    try:
                        return np.array([float(x) for x in parts])
                    except:
                        return np.array([np.nan])
                else:
                    return np.array([float(cell)])

            line_nums_with_diff = []

            for row_idx, (v1, v2) in enumerate(zip(col1_raw, col2_raw)):
                arr1, arr2 = parse_cell(v1), parse_cell(v2)
                min_arr_len = min(len(arr1), len(arr2))
                arr1, arr2 = arr1[:min_arr_len], arr2[:min_arr_len]

                denominator = arr1**2 + arr2**2
                with np.errstate(divide='ignore', invalid='ignore'):
                    relative_errors = ((arr1 - arr2)**2) / denominator
                    relative_errors = np.nan_to_num(relative_errors, nan=0.0, posinf=0.0)

                row_max_error = np.max(relative_errors)
                max_relative_error = max(max_relative_error, row_max_error)

                if row_max_error > self.tolerance:
                    line_nums_with_diff.append(row_idx + 3)

            if line_nums_with_diff:
                col_issues.append(f"{colname}: {len(line_nums_with_diff)} diffs, lines={line_nums_with_diff[:10]}")

        info_match = (info1.strip() == info2.strip()) and (max_relative_error <= self.tolerance)

        result = {
            "Class": subfolder,
            "Commands": executed_commands if executed_commands else "",
            "Filename": filename,
            "DIM": dim,
            "Len": f"{len1} / {len2}",
            "Cols": f"{df1.shape[1]} / {df2.shape[1]}",
            "Header": "Match" if headers_match else "MisMatch",
            "Data": "Match" if info_match else "MisMatch",
            #"Error": "; ".join(col_issues) if col_issues else "None",
            "Rel_Data_Error": f"{max_relative_error:.3e}"
        }

        self.comparison_results.append(result)
        return result

    def to_dataframe(self):
        """
        Convert comparison results to DataFrame with correct column order.
        """
        cols = [
            "Class", "Commands", "Filename", "DIM", "Len",
            "Cols", "Header", "Data", "Rel_Data_Error"
        ]
        df = pd.DataFrame(self.comparison_results)
        return df[cols]


# =========================================================
# Run Comparison and Generate Report
# =========================================================

def run_comparison_and_generate_report(test_ver, stand_ver, tolerance=1e-6):
    comparator = SpectrumComparator(tolerance=tolerance)
    results = []

    for loader_class in os.listdir(test_ver):
        v1_subdir = os.path.join(test_ver, loader_class)
        v2_subdir = os.path.join(stand_ver, loader_class)
        if not os.path.isdir(v1_subdir) or not os.path.isdir(v2_subdir):
            continue

        for file in sorted(os.listdir(v1_subdir)):
            train_path = os.path.join(v1_subdir, file)
            test_path = os.path.join(v2_subdir, file)
            if not os.path.exists(test_path):
                continue
            results.append(
                comparator.compare(file, "Pending", train_path, test_path, subfolder=loader_class)
            )

    df = pd.DataFrame(results)
    report_path = os.path.join(stand_ver, f"comparison_report_{timestamp}.csv")
    os.makedirs(stand_ver, exist_ok=True)
    df.to_csv(report_path, index=False)
    return df, report_path

# =========================================================
# Spectrum Runner
# =========================================================
def run_all_commands(commands, test_ver, stand_ver, start_index=1, show_all=False, tolerance=1e-6):
    executed_commands_dict = {}
    index = start_index

    # Run and export all commands
    for cmd in commands:
        index, executed_cmds, file_name = run_command_and_export(
            command=cmd,
            stand_ver=stand_ver,
            start_index=index,
            executed_commands_dict=executed_commands_dict
        )

    # Compare results
    df_report, report_path = run_comparison_and_generate_report(
        test_ver=test_ver,
        stand_ver=stand_ver,
        tolerance=tolerance
    )

    # ✅ Add executed command from dict
    df_report["Commands"] = df_report["Filename"].apply(
        lambda f: executed_commands_dict.get(int(re.findall(r'\d+', f)[0]), "")
    )

    if not show_all:
        df_report = df_report[(df_report["Header"] != "Match") | (df_report["Data"] != "Match")]

    return df_report, report_path