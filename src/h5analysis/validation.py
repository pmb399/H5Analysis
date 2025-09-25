import os
import re
import ast
import inspect
import datetime
import numpy as np
import pandas as pd
from io import StringIO

# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def run_command_and_export(command, version1=None, start_index=1):
    """
    Runs a given command string, detects the loader class (Load1d, Load2d, XESLoader, etc.),
    exports results into version-specific subfolders, then resets the loader after export.
    Parameters
    ----------
    command : str
        Command string to execute (can contain multiple commands separated by ';').
    version1 : str or None
        Base folder name for first export set (e.g., "Standard"). If None, skip.
    start_index : int
        Starting version number (default=1).
    Command log is printed in .csv and safe in the version2 folder.
    """
    if not version1:
        raise ValueError("Version1 folder must be provided.")
    # Detect loader class name (first word before a dot)
    match = re.match(r"^\s*([A-Za-z_]\w*)\.", command)
    if not match:
        raise ValueError(f"Could not detect loader class from command: {command}")
    loader_var_name = match.group(1)
    # Create subfolders for this loader
    if version1:
        folder1 = os.path.join(version1, loader_var_name)
        os.makedirs(folder1, exist_ok=True)
    # Create a fresh execution context from current globals
    caller_globals = inspect.currentframe().f_back.f_globals.copy()
    # Ensure the loader class is available in globals
    if loader_var_name not in caller_globals:
        raise ValueError(f"{loader_var_name} is not defined in the global scope.")
    loader_class = caller_globals[loader_var_name]
    # Add the loader instance
    caller_globals[loader_var_name] = loader_class()
    # Execute commands
    subcommands = [cmd.strip() for cmd in command.split(';') if cmd.strip()]
    for subcmd in subcommands:
        print(f"[Executing] {subcmd}")
        exec(subcmd, caller_globals)
    # Export
    loader_instance = caller_globals[loader_var_name]
    exported_files = []  # track exported file paths
    if version1:
        file_name1 = os.path.join(folder1, f"file{start_index}")
        if os.path.exists(file_name1):
            os.remove(file_name1)  # overwrite instead of append
        print(f"[Exporting] {file_name1} -> {folder1}/")
        loader_instance.export(file_name1)
        exported_files.append(file_name1)
    # Reset loader after export
    caller_globals[loader_var_name] = loader_class()
    
    # ---- Save executed command and exported files as DataFrame in main folder ----
    log_path = os.path.join(version1, f"command_log_{timestamp}.csv")# os.path.join(os.getcwd(), "command_log.csv")
    df_new = pd.DataFrame([{
        "Start_Index": start_index,
        "Exported_Files": "; ".join(exported_files),
        "Command": command
        
    }])
    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(log_path, index=False)
    return start_index + 1

# ------------------ Report Generator ------------------ #
class SpectrumComparator:
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.comparison_results = []
    @staticmethod
    def _normalize_header_line(h: str) -> str:
        # Trim spaces around each comma-separated token
        return ",".join([p.strip() for p in h.split(",")])
    def _load_ascii(self, file, skip_header=False):
        # Support .asc, .txt, .csv; default to .asc if no extension
        if not (file.endswith('.csv') or file.endswith('.asc') or file.endswith('.txt')):
            file += '.asc'
        with open(file, 'r') as f:
            lines = [ln for ln in f.readlines() if ln.strip() != ""]  # drop blank lines
        if len(lines) < 3:
            raise ValueError(f"File {file} does not have expected 3+ lines (info, header, data).")
        # First line: file info
        file_info = lines[0].strip()
        # Second line: header (raw + normalized)
        header_line_raw = lines[1].strip()
        header_line_norm = self._normalize_header_line(header_line_raw)
        header_cols = [p.strip() for p in header_line_norm.split(",")]
        # Data lines
        data_str = "".join(lines[2:])
        df = pd.read_csv(StringIO(data_str), names=header_cols)
        if df.shape[1] < 1:
            raise ValueError(f"File {file} does not contain at least one column. Found: {list(df.columns)}")
        return df, header_cols, file_info, header_line_raw, header_line_norm
    def _headcount(self, file):
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]  # remove empty lines
            header_count = 0
            header_lines = []
            for i in range(len(lines) - 1):
                if lines[i].startswith("#") and re.match(r"^[A-Za-z]", lines[i+1]):
                    header_count += 1
                    header_lines.append(lines[i+1])
        return header_count,header_lines
    def compare(self, filename, desc, file1, file2, skip_header1=False, skip_header2=False):
        df1, hdr_cols1, info1, hdr1_raw, hdr1_norm = self._load_ascii(file1, skip_header1)
        df2, hdr_cols2, info2, hdr2_raw, hdr2_norm = self._load_ascii(file2, skip_header2)
        if len(hdr_cols1) == 1 and df1.shape[0] > 1 and self._headcount(file1)[0] >= 2:
            dim = "3D"
        elif len(hdr_cols1) == 1 and df1.shape[0] > 1 and self._headcount(file1)[0] == 1:
            dim = "2D"
        elif len(hdr_cols1) == df1.shape[1]:
            dim = "1D"
        else:
            dim = "Unknown"
        # Row counts
        len1 = len(df1)
        len2 = len(df2)
        min_len = min(len1, len2)
        df1 = df1.iloc[:min_len].reset_index(drop=True)
        df2 = df2.iloc[:min_len].reset_index(drop=True)
        # Header and file-info comparisons
        if dim== "3D":
            headers_match = (self._headcount(file1)[1] == self._headcount(file2)[1])
        else:
            headers_match = (hdr1_norm == hdr2_norm)
        # Multi-column / matrix comparison
        col_issues = []
        any_diff_found = False  # flag to detect any difference
        for col_idx in range(min(df1.shape[1], df2.shape[1])):
            colname = hdr_cols1[col_idx] if col_idx < len(hdr_cols1) else f"Col{col_idx+1}"
            col1_raw = df1.iloc[:, col_idx]
            col2_raw = df2.iloc[:, col_idx]
            def parse_cell(cell):
                if isinstance(cell, str):
                    parts = cell.strip().split()
                    try:
                        return np.array([float(x) for x in parts])
                    except:
                        return np.array([np.nan])
                else:
                    return np.array([float(cell)])
            # row-wise comparison
            line_nums_with_diff = []
            max_diff = 0.0
            for row_idx, (v1, v2) in enumerate(zip(col1_raw, col2_raw)):
                arr1 = parse_cell(v1)
                arr2 = parse_cell(v2)
                min_arr_len = min(len(arr1), len(arr2))
                arr1 = arr1[:min_arr_len]
                arr2 = arr2[:min_arr_len]
                diff = np.abs(arr1 - arr2)
                if np.any(diff > self.tolerance):
                    any_diff_found = True
                    line_nums_with_diff.append(row_idx + 3)  # data starts at line 3
                if diff.size > 0:
                    max_diff = max(max_diff, float(np.max(diff)))
            if line_nums_with_diff:
                col_issues.append(
                    f"{colname}: {len(line_nums_with_diff)} diffs, lines={line_nums_with_diff[:10]}"
                )
        info_match = (info1.strip() == info2.strip()) and not any_diff_found
        result = {
            "Filename": filename,
            "Dim": dim,
            "STD_Length": int(len1),
            "NEW_Length": int(len2),
            "STD_NumCols": df1.shape[1],
            "NEW_NumCols": df2.shape[1],
            "HeadersMatch": bool(headers_match),
            "FileInfoMatch": bool(info_match),

            "Error": desc if desc != "Pending" else ""
        }
        if col_issues:
            result["Error"] = "; ".join(col_issues)
        self.comparison_results.append(result)
        return result
    def to_dataframe(self):
        return pd.DataFrame(self.comparison_results)
def run_comparison_and_generate_report(version1, version2):
    export_dir = version2
    """
    Compare spectra organized by loader-class subfolders.
    version1: directory for the first set (e.g., Standard)
    version2: directory for the second set (e.g., New)
    export_dir: where to save the comparison report
    timestamp: string for unique report naming
    """
    tolerance = 1e-6
    comparator = SpectrumComparator(tolerance)
    exts = (".csv", ".asc", ".txt")
    for loader_class in os.listdir(version1):
        v1_subdir = os.path.join(version1, loader_class)
        v2_subdir = os.path.join(version2, loader_class)
        if not os.path.isdir(v1_subdir) or not os.path.isdir(v2_subdir):
            continue
        base_files_v1 = [f for f in os.listdir(v1_subdir) if f.endswith(exts)]
        for file in base_files_v1:
            train_path = os.path.join(v1_subdir, file)
            test_path = os.path.join(v2_subdir, file)
            if not os.path.exists(test_path):
                print(f"[Skip] No matching file in {v2_subdir} for: {file}")
                continue
            # Compare
            result = comparator.compare(file, "Pending", train_path, test_path)
            # Add subfolder info
            comparator.comparison_results[-1]["Subfolder"] = loader_class
            # Build description
            issues = []
            if result["STD_Length"] != result["NEW_Length"]:
                issues.append(f"Length mismatch ({result['STD_Length']} vs {result['NEW_Length']})")
            if result["STD_NumCols"] != result["NEW_NumCols"]:
                issues.append(f"Column count mismatch ({result['STD_NumCols']} vs {result['NEW_NumCols']})")
            if not result["FileInfoMatch"]:
                issues.append("File info line differs")
            if not result["HeadersMatch"]:
                issues.append("Headers differ")
            if result["Error"]:
                issues.append(result["Error"])
            desc_str = "None" if len(issues) == 0 else "; ".join(issues)
            comparator.comparison_results[-1]["Error"] = desc_str
    # Export report
    df = comparator.to_dataframe()
    # Reformat Length and NumCols columns
    df["Length"] = df["STD_Length"].astype(str) + " / " + df["NEW_Length"].astype(str)
    df["NumCols"] = df["STD_NumCols"].astype(str) + " / " + df["NEW_NumCols"].astype(str)

    # Drop old columns
    df = df.drop(columns=["STD_Length", "NEW_Length", "STD_NumCols", "NEW_NumCols"])
    # Option one: Using fixed column order
    cols = [
        "Subfolder",
        "Filename",
        "Dim",
        "Length",
        "NumCols",
        "HeadersMatch",
        "FileInfoMatch",
        "Error"
    ]
    # Option two. Using reordering columns: Subfolder first
    #cols = ["Subfolder"] + [c for c in df.columns if c != "Subfolder"]
    df = df[cols]
    os.makedirs(export_dir, exist_ok=True)
    out_file = os.path.join(export_dir, f"comparison_report_{timestamp}.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Report saved to: {out_file}")
    return df, out_file








