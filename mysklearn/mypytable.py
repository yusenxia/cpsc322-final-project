import copy
import csv
from tabulate import tabulate


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable."""
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Returns (N, M) where N is #rows and M is #cols."""
        n = len(self.data)
        m = len(self.column_names)
        return n, m

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list."""
        if isinstance(col_identifier, int):
            idx = col_identifier
            if idx < 0 or idx >= len(self.column_names):
                raise ValueError("Invalid column index")
        elif isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError("Invalid column name")
            idx = self.column_names.index(col_identifier)
        else:
            raise ValueError("col_identifier must be str or int")

        col = []
        for row in self.data:
            val = row[idx]
            if include_missing_values or val != "NA":
                col.append(val)
        return col

    def convert_to_numeric(self):
        """Try to convert each value to numeric (prefer int, then float)."""
        for i, row in enumerate(self.data):
            new_row = []
            for val in row:
                if isinstance(val, str) and val != "NA":
                    v = val.strip()
                    try:
                        iv = int(v)
                        new_row.append(iv)
                        continue
                    except Exception:
                        try:
                            fv = float(v)
                            new_row.append(fv)
                            continue
                        except Exception:
                            new_row.append(val)
                else:
                    new_row.append(val)
            self.data[i] = new_row

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows at the specified original indexes."""
        if not row_indexes_to_drop:
            return
        for idx in sorted(set(row_indexes_to_drop), reverse=True):
            if 0 <= idx < len(self.data):
                self.data.pop(idx)

    def load_from_file(self, filename):
        """Load from CSV (first row header) and convert numerics."""
        with open(filename, "r", encoding="utf-8", newline="") as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        if not rows:
            self.column_names = []
            self.data = []
            return self

        self.column_names = rows[0]
        self.data = rows[1:]
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save header and data to CSV."""
        with open(filename, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Return list of indexes of duplicate rows based on key columns."""
        key_indexes = []
        for name in key_column_names:
            if name not in self.column_names:
                raise ValueError("Invalid column name")
            key_indexes.append(self.column_names.index(name))

        seen = set()
        dups = []
        for i, row in enumerate(self.data):
            key = tuple(row[k] for k in key_indexes)
            if key in seen:
                dups.append(i)
            else:
                seen.add(key)
        return dups

    def remove_rows_with_missing_values(self):
        """Remove any row containing 'NA'."""
        self.data = [row for row in self.data if "NA" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """Replace 'NA' in a numeric column with the column's original average."""
        if col_name not in self.column_names:
            raise ValueError("Invalid column name")
        idx = self.column_names.index(col_name)

        vals = [row[idx] for row in self.data if row[idx] != "NA"]
        if not vals:
            return
        avg = sum(vals) / len(vals)

        for row in self.data:
            if row[idx] == "NA":
                row[idx] = avg

    def compute_summary_statistics(self, col_names):
        """Compute min, max, mid, avg, median for given numeric columns."""
        out_header = ["attribute", "min", "max", "mid", "avg", "median"]
        out_rows = []

        if not self.data:
            return MyPyTable(out_header, [])

        for name in col_names:
            if name not in self.column_names:
                raise ValueError("Invalid column name")
            idx = self.column_names.index(name)

            col = [v for v in (row[idx] for row in self.data) if v != "NA"]
            if not col:
                continue
            col_sorted = sorted(col)
            cmin = col_sorted[0]
            cmax = col_sorted[-1]
            cmid = (cmin + cmax) / 2
            avg = sum(col_sorted) / len(col_sorted)
            n = len(col_sorted)
            if n % 2 == 1:
                median = col_sorted[n // 2]
            else:
                median = (col_sorted[n // 2 - 1] + col_sorted[n // 2]) / 2
            out_rows.append([name, cmin, cmax, cmid, avg, median])

        return MyPyTable(out_header, out_rows)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table."""
        left_key_idx = [self.column_names.index(c) for c in key_column_names]
        right_key_idx = [other_table.column_names.index(c) for c in key_column_names]

        right_nonkey_idx = [
            j for j, name in enumerate(other_table.column_names)
            if name not in key_column_names
        ]
        out_header = list(self.column_names) + [
            other_table.column_names[j] for j in right_nonkey_idx
        ]
        out_data = []

        for lrow in self.data:
            for rrow in other_table.data:
                if all(lrow[li] == rrow[ri] for li, ri in zip(left_key_idx, right_key_idx)):
                    merged = list(lrow) + [rrow[j] for j in right_nonkey_idx]
                    out_data.append(merged)

        return MyPyTable(out_header, out_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table."""
        left_key_idx = [self.column_names.index(c) for c in key_column_names]
        right_key_idx = [other_table.column_names.index(c) for c in key_column_names]

        right_nonkey_idx = [
            j for j, name in enumerate(other_table.column_names)
            if name not in key_column_names
        ]
        out_header = list(self.column_names) + [
            other_table.column_names[j] for j in right_nonkey_idx
        ]
        out_data = []
        matched_right = set()

        for lrow in self.data:
            found = False
            for ridx, rrow in enumerate(other_table.data):
                if all(lrow[li] == rrow[ri] for li, ri in zip(left_key_idx, right_key_idx)):
                    merged = list(lrow) + [rrow[j] for j in right_nonkey_idx]
                    out_data.append(merged)
                    matched_right.add(ridx)
                    found = True
            if not found:
                out_data.append(list(lrow) + ["NA"] * len(right_nonkey_idx))
        for ridx, rrow in enumerate(other_table.data):
            if ridx not in matched_right:
                fake_left = ["NA"] * len(self.column_names)
                for li, ri in zip(left_key_idx, right_key_idx):
                    fake_left[li] = rrow[ri]
                merged = fake_left + [rrow[j] for j in right_nonkey_idx]
                out_data.append(merged)

        return MyPyTable(out_header, out_data)