import pandas as pd
from typing import Optional


def read_dat(dat_fp: str, header_lines: int, delimiter: Optional[str] = None) -> pd.DataFrame:
    """Reads an HGS ASCII output file as a pandas DataFrame."""

    # Search for column names.
    # For ASCII files, these names are stored in a line beginning with "variables"
    with open(dat_fp) as f:
        head = next(f)
        while 'variables' not in head.lower():
            head = next(f)
        names = [i for i in head.strip().split('=')[1].split('"') if i and ',' not in i and i != ' ']

    # Use pandas to read the data within the ASCII file.
    if delimiter:
        return pd.read_csv(dat_fp, skiprows=header_lines,
                           names=names, index_col='Time', delimiter=delimiter)
    else:
        return pd.read_csv(dat_fp, skiprows=header_lines,
                           names=names, index_col='Time', delim_whitespace=True)
