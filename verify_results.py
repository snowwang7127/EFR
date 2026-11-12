"""Check every rebuilt CSV against the released manuscript reference values."""
import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from reproduce import ROOT, verify_release


def verify(directory):
    verify_release()
    expected = sorted((ROOT / "reference").glob("*.csv"))
    actual = sorted(directory.glob("*.csv"))
    if [f.name for f in expected] != [f.name for f in actual]:
        raise ValueError(f"Missing or extra CSVs in {directory}")
    for reference, regenerated in zip(expected, actual):
        a, b = pd.read_csv(reference), pd.read_csv(regenerated)
        for col in a.select_dtypes(include="integer"):
            pd.testing.assert_series_equal(a[col], b[col], check_exact=True)
        assert_frame_equal(a, b, rtol=0, atol=1e-9, check_exact=False, obj=reference.name)
        print(f"PASS {reference.name}: {len(a)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
