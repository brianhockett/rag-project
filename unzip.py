"""Extract the CMU Movie Summary Corpus files this project uses.

Reads data/MovieSummaries.tar.gz and writes only the files we need to
data/MovieSummaries/. The character metadata and the TV Tropes / name
cluster files are skipped: our RAG only uses plot summaries and movie
metadata.

Download the tarball into data/ first:
    http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz

Run:
    python unzip.py
"""

# Imports
from pathlib import Path
import tarfile

# Data paths
DATA = Path(__file__).parent / "data"
TARBALL = DATA / "MovieSummaries.tar.gz"
OUT = DATA / "MovieSummaries"

# File name -> line count stated in the corpus README (None = don't check).
# README.txt is kept because it documents the column layout and the license.
KEEP = {
    "plot_summaries.txt": 42_306,
    "movie.metadata.tsv": 81_741,
    "README.txt": None,
}

# Main function
def main():
    # Ensure that the tarball exists before trying to extract it.
    if not TARBALL.exists():
        raise SystemExit(f"{TARBALL} not found. Download it first (see the top of unzip.py).")

    # One pass through the archive, extracting only the files in KEEP.
    # filter = "data" refuses absolute paths, "../" tricks and links outside data/.
    wanted = {f"MovieSummaries/{name}" for name in KEEP}
    found = set()
    with tarfile.open(TARBALL, "r:gz") as tar:
        for member in tar:
            if member.name in wanted:
                tar.extract(member, DATA, filter = "data")
                found.add(member.name)

    # Check that we found all the files we wanted.
    missing = wanted - found
    if missing:
        raise SystemExit(f"Not in {TARBALL.name}: {sorted(missing)}")

    # Report the results.
    print(f"Extracted to {OUT}/")
    for name, expected in KEEP.items():
        path = OUT / name
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        check = "" if expected is None else ("ok" if n_lines == expected else f"EXPECTED {expected:,}")
        print(f"  {name:<20} {path.stat().st_size / 1e6:6.1f} MB  {n_lines:>7,} lines  {check}")

# Run the main function if this script is executed directly.
if __name__ == "__main__":
    main()
