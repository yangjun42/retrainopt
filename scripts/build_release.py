"""Build with locked tools and normalize sdist archive metadata.

Set SOURCE_DATE_EPOCH to the source commit time. Setuptools honors it for
wheels, but its sdist tar/gzip metadata otherwise uses host IDs and mtimes.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    epoch = int(os.environ["SOURCE_DATE_EPOCH"])
    if epoch < 315532800:
        raise ValueError("SOURCE_DATE_EPOCH must be at least 1980-01-01 for wheel ZIP timestamps.")
    destination = args.outdir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty build output directory to exclude stale distributions.")
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(destination)],
        check=True,
    )
    archives = list(destination.glob("*.tar.gz"))
    if len(archives) != 1 or len(list(destination.glob("*.whl"))) != 1:
        raise ValueError("Expected exactly one source distribution and one wheel.")
    archive = archives[0]
    normalized = archive.with_suffix(".normalized")
    try:
        with tarfile.open(archive, "r:gz") as source, normalized.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as output:
                    for member in sorted(source.getmembers(), key=lambda item: item.name):
                        if not (member.isfile() or member.isdir()):
                            raise ValueError(f"Unsupported source archive member: {member.name}")
                        member.uid = member.gid = 0
                        member.uname = member.gname = ""
                        member.mtime = epoch
                        member.mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
                        member.pax_headers = {}
                        stream = source.extractfile(member) if member.isfile() else None
                        try:
                            output.addfile(member, stream)
                        finally:
                            if stream is not None:
                                stream.close()
        normalized.replace(archive)
    finally:
        normalized.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
