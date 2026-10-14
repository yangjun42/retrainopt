"""Compare public index artifact digests with the exact tested distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", choices=("pypi", "testpypi"), required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    host = "test.pypi.org" if args.target == "testpypi" else "pypi.org"
    with urlopen(f"https://{host}/pypi/retrainopt/{args.version}/json", timeout=30) as response:
        payload = json.load(response)
    assert payload["info"]["name"] == "retrainopt"
    assert payload["info"]["version"] == args.version
    remote = {item["filename"]: item["digests"]["sha256"] for item in payload["urls"]}
    local = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.dist.iterdir() if path.is_file()}
    assert len(local) == 2 and any(name.endswith(".whl") for name in local) and any(name.endswith(".tar.gz") for name in local)
    assert remote == local, "Public index artifacts differ from the tested build."
    print(json.dumps({"index": host, "version": args.version, "sha256": local}, sort_keys=True))


if __name__ == "__main__":
    main()
