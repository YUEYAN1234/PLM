from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


UNIREF_FASTA_URLS = {
    "uniref50": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
    "uniref90": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
}


@dataclass(frozen=True)
class ChecksumInfo:
    md5: str
    size: int | None = None


def download_file(url: str, destination: Path, *, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Found existing file: {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        with tmp_path.open("wb") as output, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                progress.update(len(chunk))
    shutil.move(str(tmp_path), str(destination))


def parse_md5(md5_path: Path) -> str:
    text = md5_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty md5 file: {md5_path}")
    return text.split()[0].lower()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_metalink_md5(metalink_path: Path, filename: str) -> str:
    return parse_metalink_info(metalink_path, filename).md5


def parse_metalink_info(metalink_path: Path, filename: str) -> ChecksumInfo:
    root = ET.parse(metalink_path).getroot()
    for file_node in root.iter():
        if local_name(file_node.tag) != "file" or file_node.attrib.get("name") != filename:
            continue
        size: int | None = None
        expected_md5: str | None = None
        for node in file_node.iter():
            if local_name(node.tag) == "size" and node.text and node.text.strip():
                size = int(node.text.strip())
            if local_name(node.tag) == "hash" and node.attrib.get("type", "").lower() == "md5":
                if node.text and node.text.strip():
                    expected_md5 = node.text.strip().lower()
        if expected_md5:
            return ChecksumInfo(md5=expected_md5, size=size)
    raise ValueError(f"Could not find MD5 for {filename} in {metalink_path}")


def download_checksum(url: str, md5_path: Path, *, force: bool = False) -> ChecksumInfo:
    filename = Path(url).name
    if md5_path.exists() and not force:
        print(f"Found existing file: {md5_path}")
        metalink_path = md5_path.with_name("RELEASE.metalink")
        size = None
        if metalink_path.exists():
            try:
                size = parse_metalink_info(metalink_path, filename).size
            except ValueError:
                size = None
        return ChecksumInfo(md5=parse_md5(md5_path), size=size)

    try:
        download_file(f"{url}.md5", md5_path, force=force)
        return ChecksumInfo(md5=parse_md5(md5_path))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        print(f"No standalone MD5 file at {url}.md5; falling back to RELEASE.metalink")

    base_url = url.rsplit("/", 1)[0]
    metalink_path = md5_path.with_name("RELEASE.metalink")
    download_file(f"{base_url}/RELEASE.metalink", metalink_path, force=force)
    checksum = parse_metalink_info(metalink_path, filename)
    md5_path.write_text(f"{checksum.md5}  {filename}\n", encoding="utf-8")
    print(f"Wrote checksum: {md5_path}")
    return checksum


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, checksum: ChecksumInfo) -> None:
    if checksum.size is not None:
        actual_size = path.stat().st_size
        if actual_size != checksum.size:
            raise ValueError(
                f"File size mismatch for {path}: expected {checksum.size} bytes, got {actual_size}. "
                "The local file is incomplete or from another release. Re-run with --force to download it again."
            )

    actual = file_md5(path)
    if actual.lower() != checksum.md5:
        raise ValueError(
            f"MD5 mismatch for {path}: expected {checksum.md5}, got {actual}. "
            "The local file is incomplete or from another release. Re-run with --force to download it again."
        )
    print(f"MD5 verified: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download UniRef FASTA data from UniProt FTP.")
    parser.add_argument("--dataset", choices=sorted(UNIREF_FASTA_URLS), default="uniref50")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    parser.add_argument("--skip-md5", action="store_true", help="Do not download or verify MD5 checksum.")
    args = parser.parse_args(argv)

    url = UNIREF_FASTA_URLS[args.dataset]
    fasta_path = args.out / Path(url).name
    md5_path = args.out / f"{Path(url).name}.md5"

    download_file(url, fasta_path, force=args.force)
    if not args.skip_md5:
        checksum = download_checksum(url, md5_path, force=args.force)
        verify_md5(fasta_path, checksum)

    print(f"Ready: {fasta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
