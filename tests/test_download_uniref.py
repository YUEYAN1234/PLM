from pathlib import Path

from plm.download_uniref import parse_metalink_md5


def test_parse_metalink_md5(tmp_path: Path) -> None:
    metalink = tmp_path / "RELEASE.metalink"
    metalink.write_text(
        """<metalink xmlns="http://www.metalinker.org/" version="3.0">
  <files>
    <file name="uniref50.fasta.gz">
      <verification>
        <hash type="md5">abc123</hash>
      </verification>
    </file>
  </files>
</metalink>
""",
        encoding="utf-8",
    )

    assert parse_metalink_md5(metalink, "uniref50.fasta.gz") == "abc123"
