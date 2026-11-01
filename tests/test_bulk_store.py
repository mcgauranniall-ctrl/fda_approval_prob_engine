from fdaengine.dataset import bulk


def test_jsonl_writer_is_reproducible_and_manifest_tracks_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk, "RAW_DIR", tmp_path)
    monkeypatch.setattr(bulk, "MANIFEST", tmp_path / "manifest.json")
    shas = []
    for _ in range(2):
        w = bulk.JsonlGzWriter(tmp_path / "ctgov" / "s.jsonl.gz")
        w.write({"nct": "NCT1", "x": "é"})
        shas.append(w.close())
    assert shas[0] == shas[1] == bulk.sha256_file(tmp_path / "ctgov" / "s.jsonl.gz")
    assert list(bulk.iter_jsonl_gz(tmp_path / "ctgov" / "s.jsonl.gz")) == [{"nct": "NCT1", "x": "é"}]

    m = bulk.load_manifest()
    assert not bulk.is_current(m, "ctgov/s.jsonl.gz", "v1")
    bulk.record(m, "ctgov/s.jsonl.gz", "ctgov", "https://example", shas[0], "v1", records=1)
    bulk.save_manifest(m)
    m = bulk.load_manifest()
    assert bulk.is_current(m, "ctgov/s.jsonl.gz", "v1")
    assert not bulk.is_current(m, "ctgov/s.jsonl.gz", "v2")          # new export → re-download
    (tmp_path / "ctgov" / "s.jsonl.gz").write_bytes(b"truncated")
    assert not bulk.is_current(m, "ctgov/s.jsonl.gz", "v1")          # size changed → re-download
    assert bulk.raw_files("ctgov", m) == [tmp_path / "ctgov" / "s.jsonl.gz"]
