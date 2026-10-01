"""Wiring: from the catalogue to the policy that says which side a lecture is on.

The side of a lecture is never written down in code. It comes from the lecture
manifest when the recordings are at hand, and otherwise from the two published
files, which carry the same split. Naming lectures in code would let a rename
silently move a lecture from one side to the other.
"""

from __future__ import annotations

from ..core.errors import SettingsError
from ..core.io import read_jsonl
from .catalog import DataCatalog
from .manifest import LectureManifest
from .splits import DatasetSplit, ManifestSplit, SplitPolicy

#: The names this builder accepts, in the order it tries them.
POLICY_NAMES = ("manifest", "dataset_split")


def build_split_policy(catalog: DataCatalog, name: str | None = None) -> SplitPolicy:
    """The split policy: the one asked for, or the manifest if it is there.

    Falling back rather than failing lets the evaluation run from the published
    datasets alone, on a machine that holds no recordings.
    """
    if name is not None and name not in POLICY_NAMES:
        raise SettingsError(f"unknown split policy {name!r}; known: {list(POLICY_NAMES)}")
    if name in (None, "manifest"):
        manifest_path = catalog.local_path("lecture_manifest")
        if manifest_path.exists():
            return ManifestSplit(LectureManifest.load(manifest_path))
        if name == "manifest":
            raise SettingsError(
                f"the manifest split needs {manifest_path.name}, which is not at {manifest_path}"
            )
    return DatasetSplit(
        read_jsonl(catalog.dataset_file("train_examples")),
        read_jsonl(catalog.dataset_file("test_examples")),
    )
