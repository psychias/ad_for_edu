"""Row schema, dataset sources, the lecture manifest, the split and the media layout."""

from .build import POLICY_NAMES, build_split_policy
from .catalog import DataCatalog, DatasetEntry
from .manifest import LectureManifest
from .media import Gap, Keyframe, LectureArtefacts, MediaLayout, SlideTextFrame, Word
from .sources import DATASET_SOURCES, DatasetSource
from .splits import (
    SPLIT_POLICIES,
    DatasetSplit,
    ManifestSplit,
    SplitPolicy,
    assert_disjoint_lectures,
    assert_disjoint_moments,
    assert_no_test_lecture,
)

__all__ = [
    "DATASET_SOURCES",
    "POLICY_NAMES",
    "SPLIT_POLICIES",
    "DataCatalog",
    "DatasetEntry",
    "DatasetSource",
    "DatasetSplit",
    "Gap",
    "Keyframe",
    "LectureArtefacts",
    "LectureManifest",
    "ManifestSplit",
    "MediaLayout",
    "SlideTextFrame",
    "SplitPolicy",
    "Word",
    "assert_disjoint_lectures",
    "assert_disjoint_moments",
    "assert_no_test_lecture",
    "build_split_policy",
]
