"""
VOC-ArchNLP: Dutch Colonial Archive Mining for Indonesian Archaeological Data
=============================================================================
Version  : 1.0.0
Author   : Mukhlis Amien (amien@ubhinus.ac.id)
ORCID    : 0000-0002-1848-167X
Affiliation : Universitas Bhinneka Nusantara (Ubhinus), Malang, Indonesia
Year     : 2026
License  : CC BY 4.0
Project  : VOLCARCH — Volcanic Taphonomic Bias in Indonesian Archaeological Records

Derived from experimental work: E091, E141, E197, E206, E207, E211.

Components:
  downloader  — fetch VOC transcriptions from GLOBALISE Dataverse (CC0)
  preprocessor — clean HTR-transcribed colonial Dutch text
  normalizer  — map pre-1947 Dutch orthography to modern Dutch
  extractor   — extract archaeological mention sentences with entity tagging
  pipeline    — end-to-end orchestrator
  cli         — unified command-line interface
"""

__version__ = "1.0.0"
__author__ = "Mukhlis Amien"
__email__ = "amien@ubhinus.ac.id"
__license__ = "CC BY 4.0"
__description__ = (
    "NLP pipeline for mining archaeological information from Dutch VOC colonial archives"
)

from .extractor import ArchaeologicalMentionExtractor
from .pipeline import VOCArchPipeline

__all__ = ["ArchaeologicalMentionExtractor", "VOCArchPipeline"]
