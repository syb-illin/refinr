"""Chargement des profils de destination (config/destination_profiles.yaml)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent.parent / "config" / "destination_profiles.yaml"


@dataclasses.dataclass
class DestinationProfile:
    key: str
    label: str
    target_lufs: float
    true_peak_ceiling_dbtp: float
    boosts_quiet: bool | None
    notes: str = ""


@dataclasses.dataclass
class GainStagingConfig:
    target_lufs: float
    true_peak_ceiling_dbtp: float


@dataclasses.dataclass
class ProfileCatalog:
    profiles: dict[str, DestinationProfile]
    gain_staging: GainStagingConfig

    def get(self, key: str) -> DestinationProfile:
        try:
            return self.profiles[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles))
            raise KeyError(f"Profil de destination inconnu: {key!r}. Disponibles: {available}") from exc


def load_profiles(path: str | Path = DEFAULT_PROFILES_PATH) -> ProfileCatalog:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    profiles = {
        key: DestinationProfile(
            key=key,
            label=entry["label"],
            target_lufs=float(entry["target_lufs"]),
            true_peak_ceiling_dbtp=float(entry["true_peak_ceiling_dbtp"]),
            boosts_quiet=entry.get("boosts_quiet"),
            notes=entry.get("notes", "").strip(),
        )
        for key, entry in data["profiles"].items()
    }
    gain_staging = GainStagingConfig(
        target_lufs=float(data["gain_staging"]["target_lufs"]),
        true_peak_ceiling_dbtp=float(data["gain_staging"]["true_peak_ceiling_dbtp"]),
    )
    return ProfileCatalog(profiles=profiles, gain_staging=gain_staging)
