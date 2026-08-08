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
    # Sample rate / bit depth de LIVRAISON pour ce profil — AVANT cette
    # fonctionnalité, la sortie passait au travers au sample rate/bit depth
    # de la SOURCE quel qu'il soit (un WAV 96kHz/32-bit restait 96kHz/32-bit
    # en sortie), ce qui n'a aucune raison de correspondre à ce qu'attend la
    # plateforme cible. Défauts 44100Hz / PCM_24 : recherche 2026 (DistroKid,
    # guides de mastering streaming) confirme que 24-bit/44.1kHz WAV est LE
    # standard de livraison quasi-universel — les agrégateurs/plateformes
    # transcodent eux-mêmes vers leur format final (AAC, Ogg Vorbis, etc.),
    # donc pas de divergence réelle constatée entre profils à ce jour. Champs
    # gardés PAR profil (pas une constante globale) pour rester ajustables
    # si un cas particulier (hi-res, club edit, etc.) l'exigeait un jour.
    output_sample_rate: int = 44100
    output_bit_depth: str = "PCM_24"  # voir audio_io.save_wav: "PCM_16"|"PCM_24"|"PCM_32"|"FLOAT"


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
            output_sample_rate=int(entry.get("output_sample_rate", 44100)),
            output_bit_depth=str(entry.get("output_bit_depth", "PCM_24")),
        )
        for key, entry in data["profiles"].items()
    }
    gain_staging = GainStagingConfig(
        target_lufs=float(data["gain_staging"]["target_lufs"]),
        true_peak_ceiling_dbtp=float(data["gain_staging"]["true_peak_ceiling_dbtp"]),
    )
    return ProfileCatalog(profiles=profiles, gain_staging=gain_staging)
