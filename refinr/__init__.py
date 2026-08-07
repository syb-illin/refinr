"""
refinr
======

Pipeline de retraitement audio pour fichiers WAV bruts :
gain staging -> EQ -> saturation -> tape -> leveling vers un profil de
destination (Spotify, YouTube, Apple Music, DistroKid générique, etc.),
en s'appuyant sur des plugins Audio Unit installés localement (FabFilter
Pro-Q4, Softube Saturn 2 / BackBox HG2, Waves J37).

Ce package est scindé en modules indépendants pour que les parties
testables hors macOS (mesure de loudness, analyse spectrale, sélection
de preset, orchestration, batch, reporting) puissent être développées et
vérifiées séparément du hosting AU réel (qui requiert AudioToolbox via
ctypes + Foundation via PyObjC, donc macOS).
"""

__version__ = "0.2.1"
