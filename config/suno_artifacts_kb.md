# Base de connaissance : artefacts caractéristiques des tracks générées par Suno

Ce document recense les artefacts audio spécifiquement associés aux
générateurs de musique par IA (Suno en particulier, parfois partagés avec
Udio/Stable Audio), sourcés depuis des retours de mastering engineers et
d'outils dédiés au nettoyage de tracks IA. Il sert de base au mode
**"Source Suno / IA générée"** (`suno_mode=True`) de `decide_bands()` dans
`refinr/proq4_control.py` — des corrections **en plus** de l'analyse
générique habituelle, activées explicitement (jamais par défaut, pour
rester fidèle au principe "jamais de traitement générique" du projet : ces
corrections ne doivent s'appliquer qu'à du contenu dont on sait qu'il vient
d'un générateur IA).

Important : ces artefacts sont décrits comme "baked into the audio at the
codec level" par plusieurs sources — l'EQ/dynamique peut les atténuer,
pas les supprimer complètement. Refinr applique une atténuation corrective
raisonnable, pas un "artifact remover" dédié (hors scope : nécessiterait
une détection spectrale bien plus fine, type analyse de flatness/aliasing,
non implémentée ici).

## 1. Boue bas-médium (250–500 Hz), pic vers 300 Hz

Suno génère tous les instruments simultanément (contrairement à un mixage
multipiste traditionnel construit couche par couche), ce qui fait
s'accumuler l'énergie bas-médium — cité comme "AI instrument clustering".
Le training data étant riche en sources home-recorded, le modèle "sur-cuit"
systématiquement les fréquences 200–400 Hz.

- **Fix rapporté** : cut large et doux, 1 à 3 dB autour de 300 Hz.
- **Couverture existante dans Refinr** : déjà couvert par la correction
  générique `already_compressed` (Bell -1.5dB à 300Hz, déclenchée sur
  crest factor < 8dB) — pas de nouvelle bande nécessaire, mais confirme
  que ce réglage est pertinent spécifiquement pour du Suno aussi.

## 2. Fizz / bruit de synthèse en haute fréquence (>14 kHz)

Le codec de génération jette le détail HF au-delà d'environ 14kHz et le
remplace par du bruit de synthèse — source de fatigue d'écoute au casque.

- **Fix rapporté** : shelf doux à 14kHz, -2 à -3dB.
- **Nouveau dans Refinr** (`suno_mode=True` uniquement) : High Shelf
  14000Hz, -2.5dB.

## 3. Harshness/métallique vocal (2.5–5 kHz, concentré 3.5–5 kHz chez Suno)

Les formants synthétiques se figent statiquement dans la zone 2–3kHz (une
voix humaine y bouge dynamiquement), et Suno concentre le "buzz" métallique
un peu plus haut que les autres générateurs, vers 3.5–5kHz.

- **Fix rapporté** : cut doux 2-4dB, Q large (0.5–1.0), centré ~4kHz ;
  plage de sécurité usuelle -2 à -5dB à 2.5kHz.
- **Nouveau dans Refinr** (`suno_mode=True` uniquement) : Bell 4000Hz,
  -2.5dB, Q 0.7 (large). Appliqué au mix entier (pas de séparation vocale
  disponible dans Refinr), donc volontairement modéré pour ne pas
  assourdir les autres éléments présents dans cette zone.

## 4. Flou de phase stéréo / élargissement artificiel

L'élargissement stéréo artificiel de Suno crée des problèmes de phase et
une perte de compatibilité mono ; le repliement mono peut révéler une
annulation audible.

- **Fix rapporté** : traitement mid-side, réduction des side de 2-3dB
  sous 200Hz, sub mono sous 80Hz.
- **Couverture existante dans Refinr** : déjà couvert par la correction
  générique `wide_stereo` (High Pass Side 150Hz, déclenchée sur
  corrélation stéréo < 0.2) — même logique (garder le bas du spectre
  compatible mono), seuil de fréquence légèrement différent (150Hz vs
  80-200Hz rapportés, dans la même fourchette).

## 5. Dynamique collapsée

Les exports Suno sortent souvent déjà avec une dynamique compressée par
défaut.

- **Couverture existante dans Refinr** : déjà couvert par la correction
  générique `already_compressed` (même mécanisme que le point 1).

## Sources

- [AI Music Fixer — MixMasterAI](https://www.mixmasterai.co/mastering/fix)
- [How to Fix AI Music Artifacts — Intrect](https://intrect.io/blog/how-to-fix-ai-music-artifacts/)
- [How to Mix and Master Suno AI Tracks — MixMasterAI](https://www.mixmasterai.co/mix-and-master-suno)
- [Suno Vocal Fix — MixMasterAI](https://www.mixmasterai.co/suno-vocal-fix)
- [Fix Metallic and Robotic Sound in AI Music — Undetectr](https://undetectr.com/blog/remove-metallic-sound-from-ai-music)
- [How to properly mix Suno stems — Peak Studios](https://www.peak-studios.de/en/suno-stems-mixen/)

## Limites connues

Ces seuils/fréquences viennent de retours d'expérience publiés, pas d'une
calibration mesurée par Refinr sur un corpus de tracks Suno réelles (aucun
accès macOS/plugins pendant le développement de cette fonctionnalité). À
valider/ajuster à l'usage. Les corrections restent conditionnées à
`suno_mode=True` (jamais activées automatiquement par l'analyse seule),
précisément parce qu'elles ne sont pas déduites de mesures faites sur CE
fichier mais d'une connaissance a priori sur la source.
