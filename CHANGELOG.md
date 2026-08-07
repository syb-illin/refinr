# Changelog

Historique des versions de Refinr — généré automatiquement par le hook
`.githooks/post-commit` à chaque commit (une version = un commit, voir
README > Versioning automatique). Les entrées ci-dessous avant la mise en
place du hook ont été reconstruites depuis l'historique git ; celles qui
suivent sont ajoutées automatiquement.

<!-- CHANGELOG_INSERT -->

## [0.1.8] - 2026-08-07 (patch)
- fix: stub `process_chain_offline` en pass-through pour les tests sur macOS CI (les plugins commerciaux réels ne sont pas installés sur les runners hébergés), la chaîne complète reste testée sauf le rendu DSP réel

## [0.1.7] - 2026-08-07 (patch)
- feat: GUI moderne (menu, drag & drop, panneau de log en mode debug, export de rapport), diagnostic complet + raisons détaillées par bande EQ dans le rapport HTML, Ruff + Black, CI lint/tests/couverture, badge de couverture

## [0.1.6] - 2026-08-07 (patch)
- feat: rapport HTML enrichi — détail avant/après (LUFS/true-peak) et liste des bandes EQ décidées par étape, pas seulement en JSON

## [0.1.5] - 2026-08-07 (patch)
- feat: EQ Pro-Q4 pilotée dynamiquement par défaut (`proq4_control.py`) — construction programmatique de l'état AU (fréquence/gain/Q/forme/pente/stéréo/dynamique) à partir de l'analyse du WAV, plus de sélection de preset EQ figé ; saturation/tape restent en sélection par tags en attendant leur reverse-engineering

## [0.1.4] - 2026-08-07 (patch)
- chore: bump version

## [0.1.3] - 2026-08-07 (patch)
- fix: permissions Release 403 + bump des actions GitHub vers les versions Node24 (checkout@v5, setup-python@v6, upload-artifact@v7, action-gh-release@v3)

## [0.1.2] - 2026-08-07 (patch)
- fix: retrait des presets de démo factices (FabFilter Pro-Q4 fictif) qui faisaient échouer le hosting AU réel sur le runner CI

## [0.1.1] - 2026-08-07 (patch)
- fix: `au_host.py` réécrit en ctypes (le package `pyobjc-framework-AudioToolbox` n'existe pas sur PyPI), retrait des mentions "Suno" du code et du README

## [0.1.0] - 2026-08-07 (minor)
- feat: version initiale de Refinr — pipeline gain staging → EQ → saturation → tape → leveling par profil de destination, GUI PyQt6, batch parallèle, reporting détaillé, packaging automatisé via GitHub Actions
