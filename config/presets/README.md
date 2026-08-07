# Bibliothèque de presets AU

Range ici tes exports `.aupreset` par rôle :

```
config/presets/
  eq/           # FabFilter Pro-Q4
  saturation/   # Saturn 2 ou BackBox HG2
  tape/         # Waves J37
```

Pour chaque `.aupreset`, un fichier compagnon optionnel `<même_nom>.meta.yaml`
indique quand ce preset doit être choisi automatiquement, en fonction des
tags produits par l'analyse du fichier (`refinr/analysis.py ->
FileAnalysis.summary_tags()`) :

```yaml
tags: [warmth, gentle]        # libre, informatif
intensity: light               # light | medium | heavy
suited_for:
  tags_any: [dark, already_compressed]   # au moins un de ces tags doit matcher
  tags_none: [very_dynamic]              # aucun de ces tags ne doit être présent
priority: 1                    # départage les égalités de score
```

Sans `meta.yaml`, un preset est traité comme "universel" (toujours éligible,
priorité 0) — pratique au début, en attendant d'annoter la bibliothèque.

Tags actuellement produits par l'analyse : `clipping_detected`,
`already_compressed`, `very_dynamic`, `bright`, `dark`, `balanced_tonal`,
`wide_stereo`, `narrow_mono_like`.

## ⚠️ Fichiers de démo à supprimer

`eq/bright_cut_for_dark.aupreset`, `eq/bright_cut_for_dark.meta.yaml` et
`eq/universal_gentle.aupreset` sont des **presets factices** (générés
pendant les tests, ce ne sont pas de vrais exports FabFilter) qui illustrent
juste le mécanisme de sélection. Supprime-les dès que tu ajoutes tes vrais
presets Pro-Q4/Saturn 2/HG2/J37 — sinon ils seront proposés comme candidats
"universels" par erreur.

## Workflow recommandé pour peupler la bibliothèque

1. Dans chaque plugin, créer plusieurs variantes couvrant une plage de
   réglages (ex: EQ "compense dark", EQ "compense bright", EQ "de-ess
   léger", Saturation "light/medium/heavy", Tape "subtle/pushed").
2. Exporter chaque variante en `.aupreset` (menu preset du plugin dans le
   AU Validator ou dans ta DAW hôte) et déposer dans le bon sous-dossier.
3. Lancer `python3 tools/inspect_aupreset.py chemin/vers/preset.aupreset`
   pour vérifier que le fichier est bien identifié (manufacturer/type/
   subtype décodés) et voir si des paramètres lisibles apparaissent dans
   le blob.
4. Écrire le `.meta.yaml` associé avec les tags pertinents.
