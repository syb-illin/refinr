<div align="center">

# 🎚️ Refinr

**Retraitement audio automatisé pour fichiers WAV bruts — via tes vrais plugins Audio Unit**

Gain staging → EQ → Saturation → Tape → Leveling par profil de destination, spécifique à chaque fichier, jamais générique.

[![Build](https://github.com/syb-illin/refinr/actions/workflows/build-macos-app.yml/badge.svg)](https://github.com/syb-illin/refinr/actions/workflows/build-macos-app.yml)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-black?logo=apple&logoColor=white)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt&logoColor=white)
![Bridge](https://img.shields.io/badge/audio%20engine-ctypes%20%2F%20AudioToolbox-0A84FF)
![Version](https://img.shields.io/badge/version-0.1.1-orange)
![Status](https://img.shields.io/badge/status-en%20d%C3%A9veloppement-yellow)

**Repo :** [github.com/syb-illin/refinr](https://github.com/syb-illin/refinr)

</div>

---

## Sommaire

- [Le concept](#le-concept)
- [Pourquoi Python + PyObjC, pas Swift/Xcode](#pourquoi-python--pyobjc-pas-swiftxcode)
- [État du projet](#état-du-projet--ce-qui-a-été-validé-où)
- [Installation ultra-rapide](#installation-ultra-rapide-build-automatique--app-packagée)
- [Build 100% automatique via GitHub Actions](#build-100-automatique-via-github-actions-sans-toucher-ton-mac)
- [Installation manuelle](#installation-manuelle-si-tu-préfères-contrôler-chaque-étape)
- [Peupler la bibliothèque de presets](#étape-2--peupler-la-bibliothèque-de-presets)
- [Traiter des fichiers](#étape-3--traiter-des-fichiers)
- [Profils de destination](#profils-de-destination)
- [Tests](#tests)
- [Structure du projet](#structure-du-projet)

---

## Le concept

Beaucoup de sources exportent des WAV chauds, souvent proches de
l'écrêtage, avec un rendu générique côté timbre et dynamique. Refinr
reprend chaque fichier individuellement :

| Étape | Ce qui se passe |
|---|---|
| 🎚️ **Gain staging** | Ramène le niveau à −18 LUFS avant tout traitement, pour ne pas exploser les plugins en aval |
| 🔍 **Analyse** | Spectre, dynamique, écrêtage, largeur stéréo — par fichier, jamais un preset unique pour tout un dossier |
| 🎛️ **EQ** | FabFilter Pro-Q4, preset choisi selon l'analyse |
| 🔥 **Saturation** | Softube Saturn 2 ou BackBox HG2 |
| 📼 **Tape** | Waves J37 |
| 📊 **Leveling** | Vers le profil cible (Spotify, YouTube, Apple Music, DistroKid...) avec limiteur true-peak |
| 📄 **Rapport** | JSON + HTML détaillé par fichier, batch parallèle |

---

## Pourquoi Python + ctypes, pas Swift/Xcode

Hoster des Audio Units nécessite l'API C **AudioToolbox** (Audio Component
Manager : recherche, instanciation, rendu). Cette API n'est PAS wrappée par
PyObjC (vérifié dans la doc officielle PyObjC — AudioToolbox/AudioUnit n'ont
aucun wrapper Python), donc `refinr/au_host.py` l'appelle directement via
**`ctypes`** (bibliothèque standard, zéro dépendance). **PyObjC** n'intervient
que pour une seule chose : convertir le dict du preset en `NSDictionary`, seul
format accepté pour restaurer l'état d'un plugin (`kAudioUnitProperty_ClassInfo`).
Seul prérequis : les **Xcode Command Line Tools** (`xcode-select --install`,
~500 Mo, gratuit — pas besoin d'ouvrir Xcode.app), nécessaires pour compiler
le bridge PyObjC et packager l'app.

---

## État du projet / ce qui a été validé où

Ce projet a été écrit et testé dans un environnement **Linux** (pas de
macOS disponible pendant le développement).

| Module | Rôle | Testé ici (Linux) |
|---|---|:---:|
| `refinr/audio_io.py` | lecture/écriture WAV | ✅ |
| `refinr/loudness.py` | mesure LUFS/true peak (BS.1770), gain staging, limiteur | ✅ |
| `refinr/analysis.py` | analyse spectrale/dynamique par fichier | ✅ |
| `refinr/preset_mapping.py` | sélection de presets par fichier via tags | ✅ |
| `refinr/chain.py` | orchestration complète (AU court-circuité hors macOS) | ✅ |
| `refinr/batch.py` | parallélisme + agrégation | ✅ |
| `refinr/report.py` | rapport JSON + HTML | ✅ |
| `tools/inspect_aupreset.py` | inspection de `.aupreset` | ✅ (plistlib pur, cross-OS) |
| **`refinr/au_host.py`** | **hosting AU réel (ctypes + AudioToolbox)** | ❌ **macOS requis — à valider chez toi** |
| **`gui/*.py`** | interface PyQt6 | ⚠️ syntaxe vérifiée, non exécutée |

<details>
<summary><strong>Détail : pourquoi <code>au_host.py</code> n'est pas garanti à 100% du premier coup</strong></summary>

<br>

`au_host.py` appelle l'API C AudioToolbox (`AudioComponentFindNext`,
`AudioComponentInstanceNew`, `AudioUnitInitialize`, `AudioUnitSetProperty`,
`AudioUnitRender`) directement via `ctypes`, avec des structures et
constantes (IDs de propriété, flags de format) tirées des headers Apple
`AUComponent.h` / `CoreAudioTypes.h` — stables depuis longtemps, mais je
n'ai pas pu les exécuter pendant l'écriture (pas de macOS disponible ici).

**Étape 1 sur ta machine :** lance `tools/au_host_smoketest.py` (utilise
l'AU système Apple `AULowpass`, donc sans dépendre de tes plugins
commerciaux) pour isoler d'éventuels problèmes bas niveau avant de
brancher FabFilter / Softube / Waves.

En cas d'erreur, `au_host.py` lève une `RuntimeError` avec le code
`OSStatus` retourné par l'appel qui a échoué (décodé en FourCC lisible
quand c'est possible, ex: `-10863 ('...')`) — utile pour chercher l'erreur
précise dans la doc Apple ou m'indiquer où corriger.

</details>

---

## Installation ultra-rapide (build automatique + app packagée)

Une seule commande gère tout : venv, dépendances, tests, build `.app`,
versioning et zip prêt à archiver/déplacer dans `/Applications`.

```bash
cd refinr
./build_and_package.sh
```

→ produit `releases/Refinr_v0.1.1_<date-heure>.zip`. Relance ce même
script à chaque fois que le code change (nouvelle version dans
`refinr/__init__.py` → nom de zip différent, historique conservé dans
`releases/`).

> Ceci ne dispense pas de valider le hosting AU (étape ci-dessous) : le
> build automatique ne peut pas remplacer un test réel de tes plugins,
> seulement compiler l'app.

---

## Build 100% automatique via GitHub Actions (sans toucher ton Mac)

Alternative à `build_and_package.sh` : un push déclenche le build sur un
runner macOS distant (Apple Silicon natif), et dépose le zip prêt à
télécharger dans les **Releases** GitHub. Zéro commande locale après le
push initial.

Repo déjà créé : **https://github.com/syb-illin/refinr**

<details open>
<summary><strong>Mise en route (à faire une fois)</strong></summary>

<br>

`main` et le tag `v0.1.0` sont déjà poussés. Pour les builds suivants :
change `__version__` dans `refinr/__init__.py`, commit, tag
(`git tag v0.1.1 && git push --tags`) — nouvelle Release automatique. Le
bouton **"Run workflow"** (onglet Actions) permet aussi un build ponctuel
sans tag (artifact téléchargeable 90 jours, pas de Release).

Je n'ai ni `gh` (pas installable : le réseau de mon environnement
d'exécution bloque le téléchargement du binaire) ni `api.github.com`
accessible pour suivre l'avancement d'un build — seul `git push`/`git tag`
en HTTPS avec un token passe par le proxy autorisé. Si tu me donnes un
Personal Access Token (scope `repo` + `workflow`, idéalement limité à ce
seul repo, collé dans le chat), je peux pousser et taguer à ta place ;
pour suivre le résultat du build (logs, statut), c'est à vérifier de ton
côté sur https://github.com/syb-illin/refinr/actions — je ne peux pas le
lire moi-même.

</details>

<details>
<summary><strong>Coût & signature — à lire avant de pousser</strong></summary>

<br>

**Coût :** GitHub Actions est gratuit et illimité pour les repos publics.
En repo privé, les minutes macOS consomment le quota à un multiplicateur
×10 (un build de ~5 min facture ~50 min de quota) — reste gérable pour un
usage perso occasionnel, à surveiller si builds fréquents.

**Signature :** le workflow signe l'app en "ad-hoc" (`codesign --sign -`),
suffisant pour qu'elle *lance* sur Apple Silicon, mais pas notarisée par
Apple (ça nécessite un compte développeur payant, 99 $/an). Au premier
lancement, macOS Gatekeeper affichera un avertissement "développeur non
identifié" — clic droit sur l'app → **Ouvrir**, une seule fois.

</details>

---

## Installation manuelle (si tu préfères contrôler chaque étape)

```bash
xcode-select --install        # une seule fois, si pas déjà fait

cd refinr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Étape 1 — valider le hosting AU

```bash
python3 tools/au_host_smoketest.py un_fichier_test.wav
```

## Étape 2 — peupler la bibliothèque de presets

1. Dans chaque plugin, crée plusieurs variantes couvrant une plage de
   réglages (voir `config/presets/README.md` pour la convention complète
   et des exemples de `.meta.yaml`).
2. Exporte chaque variante en `.aupreset`, dépose dans
   `config/presets/{eq,saturation,tape}/`.
3. Inspecte chaque fichier :
   ```bash
   python3 tools/inspect_aupreset.py config/presets/eq/mon_preset.aupreset --dump-blob
   ```
4. Écris le `.meta.yaml` associé pour indiquer quand ce preset doit être
   choisi automatiquement.
5. **Supprime les fichiers de démo factices** dans `config/presets/eq/`
   (voir `config/presets/README.md`).

<details>
<summary><strong>⚠️ Limite technique importante : presets = state opaque, pas paramètres individuels</strong></summary>

<br>

FabFilter/Softube/Waves encodent l'état de leurs plugins dans un blob
binaire propriétaire (clé `data` du `.aupreset`), pas comme des paramètres
AU nommés individuellement exposés. Concrètement, l'automatisation :

- **peut** choisir, pour chaque fichier, le **meilleur preset parmi ceux
  que tu as fournis** (sélection discrète, basée sur les tags d'analyse) ;
- **ne peut pas**, en l'état, interpoler en continu un paramètre précis
  (ex : "descends le gain de la bande 3 de l'EQ de 2 dB de plus si le
  fichier est très brillant") sans reverse-engineering du format binaire
  spécifique à chaque plugin (faisable via diff binaire — voir
  `--dump-blob` — mais pas fait ici, faute de vrais exports à disposition).

Donc plus tu fournis de variantes de presets couvrant une plage fine de
réglages (ex : EQ "dark -1dB", "dark -2dB", "dark -3dB"...), plus la
sélection automatique sera précise. C'est le modèle sur lequel tout le
pipeline (`preset_mapping.py`) est construit.

</details>

## Étape 3 — traiter des fichiers

**CLI** (pratique pour scripter) :
```bash
python3 tools/run_batch_cli.py --profile spotify --workers 6 *.wav
```

**GUI** :
```bash
python3 gui/app.py
```
Glisser-déposer des WAV, voir la preview des presets choisis par fichier,
choisir le profil de destination, lancer le batch, ouvrir le rapport HTML.

---

## Profils de destination

Définis dans `config/destination_profiles.yaml` :

| Profil | LUFS cible | True peak |
|---|:---:|:---:|
| Spotify | −14 | −1 dBTP |
| YouTube / YouTube Music | −14 | −1 dBTP |
| Apple Music | −16 | −1 dBTP |
| Amazon Music | −14 | −2 dBTP |
| TIDAL | −14 | −1 dBTP |
| Deezer | −15 | −1 dBTP |
| SoundCloud | −14 | −1 dBTP |
| DistroKid (générique) | −14 | −1 dBTP |
| Édit club/DJ (loud) | −9 | −0.3 dBTP |

Chiffres à jour à l'écriture de ce projet (2026) — à re-vérifier
périodiquement, les plateformes changent leurs specs de temps en temps.

---

## Tests

```bash
python3 -m pytest tests/ -v
```
Couvre tout sauf `au_host.py` et le GUI (macOS requis — voir tableau plus
haut). Les fixtures audio de test sont générées automatiquement au premier
lancement, via `tests/conftest.py`.

---

## Structure du projet

```
refinr/
  refinr/
    audio_io.py        lecture/écriture WAV
    loudness.py         BS.1770, gain staging, limiteur true-peak
    analysis.py          features spectrales/dynamiques par fichier
    preset_types.py      PluginPreset, chargement .aupreset (plistlib)
    preset_mapping.py    bibliothèque + sélection de presets par fichier
    au_host.py            hosting AU réel (ctypes + AudioToolbox) — macOS only
    chain.py               orchestration complète par fichier
    batch.py                parallélisation (ProcessPoolExecutor)
    report.py                rapport JSON + HTML
    profiles.py               profils de destination
  gui/
    app.py, main_window.py, worker.py    interface PyQt6
  tools/
    inspect_aupreset.py         inspecteur de .aupreset
    au_host_smoketest.py         validation du hosting AU sur un AU système
    run_batch_cli.py              CLI de batch
  config/
    destination_profiles.yaml
    presets/{eq,saturation,tape}/
  tests/
    conftest.py, generate_test_audio.py, test_*.py
  .github/workflows/
    build-macos-app.yml     CI : build + release automatique sur tag
  build_and_package.sh      build local en une commande
  setup.py                  packaging py2app
```

<div align="center">

---

Fait pour un usage personnel • Aucune notarisation Apple • Aucune donnée envoyée nulle part

</div>
