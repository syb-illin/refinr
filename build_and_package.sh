#!/usr/bin/env bash
#
# Build & package automatique — À LANCER SUR macOS (Apple Silicon).
#
# Une seule commande gère tout : venv, dépendances, build .app (py2app),
# versioning, et zip final prêt à archiver/partager.
#
#   ./build_and_package.sh
#
# Sortie : releases/Refinr_vX.Y.Z_YYYYMMDD-HHMM.zip
#
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "❌ Ce script doit tourner sur macOS (build .app via py2app)." >&2
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "❌ Xcode Command Line Tools introuvables. Lance: xcode-select --install" >&2
  exit 1
fi

VERSION=$(python3 -c "import sys; sys.path.insert(0, '.'); from refinr import __version__; print(__version__)")
TIMESTAMP=$(date +%Y%m%d-%H%M)
RELEASES_DIR="releases"
APP_NAME="Refinr"
ZIP_NAME="Refinr_v${VERSION}_${TIMESTAMP}.zip"

echo "=== Refinr — build v${VERSION} (${TIMESTAMP}) ==="

echo "[1/6] Environnement virtuel..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/6] Dépendances (peut prendre quelques minutes la 1ère fois, compilation PyObjC)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[3/6] Tests rapides (hors hosting AU réel, qui nécessite un run manuel du smoketest)..."
python3 -m pytest tests/ -q

echo "[4/6] Nettoyage des anciens builds..."
rm -rf build dist

echo "[5/6] Build de l'app (py2app)..."
python3 setup.py py2app --quiet

if [[ ! -d "dist/${APP_NAME}.app" ]]; then
  echo "❌ Le build a échoué : dist/${APP_NAME}.app introuvable." >&2
  exit 1
fi

echo "[6/7] Signature ad-hoc (requise pour lancer un binaire arm64 non notarié)..."
codesign --force --deep --sign - "dist/${APP_NAME}.app"

echo "[7/7] Packaging versionné..."
mkdir -p "${RELEASES_DIR}"
(cd dist && zip -r -q "../${RELEASES_DIR}/${ZIP_NAME}" "${APP_NAME}.app")

echo ""
echo "✅ Terminé : ${RELEASES_DIR}/${ZIP_NAME}"
echo "   Décompresse et déplace '${APP_NAME}.app' dans /Applications."
echo "   1er lancement : clic droit -> Ouvrir (Gatekeeper bloque par défaut un"
echo "   app non notarié par Apple ; ceci ne se produit qu'une fois)."
echo ""
echo "⚠️  Rappel : le hosting AU réel (FabFilter/Saturn2/HG2/J37) n'a pas pu être"
echo "   testé avant livraison (pas de macOS disponible côté build). Avant de"
echo "   traiter de vrais fichiers, lance une fois :"
echo "     python3 tools/au_host_smoketest.py un_fichier_test.wav"
echo "   et signale-moi toute erreur pour que je corrige au_host.py."
