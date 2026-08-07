"""
Packaging en .app via py2app (macOS uniquement).

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python3 setup.py py2app

L'app générée sera dans dist/Refinr.app
"""

from setuptools import setup

APP = ["gui/app.py"]
DATA_FILES = [
    ("config", ["config/destination_profiles.yaml"]),
]
OPTIONS = {
    "argv_emulation": False,
    "packages": ["refinr", "gui"],
    "includes": [
        "Foundation",
    ],
    "plist": {
        "CFBundleName": "Refinr",
        "CFBundleDisplayName": "Refinr",
        "CFBundleIdentifier": "local.refinr",
        "CFBundleShortVersionString": "0.1.4",
        "NSHumanReadableCopyright": "Usage personnel",
        # Nécessaire si l'app doit un jour lire des fichiers audio via
        # microphone/entrées ; pour du traitement offline de fichiers WAV
        # déjà présents sur disque, pas de permission particulière requise.
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
