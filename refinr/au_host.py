"""
Hosting réel des plugins Audio Unit via PyObjC + AVFoundation/AudioToolbox.

⚠️ MACOS UNIQUEMENT. Ce module ne peut pas être exécuté ni testé dans un
environnement Linux — il n'a pas pu être validé en exécution réelle pendant
son écriture (pas de macOS disponible ici). Le pattern utilisé
(AVAudioEngine en "manual rendering mode" pour un rendu offline,
AUAudioUnit.fullState pour recharger un .aupreset) est le pattern standard
documenté par Apple, mais certains noms de sélecteurs PyObjC exacts peuvent
nécessiter un ajustement mineur une fois testés sur ta machine — voir
`tools/au_host_smoketest.py` pour valider la mécanique avec un AU système
(AULowpass) AVANT de brancher tes plugins commerciaux.

En cas d'AttributeError sur un sélecteur : lance un `python3` interactif,
importe le module concerné (ex: `from AVFoundation import AVAudioEngine`)
et fais `[m for m in dir(AVAudioEngine) if "render" in m.lower()]` pour
retrouver le nom exact généré par le bridge PyObjC installé.
"""

from __future__ import annotations

import dataclasses
import struct
import sys
import threading
import time

if sys.platform != "darwin":
    raise ImportError(
        "refinr.au_host nécessite macOS (PyObjC + AVFoundation/AudioToolbox). "
        "Les autres modules (loudness, analysis, preset_mapping, batch, report) "
        "n'ont pas cette contrainte et tournent sur n'importe quel OS."
    )

from AVFoundation import (  # noqa: E402
    AVAudioEngine,
    AVAudioEngineManualRenderingModeOffline,
    AVAudioFile,
    AVAudioFormat,
    AVAudioPCMBuffer,
    AVAudioPlayerNode,
    AVAudioUnit,
)
from AudioToolbox import (  # noqa: E402
    AudioComponentDescription,
    kAudioUnitType_Effect,
)
from Foundation import NSURL, NSRunLoop, NSDate  # noqa: E402

from .preset_types import PluginPreset

MAX_FRAME_COUNT = 4096


def four_char_code(code: str) -> int:
    """Convertit un OSType 4 caractères ('aufx', 'FabF', ...) en entier."""
    raw = code.encode("mac_roman")
    if len(raw) != 4:
        raise ValueError(f"OSType invalide (doit faire 4 caractères): {code!r}")
    return struct.unpack(">I", raw)[0]


def _pump_run_loop_until(predicate, timeout_s: float = 15.0, interval_s: float = 0.01) -> None:
    """
    Attend qu'un callback async (instantiate AVAudioUnit) se déclenche, en
    faisant tourner le run loop courant par petits paliers. Nécessaire car
    ce script n'a pas de boucle d'événements Cocoa (pas de NSApplication).
    """
    deadline = time.monotonic() + timeout_s
    run_loop = NSRunLoop.currentRunLoop()
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("Timeout en attendant le chargement asynchrone de l'Audio Unit.")
        run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(interval_s))


def instantiate_au(preset: PluginPreset) -> AVAudioUnit:
    """
    Instancie l'Audio Unit correspondant à `preset` (via son manufacturer/
    subtype décodés) et lui applique l'état du preset (fullState).
    """
    desc = AudioComponentDescription(
        componentType=four_char_code(preset.component_type) if len(preset.component_type) == 4 else kAudioUnitType_Effect,
        componentSubType=four_char_code(preset.component_subtype),
        componentManufacturer=four_char_code(preset.component_manufacturer),
        componentFlags=0,
        componentFlagsMask=0,
    )

    result: dict = {"unit": None, "error": None, "done": False}

    def _completion(audio_unit, error):
        result["unit"] = audio_unit
        result["error"] = error
        result["done"] = True

    AVAudioUnit.instantiateWithComponentDescription_options_completionHandler_(desc, 0, _completion)
    _pump_run_loop_until(lambda: result["done"])

    if result["error"] is not None:
        raise RuntimeError(f"Échec instanciation AU pour preset {preset.name!r}: {result['error']}")

    audio_unit = result["unit"]
    if audio_unit is None:
        raise RuntimeError(
            f"AU introuvable pour preset {preset.name!r} "
            f"(type={preset.component_type}, subtype={preset.component_subtype}, "
            f"manufacturer={preset.component_manufacturer}). Le plugin est-il bien installé "
            f"et scanné par le système (AU Validator / `auval`) ?"
        )

    au_unit = audio_unit.AUAudioUnit()
    if preset.full_state:
        au_unit.setFullState_(preset.full_state)
    return audio_unit


@dataclasses.dataclass
class RenderResult:
    output_path: str
    frames_rendered: int


def process_chain_offline(
    input_path: str,
    output_path: str,
    presets_in_order: list[PluginPreset],
) -> RenderResult:
    """
    Charge `input_path`, fait passer le signal à travers la chaîne d'Audio
    Units (dans l'ordre donné), et écrit le résultat dans `output_path`.

    Rendu offline via AVAudioEngine en "manual rendering mode" (pas de
    lecture temps réel, pas de dépendance à un device de sortie).
    """
    input_url = NSURL.fileURLWithPath_(input_path)
    input_file = AVAudioFile.alloc().initForReading_error_(input_url, None)
    if input_file is None:
        raise RuntimeError(f"Impossible d'ouvrir le fichier d'entrée: {input_path}")

    process_format = input_file.processingFormat()
    total_frames = int(input_file.length())

    engine = AVAudioEngine.alloc().init()
    player = AVAudioPlayerNode.alloc().init()
    engine.attachNode_(player)

    au_nodes = [instantiate_au(p) for p in presets_in_order]
    for au in au_nodes:
        engine.attachNode_(au)

    chain_nodes = [player] + au_nodes
    for src, dst in zip(chain_nodes[:-1], chain_nodes[1:]):
        engine.connect_to_format_(src, dst, process_format)

    last_node = chain_nodes[-1]
    engine.connect_to_format_(last_node, engine.outputNode(), process_format)

    ok, error = engine.enableManualRenderingMode_format_maximumFrameCount_error_(
        AVAudioEngineManualRenderingModeOffline, process_format, MAX_FRAME_COUNT, None
    )
    if not ok:
        raise RuntimeError(f"enableManualRenderingMode a échoué: {error}")

    started, start_error = engine.startAndReturnError_(None)
    if not started:
        raise RuntimeError(f"engine.startAndReturnError_ a échoué: {start_error}")

    player.scheduleFile_atTime_completionHandler_(input_file, None, None)
    player.play()

    output_url = NSURL.fileURLWithPath_(output_path)
    output_settings = process_format.settings()
    output_file = AVAudioFile.alloc().initForWriting_settings_error_(output_url, output_settings, None)
    if output_file is None:
        raise RuntimeError(f"Impossible de créer le fichier de sortie: {output_path}")

    render_format = engine.manualRenderingFormat()
    max_frames = engine.manualRenderingMaximumFrameCount()

    frames_rendered = 0
    STATUS_ERROR = 2  # AVAudioEngineManualRenderingStatusError

    while frames_rendered < total_frames:
        remaining = total_frames - frames_rendered
        frames_this_pass = min(max_frames, remaining)
        buffer = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(render_format, frames_this_pass)
        status, render_error = engine.renderOffline_toBuffer_error_(frames_this_pass, buffer, None)

        if status == STATUS_ERROR:
            raise RuntimeError(f"renderOffline a échoué: {render_error}")

        rendered_now = int(buffer.frameLength())
        if rendered_now == 0:
            break

        output_file.writeFromBuffer_error_(buffer, None)
        frames_rendered += rendered_now

    engine.stop()
    return RenderResult(output_path=output_path, frames_rendered=frames_rendered)
