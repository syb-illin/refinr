"""
Hosting réel des plugins Audio Unit via l'API C AudioToolbox, appelée
directement en `ctypes`.

⚠️ MACOS UNIQUEMENT. Ce module ne peut pas être exécuté ni testé dans un
environnement Linux — il n'a pas pu être validé en exécution réelle pendant
son écriture (pas de macOS disponible ici).

## Pourquoi ctypes et pas PyObjC/AVFoundation

Une première version de ce module s'appuyait sur `AVAudioEngine` en mode
"manual rendering" via PyObjC/AVFoundation, avec un package
`pyobjc-framework-AudioToolbox` pour le type `AudioComponentDescription`.
Ce package **n'existe pas** : AudioToolbox et AudioUnit sont des API C pures
et ne sont PAS wrappées par PyObjC (vérifié dans la doc officielle PyObjC,
table "Overview of macOS frameworks and their wrappers" — AudioToolbox et
AudioUnit y sont listés avec un wrapper Python "-", c'est-à-dire aucun). Le
premier build a échoué à `pip install` pour cette raison.

La bonne approche, et en réalité la plus directe : `AudioComponentFindNext`,
`AudioComponentInstanceNew`, `AudioUnitInitialize`, `AudioUnitSetProperty`,
`AudioUnitRender` sont une API C stable (inchangée depuis macOS 10.7+) que
`ctypes` (bibliothèque standard, zéro dépendance) appelle nativement. PyObjC
n'est conservé que pour UNE chose : convertir le dict Python du preset
(`.aupreset` chargé via `plistlib`) en `NSDictionary`, seul format accepté
par `AudioUnitSetProperty(kAudioUnitProperty_ClassInfo, ...)` pour restaurer
l'état du plugin — `Foundation.NSDictionary` + `.__c_void_p__()` (API PyObjC
documentée pour l'interop ctypes) donnent le pointeur CFDictionaryRef à
passer tel quel.

## Validation

Les constantes numériques ci-dessous (property IDs, format flags, tailles
de struct) viennent des headers Apple `AudioUnit/AUComponent.h` et
`CoreAudioTypes.h`, stables depuis longtemps, mais n'ont pas pu être
exécutées ici. **Étape 1 sur ta machine : lance
`tools/au_host_smoketest.py`** (utilise l'AU système Apple `AULowpass`,
donc sans dépendre de tes plugins commerciaux) pour valider la mécanique
avant de brancher FabFilter/Softube/Waves. En cas d'erreur `OSStatus`,
le code numérique retourné (souvent un FourCC lisible, ex: `-10863` =
`kAudioUnitErr_InvalidProperty`) aide à localiser le problème précis.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import struct
import sys

import numpy as np

if sys.platform != "darwin":
    raise ImportError(
        "refinr.au_host nécessite macOS (ctypes sur AudioToolbox.framework + PyObjC Foundation). "
        "Les autres modules (loudness, analysis, preset_mapping, batch, report) "
        "n'ont pas cette contrainte et tournent sur n'importe quel OS."
    )

from Foundation import NSDictionary  # noqa: E402

from .audio_io import AudioBuffer  # noqa: E402
from .preset_types import PluginPreset  # noqa: E402

# --------------------------------------------------------------------------
# Chargement de la librairie AudioToolbox
# --------------------------------------------------------------------------

_audiotoolbox_path = "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"
AudioToolbox = ctypes.CDLL(_audiotoolbox_path)

# --------------------------------------------------------------------------
# Constantes (depuis AUComponent.h / AudioComponent.h / CoreAudioTypes.h)
# --------------------------------------------------------------------------

kAudioUnitType_Effect = None  # renseigné dynamiquement via four_char_code("aufx") plus bas

kAudioUnitProperty_ClassInfo = 0
kAudioUnitProperty_StreamFormat = 8
kAudioUnitProperty_Latency = 12
kAudioUnitProperty_MaximumFramesPerSlice = 14
kAudioUnitProperty_SetRenderCallback = 23

kAudioUnitScope_Global = 0
kAudioUnitScope_Input = 1
kAudioUnitScope_Output = 2

kAudioFormatLinearPCM = None  # four_char_code("lpcm")

kAudioFormatFlagIsFloat = 1 << 0
kAudioFormatFlagIsPacked = 1 << 3
kAudioFormatFlagIsNonInterleaved = 1 << 5

kAudioUnitRenderAction_OutputIsSilence = 1 << 4

MAX_FRAMES_PER_SLICE = 4096
TAIL_SECONDS = 1.0  # flush de silence en fin de chaîne pour laisser sortir la queue (delay/reverb) des plugins
N_CHANNELS = 2  # tout le pipeline travaille en stéréo (voir AudioBuffer.as_stereo())


def four_char_code(code: str) -> int:
    """Convertit un OSType 4 caractères ('aufx', 'FabF', ...) en entier."""
    raw = code.encode("mac_roman")
    if len(raw) != 4:
        raise ValueError(f"OSType invalide (doit faire 4 caractères): {code!r}")
    return struct.unpack(">I", raw)[0]


def _osstatus_repr(status: int) -> str:
    """Affiche un OSStatus à la fois en décimal et en FourCC lisible si possible."""
    try:
        raw = struct.pack(">i", status)
        chars = raw.decode("mac_roman")
        if chars.isprintable():
            return f"{status} ('{chars}')"
    except (struct.error, UnicodeDecodeError):
        pass
    return str(status)


def _check(status: int, context: str) -> None:
    if status != 0:
        raise RuntimeError(f"{context} a échoué avec OSStatus {_osstatus_repr(status)}")


kAudioUnitType_Effect = four_char_code("aufx")
kAudioFormatLinearPCM = four_char_code("lpcm")

# --------------------------------------------------------------------------
# Structures C (ctypes) — layouts stables depuis macOS 10.7+
# --------------------------------------------------------------------------


class AudioComponentDescription(ctypes.Structure):
    _fields_ = [
        ("componentType", ctypes.c_uint32),
        ("componentSubType", ctypes.c_uint32),
        ("componentManufacturer", ctypes.c_uint32),
        ("componentFlags", ctypes.c_uint32),
        ("componentFlagsMask", ctypes.c_uint32),
    ]


class AudioStreamBasicDescription(ctypes.Structure):
    _fields_ = [
        ("mSampleRate", ctypes.c_double),
        ("mFormatID", ctypes.c_uint32),
        ("mFormatFlags", ctypes.c_uint32),
        ("mBytesPerPacket", ctypes.c_uint32),
        ("mFramesPerPacket", ctypes.c_uint32),
        ("mBytesPerFrame", ctypes.c_uint32),
        ("mChannelsPerFrame", ctypes.c_uint32),
        ("mBitsPerChannel", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


class AudioBufferStruct(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


class AudioBufferListStereo(ctypes.Structure):
    """AudioBufferList à taille fixe (2 buffers) — le pipeline est toujours stéréo."""

    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers", AudioBufferStruct * N_CHANNELS),
    ]


class AudioTimeStamp(ctypes.Structure):
    _fields_ = [
        ("mSampleTime", ctypes.c_double),
        ("mHostTime", ctypes.c_uint64),
        ("mRateScalar", ctypes.c_double),
        ("mWordClockTime", ctypes.c_uint64),
        ("mSMPTETime", ctypes.c_byte * 18),  # struct SMPTETime, contenu non utilisé ici
        ("mFlags", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


kAudioTimeStampSampleTimeValid = 1 << 0

AURenderCallback = ctypes.CFUNCTYPE(
    ctypes.c_int32,  # OSStatus
    ctypes.c_void_p,  # inRefCon
    ctypes.POINTER(ctypes.c_uint32),  # ioActionFlags
    ctypes.POINTER(AudioTimeStamp),  # inTimeStamp
    ctypes.c_uint32,  # inBusNumber
    ctypes.c_uint32,  # inNumberFrames
    ctypes.POINTER(AudioBufferListStereo),  # ioData
)


class AURenderCallbackStruct(ctypes.Structure):
    _fields_ = [
        ("inputProc", AURenderCallback),
        ("inputProcRefCon", ctypes.c_void_p),
    ]


# --------------------------------------------------------------------------
# Signatures des fonctions AudioToolbox utilisées
# --------------------------------------------------------------------------

AudioToolbox.AudioComponentFindNext.restype = ctypes.c_void_p
AudioToolbox.AudioComponentFindNext.argtypes = [ctypes.c_void_p, ctypes.POINTER(AudioComponentDescription)]

AudioToolbox.AudioComponentInstanceNew.restype = ctypes.c_int32
AudioToolbox.AudioComponentInstanceNew.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

AudioToolbox.AudioComponentInstanceDispose.restype = ctypes.c_int32
AudioToolbox.AudioComponentInstanceDispose.argtypes = [ctypes.c_void_p]

AudioToolbox.AudioUnitInitialize.restype = ctypes.c_int32
AudioToolbox.AudioUnitInitialize.argtypes = [ctypes.c_void_p]

AudioToolbox.AudioUnitUninitialize.restype = ctypes.c_int32
AudioToolbox.AudioUnitUninitialize.argtypes = [ctypes.c_void_p]

AudioToolbox.AudioUnitSetProperty.restype = ctypes.c_int32
AudioToolbox.AudioUnitSetProperty.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint32,
]

# AudioUnitGetProperty (utilisé pour lire kAudioUnitProperty_Latency, voir
# _instantiate_au / la compensation de délai plus bas) — signature distincte
# de Set : ioDataSize est un POINTEUR (in/out), pas une valeur simple.
AudioToolbox.AudioUnitGetProperty.restype = ctypes.c_int32
AudioToolbox.AudioUnitGetProperty.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
]

AudioToolbox.AudioUnitRender.restype = ctypes.c_int32
AudioToolbox.AudioUnitRender.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(AudioTimeStamp),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(AudioBufferListStereo),
]


def _make_stream_format(sample_rate: float) -> AudioStreamBasicDescription:
    """Float32 non-interleaved (format canonique Audio Unit), stéréo."""
    fmt = AudioStreamBasicDescription()
    fmt.mSampleRate = float(sample_rate)
    fmt.mFormatID = kAudioFormatLinearPCM
    fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved
    fmt.mBytesPerPacket = 4
    fmt.mFramesPerPacket = 1
    fmt.mBytesPerFrame = 4
    fmt.mChannelsPerFrame = N_CHANNELS
    fmt.mBitsPerChannel = 32
    fmt.mReserved = 0
    return fmt


class _SourceFeeder:
    """
    Fournit les données source à un Audio Unit via un AURenderCallback :
    l'AU "tire" (pull) les frames dont il a besoin en rappelant cette
    fonction. On expose directement des pointeurs vers nos tableaux numpy
    (zero-copy) tant qu'il reste de la donnée réelle, puis du silence pour
    le flush de queue (delay/reverb) en fin de traitement.
    """

    def __init__(self, left: np.ndarray, right: np.ndarray, tail_frames: int):
        # Tableaux contigus, indépendants par canal (non-interleaved).
        self._left = np.ascontiguousarray(left, dtype=np.float32)
        self._right = np.ascontiguousarray(right, dtype=np.float32)
        self._real_frames = self._left.shape[0]
        self._total_frames = self._real_frames + tail_frames
        self._silence = np.zeros(MAX_FRAMES_PER_SLICE, dtype=np.float32)
        self._position = 0
        self._callback = AURenderCallback(self._render)

    @property
    def callback(self) -> AURenderCallback:
        return self._callback

    @property
    def exhausted(self) -> bool:
        return self._position >= self._total_frames

    def _render(self, in_ref_con, io_action_flags, in_time_stamp, in_bus_number, in_number_frames, io_data):
        n = int(in_number_frames)
        start = self._position
        end = min(start + n, self._total_frames)
        n_available = max(0, end - start)

        buffers = io_data.contents.mBuffers
        for ch, arr in enumerate((self._left, self._right)):
            n_real = max(0, min(n_available, self._real_frames - start))
            if n_real > 0:
                src_ptr = arr[start : start + n_real].ctypes.data_as(ctypes.c_void_p)
                ctypes.memmove(buffers[ch].mData, src_ptr, n_real * 4)
            if n_real < n:
                # complète avec du silence (queue au-delà de la source, ou padding de fin de bloc)
                pad_ptr = ctypes.cast(buffers[ch].mData, ctypes.POINTER(ctypes.c_float))
                for i in range(n_real, n):
                    pad_ptr[i] = 0.0
            buffers[ch].mDataByteSize = n * 4

        self._position = min(start + n, self._total_frames)
        if n_available == 0 and io_action_flags:
            io_action_flags[0] |= kAudioUnitRenderAction_OutputIsSilence
        return 0  # noErr


@dataclasses.dataclass
class _LoadedAU:
    instance: ctypes.c_void_p
    preset_name: str
    latency_seconds: float = 0.0


def _query_latency_seconds(instance: ctypes.c_void_p, preset_name: str) -> float:
    """
    Lit `kAudioUnitProperty_Latency` (délai algorithmique du plugin, en
    secondes — lookahead d'un limiteur/désesseur, groupe de retard d'un EQ
    linear-phase, etc.) APRÈS `AudioUnitInitialize` (la valeur n'est fiable
    qu'une fois le plugin initialisé avec son stream format final).

    Sans compensation de ce délai (PDC — "plugin delay compensation", ce que
    fait automatiquement tout host pro type Logic/Pro Tools), le signal en
    sortie de CHAQUE étage de la chaîne serait décalé vers l'avant de ce
    montant, et l'effet se cumule sur toute la chaîne (EQ -> saturation ->
    tape) — silence de tête artificiel, désynchronisation par rapport à la
    source. Voir le trim appliqué dans `_process_single_au`.

    Dégradation gracieuse à 0.0 si la propriété n'est pas lisible (certains
    plugins ne l'exposent pas correctement) : pas pire que le comportement
    d'avant l'introduction de la PDC, jamais une erreur bloquante pour ça.
    """
    latency_value = ctypes.c_double(0.0)
    latency_size = ctypes.c_uint32(ctypes.sizeof(latency_value))
    status = AudioToolbox.AudioUnitGetProperty(
        instance,
        kAudioUnitProperty_Latency,
        kAudioUnitScope_Global,
        0,
        ctypes.byref(latency_value),
        ctypes.byref(latency_size),
    )
    if status != 0:
        return 0.0
    value = float(latency_value.value)
    return value if value > 0.0 and np.isfinite(value) else 0.0


def _instantiate_au(preset: PluginPreset, sample_rate: float) -> _LoadedAU:
    desc = AudioComponentDescription(
        componentType=(
            four_char_code(preset.component_type) if len(preset.component_type) == 4 else kAudioUnitType_Effect
        ),
        componentSubType=four_char_code(preset.component_subtype),
        componentManufacturer=four_char_code(preset.component_manufacturer),
        componentFlags=0,
        componentFlagsMask=0,
    )

    component = AudioToolbox.AudioComponentFindNext(None, ctypes.byref(desc))
    if not component:
        raise RuntimeError(
            f"AU introuvable pour preset {preset.name!r} "
            f"(type={preset.component_type}, subtype={preset.component_subtype}, "
            f"manufacturer={preset.component_manufacturer}). Le plugin est-il bien installé "
            f"et scanné par le système (`auval -a` pour lister les AU disponibles) ?"
        )

    instance = ctypes.c_void_p()
    _check(
        AudioToolbox.AudioComponentInstanceNew(component, ctypes.byref(instance)),
        f"AudioComponentInstanceNew({preset.name!r})",
    )

    stream_format = _make_stream_format(sample_rate)
    for scope in (kAudioUnitScope_Input, kAudioUnitScope_Output):
        _check(
            AudioToolbox.AudioUnitSetProperty(
                instance,
                kAudioUnitProperty_StreamFormat,
                scope,
                0,
                ctypes.byref(stream_format),
                ctypes.sizeof(stream_format),
            ),
            f"AudioUnitSetProperty(StreamFormat, scope={scope}) sur {preset.name!r}",
        )

    max_frames = ctypes.c_uint32(MAX_FRAMES_PER_SLICE)
    _check(
        AudioToolbox.AudioUnitSetProperty(
            instance,
            kAudioUnitProperty_MaximumFramesPerSlice,
            kAudioUnitScope_Global,
            0,
            ctypes.byref(max_frames),
            ctypes.sizeof(max_frames),
        ),
        f"AudioUnitSetProperty(MaximumFramesPerSlice) sur {preset.name!r}",
    )

    if preset.full_state:
        # AudioUnitSetProperty(ClassInfo) attend un CFDictionaryRef : on
        # passe par NSDictionary (toll-free bridgé à CFDictionary) et son
        # pointeur brut via l'API PyObjC/ctypes documentée __c_void_p__().
        ns_dict = NSDictionary.dictionaryWithDictionary_(preset.full_state)
        ptr = ctypes.c_void_p(ns_dict.__c_void_p__())
        _check(
            AudioToolbox.AudioUnitSetProperty(
                instance,
                kAudioUnitProperty_ClassInfo,
                kAudioUnitScope_Global,
                0,
                ctypes.byref(ptr),
                ctypes.sizeof(ptr),
            ),
            f"AudioUnitSetProperty(ClassInfo) sur {preset.name!r} — état du preset non restauré",
        )

    _check(AudioToolbox.AudioUnitInitialize(instance), f"AudioUnitInitialize({preset.name!r})")

    latency_seconds = _query_latency_seconds(instance, preset.name)

    return _LoadedAU(instance=instance, preset_name=preset.name, latency_seconds=latency_seconds)


def _dispose_au(au: _LoadedAU) -> None:
    AudioToolbox.AudioUnitUninitialize(au.instance)
    AudioToolbox.AudioComponentInstanceDispose(au.instance)


def _process_single_au(
    preset: PluginPreset, left: np.ndarray, right: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fait passer un buffer stéréo (2 tableaux mono float32) à travers un
    seul AU. Retourne (left, right, latency_samples_compensés) — voir la PDC
    appliquée juste avant le `return` ci-dessous."""
    tail_frames = int(TAIL_SECONDS * sample_rate)
    feeder = _SourceFeeder(left, right, tail_frames)

    au = _instantiate_au(preset, sample_rate)
    try:
        callback_struct = AURenderCallbackStruct(inputProc=feeder.callback, inputProcRefCon=None)
        _check(
            AudioToolbox.AudioUnitSetProperty(
                au.instance,
                kAudioUnitProperty_SetRenderCallback,
                kAudioUnitScope_Input,
                0,
                ctypes.byref(callback_struct),
                ctypes.sizeof(callback_struct),
            ),
            f"AudioUnitSetProperty(SetRenderCallback) sur {preset.name!r}",
        )

        total_frames = feeder._total_frames
        out_left = np.zeros(total_frames, dtype=np.float32)
        out_right = np.zeros(total_frames, dtype=np.float32)

        out_buf_l = np.zeros(MAX_FRAMES_PER_SLICE, dtype=np.float32)
        out_buf_r = np.zeros(MAX_FRAMES_PER_SLICE, dtype=np.float32)

        buffer_list = AudioBufferListStereo()
        buffer_list.mNumberBuffers = N_CHANNELS
        buffer_list.mBuffers[0].mNumberChannels = 1
        buffer_list.mBuffers[0].mData = out_buf_l.ctypes.data_as(ctypes.c_void_p)
        buffer_list.mBuffers[1].mNumberChannels = 1
        buffer_list.mBuffers[1].mData = out_buf_r.ctypes.data_as(ctypes.c_void_p)

        timestamp = AudioTimeStamp()
        timestamp.mFlags = kAudioTimeStampSampleTimeValid

        position = 0
        sample_time = 0.0
        while position < total_frames:
            chunk = min(MAX_FRAMES_PER_SLICE, total_frames - position)
            buffer_list.mBuffers[0].mDataByteSize = chunk * 4
            buffer_list.mBuffers[1].mDataByteSize = chunk * 4
            timestamp.mSampleTime = sample_time

            flags = ctypes.c_uint32(0)
            status = AudioToolbox.AudioUnitRender(
                au.instance,
                ctypes.byref(flags),
                ctypes.byref(timestamp),
                0,
                chunk,
                ctypes.byref(buffer_list),
            )
            _check(status, f"AudioUnitRender sur {preset.name!r} (position={position})")

            out_left[position : position + chunk] = out_buf_l[:chunk]
            out_right[position : position + chunk] = out_buf_r[:chunk]

            position += chunk
            sample_time += chunk

        # --- PDC (plugin delay compensation) ---------------------------------
        # `au.latency_seconds` (kAudioUnitProperty_Latency, lu juste après
        # AudioUnitInitialize — voir _query_latency_seconds) est le délai
        # algorithmique introduit par CE plugin. Sans compensation, les
        # `latency_samples` premiers échantillons de sortie sont du
        # pré-silence/pré-ring dû au plugin, pas du vrai signal — on les
        # retire pour réaligner la sortie sur l'entrée, exactement ce que
        # fait la PDC automatique de tout host pro (Logic, Pro Tools). Le
        # `TAIL_SECONDS` généreux (1s, largement > toute latence de plugin de
        # mastering réaliste) absorbe cette perte côté fin de buffer sans
        # jamais couper de vrai signal.
        latency_samples = int(round(au.latency_seconds * sample_rate))
        if latency_samples > 0:
            out_left = out_left[latency_samples:]
            out_right = out_right[latency_samples:]

        return out_left, out_right, latency_samples
    finally:
        _dispose_au(au)


@dataclasses.dataclass
class RenderResult:
    frames_rendered: int
    latency_compensated_samples: int = 0  # somme des délais PDC retirés sur toute la chaîne (voir _process_single_au)


def process_chain_offline(
    buffer: AudioBuffer, presets_in_order: list[PluginPreset]
) -> tuple[AudioBuffer, RenderResult]:
    """
    Fait passer `buffer` (déjà gain-stagé) à travers la chaîne d'Audio Units
    donnée, dans l'ordre, chacun en pull via AURenderCallback (offline,
    aucune dépendance à un device de sortie). Chaque AU ajoute jusqu'à
    `TAIL_SECONDS` de flush de queue au signal, et sa PDC (plugin delay
    compensation, voir `_process_single_au`) est appliquée avant de passer
    au plugin suivant — sans quoi le délai algorithmique de chaque étage se
    cumulerait sur toute la chaîne (silence de tête artificiel croissant).
    """
    stereo = buffer.as_stereo()
    left, right = stereo[:, 0].copy(), stereo[:, 1].copy()

    total_latency_samples = 0
    for preset in presets_in_order:
        left, right, latency_samples = _process_single_au(preset, left, right, buffer.sample_rate)
        total_latency_samples += latency_samples

    out_samples = np.stack([left, right], axis=1).astype(np.float32)
    out_buffer = AudioBuffer(samples=out_samples, sample_rate=buffer.sample_rate, source_path=buffer.source_path)
    return out_buffer, RenderResult(
        frames_rendered=out_samples.shape[0], latency_compensated_samples=total_latency_samples
    )
