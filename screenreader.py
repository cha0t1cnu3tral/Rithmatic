import ctypes
import os
import platform
import time

try:
    from accessible_output2.outputs.auto import Auto
except Exception:  # Fallback when package missing or no provider
    Auto = None


class ScreenReader:
    def __init__(self):
        self._speaker = None
        self._nvda = None
        self._last_spoken = ""
        self._last_spoken_at = 0.0
        self._repeat_window = 0.2
        if Auto is not None:
            try:
                self._speaker = Auto()
            except Exception:
                self._speaker = None
        if self._speaker is None:
            self._nvda = _NVDABridge()
            if not self._nvda.available:
                self._nvda = None

    def speak(self, text, interrupt=True):
        if not text:
            return
        text = str(text).strip()
        if not text:
            return
        now = time.monotonic()
        if not interrupt:
            if (
                text == self._last_spoken
                and (now - self._last_spoken_at) < self._repeat_window
            ):
                return
        if self._speaker is not None:
            try:
                # interrupt parameter not supported by all outputs; ignore if it fails
                try:
                    self._speaker.output(text, interrupt=interrupt)
                except TypeError:
                    self._speaker.output(text)
                self._last_spoken = text
                self._last_spoken_at = now
                return
            except Exception:
                pass
        if self._nvda is not None:
            self._last_spoken = text
            self._last_spoken_at = now
            self._nvda.speak(text)


class _NVDABridge:
    def __init__(self):
        self.available = False
        self._dll = None
        dll_path = _find_nvda_dll()
        if not dll_path:
            return
        try:
            self._dll = ctypes.windll.LoadLibrary(dll_path)
            init = getattr(self._dll, "nvdaControllerClient_initialize", None)
            if init is None:
                return
            if init() != 0:
                return
            self.available = True
        except Exception:
            self._dll = None
            self.available = False

    def speak(self, text):
        if not self.available or self._dll is None:
            return
        try:
            speak_fn = getattr(self._dll, "nvdaControllerClient_speakText", None)
            if speak_fn is None:
                return
            speak_fn(ctypes.c_wchar_p(text))
        except Exception:
            pass


def _find_nvda_dll():
    env_path = os.environ.get("NVDA_CONTROLLER_DLL")
    if env_path and os.path.exists(env_path):
        return env_path
    pkg_path = _find_accessible_output_dll()
    if pkg_path:
        return pkg_path
    candidates = []
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if not base:
            continue
        candidates.append(os.path.join(base, "NVDA", "nvdaControllerClient64.dll"))
        candidates.append(os.path.join(base, "NVDA", "nvdaControllerClient32.dll"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_accessible_output_dll():
    try:
        import accessible_output2
    except Exception:
        return None
    base = os.path.dirname(accessible_output2.__file__)
    lib_dir = os.path.join(base, "lib")
    if not os.path.isdir(lib_dir):
        return None
    is_64 = "64" in platform.architecture()[0]
    name = "nvdaControllerClient64.dll" if is_64 else "nvdaControllerClient32.dll"
    path = os.path.join(lib_dir, name)
    if os.path.exists(path):
        return path
    # Fallback to any dll in lib directory
    for candidate in ("nvdaControllerClient64.dll", "nvdaControllerClient32.dll"):
        path = os.path.join(lib_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def announce(text, delay_ms=0):
    reader = ScreenReader()
    if delay_ms:
        time.sleep(delay_ms / 1000.0)
    reader.speak(text)
