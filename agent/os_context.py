"""
agent/os_context.py
-------------------
OS-native context detector for the EMG Smart Glove agent layer.

Queries Windows every 0.5 seconds to determine which application the user
is currently looking at — no screenshot, no network, no API key required.

── What it returns ────────────────────────────────────────────────────────────
A structured dict:
    {
        "process":  "chrome.exe",
        "title":    "YouTube — Google Chrome",
        "url":      "youtube.com",     # "" if not a browser
        "app_name": "youtube",         # clean key used by context_rules.py
    }

app_name is what context_rules.py looks up to find gesture remappings.
It is derived by matching process name, window title, and URL against a
priority-ordered list of known patterns.

── Three detection layers ─────────────────────────────────────────────────────
Layer 1 — Process name (win32gui + win32process + psutil)
    Always available. Gives "chrome.exe", "Code.exe", "vlc.exe" etc.

Layer 2 — Window title (win32gui.GetWindowText)
    Always available. Gives "YouTube — Google Chrome", "main.py - GLOVE" etc.
    Sufficient to distinguish many apps without Layer 3.

Layer 3 — Browser tab URL (uiautomation accessibility API)
    Only attempted for known browsers (Chrome, Edge, Firefox).
    Reads the address bar via Windows UI Automation.
    Falls back gracefully if accessibility is blocked or the bar is empty.

── Graceful degradation ───────────────────────────────────────────────────────
Every exception is caught at the method level.
If Layer 3 fails  → url = "", app_name derived from title/process only.
If Layer 2 fails  → title = "", app_name derived from process only.
If Layer 1 fails  → returns last known context (never None, never crashes).

── Dependencies ───────────────────────────────────────────────────────────────
    pip install pywin32       # win32gui, win32process (likely already installed)
    pip install psutil        # process name from PID
    pip install uiautomation  # browser URL reading via accessibility API

── Standalone usage ───────────────────────────────────────────────────────────
    python agent/os_context.py
    Prints detected context every 0.5 seconds. Switch apps and watch it update.

── Called by ──────────────────────────────────────────────────────────────────
    agent/context_agent.py — calls get_context() in _update() before API call
"""

import time
import threading

# ── Windows API imports ───────────────────────────────────────────────────────
# Wrapped in try/except so the file can at least be imported on non-Windows
# systems (e.g. CI runners) without crashing at import time.
try:
    import win32gui
    import win32process
    import psutil
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False
    print("  [OSContext] pywin32 / psutil not found — install with:")
    print("    pip install pywin32 psutil")

# uiautomation is only needed for browser URL reading — import separately
try:
    import uiautomation as auto
    _UIAUTO_AVAILABLE = True
except ImportError:
    _UIAUTO_AVAILABLE = False
    # This is non-fatal — URL reading is disabled, everything else still works

# ── Poll interval ─────────────────────────────────────────────────────────────
POLL_INTERVAL = 0.5   # seconds between OS queries — fast and free

# ── Known browser process names ───────────────────────────────────────────────
# Used to decide when to attempt Layer 3 (URL reading).
BROWSER_PROCESSES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"}

# ── Process name → clean app name ─────────────────────────────────────────────
# Checked first. Unambiguous processes (e.g. Code.exe is always VS Code).
PROCESS_APP_MAP = {
    "Code.exe":          "vscode",
    "code.exe":          "vscode",       # some installs use lowercase
    "vlc.exe":           "vlc",
    "mpv.exe":           "mpv",
    "Spotify.exe":       "spotify",
    "spotify.exe":       "spotify",
    "Discord.exe":       "discord",
    "discord.exe":       "discord",
    "slack.exe":         "slack",
    "Slack.exe":         "slack",
    "zoom.exe":          "zoom",
    "Zoom.exe":          "zoom",
    "WINWORD.EXE":       "word",
    "EXCEL.EXE":         "excel",
    "POWERPNT.EXE":      "powerpoint",
    "notepad.exe":       "notepad",
    "Notepad.exe":       "notepad",
    "explorer.exe":      "file_explorer",
    "WindowsTerminal.exe": "terminal",
    "powershell.exe":    "terminal",
    "cmd.exe":           "terminal",
    "mspaint.exe":       "paint",
    "devenv.exe":        "visual_studio",   # Visual Studio (not VS Code)
    "pycharm64.exe":     "pycharm",
    "pycharm.exe":       "pycharm",
    "idea64.exe":        "intellij",
    "figma.exe":         "figma",
    "Figma.exe":         "figma",
    "obs64.exe":         "obs",
    "obs.exe":           "obs",
}

# ── URL substring → clean app name ────────────────────────────────────────────
# Checked when process is a browser. More specific strings listed first
# because matching stops at the first hit.
URL_APP_MAP = [
    ("music.youtube.com",    "youtube_music"),   # must be before youtube.com
    ("youtube.com",          "youtube"),
    ("docs.google.com",      "google_docs"),
    ("sheets.google.com",    "google_sheets"),
    ("slides.google.com",    "google_slides"),
    ("mail.google.com",      "gmail"),
    ("calendar.google.com",  "google_calendar"),
    ("drive.google.com",     "google_drive"),
    ("meet.google.com",      "google_meet"),
    ("github.com",           "github"),
    ("netflix.com",          "netflix"),
    ("twitch.tv",            "twitch"),
    ("open.spotify.com",     "spotify"),
    ("figma.com",            "figma"),
    ("notion.so",            "notion"),
    ("trello.com",           "trello"),
    ("linear.app",           "linear"),
    ("stackoverflow.com",    "stackoverflow"),
    ("reddit.com",           "reddit"),
    ("twitter.com",          "twitter"),
    ("x.com",                "twitter"),
    ("linkedin.com",         "linkedin"),
    ("claude.ai",            "claude"),
    ("chatgpt.com",          "chatgpt"),
]

# ── Window title keyword → clean app name ─────────────────────────────────────
# Fallback when URL is unavailable. Checks if keyword appears in title.
# Checked in order — first match wins.
TITLE_APP_MAP = [
    ("YouTube",        "youtube"),
    ("Google Docs",    "google_docs"),
    ("Google Sheets",  "google_sheets"),
    ("Google Slides",  "google_slides"),
    ("Gmail",          "gmail"),
    ("GitHub",         "github"),
    ("Netflix",        "netflix"),
    ("Twitch",         "twitch"),
    ("Figma",          "figma"),
    ("Notion",         "notion"),
    ("Visual Studio Code", "vscode"),    # title fallback for VS Code web
    ("Stack Overflow", "stackoverflow"),
    ("Reddit",         "reddit"),
    ("Claude",         "claude"),
    ("ChatGPT",        "chatgpt"),
    ("Twitter",        "twitter"),
    ("LinkedIn",       "linkedin"),
]


class OSContextDetector:
    """
    Queries Windows for the currently active application and returns a
    structured context dict every POLL_INTERVAL seconds.

    The detector runs as a daemon thread. External callers use get_context()
    which always returns instantly from a cached result — no blocking.

    Usage:
        detector = OSContextDetector()
        detector.start()

        # Call from anywhere — always instant:
        ctx = detector.get_context()
        print(ctx["app_name"])   # "youtube", "vscode", "unknown", etc.

        detector.stop()
    """

    def __init__(self):
        # Shared state — written by background thread, read by get_context()
        self._context = {
            "process":  "",
            "title":    "",
            "url":      "",
            "app_name": "unknown",
        }
        self._lock    = threading.Lock()
        self._running = False

        if not _WIN32_AVAILABLE:
            print("  [OSContext] Running in stub mode — win32 not available")
        if not _UIAUTO_AVAILABLE:
            print("  [OSContext] uiautomation not found — URL detection disabled")
            print("    Install with: pip install uiautomation")

        print("  [OSContext] OSContextDetector initialised")


    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background polling thread."""
        self._running = True
        thread = threading.Thread(
            target=self._polling_loop,
            name="OSContext-Poller",
            daemon=True,    # killed automatically when main process exits
        )
        thread.start()
        print(f"  [OSContext] Polling every {POLL_INTERVAL}s")


    def stop(self):
        """Signal the polling thread to exit."""
        self._running = False


    def get_context(self):
        """
        Returns the most recently detected context dict. Always instant.
        Never raises an exception.

        Returns a dict with keys: process, title, url, app_name.
        app_name is "unknown" if the app is not recognised.
        """
        with self._lock:
            return dict(self._context)   # return a copy so caller can't mutate it


    # ── Background polling loop ───────────────────────────────────────────────

    def _polling_loop(self):
        """Daemon thread body. All exceptions caught — must never crash."""
        while self._running:
            try:
                new_ctx = self._detect()
                if new_ctx is not None:
                    with self._lock:
                        self._context = new_ctx
            except Exception as e:
                print(f"  [OSContext] Poll error: {e}")
            time.sleep(POLL_INTERVAL)


    # ── Detection pipeline ────────────────────────────────────────────────────

    def _detect(self):
        """
        Runs the three-layer detection pipeline and returns a context dict.
        Returns None if Layer 1 completely fails (preserves last known context).
        """
        if not _WIN32_AVAILABLE:
            return None   # stub mode — leave context as-is

        # ── Layer 1: process name ─────────────────────────────────────────────
        process, hwnd = self._get_process_name()
        if process is None:
            return None   # window changed between calls — keep last context

        # ── Layer 2: window title ─────────────────────────────────────────────
        title = self._get_window_title(hwnd)

        # ── Layer 3: browser URL (only for browsers) ──────────────────────────
        # We pass 'title' so the method can anchor its accessibility search
        # to the specific window rather than searching the whole desktop.
        url = ""
        if process.lower() in BROWSER_PROCESSES:
            url = self._get_browser_url(process, title)

        # ── Derive app_name ───────────────────────────────────────────────────
        app_name = self._resolve_app_name(process, title, url)

        return {
            "process":  process,
            "title":    title,
            "url":      url,
            "app_name": app_name,
        }


    def _get_process_name(self):
        """
        Layer 1: get foreground window handle + process name.
        Returns (process_name, hwnd) or (None, None) on failure.
        """
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd == 0:
                return None, None   # no window in focus (desktop, lock screen)

            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid).name()
            return process, hwnd

        except Exception:
            return None, None


    def _get_window_title(self, hwnd):
        """
        Layer 2: window title text from the handle.
        Returns "" on failure.
        """
        try:
            return win32gui.GetWindowText(hwnd)
        except Exception:
            return ""


    def _get_browser_url(self, process_name, window_title=""):
        """
        Layer 3: read the active tab URL from the browser address bar via
        Windows UI Automation (accessibility API).

        Strategy — anchor to the specific window, not the desktop root:
          1. Find the foreground browser window by its exact title.
          2. Search for an Edit control inside that window only.
          3. Read its Value (the URL text).

        This is dramatically faster than searching from the desktop root
        because the accessibility tree is scoped to one window instead of
        the entire OS. We also set a short 1-second timeout so a miss fails
        fast rather than blocking for 10 seconds.

        window_title : str — the window title from Layer 2, used to anchor
                             the search. If empty, falls back to a rootless
                             search (slower but still works).

        Returns the URL string, or "" if anything fails.
        """
        if not _UIAUTO_AVAILABLE:
            return ""

        try:
            # ── Set a short timeout so misses fail fast ────────────────────────
            # uiautomation's default is 10 seconds — way too long for a 0.5s poll.
            # 1 second is enough: if the address bar exists, it's found instantly.
            auto.SetGlobalSearchTimeout(1.0)

            proc_lower = process_name.lower()

            # ── Anchor the search to the foreground browser window ─────────────
            # Finding the window by title scopes the accessibility tree search.
            # Without this, auto.EditControl() searches the entire desktop tree.
            if window_title:
                window = auto.WindowControl(
                    Name=window_title,
                    searchDepth=1,     # top-level windows only — very fast
                )
            else:
                window = auto.GetRootControl()   # fallback: search from desktop

            # ── Find the address bar Edit control inside the window ────────────
            # Chrome and Edge both use an Edit control for the address bar.
            # We search by ClassName rather than Name because the Name changes
            # when the bar is focused vs unfocused ("Address and search bar"
            # vs the current URL text) — ClassName is stable.
            if "chrome" in proc_lower or "brave" in proc_lower:
                # Chrome/Brave address bar ClassName is "OmniboxViewViews"
                bar = window.EditControl(ClassName="OmniboxViewViews")

            elif "msedge" in proc_lower:
                # Edge uses the same Chromium-based omnibox
                bar = window.EditControl(ClassName="OmniboxViewViews")

            elif "firefox" in proc_lower:
                # Firefox address bar is inside a toolbar
                bar = window.EditControl(ClassName="urlbar-input")

            else:
                return ""

            # ── Read the URL value ─────────────────────────────────────────────
            # GetValuePattern().Value reads whatever text is currently in the bar.
            # This is the URL when the tab is not being edited.
            url = bar.GetValuePattern().Value
            return url if url else ""

        except Exception:
            # Common failure reasons (all harmless):
            # - Browser window is animating / transitioning between tabs
            # - User is actively typing in the address bar (bar is focused/empty)
            # - ClassName changed in a browser update
            # - Accessibility APIs disabled by group policy
            return ""

        finally:
            # ── Always restore the global timeout ──────────────────────────────
            # Other uiautomation callers (if any) should get the default back.
            try:
                auto.SetGlobalSearchTimeout(10.0)
            except Exception:
                pass


    def _resolve_app_name(self, process, title, url):
        """
        Derives the clean app_name key from the three available signals.
        Priority: URL (most specific) → title keywords → process name.

        Returns a short snake_case string like "youtube", "vscode", "unknown".
        """
        # ── URL matching (browser only) ───────────────────────────────────────
        # More specific entries in URL_APP_MAP are listed first, so we check
        # "music.youtube.com" before "youtube.com" — first match wins.
        if url:
            url_lower = url.lower()
            for pattern, name in URL_APP_MAP:
                if pattern in url_lower:
                    return name

        # ── Process name exact match ──────────────────────────────────────────
        if process in PROCESS_APP_MAP:
            return PROCESS_APP_MAP[process]

        # ── Window title keyword match ────────────────────────────────────────
        if title:
            for keyword, name in TITLE_APP_MAP:
                if keyword.lower() in title.lower():
                    return name

        # ── Browser new tab — no URL, generic title ───────────────────────────
        # "New Tab" / "New tab" appears in Chrome, Edge, Brave when no page
        # is loaded. It's not "unknown" — it's a known state with no context.
        if process.lower() in BROWSER_PROCESSES:
            title_lower = title.lower()
            if "new tab" in title_lower or title_lower == "":
                return "browser_newtab"

        # ── Nothing matched ───────────────────────────────────────────────────
        return "unknown"


# ── Standalone test ───────────────────────────────────────────────────────────
# Run:  python agent/os_context.py
# Then switch between apps and watch the context update every 0.5 seconds.
# Press Ctrl+C to stop.

if __name__ == "__main__":
    print("\n  EMG Glove — OS Context Detector (standalone test)")
    print("  Switch between apps — context updates every 0.5 seconds")
    print("  Press Ctrl+C to stop\n")
    print(f"  {'PROCESS':<25} {'APP NAME':<20} {'URL / TITLE'}")
    print(f"  {'─'*25} {'─'*20} {'─'*40}")

    detector = OSContextDetector()
    detector.start()

    last_app = ""
    try:
        while True:
            ctx = detector.get_context()
            app = ctx["app_name"]

            # Only print when something changes — keeps output readable
            if app != last_app:
                url_or_title = ctx["url"] if ctx["url"] else ctx["title"][:40]
                print(
                    f"  {ctx['process']:<25} "
                    f"{app:<20} "
                    f"{url_or_title}"
                )
                last_app = app

            time.sleep(0.1)

    except KeyboardInterrupt:
        detector.stop()
        print("\n  Stopped.")