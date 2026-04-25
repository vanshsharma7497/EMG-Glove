"""
agent/context_agent.py
-----------------------
Context-aware gesture remapping agent for the EMG Smart Glove.

── Detection pipeline (three tiers, fastest first) ───────────────────────────

  Tier 1 — OS rules (free, ~0ms, offline)
    OSContextDetector asks Windows which app is in front.
    context_rules.get_rules_for_app() looks up pre-built gesture remappings.
    If a rule exists → apply instantly. No screenshot. No network call.
    Covers: youtube, netflix, vscode, github, claude, spotify, discord, etc.

  Tier 2 — Vision API cache (free on repeat, online on first visit)
    If the app has no built-in rule, check _vision_cache keyed by app_name.
    Cache hit  → apply cached mappings from a previous API call. No new call.
    Cache miss → fall through to Tier 3.

  Tier 3 — Nvidia NIM vision API (costs API credits, ~1–2s latency)
    Takes a screenshot, sends to Llama 3.2 90B Vision, parses JSON response.
    Result cached in _vision_cache[app_name] so this app is free on next poll.
    If API fails → existing mappings stay intact (graceful fallback).

── Public attributes (read by main.py dashboard) ─────────────────────────────
  .active           bool   — False if NVIDIA_API_KEY not set
  .context_detected str    — human-readable description of current context
  .app_name         str    — clean key like "youtube", "vscode", "unknown"
  .source           str    — "rules" | "cache" | "vision" | "fallback"
  .confidence       float  — 0.0–1.0, meaningful only when source="vision"

── Fallback behaviour ────────────────────────────────────────────────────────
OS rules work without any API key — the agent is now useful even in fallback
mode. If NVIDIA_API_KEY is not set, Tier 1 and 2 still run; only Tier 3
(vision API) is disabled. get_action() always returns instantly.

── Requirements ──────────────────────────────────────────────────────────────
Optional — only needed for Tier 3 (unknown apps):
    $env:NVIDIA_API_KEY = "nvapi-..."  (Windows PowerShell)

── Called by ─────────────────────────────────────────────────────────────────
  modes/mouse_mode.py  — calls get_action() every gesture event
  main.py              — calls start(), stop(), reads .active / .context_detected
"""

import os
import io
import json
import time
import base64
import threading

from openai import OpenAI

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from core.emg_simulator   import GESTURE_PROFILES
from modes.mouse_mode      import GESTURE_ACTIONS
from agent.os_context      import OSContextDetector      # Tier 1: OS detection
from agent.context_rules   import get_rules_for_app      # Tier 1: rules lookup

# ── Constants ─────────────────────────────────────────────────────────────────
POLL_INTERVAL = 2.0       # seconds between screenshot + API calls
MODEL = "meta/llama-3.2-90b-vision-instruct"
MAX_TOKENS    = 1024        # generous enough for a JSON mapping object

# Pre-built strings used in every prompt — computed once at import time
_GESTURE_NAMES   = ", ".join(GESTURE_PROFILES.keys())
_DEFAULT_ACTIONS = json.dumps(
    {k: v for k, v in GESTURE_ACTIONS.items()},
    indent=2
)


class ContextAgent:
    """
    Daemon thread that polls every 2 seconds and keeps gesture remappings
    up to date using a three-tier pipeline: OS rules → vision cache → vision API.

    Usage:
        agent = ContextAgent()
        agent.start()

        # Every gesture event (instant — no network call):
        action = agent.get_action("fist")   # returns str or None

        # On shutdown:
        agent.stop()

    Public state (safe to read from any thread at any time):
        .active           — False if NVIDIA_API_KEY not set
        .context_detected — human-readable string shown in the dashboard
        .app_name         — clean key: "youtube", "vscode", "unknown", etc.
        .source           — which tier resolved the context last
        .confidence       — meaningful only when source == "vision"
    """

    def __init__(self):
        # ── Shared state — written by background thread, read by get_action() ─
        self._current_mappings  = {}       # gesture_name → action_name (str)
        self._lock              = threading.Lock()

        # ── Public attributes — safe to read from any thread ──────────────────
        self.context_detected   = "No context yet"
        self.app_name           = "unknown"    # clean key from os_context
        self.source             = "fallback"   # "rules"|"cache"|"vision"|"fallback"
        self.confidence         = 0.0
        self.user_rules         = {}           # loaded from glove_config.json by main.py
        self._running           = False

        # ── Tier 1: OS context detector — works with or without an API key ────
        # Start it regardless of API key — OS rules are free and offline.
        self._os_detector = OSContextDetector()

        # ── Tier 2: vision API result cache — keyed by app_name ──────────────
        # Stores the parsed JSON result from a successful vision API call so
        # the same app never triggers a second API call in the same session.
        self._vision_cache = {}   # app_name → {"mappings": {...}, "context_detected": ...}

        # ── Tier 3: Nvidia NIM vision API — optional ──────────────────────────
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            self.active  = False
            self._client = None
            print("  [Agent] No NVIDIA_API_KEY — vision API disabled")
            print("  [Agent] OS rules and cache still active")
        else:
            self._client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
            self.active = True
            print("  [Agent] ContextAgent initialised — NVIDIA_API_KEY found")

        # Note: .active now means "vision API available", not "agent active".
        # The agent always runs — OS rules work without any key.


    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """
        Start the background polling thread and the OS detector.
        Always runs — OS rules work without an API key.
        """
        # ── Start OS detector first ───────────────────────────────────────────
        self._os_detector.start()

        # ── Start the polling thread ──────────────────────────────────────────
        self._running = True
        thread = threading.Thread(
            target=self._polling_loop,
            name="ContextAgent-Poller",
            daemon=True,           # killed automatically when main thread exits
        )
        thread.start()
        api_status = "vision API active" if self.active else "OS rules only"
        print(f"  [Agent] Context agent started — {api_status}, polling every 2s")


    def stop(self):
        """Signal the polling loop and OS detector to exit."""
        self._running = False
        self._os_detector.stop()


    def get_action(self, gesture_name):
        """
        Returns the agent-suggested action for a gesture, or None.

        This is called every gesture event from the real-time loop.
        It reads from an in-memory dict under a lock — always instant.
        Never raises an exception.

        gesture_name : str  — e.g. "fist", "victory"
        Returns      : str or None  — e.g. "seek_forward", or None for fallback
        """
        try:
            with self._lock:
                return self._current_mappings.get(gesture_name)
        except Exception:
            return None


    # ── Background polling loop ───────────────────────────────────────────────

    def _polling_loop(self):
        """
        Daemon thread body. Sleeps POLL_INTERVAL seconds then calls _update().
        All exceptions are caught — a crash here must never reach the main thread.
        """
        while self._running:
            time.sleep(POLL_INTERVAL)
            if not self._running:
                break
            try:
                self._update()
            except Exception as e:
                print(f"  [Agent] Polling error (will retry): {e}")


    def _update(self):
        """
        Called every POLL_INTERVAL seconds by the polling loop.
        Runs the three-tier detection pipeline and updates shared mappings.
        If anything fails, existing mappings are left intact.
        """

        # ── Tier 1: OS detection + rules lookup ───────────────────────────────
        # Ask Windows what's in front. Instant — reads from OSContextDetector's
        # cached result (the detector runs its own 0.5s poll in the background).
        os_ctx   = self._os_detector.get_context()
        app_name = os_ctx.get("app_name", "unknown")

        # Look up pre-built rules for this app. Returns a dict or None.
        # user_rules is populated by main.py when a reload_rules command arrives.
        rules = get_rules_for_app(app_name, user_rules=self.user_rules)

        if rules is not None:
            # ── Rule found — apply instantly, skip all API calls ──────────────
            n = len(rules)
            label = os_ctx.get("url") or os_ctx.get("title", app_name)
            label = label[:50]   # cap length for terminal display

            with self._lock:
                self._current_mappings = rules
                self.context_detected  = f"{app_name} — {label}"
                self.app_name          = app_name
                self.source            = "rules"
                self.confidence        = 1.0   # rules are deterministic

            print(f"  [Agent] [{app_name}]  {n} rule{'s' if n != 1 else ''}"
                  f"  ({label})")
            return   # ← done, no screenshot needed

        # ── Tier 2: vision API cache ──────────────────────────────────────────
        # Unknown app — check if we've seen it before this session.
        if app_name in self._vision_cache:
            cached = self._vision_cache[app_name]
            new_mappings = cached.get("mappings", {})

            with self._lock:
                self._current_mappings = new_mappings
                self.context_detected  = str(cached.get("context_detected",
                                                         app_name))
                self.app_name          = app_name
                self.source            = "cache"
                self.confidence        = float(cached.get("confidence", 0.0))

            n = len(new_mappings)
            print(f"  [Agent] [{app_name}]  {n} mapping{'s' if n != 1 else ''}"
                  f"  (cached vision result)")
            return   # ← done, used cached result

        # ── Tier 3: vision API — only for unknown apps not yet cached ─────────
        if not self.active:
            # No API key — nothing more to try. Leave existing mappings.
            # Update app_name so the dashboard shows the correct context
            # even when we can't provide remappings for this app.
            with self._lock:
                self.app_name         = app_name
                self.source           = "fallback"
                self.context_detected = f"{app_name} (no rules, no API key)"
            return

        import pyautogui   # imported here to avoid cost when not needed

        try:
            # ── Screenshot → base64 ───────────────────────────────────────────
            screenshot = pyautogui.screenshot()
            buf        = io.BytesIO()
            screenshot.save(buf, format="PNG")
            image_b64  = base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"  [Agent] Screenshot failed: {e}")
            return

        # ── Query vision model ────────────────────────────────────────────────
        result = self._query_vision_model(image_b64)
        if result is None:
            return   # leave existing mappings intact on API failure

        # ── Cache the result for this app ─────────────────────────────────────
        # Next time this app is seen, Tier 2 will handle it instantly.
        self._vision_cache[app_name] = result

        # ── Update shared state under lock ────────────────────────────────────
        new_mappings = result.get("mappings", {})
        if not isinstance(new_mappings, dict):
            new_mappings = {}

        with self._lock:
            self._current_mappings = new_mappings
            self.context_detected  = str(result.get("context_detected", "Unknown"))
            self.app_name          = app_name
            self.source            = "vision"
            self.confidence        = float(result.get("confidence", 0.0))

        n = len(new_mappings)
        print(f"  [Agent] [{app_name}]  vision: \"{self.context_detected}\""
              f"  ({self.confidence*100:.0f}% conf)  "
              f"{n} remapping{'s' if n != 1 else ''}  (cached for session)")


    # ── Nvidia NIM API call ───────────────────────────────────────────────────

    def _query_vision_model(self, image_base64):
        """
        Sends a screenshot to the Nvidia NIM vision model and receives a JSON
        remapping object.

        image_base64 : str — base64-encoded PNG screenshot

        Returns a parsed dict on success, or None on any error.
        The returned dict has keys: context_detected, confidence, mappings.
        """
        user_text = (
            "Analyse this screenshot and identify:\n"
            "1. What application is currently active\n"
            "2. What UI element or content area appears to be in focus\n"
            "3. What task the user most likely appears to be doing\n\n"
            f"Available gestures: {_GESTURE_NAMES}\n\n"
            f"Default mappings for reference:\n{_DEFAULT_ACTIONS}\n\n"
            "Return a JSON object with EXACTLY this structure:\n"
            "{\n"
            '  "context_detected": "brief description of what you see",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "mappings": {\n'
            '    "gesture_name": "action_name"\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- Only include gestures in mappings where you are changing the "
            "default — omit gestures where the default is fine\n"
            "- Never remap to destructive actions without confirmation "
            "(delete, shutdown, close all, format)\n"
            "- Prefer actions that are commonly used in this specific application\n"
            "- If you cannot determine context confidently (confidence below 0.4), "
            "return empty mappings {}\n"
            "- action_name should be a short snake_case string describing the "
            'action (e.g. seek_forward, fullscreen_toggle, go_to_definition)'
        )

        try:
            response = self._client.chat.completions.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                messages   = [
                    {
                        "role"   : "system",
                        "content": (
                            "You are a gesture remapping agent for a "
                            "smart glove input device. Return ONLY valid "
                            "JSON, no explanation."
                        ),
                    },
                    {
                        "role"   : "user",
                        "content": [
                            {
                                "type"     : "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": user_text,
                            },
                        ],
                    },
                ],
            )

            # ── Parse response ────────────────────────────────────────────────
            raw_text = response.choices[0].message.content
            print(f"  [Agent] Raw response: {repr(raw_text)}")

            # Guard against empty or None response before attempting to parse
            if not raw_text or not raw_text.strip():
                print("  [Agent] Empty response from model — skipping")
                return None

            raw_text = raw_text.strip()

            # ── Robust JSON extraction ────────────────────────────────────────
            # The model sometimes adds preamble text despite being told not to.
            # Examples seen in the wild:
            #   "**Answer:**\n\n{...}"     ← markdown bold preamble
            #   "```json\n{...}\n```"      ← fenced code block
            #   "Here is the JSON:\n{...}" ← plain English preamble
            #
            # Strategy: find the first '{' and last '}' in the response and
            # extract everything between them. This is robust to any preamble
            # or postamble the model decides to add.
            brace_start = raw_text.find("{")
            brace_end   = raw_text.rfind("}")

            if brace_start == -1 or brace_end == -1:
                print("  [Agent] No JSON object found in response — skipping")
                return None

            json_text = raw_text[brace_start : brace_end + 1]
            return json.loads(json_text)

        except json.JSONDecodeError as e:
            print(f"  [Agent] JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"  [Agent] API call failed: {e}")
            return None