"""
glove_app.py
------------
EMG Smart Glove desktop status application.

Runs independently of main.py — reads glove_status.json written by the
pipeline and displays live device state. A GUI crash never touches the
gesture pipeline.

── Panels (this version: Device Status only) ─────────────────────────────────
  Device Status   — mode, agent tier, detected app, last gesture, hardware

── How live data works ────────────────────────────────────────────────────────
  main.py writes glove_status.json every frame (~12Hz in simulator mode).
  glove_app.py polls it every 1 second via CTk's .after() scheduler.
  If the file is missing or its timestamp is >5s old, glove shows as offline.
  No sockets, no shared memory — just a file. Either process can restart
  independently without affecting the other.

── Run ────────────────────────────────────────────────────────────────────────
  python glove_app.py
  (main.py does not need to be running — app shows "OFFLINE" state cleanly)

── Dependencies ───────────────────────────────────────────────────────────────
  pip install customtkinter
"""

import os
import json
import time
import customtkinter as ctk

from agent.context_rules import get_rules_for_app   # system rules lookup
from modes.mouse_mode    import GESTURE_ACTIONS      # 16 gestures + defaults

# ── File paths — must match main.py ──────────────────────────────────────────
STATUS_FILE   = "glove_status.json"
COMMANDS_FILE = "glove_commands.json"   # GUI writes this; pipeline reads + deletes
CONFIG_FILE   = "glove_config.json"    # user gesture overrides (persisted)

# ── How long before we consider the pipeline offline ─────────────────────────
STALE_AFTER = 5.0   # seconds

# ── Poll interval for reading the status file ─────────────────────────────────
POLL_MS = 1000      # milliseconds — 1 second

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG     = "#0d0d0d"    # main background — near black
C_PANEL  = "#1a1a2e"    # panel / card background
C_BORDER = "#2a2a3e"    # subtle border
C_TEXT   = "#e8e8f0"    # primary text
C_DIM    = "#666680"    # secondary / label text
C_GREEN  = "#66bb6a"    # OS Rules — good / active
C_BLUE   = "#4fc3f7"    # Vision Cache / mode label
C_ORANGE = "#ffa726"    # Vision API / AI action
C_RED    = "#ef5350"    # offline / error
C_GREY   = "#555566"    # fallback / inactive

# Source tier → (display label, colour)
SOURCE_STYLE = {
    "rules":    ("OS RULES",      C_GREEN),
    "cache":    ("VISION CACHE",  C_BLUE),
    "vision":   ("VISION API",    C_ORANGE),
    "fallback": ("FALLBACK",      C_GREY),
}

# Device mode → display label
MODE_LABELS = {
    "standard":      "MOUSE MODE",
    "communication": "COMM MODE",
    "tv_remote":     "TV REMOTE",
}


# ══════════════════════════════════════════════════════════════════════════════
# Main application window
# ══════════════════════════════════════════════════════════════════════════════

class GloveApp(ctk.CTk):
    """
    Main application window. Builds the UI on init, then polls glove_status.json
    every POLL_MS milliseconds and updates all labels.
    """

    def __init__(self):
        super().__init__()

        # ── Window setup ──────────────────────────────────────────────────────
        self.title("EMG Smart Glove")
        self.geometry("520x700")
        self.minsize(480, 500)
        self.configure(fg_color=C_BG)
        self.resizable(True, True)

        # ── State ─────────────────────────────────────────────────────────────
        self._current_mode  = "standard"   # updated from glove_status.json each poll
        self._mode_btns     = {}           # populated by _build_mode_switcher()
        self._last_mapped_app = None       # tracks app changes to trigger table rebuild

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_header()
        self._build_app_strip()      # always-visible app name bar
        self._build_footer()         # packed side="bottom" first so it stays anchored

        # ── Scrollable body — all cards live inside this ──────────────────────
        self._scroll_body = ctk.CTkScrollableFrame(
            self,
            fg_color=C_BG,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_DIM,
        )
        self._scroll_body.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_connection_card()
        self._build_mode_card()
        self._build_mode_switcher()      # mode selector buttons
        self._build_gesture_mappings()   # per-app gesture mapping editor
        self._build_settings()           # sliders + switch — saved to glove_config.json
        self._build_agent_card()
        self._build_gesture_card()

        # ── Start polling ─────────────────────────────────────────────────────
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_header(self):
        """Top title bar."""
        header = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=0, height=56)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="EMG SMART GLOVE",
            font=ctk.CTkFont(family="Courier New", size=16, weight="bold"),
            text_color=C_TEXT,
        ).pack(side="left", padx=20)

        # Live clock — updates every second alongside the poll
        self._lbl_clock = ctk.CTkLabel(
            header,
            text="--:--:--",
            font=ctk.CTkFont(family="Courier New", size=13),
            text_color=C_DIM,
        )
        self._lbl_clock.pack(side="right", padx=20)

    def _build_app_strip(self):
        """
        Persistent app name strip — always visible below the header, never
        scrolls away. Shows the active application and detection source badge.
        """
        strip = ctk.CTkFrame(self, fg_color=C_BORDER, corner_radius=0,
                             border_width=0, height=40)
        strip.pack(fill="x", padx=0, pady=(1, 0))
        strip.pack_propagate(False)

        # "APP" dim label on the left
        ctk.CTkLabel(
            strip,
            text="APP",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C_DIM,
        ).pack(side="left", padx=(16, 8))

        # App name — updated by _show_live / _show_offline
        self._strip_app = ctk.CTkLabel(
            strip,
            text="—",
            font=ctk.CTkFont(family="Courier New", size=14, weight="bold"),
            text_color=C_TEXT,
        )
        self._strip_app.pack(side="left")

        # Source badge on the right — e.g. "OS RULES" / "VISION API"
        self._strip_src = ctk.CTkLabel(
            strip,
            text="",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=C_DIM,
        )
        self._strip_src.pack(side="right", padx=16)

    def _build_connection_card(self):
        """Hardware connection status strip."""
        card = self._card(label="CONNECTION")

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))

        # Coloured dot indicator
        self._dot = ctk.CTkLabel(
            row, text="●",
            font=ctk.CTkFont(size=18),
            text_color=C_RED,
            width=28,
        )
        self._dot.pack(side="left", padx=(0, 8))

        self._lbl_connection = ctk.CTkLabel(
            row,
            text="OFFLINE",
            font=ctk.CTkFont(family="Courier New", size=14, weight="bold"),
            text_color=C_RED,
        )
        self._lbl_connection.pack(side="left")

        self._lbl_hw_mode = ctk.CTkLabel(
            row,
            text="",
            font=ctk.CTkFont(family="Courier New", size=12),
            text_color=C_DIM,
        )
        self._lbl_hw_mode.pack(side="right", padx=4)

    def _build_mode_card(self):
        """Current device mode."""
        card = self._card(label="DEVICE MODE")

        self._lbl_mode = ctk.CTkLabel(
            card,
            text="—",
            font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
            text_color=C_BLUE,
        )
        self._lbl_mode.pack(pady=(4, 10))

    def _build_mode_switcher(self):
        """Mode selector — 4 buttons that write glove_commands.json on click."""
        card = self._card(label="MODE SWITCHER")

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", pady=(4, 8))

        # (label, mode key)  — None = future placeholder, non-clickable
        modes = [
            ("STANDARD",  "standard"),
            ("COMM",      "communication"),
            ("TV REMOTE", "tv_remote"),
            ("—",         None),   # future placeholder
        ]

        for label, mode in modes:
            is_placeholder = mode is None
            btn = ctk.CTkButton(
                btn_row,
                text=label,
                font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
                fg_color=C_BORDER,          # default dim; _refresh_mode_buttons sets active
                hover_color=C_BLUE if not is_placeholder else C_BORDER,
                text_color=C_DIM if is_placeholder else C_TEXT,
                corner_radius=6,
                height=32,
                state="disabled" if is_placeholder else "normal",
                command=(lambda m=mode: self._send_mode_command(m)) if not is_placeholder else None,
            )
            btn.pack(side="left", padx=(0, 6), expand=True, fill="x")
            if not is_placeholder:
                self._mode_btns[mode] = btn

        self._refresh_mode_buttons()   # apply initial highlight

    def _send_mode_command(self, mode):
        """Atomically write glove_commands.json so the pipeline switches mode."""
        try:
            payload = {"command": "set_mode", "value": mode}
            tmp = COMMANDS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, COMMANDS_FILE)   # atomic — matches write_status() in main.py
            print(f"  [GUI] Mode command sent → '{mode}'")
        except Exception as e:
            print(f"  [GUI] Failed to send mode command: {e}")

    def _refresh_mode_buttons(self):
        """Highlight the active mode button; dim all others."""
        for mode, btn in self._mode_btns.items():
            if mode == self._current_mode:
                btn.configure(fg_color=C_BLUE, text_color=C_BG)   # highlighted
            else:
                btn.configure(fg_color=C_BORDER, text_color=C_TEXT)  # inactive

    def _build_gesture_mappings(self):
        """
        Scrollable table showing all 16 gestures and their current actions
        for the active app. Refreshes automatically when the app changes.
        """
        card = self._card(label="GESTURE MAPPINGS")

        # Sub-header: shows which app's rules are currently displayed
        self._map_app_label = ctk.CTkLabel(
            card,
            text="App: —",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C_DIM,
        )
        self._map_app_label.pack(anchor="w", pady=(0, 4))

        # Scrollable inner table — fixed height so card doesn't dominate window
        self._map_scroll = ctk.CTkScrollableFrame(
            card,
            fg_color=C_BG,
            height=160,
            corner_radius=4,
        )
        self._map_scroll.pack(fill="x", pady=(0, 4))

        # Populate immediately with default (no-app) mappings
        self._refresh_gesture_mappings("unknown")

    def _refresh_gesture_mappings(self, app_name):
        """
        Rebuilds the gesture table rows for app_name.
        Colour coding:  orange = user override  |  green = system rule  |  dim = default
        Called when app_name changes (detected in _show_live) or after a save.
        """
        # ── Clear existing rows ───────────────────────────────────────────────
        for w in self._map_scroll.winfo_children():
            w.destroy()

        # ── Gather rules from all sources ─────────────────────────────────────
        user_rules     = self._load_user_rules()
        user_app_rules = user_rules.get(app_name, {}) if user_rules else {}
        merged_rules   = get_rules_for_app(app_name, user_rules=user_rules) or {}

        # ── One row per gesture ───────────────────────────────────────────────
        for gesture, default_action in GESTURE_ACTIONS.items():
            if gesture in user_app_rules:
                action = user_app_rules[gesture]
                color  = C_ORANGE   # user-customised override
            elif gesture in merged_rules:
                action = merged_rules[gesture]
                color  = C_GREEN    # system rule from context_rules.py
            else:
                action = default_action
                color  = C_DIM      # static default (or None → "—")

            display = action if action else "—"

            row = ctk.CTkFrame(self._map_scroll, fg_color="transparent")
            row.pack(fill="x", pady=1)

            ctk.CTkLabel(
                row,
                text=gesture,
                font=ctk.CTkFont(family="Courier New", size=10),
                text_color=C_TEXT,
                width=170,
                anchor="w",
            ).pack(side="left")

            ctk.CTkLabel(
                row,
                text=display,
                font=ctk.CTkFont(family="Courier New", size=10),
                text_color=color,
                anchor="w",
            ).pack(side="left", padx=(4, 0), expand=True, fill="x")

            ctk.CTkButton(
                row,
                text="Edit",
                font=ctk.CTkFont(family="Courier New", size=9),
                fg_color=C_BORDER,
                hover_color=C_BLUE,
                text_color=C_TEXT,
                corner_radius=4,
                height=22,
                width=44,
                command=lambda g=gesture, a=display, ap=app_name:
                    self._open_edit_dialog(g, a, ap),
            ).pack(side="right")

        self._map_app_label.configure(text=f"App: {app_name}")
        self._last_mapped_app = app_name

    def _load_user_rules(self):
        """Read user_rules from glove_config.json. Returns dict or {} on any error."""
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            return cfg.get("user_rules", {})
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _open_edit_dialog(self, gesture, current_action, app_name):
        """Open a CTkInputDialog for the user to type a new action name."""
        dialog = ctk.CTkInputDialog(
            title=f"Edit — {gesture}",
            text=(
                f"Gesture:  {gesture}\n"
                f"Current:  {current_action}\n\n"
                f"Enter new action (snake_case), or clear to remove override:"
            ),
        )
        new_action = dialog.get_input()
        if new_action is None:
            return   # user cancelled
        self._save_mapping(app_name, gesture, new_action.strip())

    def _save_mapping(self, app_name, gesture, new_action):
        """
        Persist a user override to glove_config.json (atomic write), then
        send a reload_rules command so the pipeline picks up the change.
        An empty new_action string removes the override for that gesture.
        """
        try:
            # ── Load existing config ──────────────────────────────────────────
            try:
                with open(CONFIG_FILE, "r") as f:
                    cfg = json.load(f)
            except FileNotFoundError:
                cfg = {}
            except Exception:
                cfg = {}

            user_rules = cfg.get("user_rules", {})
            if app_name not in user_rules:
                user_rules[app_name] = {}

            if new_action:
                user_rules[app_name][gesture] = new_action
            else:
                # Empty string — remove override so system rule or default takes over
                user_rules[app_name].pop(gesture, None)
                if not user_rules[app_name]:   # prune empty app dict
                    user_rules.pop(app_name, None)

            cfg["user_rules"] = user_rules

            # ── Atomic write ──────────────────────────────────────────────────
            tmp = CONFIG_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, CONFIG_FILE)
            print(f"  [GUI] Mapping saved — {app_name}/{gesture} → '{new_action}'")

        except Exception as e:
            print(f"  [GUI] Failed to save mapping: {e}")
            return

        # ── Send reload_rules command to pipeline ─────────────────────────────
        try:
            tmp = COMMANDS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"command": "reload_rules"}, f)
            os.replace(tmp, COMMANDS_FILE)
            print(f"  [GUI] reload_rules command sent")
        except Exception as e:
            print(f"  [GUI] Failed to send reload command: {e}")

        # ── Refresh table immediately ─────────────────────────────────────────
        self._refresh_gesture_mappings(app_name)

    def _build_settings(self):
        """
        Settings panel — 5 sliders + 1 switch.
        Values are saved to glove_config.json["settings"] and applied to the
        pipeline via the apply_settings command.
        """
        card = self._card(label="SETTINGS")

        s = self._load_settings()

        # Scrollable inner frame keeps the card compact
        sf = ctk.CTkScrollableFrame(card, fg_color=C_BG, height=200, corner_radius=4)
        sf.pack(fill="x", pady=(0, 4))

        def _section_hdr(text):
            ctk.CTkLabel(
                sf,
                text=text,
                font=ctk.CTkFont(family="Courier New", size=9),
                text_color=C_DIM,
            ).pack(anchor="w", pady=(8, 2))

        def _slider_row(label_text, key, from_, to, n_steps, fmt):
            row = ctk.CTkFrame(sf, fg_color="transparent")
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(
                row,
                text=label_text,
                font=ctk.CTkFont(family="Courier New", size=10),
                text_color=C_TEXT,
                width=140,
                anchor="w",
            ).pack(side="left")

            val_lbl = ctk.CTkLabel(
                row,
                text=fmt(s[key]),
                font=ctk.CTkFont(family="Courier New", size=10),
                text_color=C_BLUE,
                width=70,
                anchor="e",
            )
            val_lbl.pack(side="right")

            sld = ctk.CTkSlider(
                row,
                from_=from_,
                to=to,
                number_of_steps=n_steps,
                command=lambda v, lbl=val_lbl, f=fmt: lbl.configure(text=f(v)),
            )
            sld.set(s[key])
            sld.pack(side="left", fill="x", expand=True, padx=(8, 8))
            return sld

        # ── Mouse ──────────────────────────────────────────────────────────────
        _section_hdr("MOUSE")
        self._sld_click_cooldown = _slider_row(
            "Click cooldown", "click_cooldown",
            0.1, 1.0, 9, lambda v: f"{v:.1f}s",
        )
        self._sld_scroll_amount = _slider_row(
            "Scroll amount", "scroll_amount",
            1, 10, 9, lambda v: f"{int(round(v))} units",
        )
        self._sld_cursor_speed = _slider_row(
            "Cursor speed", "cursor_speed",
            0.5, 3.0, 25, lambda v: f"{v:.1f}x",
        )

        # ── Agent ──────────────────────────────────────────────────────────────
        _section_hdr("AGENT")
        self._sld_os_poll = _slider_row(
            "OS poll interval", "os_poll_interval",
            0.2, 2.0, 18, lambda v: f"{v:.1f}s",
        )
        self._sld_vision_poll = _slider_row(
            "Vision poll", "vision_poll",
            1, 10, 9, lambda v: f"{int(round(v))}s",
        )

        # ── Input ──────────────────────────────────────────────────────────────
        _section_hdr("INPUT")
        sw_row = ctk.CTkFrame(sf, fg_color="transparent")
        sw_row.pack(fill="x", pady=2)

        ctk.CTkLabel(
            sw_row,
            text="Keyboard test mode",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C_TEXT,
            width=140,
            anchor="w",
        ).pack(side="left")

        self._sw_keyboard_mode = ctk.CTkSwitch(sw_row, text="", onvalue=True, offvalue=False)
        if s.get("keyboard_mode", True):
            self._sw_keyboard_mode.select()
        else:
            self._sw_keyboard_mode.deselect()
        self._sw_keyboard_mode.pack(side="right")

        # Apply button — outside the scroll frame so it's always visible
        ctk.CTkButton(
            card,
            text="APPLY SETTINGS",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=C_BLUE,
            hover_color=C_GREEN,
            text_color=C_BG,
            corner_radius=6,
            height=32,
            command=self._apply_settings,
        ).pack(fill="x", pady=(4, 8))

    def _load_settings(self):
        """Read glove_config.json["settings"]. Returns merged dict with defaults."""
        defaults = {
            "click_cooldown":   0.4,
            "scroll_amount":    3,
            "cursor_speed":     1.0,
            "os_poll_interval": 0.5,
            "vision_poll":      2,
            "keyboard_mode":    True,
        }
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            saved = cfg.get("settings", {})
            return {**defaults, **saved}
        except FileNotFoundError:
            return defaults
        except Exception:
            return defaults

    def _apply_settings(self):
        """
        Read current slider/switch values, save to glove_config.json,
        and send apply_settings command to the pipeline.
        """
        settings = {
            "click_cooldown":   round(self._sld_click_cooldown.get(), 2),
            "scroll_amount":    int(round(self._sld_scroll_amount.get())),
            "cursor_speed":     round(self._sld_cursor_speed.get(), 2),
            "os_poll_interval": round(self._sld_os_poll.get(), 2),
            "vision_poll":      int(round(self._sld_vision_poll.get())),
            "keyboard_mode":    bool(self._sw_keyboard_mode.get()),
        }

        # ── Persist to config ─────────────────────────────────────────────────
        try:
            try:
                with open(CONFIG_FILE, "r") as f:
                    cfg = json.load(f)
            except FileNotFoundError:
                cfg = {}
            except Exception:
                cfg = {}

            cfg["settings"] = settings

            tmp = CONFIG_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, CONFIG_FILE)
            print(f"  [GUI] Settings saved → {settings}")
        except Exception as e:
            print(f"  [GUI] Failed to save settings: {e}")
            return

        # ── Send command to pipeline ──────────────────────────────────────────
        try:
            payload = {"command": "apply_settings", "settings": settings}
            tmp = COMMANDS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, COMMANDS_FILE)
            print(f"  [GUI] apply_settings command sent")
        except Exception as e:
            print(f"  [GUI] Failed to send apply_settings command: {e}")

    def _build_agent_card(self):
        """Context agent status — tier, app name, context string."""
        card = self._card(label="CONTEXT AGENT")

        # Top row: tier badge + app name
        tier_row = ctk.CTkFrame(card, fg_color="transparent")
        tier_row.pack(fill="x", pady=(4, 4))

        self._lbl_tier = ctk.CTkLabel(
            tier_row,
            text="—",
            font=ctk.CTkFont(family="Courier New", size=13, weight="bold"),
            text_color=C_DIM,
            width=120,
            anchor="w",
        )
        self._lbl_tier.pack(side="left")

        self._lbl_app = ctk.CTkLabel(
            tier_row,
            text="",
            font=ctk.CTkFont(family="Courier New", size=13),
            text_color=C_TEXT,
        )
        self._lbl_app.pack(side="left", padx=(8, 0))

        # Context string (URL / window title) — wraps if long
        self._lbl_context = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=C_DIM,
            wraplength=420,
            justify="left",
            anchor="w",
        )
        self._lbl_context.pack(fill="x", pady=(0, 4))

        # Confidence bar — only visible when source = "vision"
        self._conf_frame = ctk.CTkFrame(card, fg_color="transparent")
        # Not packed initially — shown only when source == "vision"

        ctk.CTkLabel(
            self._conf_frame,
            text="confidence",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C_DIM,
            width=80,
            anchor="w",
        ).pack(side="left")

        self._conf_bar = ctk.CTkProgressBar(
            self._conf_frame,
            width=200, height=8,
            fg_color=C_BORDER,
            progress_color=C_ORANGE,
        )
        self._conf_bar.pack(side="left", padx=(6, 8))
        self._conf_bar.set(0)

        self._lbl_conf_pct = ctk.CTkLabel(
            self._conf_frame,
            text="",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C_DIM,
        )
        self._lbl_conf_pct.pack(side="left")

    def _build_gesture_card(self):
        """Last gesture + action."""
        card = self._card(label="LAST GESTURE")

        gesture_row = ctk.CTkFrame(card, fg_color="transparent")
        gesture_row.pack(fill="x", pady=(4, 2))

        self._lbl_gesture = ctk.CTkLabel(
            gesture_row,
            text="—",
            font=ctk.CTkFont(family="Courier New", size=18, weight="bold"),
            text_color=C_TEXT,
        )
        self._lbl_gesture.pack(side="left")

        self._lbl_action_src = ctk.CTkLabel(
            gesture_row,
            text="",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=C_DIM,
        )
        self._lbl_action_src.pack(side="right", padx=4)

        self._lbl_action = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(family="Courier New", size=13),
            text_color=C_ORANGE,
        )
        self._lbl_action.pack(anchor="w", pady=(0, 8))

    def _build_footer(self):
        """Thin status bar at the bottom of the window."""
        footer = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=0, height=32)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self._lbl_footer = ctk.CTkLabel(
            footer,
            text="Waiting for pipeline...",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C_DIM,
        )
        self._lbl_footer.pack(side="left", padx=16)

    # ── Card helper ───────────────────────────────────────────────────────────

    def _card(self, label=""):
        """
        Creates a labelled card and packs it into the window.
        Returns the inner content frame for child widgets.
        """
        outer = ctk.CTkFrame(
            self._scroll_body,
            fg_color=C_PANEL,
            corner_radius=8,
            border_width=1,
            border_color=C_BORDER,
        )
        outer.pack(fill="x", padx=16, pady=(10, 0))

        if label:
            ctk.CTkLabel(
                outer,
                text=label,
                font=ctk.CTkFont(family="Courier New", size=9),
                text_color=C_DIM,
            ).pack(anchor="w", padx=14, pady=(8, 0))

            # Thin separator under the card label
            ctk.CTkFrame(outer, fg_color=C_BORDER, height=1).pack(
                fill="x", padx=14, pady=(4, 0)
            )

        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=(4, 4))
        return inner

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self):
        """
        Called every POLL_MS milliseconds by the CTk event loop.
        Reads glove_status.json and updates all labels.
        Never blocks — schedules the next call and returns immediately.
        """
        self._lbl_clock.configure(text=time.strftime("%H:%M:%S"))

        status = self._read_status()

        if status is None:
            self._show_offline("No status file — run: python main.py")
        elif time.time() - status.get("timestamp", 0) > STALE_AFTER:
            self._show_offline("Pipeline stopped")
        else:
            self._show_live(status)

        self.after(POLL_MS, self._poll)   # schedule next poll

    def _read_status(self):
        """
        Reads and parses glove_status.json.
        Returns the parsed dict, or None on any error.
        """
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None

    # ── Display update ────────────────────────────────────────────────────────

    def _show_offline(self, reason=""):
        """Set all labels to the disconnected state."""
        self._strip_app.configure(text="—", text_color=C_DIM)
        self._strip_src.configure(text="OFFLINE", text_color=C_RED)
        self._dot.configure(text_color=C_RED)
        self._lbl_connection.configure(text="OFFLINE", text_color=C_RED)
        self._lbl_hw_mode.configure(text="")
        self._lbl_mode.configure(text="—", text_color=C_DIM)
        self._current_mode = None        # unknown — dim all switcher buttons
        self._refresh_mode_buttons()
        self._lbl_tier.configure(text="—", text_color=C_DIM)
        self._lbl_app.configure(text="")
        self._lbl_context.configure(text="")
        self._lbl_gesture.configure(text="—", text_color=C_DIM)
        self._lbl_action.configure(text="")
        self._lbl_action_src.configure(text="")
        self._conf_frame.pack_forget()
        self._lbl_footer.configure(text=f"Offline — {reason}")

    def _show_live(self, s):
        """
        Update all labels from a fresh status dict.
        s : dict — parsed from glove_status.json
        """
        # ── App strip (always-visible bar) ───────────────────────────────────
        app_name               = s.get("app_name", "unknown")
        source                 = s.get("source", "fallback")
        tier_label, tier_col   = SOURCE_STYLE.get(source, ("UNKNOWN", C_DIM))
        app_color              = C_TEXT if app_name != "unknown" else C_DIM
        self._strip_app.configure(text=app_name, text_color=app_color)
        self._strip_src.configure(text=tier_label, text_color=tier_col)

        # ── Connection row ────────────────────────────────────────────────────
        hw = s.get("hardware", "SIMULATOR")
        self._dot.configure(text_color=C_GREEN)
        self._lbl_connection.configure(text="CONNECTED", text_color=C_GREEN)
        self._lbl_hw_mode.configure(text=hw)

        # ── Device mode ───────────────────────────────────────────────────────
        raw_mode   = s.get("device_mode", "standard")
        mode_label = MODE_LABELS.get(raw_mode, raw_mode.upper())
        self._lbl_mode.configure(text=mode_label, text_color=C_BLUE)
        self._current_mode = raw_mode    # keep switcher buttons in sync
        self._refresh_mode_buttons()

        # ── Gesture mappings — refresh when active app changes ───────────────
        if app_name != self._last_mapped_app:
            self._refresh_gesture_mappings(app_name)

        # ── Agent tier ────────────────────────────────────────────────────────
        self._lbl_tier.configure(text=tier_label, text_color=tier_col)

        self._lbl_app.configure(text=s.get("app_name", "unknown"))

        ctx = s.get("context_detected", "")
        self._lbl_context.configure(
            text=ctx if len(ctx) <= 60 else ctx[:57] + "..."
        )

        # Confidence bar — only shown for live vision API results
        if source == "vision":
            conf = float(s.get("confidence", 0.0))
            self._conf_frame.pack(fill="x", pady=(0, 4))
            self._conf_bar.set(conf)
            self._lbl_conf_pct.configure(text=f"{conf*100:.0f}%")
        else:
            self._conf_frame.pack_forget()

        # ── Last gesture ──────────────────────────────────────────────────────
        gesture = s.get("last_gesture", "rest")
        action  = s.get("last_action", "")
        asrc    = s.get("action_source", "static")

        if gesture and gesture != "rest":
            self._lbl_gesture.configure(text=gesture, text_color=C_TEXT)
            self._lbl_action.configure(
                text=f"→  {action}" if action else "",
                text_color=C_ORANGE if asrc == "agent" else C_GREEN,
            )
            self._lbl_action_src.configure(
                text="[AI]" if asrc == "agent" else "[static]",
                text_color=C_DIM,
            )
        else:
            self._lbl_gesture.configure(text="rest", text_color=C_DIM)
            self._lbl_action.configure(text="")
            self._lbl_action_src.configure(text="")

        # ── Footer ────────────────────────────────────────────────────────────
        age = time.time() - s.get("timestamp", time.time())
        self._lbl_footer.configure(
            text=f"Last update: {age:.1f}s ago  |  {hw}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = GloveApp()
    app.mainloop()