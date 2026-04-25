"""
agent/context_rules.py
-----------------------
Static gesture remapping rules for known applications.

This file is pure data — no logic, no threads, no side effects.
It maps app_name strings (from os_context.py) to gesture remapping dicts.

── How it is used ─────────────────────────────────────────────────────────────
context_agent.py calls get_rules_for_app(app_name) in _update().
If a rule exists → apply it instantly, skip the vision API entirely.
If no rule exists → fall through to vision API (or static fallback).

── Structure ──────────────────────────────────────────────────────────────────
CONTEXT_RULES = {
    "app_name": {
        "gesture_name": "action_name",   # overrides the static default
        ...
    },
    ...
}

Only gestures being *changed from default* need to be listed.
Omitted gestures fall through to GESTURE_ACTIONS in mouse_mode.py.

── Action names ───────────────────────────────────────────────────────────────
The 5 built-in actions (wired to pyautogui in mouse_mode.py):
    left_click, right_click, scroll_up, scroll_down, double_click

Custom action names (logged to terminal, not yet wired to keystrokes):
    These are meaningful labels — they will be wired to actual keyboard
    shortcuts when mouse_mode.py is extended in a future session.
    Examples: fullscreen_toggle, seek_forward_10s, go_to_definition, etc.

── Adding new apps ────────────────────────────────────────────────────────────
1. Add an entry to CONTEXT_RULES with the app_name key.
2. Map whichever gestures you want to change.
3. The user-facing GUI will write to USER_RULES (glove_config.json) which
   takes priority over everything here — this file is the system default.

── Called by ──────────────────────────────────────────────────────────────────
    agent/context_agent.py — get_rules_for_app()
"""

# ── System default rules ──────────────────────────────────────────────────────
# These ship with the glove. User-customised rules (from glove_config.json)
# are loaded separately and take priority over these — see get_rules_for_app().

CONTEXT_RULES = {

    # ── YouTube ───────────────────────────────────────────────────────────────
    # Watching video: gestures become playback controls.
    # thumb_index_tap (left click) and scroll kept as-is — still useful for UI.
    "youtube": {
        "fist":             "fullscreen_toggle",    # F key
        "victory":          "seek_forward_10s",     # L key / right arrow
        "index_fold":       "seek_back_10s",        # J key / left arrow
        "thumbs_up":        "like_video",           # no standard shortcut — logged
        "flex":             "theatre_mode",         # T key
        "spiderman":        "captions_toggle",      # C key
        "middle_double_tap":"volume_up",            # up arrow
        "ring_double_tap":  "volume_down",          # down arrow
        "middle_pinky_tap": "mute_toggle",          # M key
        "index_ring_tap":   "next_video",           # SHIFT+N
        "index_double_tap": "pause_play",           # K key / Space (override double-click)
    },

    # ── Netflix ───────────────────────────────────────────────────────────────
    # Similar to YouTube but Netflix has different keyboard shortcuts.
    "netflix": {
        "fist":             "fullscreen_toggle",    # F key
        "victory":          "seek_forward_10s",     # right arrow (10s in Netflix)
        "index_fold":       "seek_back_10s",        # left arrow
        "flex":             "volume_up",            # up arrow
        "index_ring_tap":   "volume_down",          # down arrow
        "middle_pinky_tap": "mute_toggle",          # M key
        "spiderman":        "captions_toggle",      # logged (no Netflix shortcut)
        "index_double_tap": "pause_play",           # Space (override double-click)
        "middle_double_tap":"next_episode",         # logged — no standard shortcut
    },

    # ── Spotify (desktop app) ─────────────────────────────────────────────────
    "spotify": {
        "victory":          "next_track",           # CTRL+right
        "index_fold":       "previous_track",       # CTRL+left
        "middle_double_tap":"volume_up",            # CTRL+up
        "ring_double_tap":  "volume_down",          # CTRL+down
        "middle_pinky_tap": "mute_toggle",          # logged
        "fist":             "pause_play",           # Space
        "thumbs_up":        "like_track",           # ALT+SHIFT+B (Save to library)
        "flex":             "shuffle_toggle",       # CTRL+SHIFT+S
        "spiderman":        "repeat_toggle",        # CTRL+R
    },

    # ── VS Code ───────────────────────────────────────────────────────────────
    # Coding gestures: navigation, editing, terminal.
    "vscode": {
        "fist":             "go_to_definition",     # F12
        "victory":          "find_all_references",  # SHIFT+F12
        "thumbs_up":        "open_terminal",        # CTRL+`
        "flex":             "command_palette",      # CTRL+SHIFT+P
        "index_fold":       "go_back",              # ALT+left
        "spiderman":        "go_forward",           # ALT+right
        "index_ring_tap":   "find_in_file",         # CTRL+F
        "middle_pinky_tap": "format_document",      # SHIFT+ALT+F
        "middle_double_tap":"undo",                 # CTRL+Z
        "ring_double_tap":  "redo",                 # CTRL+Y
    },

    # ── GitHub (browser) ──────────────────────────────────────────────────────
    # Code review / repo browsing.
    "github": {
        "fist":             "go_back",              # browser back
        "victory":          "go_forward",           # browser forward
        "thumbs_up":        "star_repo",            # logged — no shortcut
        "flex":             "open_file_finder",     # T key (GitHub shortcut)
        "index_fold":       "focus_search",         # / key (GitHub shortcut)
        "spiderman":        "keyboard_shortcuts",   # ? key
    },

    # ── Claude (browser) ──────────────────────────────────────────────────────
    # Chatting: scroll is most useful, some navigation.
    "claude": {
        "fist":             "new_conversation",     # logged
        "victory":          "scroll_down",          # keep reading — maps to default
        "index_fold":       "scroll_up",            # scroll back up
        "thumbs_up":        "copy_last_response",   # logged
        "flex":             "focus_input",          # logged — click the text box
    },

    # ── Google Docs ───────────────────────────────────────────────────────────
    # Document editing.
    "google_docs": {
        "fist":             "undo",                 # CTRL+Z
        "victory":          "redo",                 # CTRL+Y
        "thumbs_up":        "comment",              # CTRL+ALT+M
        "flex":             "find_and_replace",     # CTRL+H
        "index_fold":       "go_to_top",            # CTRL+Home
        "spiderman":        "go_to_bottom",         # CTRL+End
        "middle_double_tap":"bold_toggle",          # CTRL+B
        "ring_double_tap":  "italic_toggle",        # CTRL+I
        "index_ring_tap":   "word_count",           # CTRL+SHIFT+C
    },

    # ── Google Sheets ─────────────────────────────────────────────────────────
    "google_sheets": {
        "fist":             "undo",                 # CTRL+Z
        "victory":          "redo",                 # CTRL+Y
        "flex":             "find_and_replace",     # CTRL+H
        "thumbs_up":        "insert_row",           # logged
        "index_fold":       "go_to_cell",           # CTRL+G
        "middle_double_tap":"bold_toggle",          # CTRL+B
    },

    # ── Discord ───────────────────────────────────────────────────────────────
    "discord": {
        "fist":             "mute_toggle",          # CTRL+SHIFT+M
        "victory":          "deafen_toggle",        # CTRL+SHIFT+D
        "thumbs_up":        "push_to_talk",         # logged — depends on user binding
        "flex":             "next_unread_channel",  # ALT+SHIFT+down
        "index_fold":       "prev_channel",         # ALT+up
        "index_ring_tap":   "jump_to_inbox",        # CTRL+I (Discord shortcut)
    },

    # ── File Explorer ─────────────────────────────────────────────────────────
    "file_explorer": {
        "fist":             "go_back",              # ALT+left
        "victory":          "go_forward",           # ALT+right
        "thumbs_up":        "go_up",               # ALT+up (go to parent folder)
        "flex":             "address_bar",          # ALT+D (focus address bar)
        "index_fold":       "new_folder",           # CTRL+SHIFT+N
        "index_ring_tap":   "rename",               # F2
        "middle_pinky_tap": "delete_file",          # Delete key
        "spiderman":        "properties",           # ALT+Enter
    },

    # ── Browser new tab ───────────────────────────────────────────────────────
    # No page loaded — revert to pure mouse behaviour.
    # We explicitly set all overrides to None so they pass through to static.
    # (An empty dict would also work — this is more explicit/readable.)
    "browser_newtab": {
        # No remapping — use default mouse mode for everything
    },

}


# ── Public helper ─────────────────────────────────────────────────────────────

def get_rules_for_app(app_name, user_rules=None):
    """
    Returns the gesture remapping dict for the given app_name, or None if
    no rules exist for that app.

    Priority (highest first):
      1. user_rules  — loaded from glove_config.json by the desktop app
      2. CONTEXT_RULES — the system defaults in this file

    user_rules : dict or None
        If provided, a dict of the same shape as CONTEXT_RULES, containing
        user-customised mappings. These override the system defaults.
        Example: {"youtube": {"fist": "scroll_down"}}

    Returns a merged dict of gesture → action, or None if the app has
    no rules at all (triggers vision API fallback in context_agent.py).
    """
    # ── Check user rules first ────────────────────────────────────────────────
    user_app_rules = None
    if user_rules and isinstance(user_rules, dict):
        user_app_rules = user_rules.get(app_name)

    # ── Check system rules ────────────────────────────────────────────────────
    system_app_rules = CONTEXT_RULES.get(app_name)

    # ── Neither exists → no rules → caller falls through to vision API ────────
    if user_app_rules is None and system_app_rules is None:
        return None

    # ── Merge: start with system rules, overlay user rules on top ────────────
    merged = {}
    if system_app_rules:
        merged.update(system_app_rules)    # system defaults as base
    if user_app_rules:
        merged.update(user_app_rules)      # user overrides win

    return merged if merged else None      # empty dict treated as None


# ── Standalone inspection tool ────────────────────────────────────────────────
# Run:  python agent/context_rules.py
# Prints all rules in a readable format for quick reference.

if __name__ == "__main__":
    print("\n  EMG Glove — Context Rules (system defaults)\n")

    for app_name, rules in sorted(CONTEXT_RULES.items()):
        if not rules:
            print(f"  [{app_name}]  (no remapping — pure mouse mode)")
            continue

        print(f"  [{app_name}]")
        for gesture, action in rules.items():
            # Format: gesture name padded, arrow, action name
            print(f"    {gesture:<25} → {action}")
        print()

    total_apps    = len(CONTEXT_RULES)
    total_rules   = sum(len(r) for r in CONTEXT_RULES.values())
    print(f"  {total_apps} apps  |  {total_rules} gesture remappings total")    