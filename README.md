# EMG Smart Glove v7.0
### Your hand is the computer.

> A wearable gesture-recognition system that translates muscle signals into real device control — across your PC, TV, and communication needs — with an AI agent that adapts automatically to whatever you're doing.

---


## What It Does

The EMG Smart Glove reads the electrical signals your muscles produce when you move your fingers, classifies them into one of **16 distinct gestures** using a trained SVM classifier, and maps them to real actions on your devices — in real time, with no physical contact required.

What makes it different from other gesture systems: **the glove knows what you're doing.** A background AI agent monitors your active application and automatically remaps the same gestures to context-appropriate actions. The same finger tap that scrolls a webpage will fast-forward a YouTube video — without you switching modes manually.

---

## Modes

### 🖱️ Mouse Mode
Control your PC without touching anything. Air-click with a pinch gesture, scroll with finger flicks, and navigate your desktop entirely through hand movements.

### ✍️ Pen Mode
Clip the pen accessory to any normal pen and write naturally on any surface. The glove converts your handwriting into digital text in real time using a 49-template character recognition library covering the full English alphabet, numbers, and symbols.

### 📺 TV Remote Mode
Swipe your hand in the air to change channels. Rotate your wrist like a volume knob to adjust audio. Control smart devices from the couch without reaching for a remote.

### 🗣️ Communication Mode
Designed for users who cannot speak but retain hand movement. Pre-set finger gestures map to spoken phrases output through the system speaker. Air-written words are converted directly to speech. Breaks down barriers between technology and human connection.

---

## The AI Agent Layer

The most technically sophisticated component of the glove is its **three-tier AI agent system** that handles context-aware gesture remapping:

```
Tier 1 — OS-Level Detection (Primary)
  Uses Python's OS and process APIs to identify the active
  foreground application instantly. Fast, deterministic,
  zero API cost. Handles ~90% of use cases.

Tier 2 — Nvidia NIM Vision AI (Fallback)
  When OS detection is insufficient (e.g. detecting what's
  inside a browser tab), screenshots the screen every 2 seconds
  and sends it to Llama 3.2 90B Vision via Nvidia NIM API
  for richer contextual understanding.

Tier 3 — Default Gesture Map
  If both layers fail or the app is unrecognised,
  falls back to the universal gesture configuration.
```

This layered approach — fast/cheap first, smart/expensive as fallback — mirrors how production AI systems are architected.

---

## Technical Specifications

| Component | Detail |
|---|---|
| Gesture classifier | Support Vector Machine (SVM) |
| Training dataset | 3,200 samples across 16 gestures |
| Sensors | EMG (electromyography) + IMU (inertial measurement unit) |
| Handwriting library | 49 templates — full alphabet, digits, symbols |
| AI vision model | Llama 3.2 90B via Nvidia NIM API |
| OS detection | Python `psutil` + `win32gui` (Windows) |
| Companion app | CustomTkinter prototype (Flutter port in progress) |
| Version | v7.0 (iterated from v1 over ~6 months) |

---

## Repository Structure

```
EMG-Glove/
│
├── agent/              # AI agent layer — OS detection + Nvidia NIM vision
├── core/               # Core EMG signal processing and SVM classifier
├── models/             # Trained gesture models and handwriting templates
├── modes/              # Mode-specific logic (mouse, pen, TV, communication)
│
├── main.py             # Entry point — launches full glove system
├── main_pen_cursor.py  # Pen mode entry point
├── main_tv_remote.py   # TV remote mode entry point
├── main_writing.py     # Handwriting recognition entry point
├── main_communication.py # Communication mode entry point
├── mode_manager.py     # Handles mode switching logic
├── glove_app.py        # CustomTkinter companion app (prototype)
├── glove_config.json   # Gesture-to-action mapping configuration
├── tv_simulator.py     # TV remote simulation for testing
└── find_ports.py       # Serial port detection utility
```

---

## Setup & Installation

### Requirements
- Python 3.9+
- Windows (OS-level detection uses Win32 APIs)
- EMG + IMU hardware connected via serial port
- Nvidia NIM API key (for vision fallback — free tier available)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure
1. Run `find_ports.py` to identify your hardware serial port
2. Add your Nvidia NIM API key to environment variables:
   ```bash
   set NIM_API_KEY=your_key_here
   ```
3. Update `glove_config.json` with your gesture-to-action mappings

### Run
```bash
# Full system (mouse mode default)
python main.py

# Specific modes
python main_pen_cursor.py
python main_tv_remote.py
python main_writing.py
python main_communication.py
```

---

## Inspiration

This project was inspired by **CTRL-labs** (acquired by Meta) and their vision of neural wristbands as a universal human-computer interface. The EMG Smart Glove is an attempt to build an accessible, open version of that vision — running on consumer hardware, powered by open-source ML, with an AI layer sophisticated enough to understand context.

Meta's neural band costs thousands and isn't publicly available. This glove is built in a bedroom.

---

## What's Next

- [ ] Bluetooth HID support (wireless, no cable)
- [ ] Flutter companion app (cross-platform mobile)
- [ ] Myo sensor integration for forearm EMG readings
- [ ] Expanded gesture set beyond 16
- [ ] Web dashboard for gesture remapping without editing JSON

---

## Built By

**Vansh Sharma** 

Built entirely independently as a personal project.  
Currently on version 7.0, iterated over approximately 6 months.

*Interested in collaborating, giving feedback, or just curious about the architecture? Open an issue or reach out.*

---

> *"Technology should adapt to humans — not the other way around."*
