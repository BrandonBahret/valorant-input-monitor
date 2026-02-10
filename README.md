# Input Monitor

A real-time input monitoring application that visualizes keyboard (A/D keys) and mouse clicks in a scrolling chart.

![Preview image of the application](https://github.com/BrandonBahret/valorant-input-monitor/blob/main/input_monitor_graph.png "Preview")

## Features

- **A Key**: Blue line in upper half of chart
- **D Key**: Red line in lower half of chart
- **Mouse Click**: White dot at center line
- **Accuracy Detection**: Red line drawn when shooting while moving
- **Pause/Unpause**: Press TAB to pause or resume the chart
- **Real-time Scrolling**: Chart scrolls from right to left showing the last 5 seconds

## Binaries

[Download: windows executable](https://github.com/BrandonBahret/valorant-input-monitor/raw/refs/heads/main/dist/InputMonitor.exe)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python input_monitor.py
```

## Configuration

The application supports optional key remapping via a `config.json` file.

### Config file location

- Place `config.json` **next to the executable** (for the Windows binary), or  
- In the same directory as `input_monitor.py` when running from source.

If no config file is found, **built-in defaults are used automatically**.

---

### `config.json` format

```json
{
  "keys": {
    "left": "a",
    "right": "d",
    "walk": "shift",
    "crouch": "ctrl",
    "pause": "tab"
  },
  "video": {
    "enabled": true,  // Enabled: Run headless (true, false)
    "vsync": true,    // VSync: (true, false)
    "target_fps": 165 // Target FPS: (e.g, 60fps)
  },
  "audio": {
    "volume": 1.0,        // Volume: (0 .. 1)
    "sound_type": 3,      // Sound Type: (1, 2, .. 6)
    "loop_duration": 1000 // Loop Duration: (e.g, 1000ms)
  }
}
```
