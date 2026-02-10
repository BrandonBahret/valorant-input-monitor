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
    "enabled": true,  // Video Enabled: Run with window (true, false)
    "vsync": true,    // VSync: Sync to monitor refresh rate (true, false)
    "target_fps": 60  // Target FPS: Used when vsync is false (e.g., 30, 60, 144)
  },
  "audio": {
    "volume": 1.0,        // Volume: Audio volume level (0.0 to 1.0)
    "sound_type": 1,      // Sound Type: Audio feedback style (1, 2, 3, 4, 5, 6)
    "loop_duration": 1000 // Loop Duration: Audio loop timing in milliseconds (e.g., 500, 1000, 1500)
  }
}
```
