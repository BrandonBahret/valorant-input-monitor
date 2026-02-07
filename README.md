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
