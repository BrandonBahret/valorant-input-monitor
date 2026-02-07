# Input Monitor

A real-time input monitoring application that visualizes keyboard (A/D keys) and mouse clicks in an scrolling chart.

## Features

- **A Key**: Blue line in upper half of chart
- **D Key**: Red line in lower half of chart
- **Mouse Click**: White dot at center line
- **Accuracy Detection**: Red line drawn when shooting while moving
- **Pause/Unpause**: Press TAB to pause or resume the chart
- **Real-time Scrolling**: Chart scrolls from right to left showing the last 5 seconds

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

### Controls

- **TAB**: Pause/unpause the chart
