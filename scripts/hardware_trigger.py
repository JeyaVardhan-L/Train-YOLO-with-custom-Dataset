#!/usr/bin/env python3
"""Arduino Ultrasonic Sensor Trigger for YOLO Airborne Detection & Radar HUD.

Listens on serial port for hardware detection signal ('DETECT'), then automatically
launches the radar detection interface.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Listen for Arduino ultrasonic hardware detection triggers and launch YOLO radar HUD."
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port for Arduino microcontroller (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Serial baud rate (default: 9600)",
    )
    parser.add_argument(
        "--trigger-cmd",
        type=str,
        default="scripts/radar_detect.py",
        help="Python script to launch upon detection trigger (default: scripts/radar_detect.py)",
    )
    parser.add_argument(
        "--trigger-signal",
        type=str,
        default="DETECT",
        help="Keyword signal sent by Arduino over serial to trigger detection (default: 'DETECT')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import serial
    except ImportError:
        print("Error: 'pyserial' package is not installed. Please install it via:", file=sys.stderr)
        print("  pip install pyserial", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("HARDWARE ULTRASONIC TRIGGER LISTENER")
    print(f"Port:           {args.port}")
    print(f"Baud rate:      {args.baud}")
    print(f"Trigger script: {args.trigger_cmd}")
    print(f"Signal keyword: '{args.trigger_signal}'")
    print("=" * 60)

    try:
        arduino = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2)  # Allow serial connection to settle
        print(f"Connected to Arduino on {args.port}. Listening for triggers...")
    except serial.SerialException as e:
        print(f"Error opening serial port {args.port}: {e}", file=sys.stderr)
        print("Check that the device is connected and port matches in Device Manager.", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            raw_line = arduino.readline()
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            print(f"[Arduino]: {line}")

            if args.trigger_signal in line:
                print("\n>>> Trigger condition met! Temporarily closing serial and launching detection...")
                arduino.close()

                # Launch radar detector subprocess
                cmd = [sys.executable, args.trigger_cmd]
                subprocess.run(cmd)

                print(">>> Detection session closed. Re-opening serial listener...")
                arduino = serial.Serial(args.port, args.baud, timeout=1)
                time.sleep(2)
                print("Listening for next trigger...")

    except KeyboardInterrupt:
        print("\nListener stopped by user.")
    finally:
        if "arduino" in locals() and arduino.is_open:
            arduino.close()


if __name__ == "__main__":
    main()
