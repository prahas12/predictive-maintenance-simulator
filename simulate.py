# src/simulate.py
"""Sensor data simulator for predictive maintenance starter project."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_device_baseline(seed=0):
    rng = np.random.RandomState(seed)
    return {
        "vib_amp": rng.uniform(0.5, 1.5),
        "temp_base": rng.uniform(40, 70),
        "pressure_base": rng.uniform(1.0, 2.0),
        "rpm_base": rng.uniform(800, 2000),
        "current_base": rng.uniform(5, 15)
    }

def simulate_stream(device_id, start_time, seconds=3600, freq_hz=1, seed=0, failure_at=None, failure_type=None):
    rng = np.random.RandomState(seed)
    dt = 1.0 / freq_hz
    timestamps = [start_time + timedelta(seconds=i) for i in range(seconds)]
    baseline = generate_device_baseline(seed + int(device_id) if str(device_id).isdigit() else seed)
    vib_amp = baseline["vib_amp"]
    temp_base = baseline["temp_base"]
    pressure_base = baseline["pressure_base"]
    rpm_base = baseline["rpm_base"]
    current_base = baseline["current_base"]

    rows = []
    for i, ts in enumerate(timestamps):
        t = i * dt
        # Vibration: multiple sinusoids + noise
        vibration_x = vib_amp * (0.3*np.sin(2*np.pi*5*t) + 0.7*np.sin(2*np.pi*23*t)) + rng.normal(0, 0.05)
        vibration_y = vib_amp * (0.2*np.sin(2*np.pi*7*t) + 0.6*np.sin(2*np.pi*19*t)) + rng.normal(0, 0.05)
        vibration_z = vib_amp * (0.25*np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*11*t)) + rng.normal(0, 0.05)

        # Temperature: slow drift + noise
        temperature = temp_base + 0.0005*i + rng.normal(0, 0.2)

        # Pressure: mild fluctuations
        pressure = pressure_base + 0.01*np.sin(2*np.pi*0.01*t) + rng.normal(0, 0.01)

        rpm = rpm_base + rng.normal(0, 10)
        current = current_base + rng.normal(0, 0.5)

        operational_state = 0
        time_to_failure = None
        ftype = None
        # inject failure ramp starting `failure_at - pre_fail_window`
        pre_fail_window = 300  # seconds before failure to start ramp
        if failure_at is not None and i >= max(0, failure_at - pre_fail_window) and i < failure_at:
            # pre-failure signature
            if failure_type == "bearing":
                vibration_x += 0.005*(i - max(0, failure_at - pre_fail_window)) * 10
                vibration_y += 0.004*(i - max(0, failure_at - pre_fail_window)) * 10
            elif failure_type == "overheating":
                temperature += 0.02*(i - max(0, failure_at - pre_fail_window))
            elif failure_type == "leak":
                pressure -= 0.002*(i - max(0, failure_at - pre_fail_window))
            elif failure_type == "electrical":
                current += 0.05*(i - max(0, failure_at - pre_fail_window))
            operational_state = 1
            time_to_failure = max(0, failure_at - i)
            ftype = failure_type

        if failure_at is not None and i >= failure_at:
            # failure moment and afterwards for a while
            operational_state = 2
            time_to_failure = 0
            ftype = failure_type

        rows.append({
            "device_id": device_id,
            "timestamp": ts.isoformat(),
            "vibration_x": float(vibration_x),
            "vibration_y": float(vibration_y),
            "vibration_z": float(vibration_z),
            "temperature": float(temperature),
            "pressure": float(pressure),
            "rpm": float(rpm),
            "current": float(current),
            "operational_state": operational_state,
            "failure_type": ftype,
            "time_to_failure": time_to_failure
        })

    return pd.DataFrame(rows)

def generate_fleet(num_devices=10, days=1, freq_hz=1, seed=42):
    start = datetime.utcnow()
    all_dfs = []
    rng = np.random.RandomState(seed)
    seconds = max(1, int(days * 24 * 3600))
    for d in range(num_devices):
        device_id = d + 1
        # decide if and when failure occurs within timespan
        if rng.rand() < 0.6 and seconds > 3600:
            failure_at = rng.randint(3600, seconds-1)  # some failure later
            ftype = rng.choice(["bearing", "overheating", "leak", "electrical"])
        else:
            failure_at = None
            ftype = None
        df = simulate_stream(device_id, start, seconds=seconds, freq_hz=freq_hz, seed=seed+d, failure_at=failure_at, failure_type=ftype)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

if __name__ == "__main__":
    df = generate_fleet(num_devices=2, days=0.001, freq_hz=1, seed=0)  # small sample
    df.to_csv("data/sample_dataset_generated.csv", index=False)
    print("Wrote data/sample_dataset_generated.csv rows=", len(df))
