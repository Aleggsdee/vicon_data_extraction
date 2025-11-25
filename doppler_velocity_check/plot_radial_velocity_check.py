#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_CHECK = 'radial_velocity_check.csv'

def main():
    df = pd.read_csv(CSV_CHECK)

    t         = df['t'].values
    r         = df['range'].values
    v_r_model = df['v_r_model'].values
    v_r_fd    = df['v_r_fd'].values
    diff      = df['diff'].values

    print("Loaded", len(t), "samples from", CSV_CHECK)
    print("Range stats [m]: min={:.3f}, max={:.3f}".format(r.min(), r.max()))
    print("Radial velocity (model) [m/s]: min={:.3f}, max={:.3f}".format(
        v_r_model.min(), v_r_model.max()))
    print("Radial velocity (FD)    [m/s]: min={:.3f}, max={:.3f}".format(
        v_r_fd.min(), v_r_fd.max()))
    print("RMS error between model and FD [m/s]:",
          np.sqrt(np.mean(diff**2)))

    # --- Plot radial velocities vs time ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_r_model, label='Analytic v_r (model)', linewidth=2)
    plt.plot(t, v_r_fd, '--', label='Finite-diff v_r (range)', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Radial velocity [m/s]')
    plt.title('Radial velocity comparison for single BODY point')
    plt.legend()
    plt.grid(True)

    # --- Plot error vs time ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, diff, label='v_r_model - v_r_fd')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m/s]')
    plt.title('Radial velocity error')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
