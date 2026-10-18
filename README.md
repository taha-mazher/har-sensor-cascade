# Sensor-gated early-exit cascade for human activity recognition on FPGA

A two-stage activity classifier for wearables, written in Verilog and verified bit for bit against a Python model, then synthesised for an Artix-7 FPGA (xc7a35t) in Vivado 2024.2.

Stage 1 is a tiny 7-6-5 MLP that runs on a single accelerometer for every window. When it is unsure, and only then, the hardware switches on a second sensor (a gyroscope on UCI-HAPT, a wrist accelerometer on PAMAP2) and a stage-2 network makes the call. The second sensor stays off the rest of the time, which is where the power saving comes from: a gyroscope draws about 5 times the current of an accelerometer.

This work extends:

- Gaikwad et al., IEEE Access 2019 (doi 10.1109/ACCESS.2019.2900084): the original 7-6-5 MLP on FPGA.
- Our AEJ 16(2) 2026 paper: the same network with ReLU6 activation. This repo also contains the corrected, bit-exact ReLU6 hardware for that paper (see "Stage 1 on its own" below).

## Headline results

All accuracies use subject-independent splits: no person is in both training and test.

| Protocol | Stage 1 alone | Cascade | Always-on bigger network | Second sensor on |
|---|---|---|---|---|
| UCI-HAPT, 5-fold by subject | 93.11 ± 0.43 | 94.61 ± 0.49 | 94.69 ± 0.51 | 11.3% of the time |
| UCI-HAPT, official split | 92.22 ± 0.62 | 93.41 ± 0.63 | 93.82 ± 1.08 | 7.8% |
| PAMAP2, leave one subject out | 79.44 ± 1.39 | 87.19 ± 1.67 | 87.89 ± 1.77 | 60.0% |

Mean ± standard deviation over 5 seeds. The gate threshold is chosen on training subjects only (nested selection, `nested_tau.py`). The cascade matches the always-on network within seed noise while the second sensor is off most of the time on UCI. On PAMAP2 the chest accelerometer is a weak primary sensor (the chest stays upright when sitting and standing), so the wrist sensor is needed more often.

Sensor saving on UCI: with BMI160 datasheet currents (accelerometer 180 µA, gyroscope 850 µA), the cascade averages about 297 µA against 925 µA with both always on, about 68% less sensor current.

## Hardware results

Artix-7 xc7a35ticsg324-1L, 50 MHz, Vivado 2024.2, post-route. Power is from SAIF switching activity captured in gate-level simulation of the real test windows (Vivado reports High confidence). Every build is bit-exact against Python in gate-level simulation.

| Design | LUTs | FFs | Slices | DSP | WNS (ns) | Power (dynamic) |
|---|---|---|---|---|---|---|
| Cascade, UCI (9-8-5 stage 2) | 358 | 286 | 126 | 1 | +5.63 | 63 mW (3 mW) |
| Cascade, PAMAP2 (14-16-5 stage 2, 8 models) | 396 to 413 | 368 | 146 to 159 | 1 | +4.32 to +5.44 | 64 mW (3 mW) |

Latency: 86 cycles when stage 1 exits, 211 cycles with stage 2 on UCI, 411 on PAMAP2.

### Stage 1 on its own (corrected AEJ hardware)

The hardware numbers in the AEJ paper came from an older PLAN-activation circuit that was never re-run after the switch to ReLU6, and that circuit did not compute the network correctly. `rtl/mlp_relu6.v` is the corrected design: bit-exact against Python on all three splits (1555/1640, 1037/1093, 2073/2186 correct).

| Variant (split A) | LUTs | FFs | DSP | WNS (ns) | Power (dynamic) | Cycles | Energy per inference (dynamic) |
|---|---|---|---|---|---|---|---|
| Serial, one MAC | 220 | 189 | 1 | +5.71 | 62.9 mW (2.60) | 85 | 107 nJ (4.47) |
| Neuron-parallel | 699 | 409 | 6 | +4.08 | 72.6 mW (12.24) | 18 | 26.1 nJ (4.65) |
| Fully parallel | 465 | 295 | 72 | +0.55 | 74.9 mW (14.50) | 7 | 10.5 nJ (2.32) |

## How it works

**Features.** Seven time-domain features per 2.56 s window (128 samples at 50 Hz, 50% overlap), computed with UCI's own filtering recipe (`features.py`). They match UCI's precomputed features at r ≥ 0.9999.

**Gate.** Stage 1 outputs 5 class scores. If the gap between the top two is below a threshold τ, and the top class is in a class mask (Sitting/Standing for UCI, any class for PAMAP2), the window is flagged.

**Sensor policy ("sticky").** A flagged window turns the second sensor on. It stays on while stage 1 keeps flagging, and turns off once it has covered two whole windows. A sensor switched on after window w first covers a whole window at w + 2, and the simulation models that delay, so results do not assume the gyroscope can see the past.

**Number format.** Inputs and hidden activations are Q4.12 (16-bit). Weights and biases are scaled per layer so no sum can overflow a 32-bit accumulator (`cascade_hw.py`). The integer model equals the float model on the test sets.

**RTL.** `rtl/cascade.v` uses one multiply-accumulate unit for all four layers, reading a single weight ROM in neuron order. Features stream in 16 bits per cycle so the PAMAP2 version fits the package pins. Parameters set the gate (TAU, MASK, keep policy) and network sizes, so one Verilog file serves both datasets.

## Repository layout

```
features.py, extract_all.py      raw sensor data -> 7 features per window (data/*_features.npz)
validate_features_uci.py         checks features.py against UCI's precomputed features
baseline.py                      stage 1 alone on UCI, PAMAP2 and WISDM, subject-independent
stage2_uci.py, stage2_delay.py   which extra features help stage 2; effect of the switch-on delay
gyro_policies.py                 sensor on/off policies simulated in time order
pamap2_cascade.py                cascade on PAMAP2 with different second sensors
uci_upper_bound.py,
pamap2_upper_bound.py            always-on comparison networks
cascade_sizes.py                 bigger stage 1 / MLP stage 2 sweep
general_gate.py, better_gate.py  gate on any class; margin vs entropy vs learned gate
nested_tau.py                    final 5-seed evaluation with nested threshold choice
wisdm_gate.py                    does the gate transfer to a third dataset (WISDM)
select7_uci.py, feature_gap.py   side studies: other 7-feature sets, 7 vs 19 features
fixed_point_reference.py         integer model of the AEJ 7-6-5 network, exports weights and test vectors
make_rom_files.py                splits weights into ROM files, checks 32-bit accumulators
cascade_hw.py                    integer cascade model, ROMs (rtl/mem/) and test vectors (out/)
paper_figures.py                 figures in paper/

rtl/          Verilog: mlp_relu6*.v (stage 1 variants), cascade.v, testbenches, iverilog scripts
rtl/mem/      weight and bias ROMs per model, plus params.txt with the gate settings
vivado/       build, simulation, SAIF and power scripts; out_*/ holds the routed reports
data/         extracted features and result JSON/logs
out/          test vectors, weights, and the Vivado result tables
paper/        figures
```

## Reproducing

**Python** (3.10+): numpy, scipy, scikit-learn, torch, matplotlib.

**Datasets** (not included, all public):

- UCI-HAPT v2.1 (Reyes-Ortiz et al., Neurocomputing 2016). The scripts look for it at `../exact_paper_implementation/data/UCI HAR Dataset/` relative to this folder.
- PAMAP2 and WISDM v1.1. Paths are set at the top of `extract_all.py`; edit them to where you unpack the data.

The extracted features are already in `data/`, so everything after feature extraction runs without the raw data.

**Simulation** (Icarus Verilog, e.g. in WSL):

```
python cascade_hw.py          # ROMs and test vectors
bash rtl/run_cascade_sim.sh   # bit-exact check of all 9 cascade models
bash rtl/run_sim.sh           # stage 1 alone, splits A, B, C
```

Each prints `RESULT: PASS` when the hardware's class, sensor-enable and stage-2 flag match Python on every window.

**Vivado** (2024.2, the free Standard edition covers Artix-7). From `vivado/`:

```
powershell -File run_cascade.ps1 cascade_uci   # synth, route, gate-level sim, SAIF power
python collect_cascade.py                      # -> out/cascade_vivado_results.txt
```

`run_all.ps1` and `run_sweep.ps1` do the same for stage 1 and the parallelism sweep. The scripts assume Vivado is at `C:\Xilinx\Vivado\2024.2`.

## Limitations

- **No board yet.** Everything is simulation plus Vivado post-route estimates. Power has not been measured on hardware.
- **Static power dominates.** The Artix-7 draws about 60 mW static, against about 3 mW dynamic for the design. The FPGA saving is tiny next to that; the real saving is in sensor current. A deployed device would want a low-static FPGA (e.g. iCE40) or an ASIC.
- **Sensor savings use datasheet currents**, not measured ones, and ignore the gyroscope's 55 ms start-up.
- **WISDM has no second sensor**, so there it only tests whether the gate flags the right windows.

## Related

The tamper-evident version of this cascade (SHA-256 / HMAC hash chain over every decision) is in the companion repo `har-hashchain-fpga`.
