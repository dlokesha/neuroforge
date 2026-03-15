---
title: Neuroforge
emoji: 🧠
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Neuroforge

Software replication of [The Biological Computing Co.](https://www.tbc.co) MEA pipeline.

**Real TBC:** image → electrical stimulation → dish of living neurons → spike readout → CNN  
**This repo:** image → electrode encoding → Echo State Network → spike readout → CNN

The key finding: filtering images through a recurrent biological-style neural network before a CNN improves classification accuracy — because the reservoir spreads information into a richer, higher-dimensional representation.

## Results

| Model | Test Accuracy |
|-------|--------------|
| Baseline CNN (raw MNIST) | ~TBD% |
| Bio-preprocessed CNN | ~TBD% |
| TBC paper reported improvement | +4.7% |

Ablation (mirrors TBC Figure 4 right):

| Signal region | Accuracy |
|---------------|----------|
| Whole array | ~TBD% |
| Center (stimulated) | ~TBD% |
| Periphery (spread) | ~TBD% |
| Chance | 10.0% |

## Pipeline

```
MNIST digit (28×28)
      ↓
Binarize + resize to 64×64 electrode grid
      ↓
Echo State Network (1024 units, fixed random weights, sr=0.9)
      ↓  stimulate for 10 timesteps
Spike-rate readout (fraction of time each unit was active)
      ↓
CNN classifier (same architecture as baseline)
```

## Quickstart

```bash
pip install -r requirements.txt

# Smoke test (fast — 2 min)
python train.py --n_samples 500 --epochs 5

# Full run (matches paper scale)
python train.py --n_samples 2000 --epochs 10
```

## File structure

```
neuroforge/
├── reservoir.py      # MEA encoder + ESN reservoir layer
├── models.py         # CNN architectures (baseline + bio + ablation)
├── train.py          # Full experiment runner
├── .cursorrules      # Cursor AI context for this project
├── requirements.txt
├── data/             # MNIST auto-downloads here
└── outputs/          # Plots and results saved here
```

## Key design decisions

**Why an Echo State Network?**  
ESNs have fixed random recurrent weights — never trained. This mirrors how TBC's biological neurons self-wire. The reservoir spreads input through recurrent dynamics and produces richer representations. It's the simplest software analog to what the neurons are doing.

**Why keep CNN architectures identical?**  
TBC's key point is that bio preprocessing improves learning without adding model complexity. If we gave the bio model a bigger CNN, any improvement could come from model capacity, not the preprocessing. Identical architectures isolate the effect.

**Why the ablation study?**  
TBC's most mechanistically interesting finding: even signals from *outside* the stimulated electrode region carry useful information. This proves the reservoir genuinely spreads information (it's not just passing through). If our periphery signal beats chance, we've independently confirmed this.

## Stretch goals

- [ ] Swap ESN for spiking neural network (`snnTorch`)  
- [ ] Scale to CIFAR-10  
- [ ] Track effective rank of CNN activations during training  
- [ ] Animate reservoir spread across timesteps (gif)

## References

- [TBC Post 1 — Computer Vision](https://www.tbc.co/post/biological-neural-dynamics-for-computer-vision-post-1-of-4)
- [TBC Post 4 — Algorithm Discovery](https://www.tbc.co/post/beyond-weight-recycling-algorithm-discovery-for-sustained-plasticity-in-neural-networks-post-4-4)
