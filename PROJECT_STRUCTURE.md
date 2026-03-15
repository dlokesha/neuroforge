# Project structure

```
neuroforge/
├── README.md
├── requirements.txt
├── PROJECT_STRUCTURE.md
│
├── app.py                 # Web app
├── db.py                  # Database
├── models.py              # BaselineCNN, BioCNN, AblationCNN
├── reservoir.py           # BioPreprocessor, MEAEncoder
├── train.py               # TBC experiment (MNIST → bio preprocess → train & plot)
│
├── continual_train.py     # Continual learning training (Permuted MNIST)
├── permuted_mnist.py      # Permuted MNIST dataset / tasks
├── plasticity.py          # Plasticity / EWC for continual learning
│
├── data/                  # MNIST (downloaded by train / continual_train)
├── outputs/               # Run artifacts (plots, results.json, continual_results.json, .pkl cache)
│
├── venv/                  # Virtual environment (optional; use any venv)
└── .git/
```

## Quick run

- **TBC replication:** `python train.py --n_samples 2000 --epochs 10 --skip_ablation`
- **Continual learning:** `python continual_train.py --n_tasks 3 --epochs 3` (add `--use_ewc` for EWC)
