# SOC Project Layout

układ projektu:

```text
SOC_project/
├── src/
│   ├── main.py
│   ├── model.py
│   └── data_utils.py
├── data/
│   └── LGHG2@n10C_to_25degC/
│       ├── Train/
│       └── Test/
├── outputs/
├── models/
└── README.md
```

## Jak uruchomić

 dataset tutaj:

```text
data/LGHG2@n10C_to_25degC/
```

a potem uruchom:

```bash
python src/main.py
```

 można nadpisać ścieżkę:

```bash
python src/main.py --data_root "ścieżka/do/LGHG2@n10C_to_25degC"
```


```bash
python src/main.py --stride 32 --epochs 20
```
Symulacja rzeczywistego modelu
```bash
python src/predict_realtime_sim.py --max_samples 5000 --print_every 500
```

Stworzenie wykresów
```bash
python src/plot_realtime_sim.py
```