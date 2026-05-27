# SOC Project Layout

Proponowany układ projektu:

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

Umieść dataset tutaj:

```text
data/LGHG2@n10C_to_25degC/
```

a potem uruchom:

```bash
python src/main.py
```

Opcjonalnie możesz nadpisać ścieżkę:

```bash
python src/main.py --data_root "ścieżka/do/LGHG2@n10C_to_25degC"
```
