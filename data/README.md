# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it

Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

Version used: the archive contains a single file `house_sale.csv`
(100,775 rows, 51 columns, file dated 2025-02-03, scraped in October 2024).
Downloaded on 2026-10-14.

## How to place it

1. Download the archive from Kaggle (button "Download"), or with curl:
   ```
   curl -L -o binaaz.zip https://www.kaggle.com/api/v1/datasets/download/sehriyarmemmedli/binaaz-sale-project
   ```
2. Extract it so the CSV lives **directly in this folder**:
   ```
   data/house_sale.csv
   ```
3. The path is set once at the top of `src/data_prep.py` (`DATA_PATH`).

## Notes / gotchas

- Column names are in **Azerbaijani** (`Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, ...). The file is read as UTF-8.
- Numbers arrive as text with units: `Sahə = "145 m²"`,
  `Torpaq sahəsi = "1.3 sot"`, `Mərtəbə = "7 / 9"`.
- All prices are in AZN (`currency_x`), but the same listing appears several
  times (36,321 repeated listing URLs).
- **Leakage:** `unit_price` (price per m²) and `total_price` (equal to
  `price`) are derived from the target and are dropped in `data_prep.clean`.
