import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


DATA_BP = Path("hw_data_composite/X_bp.xlsx")
DATA_NUP = Path("hw_data_composite/X_nup.xlsx")
MARKED_DATASET = Path("Dataset_composites_inner_marked.csv")
CLEAN_DATASET = Path("Dataset_composites_inner_clean.csv")
SUSPECTED_COLUMN = "is_suspected_artificial"

TARGETS = {
    "modulus": "Модуль упругости при растяжении, ГПа",
    "strength": "Прочность при растяжении, МПа",
    "ratio": "Соотношение матрица-наполнитель",
}

META_COLUMNS = {
    SUSPECTED_COLUMN,
    "suspected_iqr",
    "suspected_isolation_forest",
    "suspected_lof",
    "suspected_invalid_values",
    "suspected_decimal_precision",
    "suspected_votes",
}


def read_composite_part(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    # В некоторых версиях датасета сохраненный Excel-индекс попадает в колонку
    # Unnamed: 0. Используем его как индекс, чтобы выполнить INNER JOIN по ТЗ.
    if "Unnamed: 0" in df.columns:
        return df.set_index("Unnamed: 0")

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    return df


def load_dataset() -> pd.DataFrame:
    if not DATA_BP.exists() or not DATA_NUP.exists():
        raise FileNotFoundError(
            "Не найдены файлы датасета. Ожидаются: "
            "hw_data_composite/X_bp.xlsx и hw_data_composite/X_nup.xlsx"
        )

    bp = read_composite_part(DATA_BP)
    nup = read_composite_part(DATA_NUP)

    df = bp.join(nup, how="inner").reset_index(drop=True)
    return df


def has_long_decimal_part(value: float) -> bool:
    if pd.isna(value):
        return False
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        return False
    return len(text.split(".", 1)[1]) > 3


def mark_suspected_artificial(df: pd.DataFrame) -> pd.DataFrame:
    marked = df.copy()
    numeric = marked.select_dtypes(include=[np.number]).copy()

    if numeric.empty:
        marked[SUSPECTED_COLUMN] = False
        return marked

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    iqr_mask = ((numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))).any(axis=1)

    invalid_mask = pd.Series(False, index=marked.index)
    for col in ["Шаг нашивки", "Плотность нашивки"]:
        if col in marked.columns:
            invalid_mask = invalid_mask | (marked[col] <= 0)

    decimal_precision_mask = numeric.apply(
        lambda col: col.map(has_long_decimal_part)
    ).mean(axis=1) > 0.35

    if len(numeric) >= 20:
        scaled = StandardScaler().fit_transform(numeric)
        contamination = min(0.15, max(0.02, 10 / len(numeric)))

        isolation_mask = pd.Series(
            IsolationForest(contamination=contamination, random_state=42).fit_predict(scaled)
            == -1,
            index=marked.index,
        )

        n_neighbors = min(20, len(numeric) - 1)
        lof_mask = pd.Series(
            LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(
                scaled
            )
            == -1,
            index=marked.index,
        )
    else:
        isolation_mask = pd.Series(False, index=marked.index)
        lof_mask = pd.Series(False, index=marked.index)

    votes = (
        iqr_mask.astype(int)
        + isolation_mask.astype(int)
        + lof_mask.astype(int)
        + invalid_mask.astype(int)
        + decimal_precision_mask.astype(int)
    )

    marked["suspected_iqr"] = iqr_mask
    marked["suspected_isolation_forest"] = isolation_mask
    marked["suspected_lof"] = lof_mask
    marked["suspected_invalid_values"] = invalid_mask
    marked["suspected_decimal_precision"] = decimal_precision_mask
    marked["suspected_votes"] = votes
    marked[SUSPECTED_COLUMN] = invalid_mask | (votes >= 2)
    return marked


def prepare_dataset(df: pd.DataFrame, data_mode: str) -> tuple[pd.DataFrame, dict[str, int]]:
    marked = mark_suspected_artificial(df)
    marked.to_csv(MARKED_DATASET, index=False)

    clean = marked.loc[~marked[SUSPECTED_COLUMN], df.columns].copy()
    clean.to_csv(CLEAN_DATASET, index=False)

    stats = {
        "all_rows": len(marked),
        "suspected_rows": int(marked[SUSPECTED_COLUMN].sum()),
        "clean_rows": len(clean),
    }

    if data_mode == "full":
        return df.copy(), stats
    return clean, stats


def feature_columns(df: pd.DataFrame, target_key: str) -> list[str]:
    target_col = TARGETS[target_key]
    if target_key in {"modulus", "strength"}:
        # Для прогноза свойств используем технологические признаки без целевых колонок.
        excluded = {TARGETS["modulus"], TARGETS["strength"]}
        return [c for c in df.columns if c not in excluded and c not in META_COLUMNS]
    return [c for c in df.columns if c != target_col and c not in META_COLUMNS]


def train_model(df: pd.DataFrame, target_key: str):
    target_col = TARGETS[target_key]
    x_cols = feature_columns(df, target_key)
    x = df[x_cols]
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        cv=10,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)

    y_pred_train = grid.predict(x_train)
    y_pred_test = grid.predict(x_test)

    metrics = {
        "train_mse": mean_squared_error(y_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_mse": mean_squared_error(y_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_r2": r2_score(y_test, y_pred_test),
    }

    return grid, x_cols, metrics


def load_input_values(path: Path | None, x_cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        values = {c: float(df[c].median()) for c in x_cols}
        return pd.DataFrame([values])

    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for c in x_cols:
        if c in payload:
            values[c] = float(payload[c])
        else:
            values[c] = float(df[c].median())
    return pd.DataFrame([values])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI-приложение для прогноза свойств композитов."
    )
    parser.add_argument(
        "--target",
        choices=["modulus", "strength", "ratio"],
        default="modulus",
        help="Что прогнозировать: modulus / strength / ratio",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Путь к JSON с входными признаками. Пример: {\"Плотность, кг/м3\": 2030}",
    )
    parser.add_argument(
        "--data-mode",
        choices=["clean", "full"],
        default="clean",
        help="clean — исключить подозрительно искусственные строки, full — использовать все строки",
    )
    args = parser.parse_args()

    df_raw = load_dataset()
    df, data_stats = prepare_dataset(df_raw, args.data_mode)
    model, x_cols, metrics = train_model(df, args.target)
    x_user = load_input_values(args.input_json, x_cols, df)
    pred = float(model.predict(x_user)[0])

    print(f"\nЦелевой признак: {TARGETS[args.target]}")
    print(f"Режим данных: {args.data_mode}")
    print(
        "Строки: всего={all_rows}, подозрительные={suspected_rows}, после очистки={clean_rows}".format(
            **data_stats
        )
    )
    print(f"Файл с метками: {MARKED_DATASET}")
    print(f"Очищенный файл: {CLEAN_DATASET}")
    print("Лучшие параметры модели:", model.best_params_)
    print("Метрики:")
    print(f"  Train MSE:  {metrics['train_mse']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Train MAE:  {metrics['train_mae']:.4f}")
    print(f"  Train R2:   {metrics['train_r2']:.4f}")
    print(f"  Test MSE:   {metrics['test_mse']:.4f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
    print(f"  Test MAE:   {metrics['test_mae']:.4f}")
    print(f"  Test R2:    {metrics['test_r2']:.4f}")
    print(f"\nПрогноз: {pred:.6f}")


if __name__ == "__main__":
    main()
