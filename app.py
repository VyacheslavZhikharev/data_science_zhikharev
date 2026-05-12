import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


DATA_BP = Path("hw_data_composite/X_bp.xlsx")
DATA_NUP = Path("hw_data_composite/X_nup.xlsx")

TARGETS = {
    "modulus": "Модуль упругости при растяжении, ГПа",
    "strength": "Прочность при растяжении, МПа",
    "ratio": "Соотношение матрица-наполнитель",
}


def load_dataset() -> pd.DataFrame:
    if not DATA_BP.exists() or not DATA_NUP.exists():
        raise FileNotFoundError(
            "Не найдены файлы датасета. Ожидаются: "
            "hw_data_composite/X_bp.xlsx и hw_data_composite/X_nup.xlsx"
        )

    bp = pd.read_excel(DATA_BP).reset_index(drop=True)
    nup = pd.read_excel(DATA_NUP).reset_index(drop=True)
    df = bp.join(nup, how="inner")
    return df


def feature_columns(df: pd.DataFrame, target_key: str) -> list[str]:
    target_col = TARGETS[target_key]
    if target_key in {"modulus", "strength"}:
        # Для прогноза свойств используем технологические признаки без целевых колонок.
        excluded = {TARGETS["modulus"], TARGETS["strength"]}
        return [c for c in df.columns if c not in excluded]
    return [c for c in df.columns if c != target_col]


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
    args = parser.parse_args()

    df = load_dataset()
    model, x_cols, metrics = train_model(df, args.target)
    x_user = load_input_values(args.input_json, x_cols, df)
    pred = float(model.predict(x_user)[0])

    print(f"\nЦелевой признак: {TARGETS[args.target]}")
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
