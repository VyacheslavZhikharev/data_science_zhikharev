import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def ensure_gdown():
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Не найден пакет gdown. Установи его командой: python3 -m pip install gdown"
        ) from exc
    return gdown


def copy_required_files(extract_dir: Path, target_dir: Path) -> None:
    needed = ["X_bp.xlsx", "X_nup.xlsx"]
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in needed:
        matches = list(extract_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Файл {filename} не найден в скачанном архиве.")
        shutil.copy2(matches[0], target_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачать датасет композитов из Google Drive и подготовить hw_data_composite."
    )
    parser.add_argument(
        "--file-id",
        default="1B1s5gBlvgU81H9GGolLQVw_SOi-vyNf2",
        help="Google Drive file id",
    )
    parser.add_argument(
        "--out-dir",
        default="hw_data_composite",
        help="Папка для итоговых файлов X_bp.xlsx и X_nup.xlsx",
    )
    args = parser.parse_args()

    gdown = ensure_gdown()
    out_dir = Path(args.out_dir)
    url = f"https://drive.google.com/uc?id={args.file_id}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "dataset.zip"
        print("Скачивание архива...")
        gdown.download(url, str(zip_path), quiet=False)

        print("Распаковка архива...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir / "extracted")

        print("Копирование X_bp.xlsx и X_nup.xlsx...")
        copy_required_files(tmp_dir / "extracted", out_dir)

    print(f"Готово. Файлы сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
