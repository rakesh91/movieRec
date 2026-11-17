"""Download the non-redistributable MovieLens 1M data from GroupLens."""

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ml-1m"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REQUIRED = ("ratings.dat", "movies.dat", "users.dat", "README")


def main():
    if all((DATA_DIR / name).exists() for name in REQUIRED):
        print(f"MovieLens 1M already present at {DATA_DIR}")
        return

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR.parent / "ml-1m.zip"
    print(f"Downloading {URL}")
    with urlopen(URL, timeout=60) as response, archive_path.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)

    with ZipFile(archive_path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        missing = [name for name in REQUIRED if name not in members]
        if missing:
            raise RuntimeError(f"MovieLens archive is missing: {', '.join(missing)}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED:
            target = DATA_DIR / name
            with archive.open(members[name]) as source, target.open("wb") as destination:
                destination.write(source.read())
    archive_path.unlink()
    print(f"Installed MovieLens 1M files in {DATA_DIR}")


if __name__ == "__main__":
    main()
