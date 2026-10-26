from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"


def sql_path(path):
    return path.resolve().as_posix().replace("'", "''")


def main():
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW}")

    con = duckdb.connect()

    try:
        for file in files:
            source = f"read_csv_auto('{sql_path(file)}')"
            rows = con.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
            columns = [c[0] for c in con.execute(
                f"SELECT * FROM {source} LIMIT 0"
            ).description]

            print("\n" + "=" * 65)
            print(f"FILE: {file.name}")
            print(f"Rows: {rows:,}")
            print(f"Columns: {len(columns):,}")
            print(f"First columns: {columns[:8]}")
            print(f"Last columns: {columns[-5:]}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
