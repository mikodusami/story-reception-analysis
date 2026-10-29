"""Build the DuckDB warehouse from data/cleaned/ and run the SQL data-quality checks.

    python -m story_reception.warehouse                  # data/cleaned -> data/reception.duckdb
    python -m story_reception.warehouse --db other.duckdb --clean-dir some/dir

Order: sql/staging/*.sql (load CSVs), sql/models/*.sql (tables, run in file-name order),
sql/checks/*.sql (each returns the rows that break a rule; any row fails the build).
"""
import argparse
import sys
import time
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
SQL_DIR = REPO / 'sql'


def run_sql_files(con, folder: Path, replacements: dict[str, str] | None = None) -> list[str]:
    ran = []
    for path in sorted(folder.glob('*.sql')):
        sql = path.read_text()
        for key, value in (replacements or {}).items():
            sql = sql.replace('{{' + key + '}}', value)
        con.execute(sql)
        ran.append(path.stem)
    return ran


def run_checks(con, folder: Path) -> dict[str, list[tuple]]:
    """Return {check name: offending rows} for every check that found something."""
    failures = {}
    for path in sorted(folder.glob('*.sql')):
        rows = con.execute(path.read_text()).fetchall()
        if rows:
            failures[path.stem] = rows
    return failures


def build(db_path: Path, clean_dir: Path, sql_dir: Path = SQL_DIR, quiet: bool = False) -> dict[str, list[tuple]]:
    say = (lambda *a: None) if quiet else print
    start = time.time()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        run_sql_files(con, sql_dir / 'staging', {'clean_dir': clean_dir.resolve().as_posix()})
        for name in run_sql_files(con, sql_dir / 'models'):
            table = name.split('_', 1)[1]
            say(f'  {table:18s} {con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:>9,} rows')
        failures = run_checks(con, sql_dir / 'checks')
    checks = sorted(p.stem for p in (sql_dir / 'checks').glob('*.sql'))
    for name in checks:
        detail = f'  {failures[name][:5]}' if name in failures else ''
        say(f"  {'FAIL' if name in failures else 'PASS'}  {name}{detail}")
    say(f'Built {db_path} in {time.time() - start:.0f}s')
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', type=Path, default=REPO / 'data' / 'reception.duckdb')
    parser.add_argument('--clean-dir', type=Path, default=REPO / 'data' / 'cleaned')
    args = parser.parse_args(argv)
    if not args.clean_dir.exists():
        print(f'{args.clean_dir} not found. Run data-cleaning.ipynb (make clean-data) first.', file=sys.stderr)
        return 2
    return 1 if build(args.db, args.clean_dir) else 0


if __name__ == '__main__':
    sys.exit(main())
