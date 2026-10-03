#!/usr/bin/env python3
"""Download Binance public spot daily (1d) klines and write the CSV layout ``load_csv_dir`` reads.

NOT RUN YET: downloading waits for the repository owner's permission. Nothing in the tests or
the demo calls ``fetch``; the tests exercise URL building, checksum verification and the
kline conversion offline with in-memory archives.

Source: the public bulk-data archive at data.binance.vision. Each archive has a sibling
``.CHECKSUM`` file ("<sha256>  <file name>"); an archive whose SHA-256 does not match is
refused and nothing is written for it.

    python3 scripts/fetch_binance_daily.py --symbols BTCUSDT,ETHUSDT --start 2024-01 --end 2024-06 --out data/binance

writes ``data/binance/BTCUSDT.csv`` (date,open,high,low,close,volume), one row per UTC day.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import re
import sys
import urllib.request
import zipfile

BASE = 'https://data.binance.vision/data/spot'
_SYMBOL = re.compile(r'^[A-Z0-9]{2,20}$')
_MONTH = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
_DAY = re.compile(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$')


def archive_url(symbol, period, interval='1d'):
    """Monthly archive for 'YYYY-MM', daily archive for 'YYYY-MM-DD'."""
    if not _SYMBOL.match(symbol):
        raise ValueError(f'symbol must be upper-case letters/digits, got {symbol!r}')
    if _MONTH.match(period):
        kind = 'monthly'
    elif _DAY.match(period):
        kind = 'daily'
    else:
        raise ValueError(f'period must be YYYY-MM or YYYY-MM-DD, got {period!r}')
    return f'{BASE}/{kind}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip'


def checksum_url(url):
    return url + '.CHECKSUM'


def parse_checksum(text, file_name):
    """Return the hex digest from a '<sha256>  <file name>' line, refusing any other file name."""
    parts = text.strip().split()
    if len(parts) != 2 or not re.fullmatch(r'[0-9a-f]{64}', parts[0]):
        raise ValueError('malformed CHECKSUM file')
    if parts[1] != file_name:
        raise ValueError(f'CHECKSUM names {parts[1]!r}, expected {file_name!r}')
    return parts[0]


def verify_sha256(data, expected):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch: expected {expected}, got {actual}')


def klines_to_rows(csv_text):
    """Kline CSV -> [(date, open, high, low, close, volume)]. Open time is ms, or µs in newer files."""
    rows = []
    for rec in csv.reader(io.StringIO(csv_text)):
        if not rec or not rec[0].strip().isdigit():
            continue   # header line, present in some archives
        stamp = int(rec[0])
        seconds = stamp / 1_000_000 if stamp > 10**14 else stamp / 1000
        day = datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
        rows.append((day, *(float(x) for x in rec[1:6])))
    return rows


def months(start, end):
    y, m = map(int, start.split('-'))
    ey, em = map(int, end.split('-'))
    out = []
    while (y, m) <= (ey, em):
        out.append(f'{y:04d}-{m:02d}')
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(url, opener=urllib.request.urlopen):
    with opener(url) as response:
        return response.read()


def download_symbol(symbol, periods, out_dir, opener=urllib.request.urlopen):
    """Fetch, verify and convert every period for one symbol; write ``<out_dir>/<symbol>.csv``."""
    rows = {}
    for period in periods:
        url = archive_url(symbol, period)
        name = url.rsplit('/', 1)[1]
        data = fetch(url, opener)
        verify_sha256(data, parse_checksum(fetch(checksum_url(url), opener).decode(), name))
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = zf.namelist()
            if len(members) != 1:
                raise ValueError(f'{name}: expected one CSV member, found {members}')
            for row in klines_to_rows(zf.read(members[0]).decode()):
                rows[row[0]] = row
    out = Path(out_dir) / f'{symbol}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
        writer.writerows(rows[d] for d in sorted(rows))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--symbols', required=True, help='comma-separated, e.g. BTCUSDT,ETHUSDT')
    parser.add_argument('--start', required=True, help='first month, YYYY-MM')
    parser.add_argument('--end', required=True, help='last month, YYYY-MM')
    parser.add_argument('--out', required=True, help='output directory for <SYMBOL>.csv files')
    args = parser.parse_args(argv)
    for value in (args.start, args.end):
        if not _MONTH.match(value):
            parser.error('--start/--end must be YYYY-MM')
    for symbol in args.symbols.split(','):
        print(download_symbol(symbol.strip(), months(args.start, args.end), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
