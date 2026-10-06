"""Offline tests of scripts/fetch_binance_daily.py. No network: the opener is an in-memory fake."""
import hashlib
import importlib.util
import io
import socket
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from helpers import ROOT
from alpha_gp_lab.data import load_csv_dir

spec = importlib.util.spec_from_file_location('fetch_binance_daily', ROOT / 'scripts' / 'fetch_binance_daily.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def archive(name, csv_text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(name.replace('.zip', '.csv'), csv_text)
    return buf.getvalue()


class FakeOpener:
    def __init__(self, files):
        self.files, self.urls, self.timeouts = files, [], []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        self.timeouts.append(timeout)
        return io.BytesIO(self.files[url])


KLINES = {  # invented round numbers; layout: open_time, open, high, low, close, volume, close_time, ...
    'BTCUSDT': '1704067200000,100.0,110.0,95.0,105.0,1000.0,1704153599999,0,0,0,0,0\n'      # ms timestamps
               '1704153600000,105.0,112.0,104.0,108.0,1200.0,1704239999999,0,0,0,0,0\n',
    'ETHUSDT': 'open_time,open,high,low,close,volume,close_time,a,b,c,d,e\n'                  # header, µs timestamps
               '1704067200000000,20.0,21.0,19.0,20.5,500.0,1704153599999999,0,0,0,0,0\n'
               '1704153600000000,20.5,22.0,20.0,21.5,650.0,1704239999999999,0,0,0,0,0\n',
}


def files_for(symbol, text, corrupt=False):
    url = fetch.archive_url(symbol, '2024-01')
    name = url.rsplit('/', 1)[1]
    data = archive(name, text)
    digest = hashlib.sha256(b'x' if corrupt else data).hexdigest()
    return {url: data, fetch.checksum_url(url): f'{digest}  {name}\n'.encode()}


class FetchScript(unittest.TestCase):
    def test_urls(self):
        self.assertEqual(fetch.archive_url('BTCUSDT', '2024-01'),
                         'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip')
        self.assertEqual(fetch.archive_url('ETHUSDT', '2024-01-31'),
                         'https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1d/ETHUSDT-1d-2024-01-31.zip')
        self.assertTrue(fetch.checksum_url('u.zip').endswith('u.zip.CHECKSUM'))
        for symbol, period in [('btcusdt', '2024-01'), ('BTC/USDT', '2024-01'), ('BTCUSDT', '2024-13'), ('BTCUSDT', '24-01')]:
            with self.assertRaises(ValueError):
                fetch.archive_url(symbol, period)
        self.assertEqual(fetch.months('2023-11', '2024-02'), ['2023-11', '2023-12', '2024-01', '2024-02'])

    def test_checksum_logic(self):
        data = b'payload'
        good = hashlib.sha256(data).hexdigest()
        self.assertEqual(fetch.parse_checksum(f'{good}  f.zip\n', 'f.zip'), good)
        fetch.verify_sha256(data, good)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            fetch.verify_sha256(data + b'!', good)
        for text in [f'{good}  other.zip', 'nothex  f.zip', good]:
            with self.assertRaises(ValueError):
                fetch.parse_checksum(text, 'f.zip')

    def test_kline_conversion_handles_ms_and_us(self):
        self.assertEqual(fetch.klines_to_rows(KLINES['BTCUSDT'])[0][:2], ('2024-01-01', 100.0))
        rows = fetch.klines_to_rows(KLINES['ETHUSDT'])
        self.assertEqual([r[0] for r in rows], ['2024-01-01', '2024-01-02'])

    def test_download_writes_loader_format_and_refuses_bad_checksums(self):
        files = {**files_for('BTCUSDT', KLINES['BTCUSDT']), **files_for('ETHUSDT', KLINES['ETHUSDT'])}
        with tempfile.TemporaryDirectory() as tmp:
            opener = FakeOpener(files)
            for symbol in ('BTCUSDT', 'ETHUSDT'):
                fetch.download_symbol(symbol, ['2024-01'], tmp, opener)
            self.assertTrue(all('data.binance.vision' in u for u in opener.urls))
            self.assertEqual(set(opener.timeouts), {60})   # every archive and .CHECKSUM request is bounded
            panel, dropped = load_csv_dir(tmp)
            self.assertEqual(panel.dates, ('2024-01-01', '2024-01-02'))
            self.assertEqual(panel.bars['close'][1], [108.0, 21.5])
            bad = files_for('BTCUSDT', KLINES['BTCUSDT'], corrupt=True)
            out = Path(tmp) / 'bad'
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                fetch.download_symbol('BTCUSDT', ['2024-01'], out, FakeOpener(bad))
            self.assertFalse(out.exists())

    def test_a_stalled_server_times_out_instead_of_hanging(self):
        """A server that accepts the connection and never answers: the real urlopen must give up."""
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            started = time.monotonic()
            with self.assertRaises(OSError):   # socket timeout, possibly wrapped in URLError
                fetch.fetch(f'http://127.0.0.1:{server.getsockname()[1]}/x.zip', timeout=0.5)
            self.assertLess(time.monotonic() - started, 10)


if __name__ == '__main__':
    unittest.main()
