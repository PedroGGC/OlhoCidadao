import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import publish_snapshot


class SnapshotPackagingTests(unittest.TestCase):
    def test_archive_is_deterministic_and_preserves_csv_bytes(self):
        with tempfile.TemporaryDirectory(prefix='olho-pack-') as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            value = 'município;valor\r\nSão Luís;120,50\r\n'.encode('utf-8')
            (root / 'data/sample.csv').write_bytes(value)
            with patch.object(publish_snapshot, 'ROOT', root):
                entries = [publish_snapshot.entry('data/sample.csv', 'research')]
                first, second = root / 'first.tar.gz', root / 'second.tar.gz'
                publish_snapshot.create_archive(first, entries)
                publish_snapshot.create_archive(second, entries)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                with tarfile.open(first, 'r:gz') as archive:
                    self.assertEqual(archive.getnames(), ['data/sample.csv'])
                    self.assertEqual(archive.extractfile('data/sample.csv').read(), value)

    def test_concurrent_file_change_is_detected_before_publication(self):
        with tempfile.TemporaryDirectory(prefix='olho-pack-') as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            source = root / 'data/sample.csv'
            source.write_bytes(b'valor\n100\n')
            with patch.object(publish_snapshot, 'ROOT', root):
                entries = [publish_snapshot.entry('data/sample.csv', 'research')]
                source.write_bytes(b'valor\n900\n')
                with self.assertRaisesRegex(ValueError, 'mudou durante'):
                    publish_snapshot.create_archive(root / 'snapshot.tar.gz', entries)

    def test_lfs_pointer_is_never_published_as_csv(self):
        with tempfile.TemporaryDirectory(prefix='olho-pack-') as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            (root / 'data/sample.csv').write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
            with patch.object(publish_snapshot, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'ponteiro LFS'):
                    publish_snapshot.entry('data/sample.csv', 'research')


if __name__ == '__main__':
    unittest.main()
