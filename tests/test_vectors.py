from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

import numpy as np
from usearch.index import Index

from chatrag import knowledge as rk
from chatrag import vectors as rv


class FakeModel:
    def embed(self, texts, **kwargs):
        for text in texts:
            vector = np.zeros(384, dtype='float32')
            vector[0 if 'python' in text else 1] = 1
            yield vector


class VectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.chats = self.root / 'chats'
        self.chats.mkdir()
        (self.chats / 'chat_01.txt').write_text(
            '== 2023-01-01 ==\n00:00 Other: вопрос о python\n'
            '00:01 Alice Example: python хорош для проверки этой функции\n'
            '00:02 Alice Example: да\n00:03 Alice Example: javascript другая технология\n')
        self.db = self.root / 'index.sqlite'
        rk.build(self.chats, self.db, subject='Alice Example')

    def test_persistent_index_includes_short_messages_and_is_reusable(self):
        with rk.connect(self.db) as con, patch.object(rv, 'embedder', return_value=FakeModel()):
            result = rv.build_vectors(con, self.db, self.root)
            self.assertEqual(result['vectors'], 3)
            path = rv.index_path(self.db, con)
            one = Index.restore(str(path), view=True)
            two = Index.restore(str(path), view=True)
            self.assertEqual(len(one), len(two))
            self.assertTrue(rv.build_vectors(con, self.db, self.root)['reused'])
            self.assertTrue(rv.candidates(con, self.db, 'python', self.root, 2))

    def test_changed_corpus_requires_new_generation(self):
        with rk.connect(self.db) as con, patch.object(rv, 'embedder', return_value=FakeModel()):
            rv.build_vectors(con, self.db, self.root)
            old = rv.index_path(self.db, con)
        with (self.chats / 'chat_01.txt').open('a') as out:
            out.write('00:04 Alice Example: новое сообщение про python\n')
        rk.build(self.chats, self.db, subject='Alice Example')
        with rk.connect(self.db) as con, patch.object(rv, 'embedder', return_value=FakeModel()):
            self.assertNotEqual(old, rv.index_path(self.db, con))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                rv.candidates(con, self.db, 'python', self.root)
            updated = rv.build_vectors(con, self.db, self.root)
            self.assertEqual(updated['vectors'], 4)
            self.assertLess(updated['generated'], 4)

    def test_corrupted_index_is_detected_and_rebuilt_from_cache(self):
        with rk.connect(self.db) as con, patch.object(rv, 'embedder', return_value=FakeModel()):
            rv.build_vectors(con, self.db, self.root)
            path = rv.index_path(self.db, con)
            path.write_bytes(b'broken')
            rv.load_index.cache_clear()
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                rv.load_index(str(path))
            result = rv.build_vectors(con, self.db, self.root)
            self.assertEqual(result['generated'], 0)
            self.assertEqual(len(rv.load_index(str(path))), 3)

    def test_model_manifest_is_verified_before_loading_onnx(self):
        model_dir = self.root / 'model'
        asset = model_dir / rv.RERANK_FILE
        asset.parent.mkdir(parents=True)
        asset.write_text('fake-onnx')
        (model_dir / 'provenance.json').write_text(json.dumps(dict(
            repository=rv.RERANK_REPO, revision=rv.RERANK_REVISION,
            files={rv.RERANK_FILE:'bad-hash','tokenizer.json':'bad-hash'})))
        rv.reranker.cache_clear()
        with patch.object(rv, 'MODEL_DIR', model_dir):
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                rv.reranker()
        rv.reranker.cache_clear()


if __name__ == '__main__':
    unittest.main()
