import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chatrag import knowledge as rk
from chatrag import telegram
from chatrag import vectors


class Model:
    def embed(self, texts, **kwargs):
        for text in texts:
            value = np.zeros(384, dtype='float32')
            value[0 if 'python' in text else 1] = 1
            yield value


class ImportIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.chats = self.root/'chats'
        self.chats.mkdir()
        (self.chats/'chat_01.txt').write_text('== 2022-01-01 ==\n00:00 Alice Example: старое подтверждённое сообщение\n')
        self.imports = self.root/'imports'
        self.source = self.root/'result.json'
        self.data = dict(id=123, name='test', messages=[
            dict(id=1,type='message',date='2020-01-01T12:00:00',from_id='u1',text='вопрос собеседника',**{'from':None}),
            dict(id=2,type='message',date='2020-01-01T12:01:00',from_id='u2',text='python гипотезатермин',reply_to_message_id=1,**{'from':None}),
            dict(id=3,type='message',date='2020-01-01T12:02:00',from_id='u3',text='python подтверждённый ответ',**{'from':'Alice Example'}),
            dict(id=4,type='message',date='2020-01-01T12:03:00',from_id='u2',text='пересылкатермин',forwarded_from='Other',**{'from':None}),
            dict(id=5,type='message',date='2020-01-01T12:04:00',from_id='u3',text='',photo='photo.jpg',**{'from':'Alice Example'})])
        self.source.write_text(json.dumps(self.data))
        self.imported = telegram.import_export(self.source,self.imports,'Alice Example',{'u2'})
        self.db = self.root/'index.sqlite'
        rk.build(self.chats,self.db,self.imports,'Alice Example')

    def test_uncertain_opt_in_and_exact_original(self):
        with rk.connect(self.db) as con:
            self.assertEqual(rk.search(con,'гипотезатермин',self.root,all_words=True)['results'],[])
            result = rk.search(con,'гипотезатермин',self.root,include_uncertain=True,all_words=True)
            hit = result['results'][0]
            self.assertEqual(hit['authorship'],'probable')
            self.assertEqual(hit['provenance']['sender_id'],'u2')
            self.assertEqual(hit['provenance']['message_id'],2)
            self.assertEqual(hit['reply_to']['body'],'вопрос собеседника')
            self.assertEqual(hit['reply_to']['authorship'],'other')
            self.assertIn('PROBABLE',rk.render(result)['prompt'])
            self.assertEqual(rk.search(con,'пересылкатермин',self.root,include_uncertain=True,all_words=True)['results'],[])

    def test_reimport_preserves_selection_and_detects_original_corruption(self):
        repeat = telegram.import_export(self.source,self.imports,'Alice Example',None)
        self.assertFalse(repeat['changed'])
        self.assertEqual(repeat['probable_ids'],['u2'])
        archive = Path(repeat['origin']['path'])
        archive.write_text('{}')
        with rk.connect(self.db) as con:
            with self.assertRaisesRegex(ValueError,'archive is corrupted'):
                rk.check_sources(con,full=True)

    def test_uncertainty_requires_literal_boolean_opt_in(self):
        with rk.connect(self.db) as con:
            for invalid in ['false', 'true', 0, 1, None]:
                with self.assertRaisesRegex(ValueError,'boolean'):
                    rk.search(con,'гипотезатермин',self.root,include_uncertain=invalid)

    def test_unicode_separators_do_not_split_jsonl_records(self):
        text = 'python гипотезатермин\u2028продолжение\u0085конец'
        self.data['messages'][1]['text'] = text
        self.source.write_text(json.dumps(self.data,ensure_ascii=False))
        telegram.import_export(self.source,self.imports,'Alice Example',{'u2'})
        rk.build(self.chats,self.db,self.imports,'Alice Example')
        with rk.connect(self.db) as con:
            hit = rk.search(con,'гипотезатермин',self.root,include_uncertain=True,all_words=True)['results'][0]
        self.assertEqual(hit['quote'],text)
        self.assertEqual(hit['provenance']['message_id'],2)

    def test_invalid_stored_selection_does_not_silently_reclassify(self):
        metadata = Path(self.imported['metadata'])
        data = json.loads(metadata.read_text())
        data['probable_ids'] = 'u2'
        metadata.write_text(json.dumps(data))
        projection = Path(self.imported['projection'])
        before = projection.read_bytes()
        with self.assertRaisesRegex(ValueError,'probable_ids'):
            telegram.import_export(self.source,self.imports,'Alice Example',None)
        self.assertEqual(projection.read_bytes(),before)

    def test_vector_indices_separate_confirmed_and_probable(self):
        with rk.connect(self.db) as con, patch.object(vectors,'embedder',return_value=Model()):
            result = vectors.build_vectors(con,self.db,self.root)
            self.assertEqual(result['vectors'],3)
            self.assertEqual(result['confirmed_vectors'],2)
            confirmed = vectors.candidates(con,self.db,'python',self.root,limit=10)
            broad = vectors.candidates(con,self.db,'python',self.root,limit=10,include_uncertain=True)
            possible = con.execute("SELECT message_id FROM provenance WHERE authorship='probable'").fetchone()[0]
            self.assertNotIn(possible,confirmed)
            self.assertIn(possible,broad)


if __name__ == '__main__':
    unittest.main()
