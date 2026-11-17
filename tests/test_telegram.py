import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from chatrag import telegram


class TelegramTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'export'
        self.source.mkdir()
        self.destination = self.root / 'imports'
        self.data = dict(id=123, name='fixture', messages=[
            dict(id=1, type='service', action='migrate_from_group'),
            dict(id=2, type='message', date='2020-07-21T19:50:17',
                 **{'from':None, 'from_id':'user-old'}, text=['часть ', {'type':'bold','text':'два'}, '\n00:00 Other: это текст']),
            dict(id=3, type='message', date='2021-02-22T19:06:17',
                 **{'from':'Alice Example', 'from_id':'user-subject'}, text='ответ', reply_to_message_id=2),
            dict(id=4, type='message', date='2021-02-22T19:07:17',
                 **{'from':None, 'from_id':'user-other'}, text='чужое мнение'),
            dict(id=5, type='message', date='2021-02-22T19:08:17',
                 **{'from':'Alice Example', 'from_id':'user-subject'}, text='чужая пересылка', forwarded_from='Другой'),
            dict(id=6, type='message', date='2021-02-22T19:09:17',
                 **{'from':'Alice Example', 'from_id':'user-subject'}, text='', photo='not exported')])
        self.write()

    def write(self):
        (self.source/'result.json').write_text(json.dumps(self.data, ensure_ascii=False))

    def import_it(self):
        return telegram.import_export(self.source, self.destination, 'Alice Example', {'user-old'})

    def test_rich_text_multiline_sender_ids_and_authorship(self):
        result = self.import_it()
        rows, raw = telegram.read_projection(Path(result['projection']))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]['body'], 'часть два\n00:00 Other: это текст')
        self.assertEqual(rows[0]['author'], 'Deleted Account [user-old]')
        self.assertEqual(rows[2]['author'], 'Deleted Account [user-other]')
        self.assertEqual(rows[0]['provenance']['authorship'], 'probable')
        self.assertEqual(rows[1]['provenance']['authorship'], 'confirmed')
        self.assertTrue(rows[0]['subject'])
        self.assertFalse(rows[2]['subject'])
        self.assertEqual(rows[0]['line'], rows[0]['end_line'])
        self.assertEqual(rows[0]['provenance']['position'], 1)
        self.assertEqual(rows[1]['provenance']['reply_to_message_id'], 2)
        self.assertEqual(result['counts'], dict(probable=1,confirmed=2,other=1,forwarded=1))

    def test_forwarded_and_bot_content_is_not_confirmed(self):
        for field in ('forwarded_from', 'saved_from', 'via_bot'):
            with self.subTest(field=field):
                self.data['messages'][2][field] = 'другой'
                self.write()
                result = self.import_it()
                rows, _ = telegram.read_projection(Path(result['projection']))
                self.assertEqual(rows[1]['provenance']['authorship'], 'forwarded')
                self.assertFalse(rows[1]['subject'])
                del self.data['messages'][2][field]

    def test_empty_media_text_is_preserved_without_invented_quote(self):
        result = self.import_it()
        rows, _ = telegram.read_projection(Path(result['projection']))
        self.assertEqual(rows[-1]['body'], '')

    def test_archive_hash_mode_and_idempotence(self):
        original = (self.source/'result.json').read_bytes()
        first = self.import_it()
        archive = Path(first['origin']['path'])
        projection = Path(first['projection'])
        metadata = Path(first['metadata'])
        before = (archive.stat().st_mtime_ns, projection.stat().st_mtime_ns, metadata.read_bytes())
        second = self.import_it()
        self.assertFalse(second['changed'])
        self.assertEqual(before, (archive.stat().st_mtime_ns,projection.stat().st_mtime_ns,metadata.read_bytes()))
        self.assertEqual(archive.read_bytes(), original)
        self.assertEqual((self.source/'result.json').read_bytes(), original)
        self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
        self.assertEqual(telegram._hash(archive.read_bytes()), first['origin']['sha256'])

    def test_malformed_input_does_not_replace_projection(self):
        result = self.import_it()
        projection = Path(result['projection'])
        before = projection.read_bytes()
        bad = copy.deepcopy(self.data)
        bad['messages'][-1]['text'] = [dict(type='bold',text=123)]
        self.data = bad
        self.write()
        with self.assertRaises(ValueError):
            self.import_it()
        self.assertEqual(projection.read_bytes(), before)

    def test_root_id_duplicates_sender_and_date_validation(self):
        mutations = [lambda d:d.update(messages={}), lambda d:d.update(id=True),
                     lambda d:d['messages'].append(copy.deepcopy(d['messages'][1])),
                     lambda d:d['messages'][1].update(date='2020-99-01T01:00:00'),
                     lambda d:d['messages'][1].pop('from_id')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                (self.source/'result.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.import_it()

    def test_record_verification_checks_original_and_metadata(self):
        result = self.import_it()
        rows, _ = telegram.read_projection(Path(result['projection']))
        record = rows[0]['provenance']
        self.assertEqual(telegram.verify_record(record)['id'], 2)
        for field, value in [('body','придуманный текст'),('sender_id','другой'),('timestamp','2020-07-21T00:00:00'),('message_id',999),('chat_id',456)]:
            bad = copy.deepcopy(record)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                telegram.verify_record(bad)
        archive = Path(result['origin']['path'])
        archive.write_bytes(archive.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            telegram.verify_record(record)

    def test_changed_import_keeps_previous_immutable_archive(self):
        first = self.import_it()
        archive = Path(first['origin']['path'])
        before = archive.read_bytes()
        self.data['messages'][2]['text']='другая версия'
        self.write()
        second = self.import_it()
        self.assertNotEqual(first['origin']['path'],second['origin']['path'])
        self.assertEqual(archive.read_bytes(),before)
        self.assertTrue(second['changed'])

    def test_projection_cannot_promote_deleted_or_forwarded_author(self):
        result = self.import_it()
        rows, _ = telegram.read_projection(Path(result['projection']))
        for record in [rows[0]['provenance'], rows[3]['provenance']]:
            bad = copy.deepcopy(record)
            bad.update(authorship='confirmed',attribution_basis='export_name')
            with self.assertRaises(ValueError):
                telegram.verify_record(bad)

    def test_subject_name_cannot_be_rewritten_in_the_projection(self):
        result = self.import_it()
        rows, _ = telegram.read_projection(Path(result['projection']))
        # A sender that is not the subject must not become 'confirmed' by editing subject_name,
        # neither to the sender's own name nor to null (null sender == null subject).
        record = copy.deepcopy(rows[2]['provenance'])
        record.update(subject_name=None, authorship='confirmed', attribution_basis='export_name')
        with self.assertRaises(ValueError):
            telegram.verify_record(record)
        record = copy.deepcopy(rows[0]['provenance'])
        record.update(subject_name='Mallory')
        with self.assertRaisesRegex(ValueError, 'subject_name'):
            telegram.verify_record(record, 'Alice Example')

    def test_filesystem_error_is_actionable_value_error(self):
        with patch.object(telegram, '_archive', side_effect=PermissionError('fixture')):
            with self.assertRaisesRegex(ValueError, 'Cannot save the import'):
                self.import_it()

    def test_boolean_provenance_ids_are_rejected(self):
        result = self.import_it()
        rows, _ = telegram.read_projection(Path(result['projection']))
        record = copy.deepcopy(rows[0]['provenance'])
        record['message_id'] = True
        with self.assertRaises(ValueError):
            telegram.verify_record(record)


if __name__ == '__main__':
    unittest.main()
