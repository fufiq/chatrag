from pathlib import Path
import tempfile
import unittest

from chatrag import knowledge as rk
from chatrag import quote

SUBJECT = 'Alice Example'


class QuoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        chats = root / 'chats'
        chats.mkdir()
        (chats / 'chat_01.txt').write_text(
            '== 2022-11-03 ==\n'
            '10:46 Other: как это сделать проще\n'
            '10:49 Alice Example: в этом плане жит ничем не отличается от обычной компиляции\n'
            '10:51 Alice Example: единственная разница в том, что жит не требует компиляции программы полностью\n'
            '10:52 Other: ясно\n'
            '10:53 Alice Example: производительность вообще не важна\n'
            '10:54 Other: совсем другая тема про погоду и яблоки\n'
            '10:55 Other: ещё одна реплика про яблоки и груши\n')
        self.db = root / 'index.sqlite'
        rk.build(chats, self.db, subject=SUBJECT)
        self.con = rk.connect(self.db)
        self.addCleanup(self.con.close)

    def test_misquoted_phrase_finds_window_with_context(self):
        r = quote.find_quote(self.con, 'жит не отличается от компиляции, не нужно компилировать всю программу', context=2)
        top = r['results'][0]
        self.assertIn(top['line'], (3, 4))  # both messages of the subject contain query words
        self.assertGreaterEqual(top['end_line'], top['line'])
        self.assertEqual(top['window'][0]['author'], 'Alice Example')
        self.assertEqual(top['window'][0]['authorship'], 'confirmed')
        around = top['context_before'] + top['context_after']
        self.assertTrue(around)
        self.assertTrue(any(m['author'] == 'Other' and m['authorship'] == 'other' for m in around))

    def test_phrase_spread_over_neighbouring_messages_is_one_window(self):
        r = quote.find_quote(self.con, 'жит ничем не отличается от компиляции единственная разница жит не требует компиляции программы')
        top = r['results'][0]
        self.assertEqual((top['line'], top['end_line']), (3, 4))

    def test_absent_phrase_has_low_score_and_note(self):
        r = quote.find_quote(self.con, 'данным ненужны типы, типы нужны операциям')
        self.assertTrue(all(x['score'] < 0.5 for x in r['results']))
        self.assertIn('not a verbatim quote', r['note'])

    def test_short_query_is_rejected_with_warning(self):
        r = quote.find_quote(self.con, 'жит')
        self.assertEqual(r['results'], [])
        self.assertTrue(r['warnings'])


if __name__ == '__main__':
    unittest.main()
