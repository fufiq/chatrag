import sqlite3
import unittest
from unittest.mock import patch

from chatrag import terms


class TermsTest(unittest.TestCase):
    def test_python_aliases_are_separate_from_other_languages(self):
        expected = ['zconceptpython']
        for alias in ['python', 'питон', 'питона', 'питоне', 'piton']:
            self.assertEqual(terms.normalize(alias), expected, alias)
        self.assertEqual(terms.normalize('js'), ['zconceptjavascript'])
        self.assertEqual(terms.normalize('ts'), ['zconcepttypescript'])
        self.assertNotIn('zconceptpython', terms.normalize('скриптовые языки'))

    def test_iouring_aliases(self):
        for alias in ['io_uring', 'io-uring', 'io uring', 'uring']:
            self.assertEqual(terms.normalize(alias), ['zconceptiouring'], alias)

    def test_transliteration_and_morphology(self):
        self.assertEqual(terms.normalize('жит zhit jit'), ['zconceptjit'] * 3)
        self.assertEqual(terms.normalize('редактором'), terms.normalize('редактор'))
        self.assertEqual(terms.normalize('функциями'), terms.normalize('функция'))
        self.assertEqual(terms.normalize('shlang'), ['zconceptclang'])
        self.assertNotEqual(terms.normalize('жить'), terms.normalize('жит'))

    def test_cpp_symbol_is_not_lost(self):
        self.assertEqual(terms.normalize('C++ cpp плюсы'), ['zconceptcpp'] * 3)
        self.assertNotEqual(terms.normalize('C'), terms.normalize('C++'))

    def test_intermediate_representation_aliases_remain_separate_from_llvm(self):
        for term in ['ir', 'ир', 'intermediate representation', 'промежуточное представление']:
            self.assertEqual(terms.normalize(term), ['zconceptir'])
        self.assertNotEqual(terms.normalize('ir'), terms.normalize('llvm'))
        self.assertIn('intermediate representation', terms.semantic_text('для жит ненужны никакие ир'))

    def test_aliases_inside_compound_words(self):
        self.assertIn('zconceptpython', terms.normalize('питон-подобный'))
        self.assertIn('zconceptrust', terms.normalize('раст-модель'))
        self.assertEqual(terms.normalize('io_uring'), terms.normalize('io uring'))
        self.assertEqual(terms.semantic_text('питон-подобный'), 'python-подобный')

    def test_typo_correction_is_limited_and_explained(self):
        result = terms.expand_query('pythoon')
        self.assertEqual(result['normalized'], ['zconceptpython'])
        self.assertEqual(result['changes'][0]['method'], 'technical_typo')
        self.assertGreaterEqual(result['changes'][0]['score'], 90)
        self.assertNotEqual(terms.normalize('jti'), terms.normalize('jit'))
        self.assertNotIn('zconceptpython', terms.normalize('питание'))

    def test_ambiguous_typo_is_not_corrected(self):
        dictionary = ({('foobarx',): {'first'}, ('foobary',): {'second'}}, {}, {})
        terms._resolve_single.cache_clear()
        with patch.object(terms, '_dictionary', return_value=dictionary):
            self.assertIsNone(terms._resolve_single('foobar', True)[0])
        terms._resolve_single.cache_clear()

    def test_explain_preserves_original_and_canonical_dense_variant(self):
        result = terms.expand_query('Как работает io_uring и питон?')
        self.assertEqual(result['original'], 'Как работает io_uring и питон?')
        self.assertEqual(result['concepts'], ['iouring', 'python'])
        self.assertIn('Как работает io_uring и python?', result['variants'])
        self.assertEqual(result['changes'][0]['method'], 'stopword')

    def test_document_normalization_does_not_correct_typos(self):
        self.assertNotEqual(terms.normalize('pythoon'), terms.normalize('python'))
        self.assertEqual(terms.normalize('pythoon', fuzzy=True), terms.normalize('python'))

    def test_semantic_text_preserves_readable_text_and_punctuation(self):
        self.assertEqual(terms.semantic_text('Как работает IO_URING, а питон?'),
                         'Как работает io_uring, а python?')
        self.assertEqual(terms.semantic_text('Нет, функции меняются!'), 'Нет, функции меняются!')
        self.assertEqual(terms.semantic_text('pythoon', fuzzy=False), 'pythoon')
        self.assertEqual(terms.semantic_text('pythoon'), 'python')

    def test_fts_and_requires_each_meaningful_word(self):
        con = sqlite3.connect(':memory:')
        self.addCleanup(con.close)
        con.execute('CREATE VIRTUAL TABLE docs USING fts5(body)')
        for text in ['Python работает', 'JS работает', 'io_uring работает', 'скрипт работает']:
            con.execute('INSERT INTO docs VALUES (?)', (' '.join(terms.normalize(text)),))
        result = con.execute('SELECT rowid FROM docs WHERE docs MATCH ?',
                             (terms.query_expression('питон работает', True),)).fetchall()
        self.assertEqual(result, [(1,)])
        result = con.execute('SELECT rowid FROM docs WHERE docs MATCH ?',
                             (terms.query_expression('python io_uring', True),)).fetchall()
        self.assertEqual(result, [])
        # FTS operators and quotes become plain search text.
        con.execute('SELECT rowid FROM docs WHERE docs MATCH ?',
                    (terms.query_expression('python" OR * : ( )'),)).fetchall()

    def test_empty_queries_fail_and_content_negation_stays(self):
        for query in ['', ' и в на ', '*** : ()']:
            with self.assertRaises(ValueError):
                terms.expand_query(query)
        self.assertIn(terms.normalize('без')[0], terms.normalize('без python'))
        self.assertIn(terms.normalize('не')[0], terms.normalize('не python'))


if __name__ == '__main__':
    unittest.main()
