import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from chatrag import knowledge as rk
from chatrag import terms

SUBJECT = 'Alice Example'


class KnowledgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.chats = self.root / 'chats'
        self.chats.mkdir()
        (self.chats / 'chat_01.txt').write_text(
            '== 2022-09-21 ==\n'
            '23:07 Other: редактор без AST?\n'
            '23:08 Alice Example (→Other: "редактор без AST?"): редактор должен редактировать аст\n'
            'это продолжение\n'
            '23:09 Other (→Alice Example: "ложная цитата"): чужая позиция\n'
            '23:10 Alice Example: да\n')
        (self.chats / 'chat_02.txt').write_text(
            '23:11 Alice Example: jit компилирует функцию\n'
            '== 2023-05-29 ==\n'
            '19:24 Alice Example: ссылки остаются рабочими\n')
        self.db = self.root / 'index.sqlite'
        self.stats = rk.build(self.chats, self.db, subject=SUBJECT)
        self.con = rk.connect(self.db)
        self.addCleanup(self.con.close)

    def find(self, query, **kwargs):
        return rk.search(self.con, query, self.root, **kwargs)

    def test_provenance_and_context(self):
        result = self.find('редактор аст', all_words=True)['results'][0]
        self.assertEqual(result['line'], 3)
        self.assertEqual(result['end_line'], 4)
        self.assertEqual(result['date'], '2022-09-21')
        self.assertEqual(result['quote'], 'редактор должен редактировать аст\nэто продолжение')
        self.assertEqual(result['context_before'][0]['author'], 'Other')
        self.assertTrue(result['verified'])
        self.assertIn('→Other:', result['header'])

    def test_reply_previews_are_not_subject_quotes(self):
        self.assertEqual(self.find('ложная цитата')['results'], [])
        self.assertEqual(self.stats['subject_messages'], 4)
        self.assertEqual(self.find('без', all_words=True)['results'], [])
        self.assertTrue(all(r['matched_in'] == 'context_or_dense' for r in self.find('без')['results']))

    def test_single_concept_query_returns_mentions_not_neighboring_replies(self):
        result = self.find('ast')
        self.assertEqual(result['query_policy'], 'explicit_concept')
        self.assertTrue(result['results'])
        self.assertTrue(all('zconceptast' in terms.normalize(r['quote']) for r in result['results']))

    def test_reply_header_does_not_consume_parentheses_in_body(self):
        p = self.chats / 'chat_03.txt'
        p.write_text('10:00 Alice Example (→Other: "вопрос: почему?"): foo(x): это ответ\n')
        rows, _, _ = rk.parse_file(p, '', SUBJECT)
        self.assertEqual(rows[0]['body'], 'foo(x): это ответ')
        self.assertEqual(rows[0]['author'], 'Alice Example')

    def test_dates_short_messages_dictionary_and_filters(self):
        self.assertEqual(self.find('жит')['results'][0]['date'], '2022-09-21')
        self.assertEqual(self.find('да')['results'][0]['quote'], 'да')
        self.assertEqual(self.find('жит', since='2023')['results'], [])
        self.assertEqual(self.find('ссылки', until='2022')['results'], [])

    def test_stale_sources_fail_closed(self):
        with (self.chats / 'chat_01.txt').open('a') as stream:
            stream.write('23:12 Alice Example: изменено\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.find('редактор')

    def test_added_sources_require_rebuild(self):
        (self.chats / 'chat_03.txt').write_text('00:00 Alice Example: новый чат\n')
        with self.assertRaisesRegex(ValueError, 'set of chat files'):
            self.find('редактор')

    def test_chronology_applies_before_candidate_limit(self):
        # A rare word inside a long old message gets a low BM25 score.
        (self.chats / 'chat_03.txt').write_text(
            '== 2020-01-01 ==\n00:00 Alice Example: ключ ' + 'длинный ' * 500 + '\n'
            '== 2024-01-01 ==\n' + '00:01 Alice Example: ключ\n' * 310)
        rk.build(self.chats, self.db, subject=SUBJECT)
        with rk.connect(self.db) as con:
            result = rk.search(con, 'ключ', self.root, limit=1, order='oldest')
        self.assertEqual(result['results'][0]['date'], '2020-01-01')

    def test_until_year_is_inclusive(self):
        self.assertTrue(self.find('редактор', until='2022')['results'])

    def test_invalid_dates_are_rejected(self):
        for value in ['2023-13', '2023-02-30', '0000']:
            with self.assertRaises(ValueError):
                self.find('редактор', until=value)
        with self.assertRaises(ValueError):
            self.find('редактор', since='2024', until='2022')

    def test_live_process_refreshes_aliases_after_index_update(self):
        alias_path = self.root / 'aliases.json'
        alias_path.write_bytes(terms.ALIASES_PATH.read_bytes())
        def fingerprint():
            return rk.digest(alias_path.read_bytes())
        try:
            with (patch.object(terms, 'ALIASES_PATH', alias_path),
                  patch.object(rk, 'terms_fingerprint', side_effect=fingerprint)):
                rk.build(self.chats, self.db, subject=SUBJECT)
                with rk.connect(self.db) as con:
                    self.assertEqual(rk.search(con, 'новоредактор', self.root)['results'], [])
                data = json.loads(alias_path.read_text())
                next(c for c in data['concepts'] if c['id']=='editor')['aliases'].append('новоредактор')
                alias_path.write_text(json.dumps(data))
                rk.build(self.chats, self.db, subject=SUBJECT)
                with rk.connect(self.db) as con:
                    self.assertTrue(rk.search(con, 'новоредактор', self.root)['results'])
        finally:
            terms.refresh_dictionary()

    def test_changed_normalization_code_requires_process_restart(self):
        with patch.object(rk, 'LOADED_TERMS_CODE_HASH', 'old-code'):
            with self.assertRaisesRegex(ValueError, 'Restart'):
                rk.connect(self.db)

    def test_backend_exceptions_are_visible_and_do_not_hide_lexical_results(self):
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf
        with (patch('chatrag.vectors.candidates', side_effect=InvalidProtobuf('broken fixture')),
              patch('chatrag.vectors.rerank', side_effect=InvalidProtobuf('broken reranker'))):
            bundle = self.find('редактор', semantic=True)
        self.assertTrue(bundle['results'])
        self.assertEqual(len(bundle['warnings']), 2)
        self.assertNotIn('reranker', bundle['stages'])

    def test_reranker_sees_matches_at_end_of_long_messages(self):
        p = self.chats / 'chat_03.txt'
        p.write_text('== 2023-01-01 ==\n00:00 Alice Example: ' + 'погода ' * 500 + 'маркерфункции\n')
        rk.build(self.chats, self.db, subject=SUBJECT)
        def check(query, passages):
            return [1.0 if 'маркерфункции' in p else -9.0 for p in passages]
        with rk.connect(self.db) as con, patch('chatrag.vectors.candidates', return_value=[]), patch('chatrag.vectors.rerank', side_effect=check):
            bundle = rk.search(con, 'маркерфункции', self.root, semantic=True)
        self.assertTrue(bundle['results'])
        self.assertEqual(bundle['results'][0]['id'], 'chat_03.txt:2')

    def test_fts_input_and_budget(self):
        self.assertEqual(self.find('" OR * : ( )')['results'], [])
        bundle = self.find('редактор')
        budget = rk.render(dict(bundle, results=[]), 10000)['used_chars'] + 50
        rendered = rk.render(bundle, budget)
        self.assertLessEqual(rendered['used_chars'], budget)
        self.assertTrue(rendered['omitted'])

    def test_large_source_is_excerpted_inside_prompt_budget(self):
        bundle = self.find('редактор')
        bundle['results'][0]['quote'] = 'погода ' * 3000 + 'редактор аст'
        rendered = rk.render(bundle, 2400)
        self.assertLessEqual(rendered['used_chars'], 2400)
        self.assertTrue(rendered['included'])
        self.assertIn('редактор аст', rendered['prompt'])

    def test_conversation_keeps_answer_several_replies_after_anchor(self):
        p = self.chats / 'chat_03.txt'
        p.write_text('== 2023-01-01 ==\n10:00 Alice Example: завязка разговора\n'
                     + ''.join(f'10:0{i} Other: промежуточная реплика {i}\n' for i in range(1,7))
                     + '10:07 Alice Example: доска на сервере, клиент рисует\n')
        rk.build(self.chats, self.db, subject=SUBJECT)
        with rk.connect(self.db) as con:
            result = rk.search(con, 'завязка', self.root)
        anchor = next(r for r in result['results'] if r['line']==2)
        self.assertTrue(any('клиент рисует' in m['body'] for m in anchor['context_after']))
        anchor['quote'] = 'введение ' * 2000
        result['results'] = [anchor]
        result['query'] = 'доска клиент'
        prompt = rk.render(result, 3000)['prompt']
        self.assertIn('доска на сервере, клиент рисует', prompt)

    def test_context_does_not_include_distant_unrelated_day(self):
        result = self.find('ссылки')['results'][0]
        self.assertEqual(result['context_before'], [])

    def test_offline_failure_is_visible(self):
        # No vector index was built, so the dense channel must fail loudly and lexical results survive.
        bundle = self.find('редактор', semantic=True, rerank=False)
        self.assertEqual(bundle['retrieval'], 'lexical')
        self.assertTrue(bundle['warnings'])
        self.assertTrue(bundle['results'])

    def test_mcp_end_to_end(self):
        requests = [dict(jsonrpc='2.0', id=1, method='initialize'),
                    dict(jsonrpc='2.0', method='notifications/initialized'),
                    dict(jsonrpc='2.0', id=2, method='tools/list'),
                    dict(jsonrpc='2.0', id=3, method='tools/call', params=dict(
                        name='chat_search', arguments={'query': 'редактор'})),
                    dict(jsonrpc='2.0', id=4, method='tools/call', params=dict(
                        name='chat_source', arguments={'id': 'chat_01.txt:3'})),
                    dict(jsonrpc='2.0', id=5, method='tools/call', params=dict(
                        name='chat_quote', arguments={'phrase': 'редактор должен редактировать аст', 'context': 1, 'limit': 1}))]
        proc = subprocess.run([sys.executable, '-m', 'chatrag.knowledge', '--db', str(self.db),
                               '--models-dir', str(self.root), '--lexical', 'serve'],
                              input='\n'.join(json.dumps(r) for r in requests)+'\n',
                              capture_output=True, text=True, check=True)
        responses = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([r['id'] for r in responses], [1, 2, 3, 4, 5])
        self.assertEqual(len(responses[1]['result']['tools']), 4)
        quote = json.loads(responses[4]['result']['content'][0]['text'])
        self.assertTrue(any(m['id'] == 'chat_01.txt:3' for m in quote['results'][0]['window']))
        source = json.loads(responses[3]['result']['content'][0]['text'])
        self.assertEqual(source['line'], 3)

    def test_mcp_malformed_input_does_not_stop_server(self):
        proc = subprocess.run([sys.executable, '-m', 'chatrag.knowledge', 'serve'],
                              input='[]\n{"jsonrpc":"2.0","id":5,"method":"ping"}\n',
                              capture_output=True, text=True, check=True)
        responses = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertIn('error', responses[0])
        self.assertEqual(responses[1]['id'], 5)


if __name__ == '__main__':
    unittest.main()
