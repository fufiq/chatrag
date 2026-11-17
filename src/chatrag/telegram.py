"""Telegram JSON import that preserves original authorship and links back to the source."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _decode(raw, label):
    try:
        value = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid JSON: {label}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {label}')
    return value


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label}: a non-negative integer ID is required')
    return value


def _root(data):
    chat_id = _integer(data.get('id'), 'chat id')
    messages = data.get('messages')
    if not isinstance(messages, list):
        raise ValueError('The export must contain a messages array')
    seen = set()
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f'messages/{position}: expected an object')
        message_id = _integer(message.get('id'), f'messages/{position}/id')
        if message_id in seen:
            raise ValueError(f'Duplicate message id: {message_id}')
        seen.add(message_id)
        if not isinstance(message.get('type'), str):
            raise ValueError(f'messages/{position}: type is required')
    return chat_id, messages


def text_of(message):
    """Telegram text is made of strings and formatted fragments joined without a separator."""
    if 'text' not in message:
        raise ValueError('The message has no text')
    text = message['text']
    if isinstance(text, str):
        return text
    if not isinstance(text, list):
        raise ValueError('text must be a string or an array of fragments')
    result = []
    for part in text:
        if isinstance(part, str):
            result.append(part)
        elif isinstance(part, dict) and isinstance(part.get('text'), str):
            result.append(part['text'])
        else:
            raise ValueError('Invalid Telegram text fragment')
    return ''.join(result)


def _timestamp(value):
    if not isinstance(value, str) or 'T' not in value:
        raise ValueError('The message needs a full ISO date with time')
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'Invalid ISO date: {value}') from exc
    return stamp.date().isoformat(), stamp.strftime('%H:%M')


def _record(message, position, chat_id, origin, probable_ids, subject_name):
    message_id = _integer(message.get('id'), 'message id')
    date, time = _timestamp(message.get('date'))
    sender_id = message.get('from_id')
    sender_name = message.get('from')
    if not isinstance(sender_id, str) or not sender_id:
        raise ValueError(f'message {message_id}: from_id is required')
    if sender_name is not None and not isinstance(sender_name, str):
        raise ValueError(f'message {message_id}: from must be a string or null')
    reply = message.get('reply_to_message_id')
    if reply is not None:
        _integer(reply, f'message {message_id}/reply_to_message_id')
    forwarded = any(name in message for name in ('forwarded_from', 'saved_from', 'via_bot'))
    if forwarded:
        authorship, basis = 'forwarded', 'forwarded_or_generated_content'
    elif sender_name is not None and sender_name == subject_name:
        authorship, basis = 'confirmed', 'export_name'
    elif sender_id in probable_ids:
        authorship, basis = 'probable', 'user_selected_probable_account'
    else:
        authorship, basis = 'other', 'unattributed_sender'
    return dict(chat_id=chat_id, message_id=message_id, position=position, subject_name=subject_name,
        date=date, time=time, timestamp=message['date'], sender_id=sender_id,
        sender_name=sender_name, author=sender_name or f'Deleted Account [{sender_id}]',
        body=text_of(message), authorship=authorship, attribution_basis=basis,
        reply_to_message_id=reply, forwarded_from=message.get('forwarded_from'),
        saved_from=message.get('saved_from'), via_bot=message.get('via_bot'),
        origin=dict(path=str(origin['path']), sha256=origin['sha256'],
                    json_pointer=f'/messages/{position}'))


def _temporary(target, raw):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.telegram-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        os.unlink(temporary)
        raise
    return Path(temporary)


def _atomic_write(target, raw):
    if target.exists() and target.read_bytes() == raw:
        return False
    temporary = _temporary(target, raw)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _archive(target, raw, expected):
    if target.exists():
        if _hash(target.read_bytes()) != expected:
            raise ValueError(f'Archive is corrupted: {target}')
        return
    temporary = _temporary(target, raw)
    try:
        try:
            # Never overwrite an already published immutable archive.
            os.link(temporary, target)
        except FileExistsError:
            if _hash(target.read_bytes()) != expected:
                raise ValueError(f'Archive is corrupted: {target}') from None
        if _hash(target.read_bytes()) != expected:
            raise ValueError(f'Copy SHA-256 mismatch: {target}')
    finally:
        temporary.unlink(missing_ok=True)


def import_export(source, destination, subject_name, probable_ids=frozenset()):
    """Archive the original, then atomically publish a verified JSONL projection."""
    if not isinstance(subject_name, str) or not subject_name:
        raise ValueError('subject_name must be a non-empty string')
    source, destination = Path(source), Path(destination).resolve()
    if source.is_dir():
        source /= 'result.json'
    if probable_ids is not None and (not isinstance(probable_ids, (set, frozenset)) or any(not isinstance(x, str) or not x for x in probable_ids)):
        raise ValueError('probable_ids must be a set of sender IDs')
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f'Cannot read the export: {source}: {exc}') from exc
    data = _decode(raw, source)
    chat_id, messages = _root(data)
    metadata_path = destination / f'import_{chat_id}.json'
    try:
        previous = _decode(metadata_path.read_bytes(), metadata_path) if metadata_path.exists() else {}
    except OSError as exc:
        raise ValueError(f'Cannot read import metadata: {metadata_path}: {exc}') from exc
    if probable_ids is None:
        stored_ids = previous.get('probable_ids', [])
        if not isinstance(stored_ids, list) or any(not isinstance(x, str) or not x for x in stored_ids):
            raise ValueError('Invalid probable_ids in the import metadata')
        probable_ids = frozenset(stored_ids)
    sha = _hash(raw)
    archive = destination / 'archives' / sha / 'result.json'
    origin = dict(path=str(archive), sha256=sha)
    records = [_record(m, i, chat_id, origin, probable_ids, subject_name)
               for i, m in enumerate(messages) if m['type'] == 'message']
    counts = dict(Counter(record['authorship'] for record in records))
    for status in ('confirmed', 'probable', 'other', 'forwarded'):
        counts.setdefault(status, 0)
    projection = destination / f'chat_telegram_{chat_id}.jsonl'
    metadata = dict(schema=1, chat_id=chat_id, name=data.get('name'), origin=origin,
        projection=str(projection), records=len(records), exported_records=len(messages),
        counts=counts, probable_ids=sorted(probable_ids))
    previous_content = {k:v for k,v in previous.items() if k not in ('created_at', 'updated_at')}
    if previous_content == metadata:
        metadata = previous
    else:
        now = datetime.now(timezone.utc).isoformat()
        metadata.update(created_at=previous.get('created_at', now), updated_at=now)
    projection_bytes = ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records).encode('utf-8')
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    # No output file is touched until the input export has been fully validated.
    try:
        _archive(archive, raw, sha)
        changed = _atomic_write(projection, projection_bytes)
        _atomic_write(metadata_path, metadata_bytes)
    except OSError as exc:
        raise ValueError(f'Cannot save the import to {destination}: {exc}') from exc
    return dict(chat_id=chat_id, origin=origin, projection=str(projection),
                metadata=str(metadata_path), messages=len(records), counts=counts,
                probable_ids=sorted(probable_ids), changed=changed)


@lru_cache(maxsize=4)
def _archive_data(path, mtime_ns, size, expected):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f'Original unavailable: {path}') from exc
    if _hash(raw) != expected:
        raise ValueError(f'Original SHA-256 mismatch: {path}')
    data = _decode(raw, path)
    _root(data)
    return data


def verify_record(provenance, subject_name=None):
    """Verify the link to the source message without trusting the name in the projection."""
    if not isinstance(provenance, dict) or not isinstance(provenance.get('origin'), dict):
        raise ValueError('The record has no origin')
    origin = provenance['origin']
    path, expected = origin.get('path'), origin.get('sha256')
    if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Invalid origin reference')
    try:
        stat = Path(path).stat()
    except OSError as exc:
        raise ValueError(f'Original unavailable: {path}') from exc
    data = _archive_data(path, stat.st_mtime_ns, stat.st_size, expected)
    position = _integer(provenance.get('position'), 'position')
    _integer(provenance.get('chat_id'), 'chat_id')
    _integer(provenance.get('message_id'), 'message_id')
    if position >= len(data['messages']) or origin.get('json_pointer') != f'/messages/{position}':
        raise ValueError('Invalid message position in the original')
    message = data['messages'][position]
    if message.get('type') != 'message':
        raise ValueError('The reference does not point to a message')
    status = provenance.get('authorship')
    if status not in ('confirmed', 'probable', 'other', 'forwarded'):
        raise ValueError('Unknown authorship status')
    probable = {provenance.get('sender_id')} if status == 'probable' else set()
    # The subject comes from the index when available, never only from the record itself.
    subject = subject_name if subject_name is not None else provenance.get('subject_name')
    if provenance.get('subject_name') != subject:
        raise ValueError('Record does not match the original: field subject_name')
    expected_record = _record(message, position, data['id'], origin, probable, subject)
    fields = ('chat_id', 'message_id', 'subject_name', 'date', 'time', 'timestamp', 'sender_id', 'sender_name',
              'author', 'body', 'reply_to_message_id', 'forwarded_from', 'saved_from', 'via_bot',
              'authorship', 'attribution_basis')
    for name in fields:
        if provenance.get(name) != expected_record[name]:
            raise ValueError(f'Record does not match the original: field {name}')
    return message


def read_projection(path, subject_name=None):
    path = Path(path)
    try:
        raw = path.read_bytes()
        lines = raw.decode('utf-8').split('\n')
        if lines and lines[-1] == '':
            lines.pop()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f'Projection unavailable: {path}') from exc
    result = []
    seen = set()
    for line, text in enumerate(lines, 1):
        record = _decode(text.encode('utf-8'), f'{path}:{line}')
        verify_record(record, subject_name)
        identity = (record['chat_id'], record['message_id'])
        if identity in seen:
            raise ValueError(f'Duplicate message in the projection: {identity}')
        seen.add(identity)
        author = record.get('author')
        if not isinstance(author, str) or not author:
            raise ValueError(f'Missing author: {path}:{line}')
        result.append(dict(date=record['date'], time=record['time'], author=author,
            body=record['body'], line=line, end_line=line,
            header=record['time'] + ' ' + author + ': ',
            subject=record['authorship'] in ('confirmed', 'probable'), provenance=record))
    return result, raw
