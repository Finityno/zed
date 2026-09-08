"""Run only after the coordinator grants this exact source-check slot."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
log_root = Path('/tmp/fincode-performance-phase-20260908/gpui-owner-compiler')
log_root.mkdir(parents=True, exist_ok=True)

def check(binary):
    command = ['cargo', 'check', '--manifest-path', str(root / 'Cargo.toml'),
               '--bin', binary, '--features', 'compile-fail', '--message-format=json', '-q']
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (log_root / (binary + '.stdout.jsonl')).write_text(result.stdout)
    (log_root / (binary + '.stderr.log')).write_text(result.stderr)
    errors = []
    for line in result.stdout.splitlines():
        record = json.loads(line)
        if record.get('reason') == 'compiler-message' and record['message']['level'] == 'error':
            message = record['message']
            if message['message'].startswith('aborting due to') and not message['spans']:
                continue
            errors.append(record)
    return result, errors

positive, errors = check('positive-control')
if positive.returncode != 0 or errors:
    sys.exit('FAIL: same-feature positive control did not compile; inspect ' + str(log_root))

for binary, code in [('reservation_clone', 'E0599'),
                     ('reservation_after_publish', 'E0382'),
                     ('published_raw_escape', 'E0616')]:
    source = root / (binary + '.rs')
    marker_lines = [number for number, line in enumerate(source.read_text().splitlines(), 1)
                    if 'expected-error:' + code in line]
    if len(marker_lines) != 1:
        sys.exit('FAIL: ambiguous expected diagnostic marker: ' + binary)
    result, errors = check(binary)
    if result.returncode == 0 or not errors:
        sys.exit('FAIL: misuse unexpectedly compiled or produced no structured error: ' + binary)
    for record in errors:
        message = record['message']
        primary = [span for span in message['spans'] if span['is_primary']]
        if (record['target']['name'] != binary or (message.get('code') or {}).get('code') != code
                or not any(Path(span['file_name']).name == source.name
                           and span['line_start'] == marker_lines[0] for span in primary)):
            sys.exit('FAIL: unrelated diagnostic must not satisfy negative control: ' + binary)
    print('PASS compiler contract: ' + binary)
