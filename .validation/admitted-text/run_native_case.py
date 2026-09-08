"""Run only after review and an explicit grant for the one native ASCII4096 case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shlex
import subprocess
import threading
import time

MAX_TOTAL = 192 * 1024 * 1024
MAX_FILE = 64 * 1024 * 1024
MAX_CHILD_RSS = 512 * 1024 * 1024
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MONITOR_INTERVAL = 0.25
DEADLINE = 60
ROOT = Path(__file__).resolve().parent

class Logs:
    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.failure = None

    def write(self, stream, data):
        with self.lock:
            available = min(MAX_FILE - stream.tell(), MAX_TOTAL - self.total)
            written = min(max(available, 0), len(data))
            stream.write(data[:written])
            stream.flush()
            self.total += written
            if written != len(data):
                self.failure = 'bounded output budget exceeded; no larger capture attempted'


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def logical_artifact_bytes(root):
    pending = [(root, 0)]
    total = 0
    entries = 0
    while pending:
        directory, depth = pending.pop()
        if depth > 16:
            raise RuntimeError('artifact directory depth exceeds safety-monitor limit')
        with os.scandir(directory) as listing:
            for entry in listing:
                entries += 1
                if entries > 8192:
                    raise RuntimeError('artifact entry count exceeds safety-monitor limit')
                try:
                    if entry.is_symlink():
                        raise RuntimeError('artifact symlink prevents complete logical-size monitoring')
                    if entry.is_dir(follow_symlinks=False):
                        pending.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                        if total > MAX_ARTIFACT_BYTES:
                            return total
                except FileNotFoundError:
                    continue
    return total


class ChildSafetyMonitor:
    def __init__(self, child, artifacts, logs):
        self.child = child
        self.artifacts = artifacts
        self.logs = logs
        self.stopped = threading.Event()
        self.samples = 0
        self.maximum_rss = 0
        self.maximum_logical_bytes = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            while not self.stopped.is_set():
                size = logical_artifact_bytes(self.artifacts)
                self.maximum_logical_bytes = max(self.maximum_logical_bytes, size)
                if size > MAX_ARTIFACT_BYTES:
                    raise RuntimeError('synthetic artifact+StackLogging logical-size ceiling exceeded (512 MiB)')
                if self.child.poll() is None:
                    sampled = subprocess.run(['/bin/ps', '-p', str(self.child.pid), '-o', 'rss='],
                                             capture_output=True, text=True, timeout=2)
                    if sampled.returncode or not sampled.stdout.strip():
                        if self.child.poll() is None:
                            raise RuntimeError('child RSS monitor could not obtain a sample')
                    else:
                        rss = int(sampled.stdout.strip()) * 1024
                        self.samples += 1
                        self.maximum_rss = max(self.maximum_rss, rss)
                        if rss > MAX_CHILD_RSS:
                            raise RuntimeError('synthetic child sampled RSS ceiling exceeded (512 MiB)')
                self.stopped.wait(MONITOR_INTERVAL)
        except Exception as error:
            self.logs.failure = str(error)
            stop_owned(self.child)

    def finish(self):
        self.stopped.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            self.logs.failure = 'child safety monitor failed to stop'
        return {'rss_samples': self.samples, 'maximum_sampled_child_rss_bytes': self.maximum_rss,
                'maximum_sampled_logical_artifact_bytes': self.maximum_logical_bytes,
                'rss_ceiling_bytes': MAX_CHILD_RSS, 'logical_artifact_ceiling_bytes': MAX_ARTIFACT_BYTES,
                'sampling_interval_seconds': MONITOR_INTERVAL,
                'limits': 'Synthetic sampled safety guards only: peaks between samples, open-unlinked files, filesystem allocation and compression are not bounded by these measurements.'}


def tool(command, destination, logs, timeout=DEADLINE):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    def drain():
        with destination.open('wb') as output:
            while chunk := process.stdout.read(65536):
                logs.write(output, chunk)
                if logs.failure:
                    return
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if logs.failure:
                raise RuntimeError(logs.failure)
            if time.monotonic() >= deadline:
                raise RuntimeError('tool deadline exceeded: ' + command[0])
            time.sleep(0.05)
        thread.join(timeout=5)
        if thread.is_alive() or logs.failure:
            raise RuntimeError(logs.failure or 'tool output did not drain')
        if process.returncode:
            raise RuntimeError('tool coverage failure: ' + ' '.join(command))
    finally:
        stop_owned(process)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stacklogs = output / 'stacklogs'
    stacklogs.mkdir()
    binary = ROOT / 'target/debug/native-calibration'
    if not binary.is_file():
        raise RuntimeError('reviewed native binary has not been built')
    manifest_path = arguments.source_manifest.resolve()
    source_manifest = json.loads(manifest_path.read_text())
    source_root = ROOT.parents[1]
    for relative, expected in source_manifest['files'].items():
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root) or file_hash(source) != expected:
            raise RuntimeError('source changed after frozen manifest: ' + relative)
    if manifest_path.stat().st_mtime_ns > binary.stat().st_mtime_ns:
        raise RuntimeError('source manifest was not captured before this native binary build')
    (output / 'source-manifest.json').write_text(json.dumps(source_manifest, indent=2))
    environment = {key: value for key, value in os.environ.items() if not key.startswith('Malloc')}
    environment.update(MallocStackLoggingNoCompact='1', MallocStackLoggingDirectory=str(stacklogs))
    command = [str(binary), '--measure-native-unbounded', '--hold-for-tools', '--case', 'ascii4096']
    provenance = {'command': command, 'binary_sha256': file_hash(binary),
                  'input_sha256': hashlib.sha256(b'x' * 4096).hexdigest(),
                  'platform': platform.platform(), 'mac_version': platform.mac_ver(),
                  'os_build': subprocess.run(['/usr/bin/sw_vers', '-buildVersion'], capture_output=True, text=True, timeout=10, check=True).stdout.strip(),
                  'logging_environment': {key: value for key, value in environment.items() if key.startswith('Malloc')},
                  'native_allowance': 'UNMEASURED', 'maximum_output_bytes': MAX_TOTAL,
                  'maximum_sampled_child_rss_bytes': MAX_CHILD_RSS, 'maximum_logical_artifact_bytes': MAX_ARTIFACT_BYTES,
                  'source_manifest_sha256': file_hash(manifest_path)}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    logs = Logs()
    records = queue.Queue(maxsize=256)
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    (output / 'child-pid.txt').write_text(str(child.pid) + '\n')
    safety = ChildSafetyMonitor(child, output, logs)
    safety.thread.start()
    def drain_child(pipe, filename, protocol):
        with (output / filename).open('wb') as destination:
            while line := pipe.readline(8193):
                logs.write(destination, line)
                if len(line) > 8192:
                    logs.failure = 'child protocol line exceeds fixed limit'
                    return
                if protocol:
                    try:
                        records.put(line.decode('utf-8', errors='strict').rstrip('\n'), timeout=1)
                    except (queue.Full, UnicodeDecodeError):
                        logs.failure = 'child protocol queue/encoding failure'
                        return
                if logs.failure:
                    return
    threads = [threading.Thread(target=drain_child, args=(child.stdout, 'child.stdout.log', True), daemon=True),
               threading.Thread(target=drain_child, args=(child.stderr, 'child.stderr.log', False), daemon=True)]
    for thread in threads:
        thread.start()
    result = {'native_allowance': 'UNMEASURED', 'phases': [], 'sentinels': [], 'markers': [],
              'tool_commands': [], 'native_outputs': [], 'identities': [], 'coverage': 'UNVERIFIED'}
    phase_deadline = time.monotonic() + DEADLINE
    try:
        while child.poll() is None or not records.empty() or any(thread.is_alive() for thread in threads):
            if logs.failure:
                raise RuntimeError(logs.failure)
            if time.monotonic() >= phase_deadline:
                raise RuntimeError('child phase deadline exceeded')
            try:
                line = records.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith('SENTINEL '):
                result['sentinels'].append(line)
            elif line.startswith('MARKER '):
                result['markers'].append(line)
            elif line.startswith(('NATIVE_OUTPUT ', 'UNAVAILABLE_NATIVE_FONT ')):
                result['native_outputs'].append(line)
            elif line.startswith('IDENTITY '):
                result['identities'].append(line)
            elif line.startswith('PHASE '):
                fields = line.split()
                ordinal = int(fields[1])
                if f'pid={child.pid}' not in fields or ordinal not in range(9) or ordinal in result['phases']:
                    raise RuntimeError('unexpected child phase identity')
                if result['phases'] and ordinal <= result['phases'][-1]:
                    raise RuntimeError('phase order changed')
                result['phases'].append(ordinal)
                commands = [['/usr/bin/heap', str(child.pid)], ['/usr/bin/vmmap', '-summary', str(child.pid)]]
                if ordinal == 8:
                    commands += [['/usr/bin/malloc_history', str(child.pid), '-allEvents', '-fullStacks', '-noContent'],
                                 ['/usr/bin/malloc_history', str(child.pid), '-allBySize', '-highWaterMark']]
                capture_deadline = time.monotonic() + DEADLINE
                for index, invocation in enumerate(commands):
                    result['tool_commands'].append(invocation)
                    remaining = capture_deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError('phase capture deadline exceeded')
                    tool(invocation, output / f'phase-{ordinal}-tool-{index}.log', logs, remaining)
                child.stdin.write(f'CONTINUE {ordinal}\n'.encode())
                child.stdin.flush()
                phase_deadline = time.monotonic() + DEADLINE
        child.wait(timeout=5)
        result['child_exit'] = child.returncode
        if child.returncode:
            raise RuntimeError('native child failed; inspect retained operation/fallback diagnostics')
        if result['phases'] != list(range(9)) or len(result['sentinels']) != 7:
            raise RuntimeError('expected phase/sentinel records incomplete')
        # Presence is only a gate for further review. It does not prove event
        # completeness, matching frees/realloc, ordering, or native peak bytes.
        history_path = output / 'phase-8-tool-2.log'
        addresses = {int(line.split()[2], 16) for line in result['sentinels']}
        seen = set()
        with history_path.open(errors='replace') as history:
            for line in history:
                found = {int(address, 16) for address in re.findall(r'\b0x[0-9a-fA-F]+\b', line)}
                seen.update(addresses & found)
        result['sentinel_addresses_missing'] = [hex(address) for address in sorted(addresses - seen)]
        result['coverage'] = 'FAILED missing sentinel addresses' if addresses - seen else 'PENDING lifecycle/event/stack review; address presence is insufficient'
        result['native_peak_bytes'] = None
        result['persistent_native_ownership'] = 'UNRESOLVED'
        for line in result['identities']:
            fields = shlex.split(line)
            if len(fields) == 3 and fields[1] in ('path', 'coretext'):
                path = Path(fields[2])
                key = fields[1] + '_file_sha256'
                result[key] = file_hash(path) if path.is_file() else 'MISSING (possibly dyld shared cache)'
        if addresses - seen:
            raise RuntimeError('tool-event coverage failed; do not substitute a different profiler/corpus')
    except Exception as error:
        result['failure'] = str(error)
        raise
    finally:
        stop_owned(child)
        for thread in threads:
            thread.join(timeout=5)
        result['safety_monitor'] = safety.finish()
        if logs.failure:
            result['failure'] = logs.failure
        result['captured_bytes'] = logs.total
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        if logs.failure:
            raise RuntimeError(logs.failure)

if __name__ == '__main__':
    main()
