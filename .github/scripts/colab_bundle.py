"""Build and package the native Linux artifacts used by the incremental T4 notebook."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time


RUST_TARGETS = {
    'native-library': ('ruda-torch-native', 'ruda_torch_native', 'library', '', []),
    'cuda-gap-rublas-kernel_ir': ('rublas', 'kernel_ir', 'test', 'cpu-reference,ruda-test-runtime/cuda', []),
    'cuda-gap-ruDNN-attention': ('ruDNN', 'attention', 'test', 'cpu-reference,ruda-test-runtime/cuda', []),
    'cuda-gap-ruDNN-convolution': ('ruDNN', 'convolution', 'test', 'cpu-reference,ruda-test-runtime/cuda', []),
    'cuda-gap-ruSPARSE-tensor': ('ruSPARSE', 'tensor', 'test', 'cuda-tests,ruda-driver-cuda/direct-ptx', []),
    'cuda-gap-ruSPARSE-tensor_level1': ('ruSPARSE', 'tensor_level1', 'test', 'cuda-tests,ruda-driver-cuda/direct-ptx', []),
    'cuda-gap-ruSPARSE-tensor_binary': ('ruSPARSE', 'tensor_binary', 'test', 'cuda-tests,ruda-driver-cuda/direct-ptx', []),
    'cuda-gap-ruda-nn-ruda_nn': ('ruda-nn', 'ruda_nn', 'lib', 'test-cuda,ruda-driver-cuda/direct-ptx', ['ruda-driver-cuda']),
    'cuda-gap-ruda-llm-stack_autotune_gpu': ('ruda-llm', 'stack_autotune_gpu', 'test', 'nvidia-ptx,stack-autotune', []),
    'cuda-gap-ruCCL-tensor_collectives': ('ruCCL', 'tensor_collectives', 'example', 'cuda,ruda-driver-cuda/direct-ptx', ['ruda-driver-cuda']),
}
LABELS = set(RUST_TARGETS) | {'cpp-extension'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(*command):
    return subprocess.check_output(command, text=True).strip()


def build(command, cwd, state_path):
    import psutil
    started = time.monotonic()
    events = queue.Queue()
    messages = []
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_output():
        for line in process.stdout:
            events.put(line)
        events.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    next_panel = 0
    while True:
        try:
            line = events.get(timeout=1)
        except queue.Empty:
            line = ''
        if line is None:
            break
        if line:
            try:
                message = json.loads(line)
            except ValueError:
                print(line, end='', flush=True)
            else:
                messages.append(message)
                if message.get('reason') == 'compiler-artifact':
                    print('Cargo artifact:', message['target']['name'],
                          '(cached)' if message['fresh'] else '(compiled)', flush=True)
                elif message.get('reason') == 'compiler-message':
                    print(message['message'].get('rendered', ''), end='', flush=True)
        now = time.monotonic()
        if now >= next_panel:
            parent = psutil.Process(process.pid)
            processes = [parent] + parent.children(recursive=True) if process.poll() is None else []
            threads, resident, active, cpu_seconds = 0, 0, 0, 0.0
            for child in processes:
                try:
                    times = child.cpu_times()
                    cpu_seconds += times.user + times.system
                    threads += child.num_threads()
                    resident += child.memory_info().rss
                    active += sum(t.user_time + t.system_time > 0 for t in child.threads())
                except psutil.Error:
                    pass
            elapsed = now - started
            state = dict(stage='compile', completed_artifacts=0, total_artifacts=1,
                         stage_seconds=elapsed, total_seconds=elapsed,
                         cargo_completed_units=sum(m.get('reason') == 'compiler-artifact' for m in messages),
                         cargo_total_units='unknown', eta='unknown; first build of this feature graph',
                         rate_artifacts_per_second=0, scheduled_cargo_jobs=os.environ.get('CARGO_BUILD_JOBS'),
                         process_threads=threads, threads_with_cpu_time=active,
                         average_cpu_cores=cpu_seconds / max(elapsed, 1), cpu_percent=psutil.cpu_percent(),
                         process_ram_bytes=resident, available_ram_bytes=psutil.virtual_memory().available,
                         swap_bytes=psutil.swap_memory().used, gpu='none on hosted build runner',
                         checkpoint='completed artifacts uploaded after this build; no mid-link checkpoint',
                         recovery='rerun failed jobs; completed job artifacts retained; unfinished build repeated')
            state_path.write_text(json.dumps(state))
            print('BUILD PANEL', json.dumps(state), flush=True)
            next_panel = now + 30
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
    return messages


def build_one(label, output):
    root = Path(__file__).resolve().parents[2]
    revision = checked('git', 'rev-parse', 'HEAD')
    assert revision == os.environ['SOURCE_REVISION']
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    output.mkdir(parents=True, exist_ok=True)
    import psutil
    print('PREFLIGHT', json.dumps(dict(artifact_workload=1, label=label,
          available_cores=len(os.sched_getaffinity(0)), cargo_jobs=os.environ.get('CARGO_BUILD_JOBS'),
          available_ram_bytes=psutil.virtual_memory().available, free_disk_bytes=shutil.disk_usage(root).free,
          eta='unknown; independent artifact build', recovery='rerun failed jobs, retain completed artifacts')))
    state_path = Path(os.environ['RUNNER_TEMP']) / ('colab-' + label + '-state.json')
    if label == 'cpp-extension':
        import torch
        assert torch.__version__ == '2.10.0+cu126' and torch.version.cuda == '12.6'
        build([sys.executable, 'setup.py', 'build_ext', '--inplace'], root/'ruda-torch/python', state_path)
        matches = list((root/'ruda-torch/python/ruda_torch').glob('_C*.so'))
        compatibility = dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
                             torch=torch.__version__, cuda=torch.version.cuda,
                             cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
        features = []
    else:
        package, target, kind, flags, companions = RUST_TARGETS[label]
        command = ['cargo', 'build' if kind in ('example', 'library') else 'test',
                   '--locked', '--release', '-p', package]
        for companion in companions:
            command += ['-p', companion]
        if kind == 'example':
            command += ['--example', target]
        elif kind != 'library':
            command += ['--lib'] if kind == 'lib' else ['--test', target]
            command += ['--no-run']
        if flags:
            command += ['--features', flags]
        messages = build(command + ['--message-format=json'], root, state_path)
        matches = []
        for message in messages:
            if message.get('reason') != 'compiler-artifact' or message['target']['name'] != target:
                continue
            if kind == 'library':
                matches += [Path(p) for p in message['filenames'] if p.endswith('.so')]
            elif message.get('executable'):
                matches.append(Path(message['executable']))
        compatibility = dict(rust=checked('rustc', '--version'), package=package, target=target, kind=kind)
        features = flags.split(',') if flags else []
    matches = sorted(set(matches))
    assert len(matches) == 1, matches
    binary = output/matches[0].name
    shutil.copy2(matches[0], binary)
    binary.chmod(0o755)
    versions = re.findall(r'GLIBC_(\d+\.\d+)', checked('readelf', '--version-info', str(binary)))
    glibc = max(versions, key=lambda value: tuple(map(int, value.split('.'))))
    entry = dict(label=label, path=f'{label}/{binary.name}', sha256=sha(binary), bytes=binary.stat().st_size,
                 source_revision=revision, platform='linux-x86_64', minimum_glibc=glibc,
                 features=features, compatibility=compatibility)
    (output/'manifest.json').write_text(json.dumps(entry, indent=2) + '\n')
    state_path.write_text(json.dumps(dict(completed_artifacts=1, total_artifacts=1,
                                        checkpoint=str(binary), sha256=entry['sha256'])))
    print('Completed compiled artifact 1/1:', binary, flush=True)


def assemble(directory, output):
    revision = os.environ['SOURCE_REVISION']
    entries = []
    for label in sorted(LABELS):
        record = json.loads((directory/label/'manifest.json').read_text())
        assert record['label'] == label and record['source_revision'] == revision
        path = directory/record['path']
        assert path.parent == directory/label and path.is_file()
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
        entries.append(record)
    manifest = dict(format=1, source_revision=revision, platform='linux-x86_64', artifacts=entries)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    bundle = output/f'ruda-colab-t4-{revision}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        archive.add(manifest_path, arcname='manifest.json')
        for entry in entries:
            archive.add(directory/entry['path'], arcname=entry['path'])
    (output/'SHA256SUMS').write_text(sha(bundle) + '  ' + bundle.name + '\n' + sha(manifest_path) + '  manifest.json\n')
    print('Packaged', len(entries), 'compiled artifacts; no logs, test results or progress files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', choices=sorted(LABELS))
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.label:
        build_one(args.label, args.output)
    else:
        assemble(args.artifacts, args.output)
