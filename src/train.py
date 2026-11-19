import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--task', choices=['NASH', 'NASH-HCC', 'NASH-to-HCC'], default='NASH-to-HCC')
    p.add_argument('--config', type=Path, help='Default: configs/<task>.json')
    p.add_argument('--print-command', action='store_true', help='Print argv as JSON without training')
    args, overrides = p.parse_known_args()
    options = json.loads((args.config or ROOT / 'configs' / (args.task + '.json')).read_text())
    if options.get('task', args.task) != args.task:
        p.error('Task/config mismatch')
    command = [sys.executable, str(ROOT / 'src/optimize_network.py')]
    for key, value in options.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if key == 'profile_memory':
                if value: command.append(flag)
            elif key == 'profile':
                if value: command.append(flag)
            else:
                command.append(flag if value else '--no-' + key.replace('_', '-'))
        elif isinstance(value, list):
            command.extend([flag] + [str(x) for x in value])
        else:
            command.extend([flag, str(value)])
    command.extend(overrides)
    if args.print_command:
        print(json.dumps(command, indent=2))
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
