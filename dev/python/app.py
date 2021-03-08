import argparse
import importlib
import sys
import os
import re


def request_module(module_name):
	try:
		module = importlib.import_module(f'commands.{module_name}')
	except ModuleNotFoundError as e:
		print(e)
		print('Invalid command')
		sys.exit(1)
	return module


def main():
    parser = argparse.ArgumentParser('App started')
    command_list = [script.split('.')[0] for script in os.listdir('commands/') \
	if script not in ['__init__.py', '__pycache__']]
    parser.add_argument('command', type=str, help='Available command include %s' % ', '.join(command_list))

    if '-h' in sys.argv[2:] and sys.argv[1] in command_list:
        print('Hi')
        module = request_module(sys.argv[1])
        getattr(module, 'Command')().run(sys.argv[1:])
    else:
        print('Hello')
        args, unknown_args = parser.parse_known_args()
        module = request_module(args.command)
        getattr(module, 'Command')().run(unknown_args)
