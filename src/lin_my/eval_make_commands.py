import numpy as np
import sys
import random
import os
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data")
	parser.add_argument("--result_folder")
	parser.add_argument("--result_file_pattern", default='')
	parser.add_argument("--output_folder_name", default='')
	parser.add_argument('--bohg4', action='store_true')
	parser.add_argument('--sherlock', action='store_true')
	parser.add_argument('--s3_mode')
	parser.add_argument('--sherlock_output_folder_name', default='')
	parser.add_argument('--ompl', action='store_true')

	args = parser.parse_args()
	if args.sherlock:
		assert args.sherlock_output_folder_name != ''

	if args.bohg4:
		args.home_dir_data = '/scr1/yifan/hang'
	if args.output_folder_name == '':
		output_folder = os.path.abspath(os.path.join(args.result_folder, 'eval_result'))
	else:
		output_folder = os.path.abspath(os.path.join(args.result_folder, args.output_folder_name))

	assert os.path.exists(args.home_dir_data)

	if not args.ompl:
		assert args.s3_mode == 's13' or args.s3_mode == 's123'

	mkdir_if_not(output_folder)
	out_file_dir = os.path.join(output_folder, 'commands_sherlock.txt' if args.sherlock else 'commands.txt')
	if args.ompl:
		command_template = '{} '.format('python3' if args.sherlock else 'python') + 'eval_ompl.py --home_dir_data {} --result_file_dir {} --output_folder {} --result_file_name {}'
	else:
		command_template = '{} '.format('python3' if args.sherlock else 'python') + 'eval_s3.py --home_dir_data {} --result_file_dir {} --output_folder {} --result_file_name {}' + ' --mode {}'.format(args.s3_mode)
	sherlock_home_dir = '/home/users/yifanyou/'
	sherlock_home_dir_data = '/scratch/groups/bohg/hang'
	repo_dir = os.path.join(sherlock_home_dir, 'geo-hook')

	result_subfolder_all = glob.glob('{}/*{}*/eval'.format(args.result_folder, args.result_file_pattern))
	result_file_dir_all = []
	for subfolder in result_subfolder_all:
		tmp = glob.glob('{}/*epoch_1_test.json'.format(subfolder))
		
		assert len(tmp) == 1
		result_file_dir_all.append(tmp[0])
	print('{}/*{}*.json'.format(args.result_folder, args.result_file_pattern))
	result_file_name_set = set()
	result_file_name_command_set = set()
	print(result_file_dir_all)
	print('number of result file', len(result_file_dir_all))
	
	command_all = []
	assert len(result_file_dir_all) > 0
	for result_file_dir in result_file_dir_all:
		result_dict_all = load_json(result_file_dir)
		result_file_name_all = result_dict_all.keys()
		
		for result_file_name in result_file_name_all:
			# assert not (result_file_name in result_file_name_set)
			tmp_dict = result_dict_all[result_file_name]
			if not args.sherlock:
				command = command_template.format(args.home_dir_data, result_file_dir, output_folder, result_file_name)
			else:
				parent_dir = os.path.abspath(os.path.join(result_file_dir, '..', '..', '..', '..'))
				result_file_dir_rel = os.path.relpath(result_file_dir, parent_dir)
				juno_result_file_dir = os.path.abspath(os.path.join(repo_dir, 'lin_my', 'runs', result_file_dir_rel))
				juno_output_folder = os.path.abspath(os.path.join(repo_dir, 'lin_my', 'runs', args.sherlock_output_folder_name, 'eval_result'))
				# print('result file', juno_result_file_dir)
				# print('output folder', juno_output_folder)

				command = command_template.format(sherlock_home_dir_data, juno_result_file_dir, juno_output_folder, result_file_name)

			if args.ompl:
				if 'sol_pose_seq' in tmp_dict and len(tmp_dict['sol_pose_seq']) > 0:
					if not (result_file_name in result_file_name_command_set):
						command_all.append(command)
						result_file_name_command_set.add(result_file_name)
			else:
				if not (result_file_name in result_file_name_command_set):
					command_all.append(command)
					result_file_name_command_set.add(result_file_name)

			result_file_name_set.add(result_file_name)
			


	if not args.sherlock:
		result_file_name_loaded = sorted(list(result_file_name_set))
		result_file_name_dir = os.path.join(output_folder, 'result_file_name_loaded.txt')
		write_txt(result_file_name_dir, result_file_name_loaded)

		out_result_file_name_command_dir = os.path.join(output_folder, 'result_file_name_command.txt')
		write_txt(out_result_file_name_command_dir, sorted(list(result_file_name_command_set)))

		result_file_name_not_in_command = sorted(list(result_file_name_set.difference(result_file_name_command_set)))
		out_result_file_name_not_in_command_dir = os.path.join(output_folder, 'result_file_name_not_in_command.txt')
		if len(result_file_name_not_in_command) > 0:
			write_txt(out_result_file_name_not_in_command_dir, result_file_name_not_in_command)	
	random.shuffle(command_all)
	write_txt(out_file_dir, command_all)

	assert len(command_all) == len(result_file_name_command_set)
	print('number of commands', len(command_all))
	print('command path', out_file_dir)
	print('result file name', len(result_file_name_set))
