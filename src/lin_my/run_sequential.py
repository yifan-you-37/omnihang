#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
MOVED FROM /orions4-zfs/projects/rqi/Code/3dcnn/lfd/run_parallel.py
PLEASE CHECK ORIGINAL FILE FOR UPDATES!
'''

import os
import sys
import datetime
from functools import partial
from subprocess import call

'''
@brief:
	run commands in parallel
@usage:
	python run_parallel.py command_list.txt
	where each line of command_list.txt is a executable command
'''
command_file = sys.argv[1]
commands = [line.rstrip() for line in open(command_file)]

for command in commands:
	call(command, shell=True)
