import re

import numpy as np
import pandas as pd

isZ = ['10000000voltthunderbolt', 'aciddownpour', 'alloutpummeling', 'blackholeeclipse',
		'bloomdoom', 'breakneckblitz', 'catastropika', 'continentalcrush',
		'corkscrewcrash', 'devastatingdrake', 'extremeevoboost', 'genesissupernova',
		'gigavolthavoc', 'guardianofalola', 'hydrovortex', 'infernooverdrive', 
		'maliciousmoonsault', 'neverendingnightmare', 'oceanicoperetta', 
		'pulverizingpancake', 'savagespinout', 'shatteredpsyche', 'sinisterarrowraid',
		'soulstealing7starstrike', 'stokedsparksurfer', 'subzeroslammer',
		'supersonicskystrike', 'tectonicrage', 'twinkletackle']

def load_move_data(path):
	name_regex = re.compile('"\w+":')
	move_list = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			match = name_regex.match(line)
			if match:
				move = match.group(0)
				move = move[1:-2] #throw out quotation marks and colon
				if move not in isZ:
					move_list.append(move)
	return move_list

moves = load_move_data('moves')

print(moves)
print(len(moves))
print(len(isZ))

