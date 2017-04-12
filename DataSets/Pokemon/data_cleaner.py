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

def load_pokemon_moves(path):
	pokemon_regex = re.compile('\w+: {learnset')
	move_regex = re.compile('\w+: \[')
	poke_moves = dict()
	with open(path, 'r') as f:
		lines = f.readlines()
		move_list = []
		name = ''
		for line in lines:
			name_match = pokemon_regex.match(line)
			if not name_match:
				move_match = move_regex.match(line)
				if move_match:
					move = move_match.group(0)
					move_list.append(move[:-3])
			else:
				if name != '':
					poke_moves[name] = move_list
					move_list = []
				name = name_match.group(0)
				name = name[:-11]

	return poke_moves
				

def load_all_moves(path):
	move_regex = re.compile('\w+: \[')
	move_list = set()
	with open(path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				move_match = move_regex.match(line)
				if move_match:
					move = move_match.group(0)
					move_list.add(move[:-3])
	return move_list
all_moves = load_all_moves('learnsets')
print(len(all_moves))
