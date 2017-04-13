import re

pokemon_regex = re.compile('{"name":".+?},')
with open('pokelist', 'r') as f:
    lines = f.readlines()
    for line in lines:
        pokemon = pokemon_regex.findall(line)

with open('pokelist_parsed', 'w') as f:
    for lines in pokemon:
        print(lines, file=f)
