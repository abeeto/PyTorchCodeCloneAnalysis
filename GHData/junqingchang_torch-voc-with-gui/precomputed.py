import json
import collections

with open('colorfliprotate_adaptive400sq2/evaloutputs.json', 'r') as fp:
    flipdata = json.load(fp)


flip_possible = {}
flip_final = {}
flip_final['Possible'] = [i for i in flipdata]
ctr = 0
for key in flipdata:
    if ctr == 0:
        flip_possible['1'] = {key:flipdata[key]['Pred'][0]}
        flip_possible['2'] = {key:flipdata[key]['Pred'][1]}
        flip_possible['3'] = {key:flipdata[key]['Pred'][2]}
        flip_possible['4'] = {key:flipdata[key]['Pred'][3]}
        flip_possible['5'] = {key:flipdata[key]['Pred'][4]}
        flip_possible['6'] = {key:flipdata[key]['Pred'][5]}
        flip_possible['7'] = {key:flipdata[key]['Pred'][6]}
        flip_possible['8'] = {key:flipdata[key]['Pred'][7]}
        flip_possible['9'] = {key:flipdata[key]['Pred'][8]}
        flip_possible['10'] = {key:flipdata[key]['Pred'][9]}
        flip_possible['11'] = {key:flipdata[key]['Pred'][10]}
        flip_possible['12'] = {key:flipdata[key]['Pred'][11]}
        flip_possible['13'] = {key:flipdata[key]['Pred'][12]}
        flip_possible['14'] = {key:flipdata[key]['Pred'][13]}
        flip_possible['15'] = {key:flipdata[key]['Pred'][14]}
        flip_possible['16'] = {key:flipdata[key]['Pred'][15]}
        flip_possible['17'] = {key:flipdata[key]['Pred'][16]}
        flip_possible['18'] = {key:flipdata[key]['Pred'][17]}
        flip_possible['19'] = {key:flipdata[key]['Pred'][18]}
        flip_possible['20'] = {key:flipdata[key]['Pred'][19]}
        ctr += 1
    else:
        flip_possible['1'][key] = flipdata[key]['Pred'][0]
        flip_possible['2'][key] = flipdata[key]['Pred'][1]
        flip_possible['3'][key] = flipdata[key]['Pred'][2]
        flip_possible['4'][key] = flipdata[key]['Pred'][3]
        flip_possible['5'][key] = flipdata[key]['Pred'][4]
        flip_possible['6'][key] = flipdata[key]['Pred'][5]
        flip_possible['7'][key] = flipdata[key]['Pred'][6]
        flip_possible['8'][key] = flipdata[key]['Pred'][7]
        flip_possible['9'][key] = flipdata[key]['Pred'][8]
        flip_possible['10'][key] = flipdata[key]['Pred'][9]
        flip_possible['11'][key] = flipdata[key]['Pred'][10]
        flip_possible['12'][key] = flipdata[key]['Pred'][11]
        flip_possible['13'][key] = flipdata[key]['Pred'][12]
        flip_possible['14'][key] = flipdata[key]['Pred'][13]
        flip_possible['15'][key] = flipdata[key]['Pred'][14]
        flip_possible['16'][key] = flipdata[key]['Pred'][15]
        flip_possible['17'][key] = flipdata[key]['Pred'][16]
        flip_possible['18'][key] = flipdata[key]['Pred'][17]
        flip_possible['19'][key] = flipdata[key]['Pred'][18]
        flip_possible['20'][key] = flipdata[key]['Pred'][19]

for class_num in flip_possible:
    sorted_dict = sorted(flip_possible[class_num].items(), key=lambda t: t[1])
    sorted_dict.reverse()
    flip_final[class_num] = [key for key in collections.OrderedDict(sorted_dict)]
    
with open('coloradjust/evaloutputs.json', 'r') as fp:
    colordata = json.load(fp)

print(colordata["2008_000002"])

color_possible = {}
color_final = {}
color_final['Possible'] = [i for i in colordata]
ctr = 0
for key in colordata:
    if ctr == 0:
        color_possible['1'] = {key:colordata[key]['Pred'][0]}
        color_possible['2'] = {key:colordata[key]['Pred'][1]}
        color_possible['3'] = {key:colordata[key]['Pred'][2]}
        color_possible['4'] = {key:colordata[key]['Pred'][3]}
        color_possible['5'] = {key:colordata[key]['Pred'][4]}
        color_possible['6'] = {key:colordata[key]['Pred'][5]}
        color_possible['7'] = {key:colordata[key]['Pred'][6]}
        color_possible['8'] = {key:colordata[key]['Pred'][7]}
        color_possible['9'] = {key:colordata[key]['Pred'][8]}
        color_possible['10'] = {key:colordata[key]['Pred'][9]}
        color_possible['11'] = {key:colordata[key]['Pred'][10]}
        color_possible['12'] = {key:colordata[key]['Pred'][11]}
        color_possible['13'] = {key:colordata[key]['Pred'][12]}
        color_possible['14'] = {key:colordata[key]['Pred'][13]}
        color_possible['15'] = {key:colordata[key]['Pred'][14]}
        color_possible['16'] = {key:colordata[key]['Pred'][15]}
        color_possible['17'] = {key:colordata[key]['Pred'][16]}
        color_possible['18'] = {key:colordata[key]['Pred'][17]}
        color_possible['19'] = {key:colordata[key]['Pred'][18]}
        color_possible['20'] = {key:colordata[key]['Pred'][19]}
        ctr += 1
    else:
        color_possible['1'][key] = colordata[key]['Pred'][0]
        color_possible['2'][key] = colordata[key]['Pred'][1]
        color_possible['3'][key] = colordata[key]['Pred'][2]
        color_possible['4'][key] = colordata[key]['Pred'][3]
        color_possible['5'][key] = colordata[key]['Pred'][4]
        color_possible['6'][key] = colordata[key]['Pred'][5]
        color_possible['7'][key] = colordata[key]['Pred'][6]
        color_possible['8'][key] = colordata[key]['Pred'][7]
        color_possible['9'][key] = colordata[key]['Pred'][8]
        color_possible['10'][key] = colordata[key]['Pred'][9]
        color_possible['11'][key] = colordata[key]['Pred'][10]
        color_possible['12'][key] = colordata[key]['Pred'][11]
        color_possible['13'][key] = colordata[key]['Pred'][12]
        color_possible['14'][key] = colordata[key]['Pred'][13]
        color_possible['15'][key] = colordata[key]['Pred'][14]
        color_possible['16'][key] = colordata[key]['Pred'][15]
        color_possible['17'][key] = colordata[key]['Pred'][16]
        color_possible['18'][key] = colordata[key]['Pred'][17]
        color_possible['19'][key] = colordata[key]['Pred'][18]
        color_possible['20'][key] = colordata[key]['Pred'][19]

for class_num in color_possible:
    sorted_dict = sorted(color_possible[class_num].items(), key=lambda t: t[1])
    sorted_dict.reverse()
    color_final[class_num] = [key for key in collections.OrderedDict(sorted_dict)]

output = {'Color Jitter Model':color_final, 'Edited Resnet Model':flip_final}

with open('precomputed_rank.json', 'w') as pr:
    json.dump(output, pr)