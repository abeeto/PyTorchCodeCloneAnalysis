import json
import collections
import matplotlib.pyplot as plt

with open('fliponly/evaloutputs.json', 'r') as fp:
    flipdata = json.load(fp)


flip_possible = {}
flip_final = {}
flip_final['Possible'] = [i for i in flipdata]
ctr = 0
for key in flipdata:
    if ctr == 0:
        flip_possible[0] = {key:flipdata[key]['Pred'][0]}
        flip_possible[1] = {key:flipdata[key]['Pred'][1]}
        flip_possible[2] = {key:flipdata[key]['Pred'][2]}
        flip_possible[3] = {key:flipdata[key]['Pred'][3]}
        flip_possible[4] = {key:flipdata[key]['Pred'][4]}
        flip_possible[5] = {key:flipdata[key]['Pred'][5]}
        flip_possible[6] = {key:flipdata[key]['Pred'][6]}
        flip_possible[7] = {key:flipdata[key]['Pred'][7]}
        flip_possible[8] = {key:flipdata[key]['Pred'][8]}
        flip_possible[9] = {key:flipdata[key]['Pred'][9]}
        flip_possible[10] = {key:flipdata[key]['Pred'][10]}
        flip_possible[11] = {key:flipdata[key]['Pred'][11]}
        flip_possible[12] = {key:flipdata[key]['Pred'][12]}
        flip_possible[13] = {key:flipdata[key]['Pred'][13]}
        flip_possible[14] = {key:flipdata[key]['Pred'][14]}
        flip_possible[15] = {key:flipdata[key]['Pred'][15]}
        flip_possible[16] = {key:flipdata[key]['Pred'][16]}
        flip_possible[17] = {key:flipdata[key]['Pred'][17]}
        flip_possible[18] = {key:flipdata[key]['Pred'][18]}
        flip_possible[19] = {key:flipdata[key]['Pred'][19]}
        ctr += 1
    else:
        flip_possible[0][key] = flipdata[key]['Pred'][0]
        flip_possible[1][key] = flipdata[key]['Pred'][1]
        flip_possible[2][key] = flipdata[key]['Pred'][2]
        flip_possible[3][key] = flipdata[key]['Pred'][3]
        flip_possible[4][key] = flipdata[key]['Pred'][4]
        flip_possible[5][key] = flipdata[key]['Pred'][5]
        flip_possible[6][key] = flipdata[key]['Pred'][6]
        flip_possible[7][key] = flipdata[key]['Pred'][7]
        flip_possible[8][key] = flipdata[key]['Pred'][8]
        flip_possible[9][key] = flipdata[key]['Pred'][9]
        flip_possible[10][key] = flipdata[key]['Pred'][10]
        flip_possible[11][key] = flipdata[key]['Pred'][11]
        flip_possible[12][key] = flipdata[key]['Pred'][12]
        flip_possible[13][key] = flipdata[key]['Pred'][13]
        flip_possible[14][key] = flipdata[key]['Pred'][14]
        flip_possible[15][key] = flipdata[key]['Pred'][15]
        flip_possible[16][key] = flipdata[key]['Pred'][16]
        flip_possible[17][key] = flipdata[key]['Pred'][17]
        flip_possible[18][key] = flipdata[key]['Pred'][18]
        flip_possible[19][key] = flipdata[key]['Pred'][19]


for class_num in flip_possible:
    sorted_dict = sorted(flip_possible[class_num].items(), key=lambda t: t[1])
    sorted_dict.reverse()
    sorted_dict = collections.OrderedDict(sorted_dict)
    flip_final[class_num] = [(key, sorted_dict[key]) for key in sorted_dict]



f_max_flip = {}
top_50_flip = {}
for class_num in flip_final:
    ctr = 0
    if(class_num != 'Possible'):
        top_50_flip[class_num] = []
        for val in flip_final[class_num]:
            best_img, best_val = val
            top_50_flip[class_num].append(best_img)
            if ctr == 0:
                f_max_flip[class_num] = best_val
            ctr += 1
            if ctr == 50:
                break



# print(f_max_flip)
# print(top_50_flip)

flip_multiple = []
flip_precision = {}
for i in range(20):
    flip_precision[i] = {'x': []}
for class_num in f_max_flip:
    val = f_max_flip[class_num]/19
    flip_multiple.append(val)

for class_num in flipdata:
    threshold = [0]*20
    img_data = flipdata[class_num]
    true_val = img_data['True']
    pred_val = img_data['Pred']
    for j in range(len(threshold)):
        for i in range(20):      
            if threshold[j] not in flip_precision[j]:
                flip_precision[j][threshold[j]] = {'TP':0, 'FP':0}
            if pred_val[j] > threshold[j]:
                if int(true_val[j]) == 1:
                    flip_precision[j][threshold[j]]['TP'] += 1
                else:
                    flip_precision[j][threshold[j]]['FP'] += 1
            if len(flip_precision[j]['x']) != 20:
                flip_precision[j]['x'].append(threshold[j])
            threshold[j] += flip_multiple[j]
        
tail_accuracy = {}
for class_num in top_50_flip:
    tail_accuracy[class_num] = {}
    for img_name in top_50_flip[class_num]:
        threshold = 0
        T_data = flipdata[img_name]['True'][class_num]
        P_data = flipdata[img_name]['Pred'][class_num]
        for i in range(20):
            if threshold not in tail_accuracy[class_num]:
                tail_accuracy[class_num][threshold] = {'TP':0, 'FP':0}
            if P_data > threshold:
                if (int(T_data) == 1):
                    tail_accuracy[class_num][threshold]['TP'] += 1
                else:
                    tail_accuracy[class_num][threshold]['FP'] += 1
            threshold += flip_multiple[class_num]
plt.figure(figsize=(20,20))
for class_num in tail_accuracy:
    plt.subplot(5,8,(class_num*2)+1)
    plt.title('Class {}'.format(class_num+1))
    plt.xlabel('threshold')
    plt.ylabel('Tail accuracy')
    x_val = []
    y_val = []
    for threshold in tail_accuracy[class_num]:
        x_val.append(threshold)
        TP = tail_accuracy[class_num][threshold]['TP']
        FP = tail_accuracy[class_num][threshold]['FP']
        if TP+FP == 0:
            precision = 0 
        else:
            precision = TP/(TP+FP)
        y_val.append(precision)
    plt.plot(x_val, y_val)

plt.show()
# print(flip_precision)

plt.figure(figsize=(20,20))
plt.title('Precision Across 20 thresholds')
plt.xlabel('threshold')
plt.ylabel('precision')
classes = []
avg_precision = 0

for class_num in flip_precision:
    classes.append(class_num)
    x_val = flip_precision[class_num]['x']
    y_val = []
    class_precision = 0
    for threshold in x_val:
        TP_data = flip_precision[class_num][threshold]['TP']
        FP_data = flip_precision[class_num][threshold]['FP']
        if (TP_data+FP_data == 0):
            precision = 0
        else:
            precision = TP_data/(TP_data+FP_data)
        y_val.append(precision)
        class_precision += precision
    avg_precision += class_precision/20
    print('Class {} Precision: {}'.format(class_num, class_precision/20))
    plt.plot(x_val, y_val)

plt.legend(classes)
plt.show()

avg_precision /= 20
print('Model 1 precision: {}'.format(avg_precision))

###############################################


with open('coloradjust/evaloutputs.json', 'r') as fp:
    colordata = json.load(fp)


color_possible = {}
color_final = {}
color_final['Possible'] = [i for i in colordata]
ctr = 0
for key in colordata:
    if ctr == 0:
        color_possible[0] = {key:colordata[key]['Pred'][0]}
        color_possible[1] = {key:colordata[key]['Pred'][1]}
        color_possible[2] = {key:colordata[key]['Pred'][2]}
        color_possible[3] = {key:colordata[key]['Pred'][3]}
        color_possible[4] = {key:colordata[key]['Pred'][4]}
        color_possible[5] = {key:colordata[key]['Pred'][5]}
        color_possible[6] = {key:colordata[key]['Pred'][6]}
        color_possible[7] = {key:colordata[key]['Pred'][7]}
        color_possible[8] = {key:colordata[key]['Pred'][8]}
        color_possible[9]  = {key:colordata[key]['Pred'][9]}
        color_possible[10] = {key:colordata[key]['Pred'][10]}
        color_possible[11] = {key:colordata[key]['Pred'][11]}
        color_possible[12] = {key:colordata[key]['Pred'][12]}
        color_possible[13] = {key:colordata[key]['Pred'][13]}
        color_possible[14] = {key:colordata[key]['Pred'][14]}
        color_possible[15] = {key:colordata[key]['Pred'][15]}
        color_possible[16] = {key:colordata[key]['Pred'][16]}
        color_possible[17] = {key:colordata[key]['Pred'][17]}
        color_possible[18] = {key:colordata[key]['Pred'][18]}
        color_possible[19] = {key:colordata[key]['Pred'][19]}
        ctr += 1
    else:
        color_possible[0][key] = colordata[key]['Pred'][0]
        color_possible[1][key] = colordata[key]['Pred'][1]
        color_possible[2][key] = colordata[key]['Pred'][2]
        color_possible[3][key] = colordata[key]['Pred'][3]
        color_possible[4][key] = colordata[key]['Pred'][4]
        color_possible[5][key] = colordata[key]['Pred'][5]
        color_possible[6][key] = colordata[key]['Pred'][6]
        color_possible[7][key] = colordata[key]['Pred'][7]
        color_possible[8][key] = colordata[key]['Pred'][8]
        color_possible[9][key] = colordata[key]['Pred'][9]
        color_possible[10][key] = colordata[key]['Pred'][10]
        color_possible[11][key] = colordata[key]['Pred'][11]
        color_possible[12][key] = colordata[key]['Pred'][12]
        color_possible[13][key] = colordata[key]['Pred'][13]
        color_possible[14][key] = colordata[key]['Pred'][14]
        color_possible[15][key] = colordata[key]['Pred'][15]
        color_possible[16][key] = colordata[key]['Pred'][16]
        color_possible[17][key] = colordata[key]['Pred'][17]
        color_possible[18][key] = colordata[key]['Pred'][18]
        color_possible[19][key] = colordata[key]['Pred'][19]

for class_num in color_possible:
    sorted_dict = sorted(color_possible[class_num].items(), key=lambda t: t[1])
    sorted_dict.reverse()
    sorted_dict = collections.OrderedDict(sorted_dict)
    color_final[class_num] = [(key,sorted_dict[key]) for key in sorted_dict]

f_max_color = {}
top_50_color = {}

for class_num in color_final:
    ctr = 0
    if(class_num != 'Possible'):
        top_50_color[class_num] = []
        for val in color_final[class_num]:
            best_img, best_val = val
            top_50_color[class_num].append(best_img)
            if ctr == 0:
                f_max_color[class_num] = best_val
            ctr += 1
            if ctr == 50:
                break

# print(f_max_color)
# print(top_50_color)

color_multiple = []
color_precision = {}
for i in range(20):
    color_precision[i] = {'x': []}
for class_num in f_max_color:
    val = f_max_color[class_num]/19
    color_multiple.append(val)

for class_num in colordata:
    threshold = [0]*20
    img_data = colordata[class_num]
    true_val = img_data['True']
    pred_val = img_data['Pred']
    for j in range(len(threshold)):
        for i in range(20):      
            if threshold[j] not in color_precision[j]:
                color_precision[j][threshold[j]] = {'TP':0, 'FP':0}
            if pred_val[j] > threshold[j]:
                if int(true_val[j]) == 1:
                    color_precision[j][threshold[j]]['TP'] += 1
                else:
                    color_precision[j][threshold[j]]['FP'] += 1
            if len(color_precision[j]['x']) != 20:
                color_precision[j]['x'].append(threshold[j])
            threshold[j] += color_multiple[j]        

tail_accuracy = {}
for class_num in top_50_color:
    tail_accuracy[class_num] = {}
    for img_name in top_50_color[class_num]:
        threshold = 0
        T_data = colordata[img_name]['True'][class_num]
        P_data = colordata[img_name]['Pred'][class_num]
        for i in range(20):
            if threshold not in tail_accuracy[class_num]:
                tail_accuracy[class_num][threshold] = {'TP':0, 'FP':0}
            if P_data > threshold:
                if (int(T_data) == 1):
                    tail_accuracy[class_num][threshold]['TP'] += 1
                else:
                    tail_accuracy[class_num][threshold]['FP'] += 1
            threshold += color_multiple[class_num]
plt.figure(figsize=(20,20))
for class_num in tail_accuracy:
    plt.subplot(5,8,(class_num*2)+1)
    plt.title('Class {}'.format(class_num+1))
    plt.xlabel('threshold')
    plt.ylabel('Tail accuracy')
    x_val = []
    y_val = []
    for threshold in tail_accuracy[class_num]:
        x_val.append(threshold)
        TP = tail_accuracy[class_num][threshold]['TP']
        FP = tail_accuracy[class_num][threshold]['FP']
        if TP+FP == 0:
            precision = 0 
        else:
            precision = TP/(TP+FP)
        y_val.append(precision)
    plt.plot(x_val, y_val)

plt.show()

# print(flip_precision)

plt.figure(figsize=(20,20))
plt.title('Precision Across 20 thresholds')
plt.xlabel('threshold')
plt.ylabel('precision')
classes = []
avg_precision = 0

for class_num in color_precision:
    class_precision = 0
    classes.append(class_num)
    x_val = color_precision[class_num]['x']
    y_val = []
    for threshold in x_val:
        TP_data = color_precision[class_num][threshold]['TP']
        FP_data = color_precision[class_num][threshold]['FP']
        if (TP_data+FP_data == 0):
            precision = 0
        else:
            precision = TP_data/(TP_data+FP_data)
        y_val.append(precision)
        class_precision += precision
    avg_precision += class_precision/20
    print('Class {} Precision: {}'.format(class_num, class_precision/20))
    plt.plot(x_val, y_val)

plt.legend(classes)
plt.show()
avg_precision /= 20
print('Model 2 precision: {}'.format(avg_precision))