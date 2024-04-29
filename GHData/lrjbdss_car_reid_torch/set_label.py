def set_label(train_txt, color_txt, model_txt):
    id2color = {}
    for line in open(color_txt, 'r'):
        v_id, color = line.strip().split(' ')
        id2color[v_id] = color

    id2model = {}
    for line in open(model_txt, 'r'):
        v_id, model = line.strip().split(' ')
        id2model[v_id] = model

    with open('3200_test_list.txt', 'w') as f:
        for line in open(train_txt, 'r'):
            line = line.strip()
            _, v_id = line.split(' ')
            try:
                line += ' ' + id2color[v_id] + ' ' + id2model[v_id]
                f.write(line + '\n')
            except KeyError:
                continue


if __name__ == '__main__':
    train_txt = '/media/lx/新加卷/datasets/VehicleID/train_test_split/test_list_3200.txt'
    color_txt = '/media/lx/新加卷/datasets/VehicleID/attribute/color_attr.txt'
    model_txt = '/media/lx/新加卷/datasets/VehicleID/attribute/model_attr.txt'
    set_label(train_txt, color_txt, model_txt)
