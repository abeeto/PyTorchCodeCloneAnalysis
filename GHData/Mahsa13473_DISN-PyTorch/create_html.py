import os.path as osp
import os
import numpy as np

base_dir = '/local-scratch/mma/DISN/DISN/datasets/ShapeNet'

filelist_dir = './filelists_chair'
model_input_dir = '/local-scratch/mma/DISN/DISN/datasets/ShapeNet/ShapeNetRendering'
model_output_dir = '/local-scratch/mma/DISN/DISN/datasets/ShapeNet/ShapeNetRendering.v1'

def writeHTML(i1, o1, o2):
    all = []
    print(len(i1))

    for i in range (len(i1)):
        # img_path_1 = 'visualization_test/'+str(i)+'_original_test'+'.png'
        # img_path_2 = 'visualization_test/'+str(i)+'_corners_pred_gt'+'.png'
        # img_path_3 = 'visualization_test/'+str(i)+'_corners_pred'+'.png'
        # img_path_4 = 'visualization_test/'+str(i)+'_heatmap_pred'+'.png'
        img_path_1 = i1[i]
        img_path_2 = o1[i]
        img_path_3 = o2[i]

        all += [(img_path_1, img_path_2, img_path_3)]


    with open('./RESULT_2D_CURVES.html', 'w') as file:
        file.write("""
            <html>
            <head>
            <title>result</title>
            <style>
            table {
            border-collapse: collapse;
            width: 100%;
            text-align: center;
            }
            th {
            color: black;
            }
            </style>
            </head>
            <body>
            <table border='1'>
            <thead>
            <tr>
            <th width='240px'>Input Image</th>
            <th width='240px'>Ground truth</th>
            <th width='240px'>Output</th>
            </tr>
            </thead>
            """)

        for (i, j, k) in all:
            file.write("<tr>\n")
            file.write("<td width='240px'>\n<img src=\"%s\" height=240 width=240>\n</td>\n" % (i))
            file.write("<td width='240px'>\n<img src=\"%s\" height=240 width=240>\n</td>\n" % (j))
            file.write("<td width='240px'>\n<img src=\"%s\" height=240 width=240>\n</td>\n" % (k))

        file.write("</tr>\n")

        file.write("</table></body>\n</html>")


def gen_obj(model_input_dir, model_output_dir, cat_id, obj_id):
    input_path = os.path.join(model_input_dir, cat_id,
                              obj_id, "rendering", "00.png")  # for v1

    output_path1 = os.path.join(model_output_dir, "albedo", cat_id, obj_id, "00.png")
    output_path2 = os.path.join(model_output_dir, "albedo", cat_id, obj_id, "08.png")
    output_path3 = os.path.join(model_output_dir, "albedo", cat_id, obj_id, "20.png")

    return input_path, output_path1, output_path2, output_path3

if __name__ == '__main__':
    '''
    i_lst = []
    o1_lst = []
    o2_lst = []
    o3_lst = []
    for filename in os.listdir(filelist_dir):
        if filename.endswith(".lst"):
            cat_id = filename.split(".")[0]
            file = os.path.join(filelist_dir, filename)
            lst = []
            with open(file) as f:
                content = f.read().splitlines()
                for line in content:
                    lst.append(line)
            model_input_dir_lst = [model_input_dir for i in range(len(lst))]
            model_output_dir_lst = [model_output_dir for i in range(len(lst))]
            cat_id_lst = [cat_id for i in range(len(lst))]
        for model_input_dir, model_output_dir, cat_id, obj_id in zip(model_input_dir_lst, model_output_dir_lst, cat_id_lst, lst):
            if len(i_lst) > 99:
                break
            else:
                i, o1, o2, o3 = gen_obj(model_input_dir, model_output_dir, cat_id, obj_id)
                # print(i, o1, o2, o3)
                # print("*********")
                i_lst.append(i)
                o1_lst.append(o1)
                o2_lst.append(o2)
                o3_lst.append(o3)
        print("Finished %s" % cat_id)
    '''
    for dir in os.walk('mlp_img/'):
        for file in dir:
            img_dirs = [img for img in file]

    img_dirs = sorted(img_dirs, key=len, reverse=False)
    for j in range(8,12):
        for i in range(6):
            if j==8 and i==0:
                i_lst = ['mlp_img/image_%d_%d_2.png'%(j, i)]
                o1_lst = ['mlp_img/image_%d_%d_1.png' % (j, i)]
                o2_lst = ['mlp_img/image_%d_%d_0.png' % (j, i)]
            else:
                i_lst = np.append(i_lst, ['mlp_img/image_%d_%d_2.png' % (j, i)])
                o1_lst = np.append(o1_lst, ['mlp_img/image_%d_%d_1.png' % (j, i)])
                o2_lst = np.append(o2_lst, ['mlp_img/image_%d_%d_0.png' % (j, i)])
    writeHTML(i_lst, o1_lst, o2_lst) # pass a list of directories of what you want to show in HTML file