import os
from PIL import Image, ImageDraw,ImageFilter
import numpy as np
from tool import utils
import imagesize
import traceback

# 数据地址
anno_src = r'D:\celebA\Anno\list_bbox_celeba.txt'
# 图片地址
img_src = r'D:\celebA\img_celeba'

# 生成文件地址
save_path = r'D:\celeba4'


def face(size):
	for face_size in [size]:
		# 样片图片存储地址
		positive_img = os.path.join(save_path, str(face_size), 'positive')
		part_img = os.path.join(save_path, str(face_size), 'part')
		negative_img = os.path.join(save_path, str(face_size), 'negative')
		for dir in [positive_img, part_img, negative_img]:
			if not os.path.exists(dir):
				os.makedirs(dir)

		# 样本数据存储地址
		positive_annofilename = os.path.join(save_path, str(face_size), 'positive.txt')
		part_annofilename = os.path.join(save_path, str(face_size), 'part.txt')
		negative_annofilename = os.path.join(save_path, str(face_size), 'negative.txt')

		positive_acount = 0
		part_acount = 0
		negative_acount = 0

		try:
			positive_anno = open(positive_annofilename, 'w')
			part_anno = open(part_annofilename, 'w')
			negative_anno = open(negative_annofilename, 'w')

			for i, line in enumerate(open(anno_src)):
				if i < 2:
					continue
				try:
					strs = line.strip().split(" ")
					strs = list(filter(bool, strs))
					img_filename = strs[0]
					img_file = os.path.join(img_src, img_filename)
					with Image.open(img_file) as img:
						# 图片坐标信息
						img_w, img_h = img.size
						x1 = int(strs[1])
						y1 = int(strs[2])
						w = int(strs[3])
						h = int(strs[4])
						x2 = x1 + w
						y2 = y1 + h
						side = np.maximum(w,h)
						print(x1, y1, w, h)
						if max(w, h) < 40 or x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
							continue
						boxes = [[x1, x2, y1, y2]]

						# 计算人脸中心点的位置
						cx = x1 + w / 2
						cy = y1 + h / 2

						# 让正样本和负样本翻倍
						for _ in range(1):
							offset_side = np.random.uniform(-0.2,0.2)*side
							offset_x = np.random.uniform(-0.2,0.2)*w/2
							offset_y = np.random.uniform(-0.2,0.2)*h/2
							_cx = cx+offset_x
							_cy = cy +offset_y
							_side = side +offset_side

							_x1 = np.maximum(cx-_side*0.5,0)
							_y1 = np.maximum(cy-_side*0.5,0)
							_x2 = _x1 +_side
							_y2 = _y1 +_side


							offsetx1_ = (x1 - _x1) / _side
							offsety1_ = (y1 - _y1) / _side
							offsetx2_ = (x2 - _x2) / _side
							offsety2_ = (y2 - _y2) / _side
							# 剪下图片
							box = np.array([x1, y1, x2, y2])
							boxes = np.array([[_x1,_y1,_x2,_y2]])
							face_crop = img.crop((x1,y1,x2,y2))
							face_resize = face_crop.resize((face_size, face_size),Image.ANTIALIAS)
							imglist = []
							imglist.append(face_resize)
							iou = utils.iou(box, boxes)[0]
							# 图片模糊化处理
							obfuscate_face = face_resize.filter(ImageFilter.BLUR)
							imglist.append(obfuscate_face)

							for _img in imglist:
								if iou > 0.65:
									positive_anno.write(
										'positive/{0}.jpg {1} {2} {3} {4} {5}\n'.format(positive_acount, 1, offsetx1_,
										                                                offsety1_, offsetx2_, offsety2_))
									positive_anno.flush()
									_img.save(os.path.join(positive_img, '{0}.jpg'.format(positive_acount)))
									positive_acount += 1

								elif (iou > 0.4) and (iou < 0.65):
									part_anno.write(
										'part/{0}.jpg {1} {2} {3} {4} {5}\n'.format(part_acount, 2, offsetx1_, offsety1_,
										                                            offsetx2_, offsety2_))
									part_anno.flush()
									_img.save(os.path.join(part_img, '{0}.jpg'.format(part_acount)))
									part_acount += 1
								elif iou < 0.3:
									negative_anno.write(
										"negative/{0}.jpg {1} 0 0 0 0\n".format(negative_acount, 0))
									negative_anno.flush()
									_img.save(os.path.join(negative_img, "{0}.jpg".format(negative_acount)))
									negative_acount += 1

							#单独生成负样本
							_boxes = np.array(boxes)
							for _ in range(1):
								side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
								x_ = np.random.randint(0, img_w - side_len)
								y_ = np.random.randint(0, img_h - side_len)
								crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

								if np.max(utils.iou(crop_box, _boxes)) < 0.3:
									face_crop = img.crop(crop_box)
									face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

									negative_anno.write(
										"negative/{0}.jpg {1} 0 0 0 0\n".format(negative_acount, 0))
									negative_anno.flush()
									face_resize.save(os.path.join(negative_img, "{0}.jpg".format(negative_acount)))
									negative_acount += 1
				except Exception as e:
					traceback.print_exc()
		finally:
			positive_anno.close()
			negative_anno.close()
			part_anno.close()

if __name__ == '__main__':
   a = face(12)
   b = face(24)
   c = face(48)
