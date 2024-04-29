import os
import numpy as np
import struct as st

class ParseFile():
    def __init__(self, 
                img_file_name, 
                label_file_name, 
                classification_label):
        self.img_file_name = img_file_name
        self.label_file_name = label_file_name
        self.classification_label = classification_label

    def parse_image(self):
        binfile = open(self.img_file_name, 'rb')
        bufferfile = binfile.read()
        # use unpack function to convert them to unsigned int
        # >: big endian ; I: unsigned int; 0: offset = 0
        # IIII: means read 4 * 4bytes
        head = st.unpack_from(">IIII", bufferfile, 0)
        img_num = head[1]
        num_row = head[2]
        num_col = head[3]
        # offset: get position where data start
        img_offset = st.calcsize(">IIII")
        byte_num = img_num * num_row * num_col
        img_fmt = ">" + str(byte_num) + "B"
        img_byte = st.unpack_from(img_fmt, bufferfile, img_offset)
        # reshape img_byte to [60000, 784]
        img_data = np.reshape(img_byte, [img_num, num_row * num_col])
        if os.path.isdir("./mnist") == False: 
            os.mkdir("./mnist")
        if str(self.img_file_name).find("train") >= 0:
            img_path = "./mnist/train_image.npy"
        else:
            img_path = "./mnist/test_image.npy"
        np.save(img_path, img_data)
        binfile.close()
        return img_data, head

    def parse_label(self):
        binfile = open(self.label_file_name, 'rb')
        bufferfile = binfile.read()
        head = st.unpack_from(">II", bufferfile, 0)
        label_num = head[1]
        label_offset = st.calcsize(">II")
        label_fmt = ">" + str(label_num) + "B"
        label = st.unpack_from(label_fmt, bufferfile, label_offset)
        if os.path.isdir("./mnist") == False: 
            os.mkdir("./mnist")
        if str(self.label_file_name).find("train") >= 0:
            label_path = "./mnist/train_label.npy"
        else:
            label_path = "./mnist/test_label.npy"
        np.save(label_path, label)
        binfile.close()
        return label, head

    def classify(self):
        img_data, img_head = ParseFile(
                        self.img_file_name, 
                        self.label_file_name,
                        self.classification_label).parse_image()
        label, label_head = ParseFile(
                        self.img_file_name, 
                        self.label_file_name,
                        self.classification_label).parse_label()
        # classify those img_data with label
        count = 0
        for i in range(np.shape(label)[0]):
            if label[i] == self.classification_label: 
                if count == 0:
                    np_img = np.array(img_data[i, :])
                else:
                    np_img = np.vstack((np_img, np.array(img_data[i, :])))
                count = count + 1
        # save file to certain path
        if os.path.isdir("./mnist_classified") == False: 
            os.mkdir("./mnist_classified")
        if str(self.img_file_name).find("train") >= 0:
            classified_img_path = "./mnist_classified/train_image" +\
                        str(self.classification_label) + ".npy"
        else:
            classified_img_path = "./mnist_classified/test_image" +\
                        str(self.classification_label) + ".npy"
        np.save(classified_img_path, np_img)
        print(self.classification_label)
        print(np.shape(np_img))