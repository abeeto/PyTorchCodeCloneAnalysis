from PyPDF2 import PdfFileReader

# This is the class based implementation of the PDF reader.

path = "data/Apocalypse_Now.pdf"


class ReadPDF(object):

    file_object = open(path, 'rb')

    def __init__(self):
        pass

    def read_data(self):

        text = PdfFileReader(self.file_object)

        if text.isEncrypted:

            text.decrypt('')

        return text

    def extract_all_data(self, text):

        my_lst = []

        for i in range(1, text.numPages):
            pageobj = text.getPage(i)

            my_lst.append(pageobj.extractText())

        self.file_object.close()

        return my_lst


def main():

    pdf_object = ReadPDF()
    my_text = pdf_object.read_data()
    data_lst = pdf_object.extract_all_data(my_text)

    p = [data.split() for data in data_lst]
    print(p[1])
    print(len(p[1]))


if __name__ == '__main__':
    main()

