from PyPDF2 import PdfFileReader

# This the function based implementation

path = "data/Apocalypse_Now.pdf"

file_object = open(path, 'rb')


def read_data(file_object):

    data = PdfFileReader(file_object)

    if data.isEncrypted:
        data.decrypt('')

    return data


def extract_all_data(data):

    text = ''

    for i in range(1, data.numPages):

        pageobj = data.getPage(i)

        text += pageobj.extractText()

    return text


def main():
    text = extract_all_data(read_data(file_object))
    print(text)


if __name__ == '__main__':
    main()

file_object.close()
