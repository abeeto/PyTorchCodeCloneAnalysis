from xml.dom.minidom import parse
import cv2 as cv

def reorganize_data(path, dest):
    """Parses the training xml and returns a dictionary of the data

    The dict has the following format:
    {
        img: [list of image paths],
        mask: [list of mask paths]
    }
    """
    print("parsing traning.xml...")
    document = parse(path + '/training.xml')

    # list of all tags for images
    images = document.getElementsByTagName("srcimg")
    # list of all tags for labels
    labels = document.getElementsByTagName("labelimg")

    for i in range(len(images)):
        image_path = images[i].getAttribute("name")
        label_path = labels[i].getAttribute("name")

        # move image to dest/images
        cv.imwrite(dest + "/images/" + str(i) + '.png', cv.imread(path + "/" + image_path))
        print(str(i) + ": " + image_path + " -> " + dest + "/images/" + str(i) + '.png')
        # move label to dest/labels
        cv.imwrite(dest + "/labels/" + str(i) + '.png', cv.imread(path + "/" + label_path))
        print(str(i) + ": " + label_path + " -> " + dest + "/labels/" + str(i) + '.png')


reorganize_data("./headsegmentation_dataset_ccncsa", "./organized_dataset")
