from data.image_dataset import ImageDataset

def main():
    path = './data/test1'
    dogcat = ImageDataset(path)
    data = dogcat[1]

    print(data['targets'])
    print(dogcat.class_to_index)

if __name__ == '__main__':
    main()