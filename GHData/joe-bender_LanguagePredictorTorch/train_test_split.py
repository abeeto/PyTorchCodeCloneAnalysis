import data
import random
import math

def create_train_test():
    dataset = data.get_dataset()
    random.shuffle(dataset)
    split_point = math.floor(.9 * len(dataset))
    training_set = dataset[:split_point]
    test_set = dataset[split_point:]

    # write training set to file
    def write_csv(filename, pairs):
        with open('datasets/{}.csv'.format(filename), 'w') as f:
            for word, language in pairs:
                f.write('{},{}\n'.format(word, language))

    write_csv('training_set', training_set)
    write_csv('test_set', test_set)
