import torch


def test(name):
    print(torch.cuda.is_available())

    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test('PyCharm')
