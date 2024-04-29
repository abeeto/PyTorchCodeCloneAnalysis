no_space = open('pku_training_no_space.txt', 'r', encoding='utf8')
label = open('pku_training_label.txt', 'r', encoding='utf8')
# orig = open('pku_training.txt', 'r', encoding='utf8')
# write_file = open('wrong.txt', 'w', encoding='utf8')
line = 1
count = 0
for line1, line2 in zip(no_space, label):
    if (len(line1) != len(line2)):
        print('at line', line)
        print('line1', len(line1), 'chars')
        print(line1)
        print('line2', len(line2), 'chars')
        print(line2)
        # write_file.write(line3)
        count += 1
    line += 1

no_space.close()
label.close()
# orig.close()

print('total', count, 'instances')
