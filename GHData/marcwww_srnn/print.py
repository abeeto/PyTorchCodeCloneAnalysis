import random
def gen_line(line,name):
    with open('./data/'+name,'a+') as f:
            print(line,file=f)

def gen_train():
    alphabet=[chr(ord('a')+i) for i in range(26)]
    for _ in range(1000):
        ch1=random.sample(alphabet,k=1)[0]
        n=int(random.random()*10)+1
        src=[ch1 for _ in range(n)]
        tar=[ch1 for _ in range(n)]
        line=' '.join(src)+'\t'+' '.join(tar)
        gen_line(line,'aa-bb')

def gen_test():
    alphabet = [chr(ord('a') + i) for i in range(26)]
    for _ in range(1000):
        ch1 = random.sample(alphabet, k=1)[0]
        n = int(random.random() * 10) + 5
        src = [ch1 for _ in range(n)]
        tar = [ch1 for _ in range(n)]
        line = ' '.join(src) + '\t' + ' '.join(tar)
        gen_line(line,'aa-bb.test')

# gen_train()
gen_test()



