import camelot

table = camelot.read_pdf('paper-pdf/P19-1001.pdf', flavor='stream', pages='7')
table[1].to_csv("hello1.csv")
table[2].to_csv("hello2.csv")
for item in table:
    print(item.df)
