import os
import json
from scipy import stats


def convert_text2dict(filename):
    last_line = file(filename, "r").readlines()[-1]
    json_acceptable_string = last_line.replace('\"', '').replace('\n', '').replace("'", "\"")
    try:
        out_dict = json.loads(json_acceptable_string)
    except ValueError:
        print filename
        print "file incomplete"
        out_dict = {}
        out_dict['books'] = 0.0
        out_dict['electronics'] = 0.0
        out_dict['kitchen'] = 0.0
        out_dict['dvd'] = 0.0
    return out_dict


def calculate_stats(dir):
    books = []
    elecs = []
    dvds = []
    ktchn = []
    files = os.listdir(dir)
    for file in files:
        out_dict = convert_text2dict(dir + '/' + file)
        books.append(out_dict['books'])
        elecs.append(out_dict['electronics'])
        dvds.append(out_dict['dvd'])
        ktchn.append(out_dict['kitchen'])

    print 'getting stats for Books:'
    print stats.describe(books)
    print 'getting stats for Electronics:'
    print stats.describe(elecs)
    print 'getting stats for Kitchen:'
    print stats.describe(ktchn)
    print 'getting stats for DVD:'
    print stats.describe(dvds)

if __name__ == '__main__':
    print "stats on domains mit topic lstm"
    calculate_stats('/home/DebanjanChaudhuri/topic_lstm_torch/mit_topic')
    print "stats on domains ohne topic lstm"
    calculate_stats('/home/DebanjanChaudhuri/topic_lstm_torch/ohne_topic')