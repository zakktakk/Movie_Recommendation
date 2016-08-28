import sys

def mapper(inputfile):
    prev_id = ''
    flg = False
    output = list()
    for line in inputfile:
        y_id, p_id, rating = line.split('\t')
        if y_id != prev_id:
            if flg:
                print_log(output)
            output = list()
            flg = False
        if int(rating) == 2 or int(rating) == 3:
            flg = True
        output.append(line)
        prev_id = y_id
    if flg:
        print_log(output)

def print_log(output):
    for element in output:
        print element

if __name__ == '__main__':
    f = open('test.txt','r')
    mapper(f)
    f.close()
