import sys, re

#USAGE:
#    python adapt_quests.py tmp_train/monophones1 quests_example.hed

phones_f = open(sys.argv[1])
phones = []
for line in phones_f:
    phones.append(line.rstrip('\n'))
phones_f.close()

questions_f = open(sys.argv[2])
output_f = open(sys.argv[2].split('.')[0] + '_pruned.hed', 'w')
for line in questions_f:
    tmp = line.rstrip('\n').split('{')
    p_l_s = tmp[1].rstrip('}')
    p_l = filter(lambda x: re.sub('[^a-z]', '', x) in phones, p_l_s.split(','))
    if len(p_l) > 0:
        output_f.write(tmp[0] + '{')
        output_f.write(','.join(p_l) + '}\n')



