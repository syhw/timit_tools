import sys, re

usage = """
    python adapt_quests.py MONOPHONES_FILE EXISTING_QUESTIONS_FILE PRUNED_QUESTIONS_FILE
    e.g.
    python adapt_quests.py tmp_train/monophones0 quests_example.hed tmp_train/quests.hed
"""
if len(sys.argv) < 4:
    print usage
    sys.exit(-1)

phones_f = open(sys.argv[1])
phones = []
for line in phones_f:
    phones.append(line.rstrip('\n'))
phones_f.close()

questions_f = open(sys.argv[2])
output_f = open(sys.argv[3], 'w')
for line in questions_f:
    tmp = line.rstrip('\n').split('{')
    p_l_s = tmp[1].rstrip('}')
    p_l = filter(lambda x: re.sub('[^a-z]', '', x) in phones, p_l_s.split(','))
    if len(p_l) > 0:
        output_f.write(tmp[0] + '{')
        output_f.write(','.join(p_l) + '}\n')



