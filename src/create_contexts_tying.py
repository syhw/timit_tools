import sys

usage = """
    python create_contexts_tying.py QUESTIONS_FILE THRESHOLDS_FILE OUTPUT_FILE [FOLDER]"
    e.g.:
    python create_contexts_tying.py tmp_train/quests_pruned.hed tmp_train/tb_contexts.hed tmp_train/tree.hed tmp_train
    """

if len(sys.argv) < 4:
    print usage
    sys.exit(-1)

folder = '.'
output = open(sys.argv[3], 'w')
if len(sys.argv) > 4:
    folder = sys.argv[4].strip(' ').strip('/')
output.write('RO 100.0 "' + folder + '/tri_stats"\n')
output.write("TR 0\n")
for line in open(sys.argv[1]):
    output.write(line)
output.write("TR 2\n")
for line in open(sys.argv[2]):
    output.write(line)
#output.write('AU "' + folder + '/fulllist"\n') # we expect to already have 
# the full list of triphones in our training set
output.write('CO "' + folder + '/tiedlist"\n')
output.write('ST "' + folder + '/trees"\n')
output.close()
