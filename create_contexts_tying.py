import sys

usage = """
    python create_contexts_tying.py QUESTIONS_FILE THRESHOLDS_FILE OUTPUT_FILE [STATS_FILE]"
    e.g.:
    python create_contexts_tying.py tmp_train/quests_pruned.hed tmp_train/tb_contexts.hed tmp_train/tree.hed tmp_train/tri_stats
    """

if len(sys.argv) < 4:
    print usage
    sys.exit(-1)

output = open(sys.argv[3], 'w')
if len(sys.argv) < 5:
    output.write("RO 100.0 stats\n")
else:
    output.write("RO 100.0 " + sys.argv[4].strip(' ') + '\n')
output.write("TR 0\n")
for line in open(sys.argv[1]):
    output.write(line)
output.write("TR 2\n")
for line in open(sys.argv[2]):
    output.write(line)
output.write('AU "fulllist"\n')
output.write('CO "tiedlist"\n')
output.write('ST "trees"\n')
output.close()
