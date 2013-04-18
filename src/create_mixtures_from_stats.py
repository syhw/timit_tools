import sys

EXACT_NB_MIXTURES = True
max_mix = {}
progression = [1, 2, 3, 5, 9, 17]
working_folder = '/'.join(sys.argv[1].split('/')[:-1])

with open(sys.argv[1]) as f:
    for line in f:
        if len(line.split()) < 6:
            continue
        (id_phn, phn, nb), nb_frames = line.split()[:3], line.split()[3:]
        nb_frames_min = min(map(float, nb_frames))
        nb_mixtures = max(1, int(nb_frames_min / 100))
        # at least 100 frames per mixture comp.
        phn = phn.strip('"')
        max_mix[phn] = nb_mixtures

for i, ind in enumerate(progression):
    with open(working_folder + "/TRMU" + str(ind) + ".hed", 'w') as f:
        for phn, mmix in max_mix.iteritems():
            n = mmix
            if mmix >= ind:
                n = ind
            else:
                if not EXACT_NB_MIXTURES:
                    max_mix[phn] = progression[max(0, i-1)]
                    n = max_mix[phn]
            f.write("MU " + str(n) + " {" + phn + ".state[2-4].mix}\n")

