DBN=save_networks/dbn_Gaussian_gpu.pickle

viterbi:
	# needs a trained HMM and a trained DBN
	python ../src/batch_viterbi.py output_viterbi.mlf /fhgfs/bootphon/scratch/gsynnaeve/TIMIT/test/test.scp ../tmp_train/hmm_final/hmmdefs --d $(DBN) to_int_and_to_state_dicts_tuple.pickle
	HResults -I /fhgfs/bootphon/scratch/gsynnaeve/TIMIT/test/test.mlf ../tmp_train/phones output_viterbi.mlf
	HResults -e n en -e aa ao -e ah ax-h -e ah ax -e ih ix -e l el -e sh zh -e uw ux -e er axr -e m em -e n nx -e ng eng -e hh hv -e pau pcl -e pau tcl -e pau kcl -e pau q -e pau bcl -e pau dcl -e pau gcl -e pau epi -e pau sil -e pau !ENTER -e pau !EXIT -I /fhgfs/bootphon/scratch/gsynnaeve/TIMIT/test/test.mlf ../tmp_train/phones output_viterbi.mlf
