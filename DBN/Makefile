DBN=save_networks/dbn_Gaussian_gpu.pickle
DICT=save_networks/to_int_and_to_state_dicts_tuple.pickle
DATASET=/fhgfs/bootphon/scratch/gsynnaeve/TIMIT

viterbi:
	# needs a trained HMM and a trained DBN
	python ../src/batch_viterbi.py output_viterbi.mlf $(DATASET)/test/test.scp ../tmp_train/hmm_final/hmmdefs --d $(DBN) $(DICT)
	HResults -I $(DATASET)/test/test.mlf ../tmp_train/phones output_viterbi.mlf
	HResults -e n en -e aa ao -e ah ax-h -e ah ax -e ih ix -e l el -e sh zh -e uw ux -e er axr -e m em -e n nx -e ng eng -e hh hv -e pau pcl -e pau tcl -e pau kcl -e pau q -e pau bcl -e pau dcl -e pau gcl -e pau epi -e pau sil -e pau !ENTER -e pau !EXIT -I $(DATASET)/test/test.mlf ../tmp_train/phones output_viterbi.mlf
