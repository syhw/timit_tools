TMP_TRAIN_FOLDER=tmp_train

help:
	@echo "Usage (in order):"
	@echo "make prepare dataset=~/postdoc/datasets/TIMIT"
	@echo "make train train_folder=~/postdoc/datasets/TIMIT/train"
	@echo "make test test_folder=~/postdoc/datasets/TIMIT/test"

prepare: wav_config mfcc_and_gammatones.py timit_to_htk_labels.py
	@echo "*** preparing the dataset ***"
	@echo "\n>>> produce MFCC from WAV files\n"
	python mfcc_and_gammatones.py --htk-mfcc $(dataset)/train
	python mfcc_and_gammatones.py --htk-mfcc $(dataset)/test
	@echo "\n>>> transform .phn files into .lab files (frames into nanoseconds)\n"
	python timit_to_htk_labels.py $(dataset)/train
	python timit_to_htk_labels.py $(dataset)/test
	@echo "\n>>> subtitles phones (61 down to 39)\n" # TODO check "q" (glottal stop) with Emmanuel
	python substitute_phones.py $(dataset)/train
	python substitute_phones.py $(dataset)/test
	@echo "\n>>> creates (train|test).mlf, (train|test).scp listings and labels (dicts)\n"
	python create_phonesMLF_list_labels.py $(dataset)/train
	python create_phonesMLF_list_labels.py $(dataset)/test

train:
	@echo "*** training the HMMs with HTK ***"
	@echo "\n>>> preparing the HMMs\n"
	mkdir tmp_train
	cp $(dataset_train_folder)/labels $(TMP_TRAIN_FOLDER)/
	cp $(dataset_train_folder)/train.mlf $(TMP_TRAIN_FOLDER)/
	cp $(dataset_train_folder)/train.scp $(TMP_TRAIN_FOLDER)/
	cp proto.hmm $(TMP_TRAIN_FOLDER)/
	mkdir $(TMP_TRAIN_FOLDER)/hmm0
	mkdir $(TMP_TRAIN_FOLDER)/hmm1
	mkdir $(TMP_TRAIN_FOLDER)/hmm2
	mkdir $(TMP_TRAIN_FOLDER)/hmm3
	# -A -D -T 1 
	HCompV -f 0.001 -m -S $(TMP_TRAIN_FOLDER)/train.scp -M $(TMP_TRAIN_FOLDER)/hmm0 $(TMP_TRAIN_FOLDER)/proto.hmm
	python create_hmmdefs_from_proto.py $(TMP_TRAIN_FOLDER)/hmm0/proto $(TMP_TRAIN_FOLDER)/labels $(TMP_TRAIN_FOLDER)/hmm0/ $(TMP_TRAIN_FOLDER)/hmm0/vFloors
	@echo "\n>>> training the HMMs\n"
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -t 250.0 150.0 1000.0 -H $(TMP_TRAIN_FOLDER)/hmm0/macros -H $(TMP_TRAIN_FOLDER)/hmm0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm1 $(TMP_TRAIN_FOLDER)/labels # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -t 250.0 150.0 1000.0 -H $(TMP_TRAIN_FOLDER)/hmm1/macros -H $(TMP_TRAIN_FOLDER)/hmm1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm2 $(TMP_TRAIN_FOLDER)/labels # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -t 250.0 150.0 1000.0 -H $(TMP_TRAIN_FOLDER)/hmm2/macros -H $(TMP_TRAIN_FOLDER)/hmm2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm3 $(TMP_TRAIN_FOLDER)/labels # check these -t parameters TODO
	@echo "\n>>> tweaking the silence model\n"
	mkdir $(TMP_TRAIN_FOLDER)/hmm4
	python create_short_pause_silence_model.py $(TMP_TRAIN_FOLDER)/hmm3/hmmdefs $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs $(TMP_TRAIN_FOLDER)/hmm4/labels
	cp $(TMP_TRAIN_FOLDER)/hmm3/macros $(TMP_TRAIN_FOLDER)/hmm4/
	mkdir $(TMP_TRAIN_FOLDER)/hmm5
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm4/macros -H $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm5 sil.hed $(TMP_TRAIN_FOLDER)/hmm4/labels
	@echo "\n>>> re-training the HMMs\n"
	mkdir $(TMP_TRAIN_FOLDER)/hmm6
	mkdir $(TMP_TRAIN_FOLDER)/hmm7
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -t 250.0 150.0 1000.0 -H $(TMP_TRAIN_FOLDER)/hmm4/macros -H $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm5 $(TMP_TRAIN_FOLDER)/labels # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -t 250.0 150.0 1000.0 -H $(TMP_TRAIN_FOLDER)/hmm5/macros -H $(TMP_TRAIN_FOLDER)/hmm5/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm6 $(TMP_TRAIN_FOLDER)/labels # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -t 250.0 150.0 1000.0 -H $(TMP_TRAIN_FOLDER)/hmm6/macros -H $(TMP_TRAIN_FOLDER)/hmm6/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm7 $(TMP_TRAIN_FOLDER)/labels # check these -t parameters TODO


test:
	@echo "*** testing the trained model ***"
	@echo $(test_folder)
