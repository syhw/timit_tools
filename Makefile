help:
	@echo "Usage:"
	@echo "make prepare dataset=~/postdoc/datasets/TIMIT"
	@echo "make train train_folder=~/postdoc/datasets/TIMIT/train"
	@echo "make test test_folder=~/postdoc/datasets/TIMIT/test"

prepare: wav_config mfcc_and_gammatones.py timit_to_htk_labels.py
	@echo ">>> preparing the dataset"
	# produce MFCC from WAV files
	python mfcc_and_gammatones.py --htk-mfcc $(dataset)/train
	python mfcc_and_gammatones.py --htk-mfcc $(dataset)/test
	# transform .phn files into .lab files (frames into nanoseconds)
	python timit_to_htk_labels.py $(dataset)/train
	python timit_to_htk_labels.py $(dataset)/test
	# subtitles phones (61 down to 39), TODO check "q" (glottal stop) with Emmanuel
	python substitute_phones.py $(dataset)/train
	python substitute_phones.py $(dataset)/test
	# creates (train|test).mlf, (train|test).scp listings and labels (dicts).
	python create_phonesMLF_list_labels.py $(dataset)/train
	python create_phonesMLF_list_labels.py $(dataset)/test

train:
	@echo ">>> training the HMMs with HTK"
	mkdir tmp_train
	cp proto.hmm tmp_train/
	cp $(train_folder)/train/train.scp tmp_train/
	HCompV -A -D -T 1 -f 0.001 -m -S tmp_train/train.scp -M tmp_train tmp_train/proto.hmm

test:
	@echo ">>> testing the trained model"
	# $(test_folder)
