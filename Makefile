prepare: wav_config mfcc_and_gammatones.py timit_to_htk_labels.py
	@echo ">>> preparing the dataset"
	python mfcc_and_gammatones.py --htk-mfcc $(dataset)/train
	python mfcc_and_gammatones.py --htk-mfcc $(dataset)/test
	python timit_to_htk_labels.py $(dataset)/train
	python timit_to_htk_labels.py $(dataset)/test
	python substitute_phones.py $(dataset)/train
	python substitute_phones.py $(dataset)/test
	python create_phonesMLF_and_labels.py $(dataset)/train
	python create_phonesMLF_and_labels.py $(dataset)/test

train:
	@echo ">>> training the HMMs with HTK"

test:
	@echo ">>> testing the trained model"
