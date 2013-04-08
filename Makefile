TMP_TRAIN_FOLDER=tmp_train

help:
	@echo "Usage (in order):"
	@echo "make prepare dataset=~/postdoc/datasets/TIMIT"
	@echo "make train dataset_train_folder=~/postdoc/datasets/TIMIT/train"
	@echo "make test dataset_test_folder=~/postdoc/datasets/TIMIT/test"

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

train: train_monophones
	@echo "\n>>> We will only train monophones, see train_triphones make cmd otherwise\n"

train_monophones_monogauss:
	@echo "*** training the HMMs with HTK ***"
	@echo "using folder $(dataset_train_folder)"
	@echo "\n>>> preparing the HMMs\n"
	mkdir $(TMP_TRAIN_FOLDER)
	cp $(dataset_train_folder)/labels $(TMP_TRAIN_FOLDER)/monophones0
	cp $(dataset_train_folder)/train.mlf $(TMP_TRAIN_FOLDER)/
	cp $(dataset_train_folder)/train.scp $(TMP_TRAIN_FOLDER)/
	cp wdnet $(TMP_TRAIN_FOLDER)/
	cp proto.hmm $(TMP_TRAIN_FOLDER)/
	mkdir $(TMP_TRAIN_FOLDER)/hmm0
	mkdir $(TMP_TRAIN_FOLDER)/hmm1
	mkdir $(TMP_TRAIN_FOLDER)/hmm2
	mkdir $(TMP_TRAIN_FOLDER)/hmm3
	# -A -D -T 1 
	HCompV -f 0.001 -m -S $(TMP_TRAIN_FOLDER)/train.scp -M $(TMP_TRAIN_FOLDER)/hmm0 $(TMP_TRAIN_FOLDER)/proto.hmm
	python create_hmmdefs_from_proto.py $(TMP_TRAIN_FOLDER)/hmm0/proto $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/hmm0/ $(TMP_TRAIN_FOLDER)/hmm0/vFloors
	@echo "\n>>> training the HMMs\n"
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm0/macros -H $(TMP_TRAIN_FOLDER)/hmm0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm1 $(TMP_TRAIN_FOLDER)/monophones0 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm1/macros -H $(TMP_TRAIN_FOLDER)/hmm1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm2 $(TMP_TRAIN_FOLDER)/monophones0 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm2/macros -H $(TMP_TRAIN_FOLDER)/hmm2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm3 $(TMP_TRAIN_FOLDER)/monophones0 # check these -t parameters TODO

tweak_silence_model: train_monophones_monogauss
	@echo "\n>>> tweaking the silence model\n"
	mkdir $(TMP_TRAIN_FOLDER)/hmm4
	#python create_short_pause_silence_model.py $(TMP_TRAIN_FOLDER)/hmm3/hmmdefs $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs $(TMP_TRAIN_FOLDER)/monophones1
	cp $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/monophones1
	cp $(TMP_TRAIN_FOLDER)/hmm3/hmmdefs $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs
	#tr "\n" " | " < $(TMP_TRAIN_FOLDER)/monophones1 > $(TMP_TRAIN_FOLDER)/gram
	#cp $(TMP_TRAIN_FOLDER)/monophones1 $(TMP_TRAIN_FOLDER)/dict # because our words are the phones
	awk '{if(!$$2) print $$1 " " $$1}' $(TMP_TRAIN_FOLDER)/monophones1 > $(TMP_TRAIN_FOLDER)/dict
	#echo "silence sil" >> $(TMP_TRAIN_FOLDER)/dict
	cp $(TMP_TRAIN_FOLDER)/hmm3/macros $(TMP_TRAIN_FOLDER)/hmm4/
	mkdir $(TMP_TRAIN_FOLDER)/hmm5
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm4/macros -H $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm5 sil.hed $(TMP_TRAIN_FOLDER)/monophones1
	@echo "\n>>> re-training the HMMs\n"
	mkdir $(TMP_TRAIN_FOLDER)/hmm6
	mkdir $(TMP_TRAIN_FOLDER)/hmm7
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm5/macros -H $(TMP_TRAIN_FOLDER)/hmm5/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm6 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO

train_monophones: tweak_silence_model
	@echo "\n>>> estimating the number of mixtures\n"
	HERest -s $(TMP_TRAIN_FOLDER)/stats -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm6/macros -H $(TMP_TRAIN_FOLDER)/hmm6/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm7 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	python create_mixtures_from_stats.py $(TMP_TRAIN_FOLDER)/stats
	mkdir $(TMP_TRAIN_FOLDER)/hmm8
	mkdir $(TMP_TRAIN_FOLDER)/hmm9
	mkdir $(TMP_TRAIN_FOLDER)/hmm10
	@echo "\n--- mixtures of 2 components ---"
	HHed -H $(TMP_TRAIN_FOLDER)/hmm7/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU2.hed $(TMP_TRAIN_FOLDER)/monophones1
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm7/macros -H $(TMP_TRAIN_FOLDER)/hmm7/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm8 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm8/macros -H $(TMP_TRAIN_FOLDER)/hmm8/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm9 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm9/macros -H $(TMP_TRAIN_FOLDER)/hmm9/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm10 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	@echo "\n--- mixtures of 3 components ---"
	HHed -H $(TMP_TRAIN_FOLDER)/hmm10/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU3.hed $(TMP_TRAIN_FOLDER)/monophones1
	mkdir $(TMP_TRAIN_FOLDER)/hmm11
	mkdir $(TMP_TRAIN_FOLDER)/hmm12
	mkdir $(TMP_TRAIN_FOLDER)/hmm13
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm10/macros -H $(TMP_TRAIN_FOLDER)/hmm10/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm11 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm11/macros -H $(TMP_TRAIN_FOLDER)/hmm11/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm12 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm12/macros -H $(TMP_TRAIN_FOLDER)/hmm12/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm13 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	@echo "\n--- mixtures of 5 components ---"
	HHed -H $(TMP_TRAIN_FOLDER)/hmm13/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU5.hed $(TMP_TRAIN_FOLDER)/monophones1
	mkdir $(TMP_TRAIN_FOLDER)/hmm14
	mkdir $(TMP_TRAIN_FOLDER)/hmm15
	mkdir $(TMP_TRAIN_FOLDER)/hmm16
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm13/macros -H $(TMP_TRAIN_FOLDER)/hmm13/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm14 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm14/macros -H $(TMP_TRAIN_FOLDER)/hmm14/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm15 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm15/macros -H $(TMP_TRAIN_FOLDER)/hmm15/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm16 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	@echo "\n--- mixtures of 9 components ---"
	HHed -H $(TMP_TRAIN_FOLDER)/hmm16/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU9.hed $(TMP_TRAIN_FOLDER)/monophones1
	mkdir $(TMP_TRAIN_FOLDER)/hmm17
	mkdir $(TMP_TRAIN_FOLDER)/hmm18
	mkdir $(TMP_TRAIN_FOLDER)/hmm19
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm16/macros -H $(TMP_TRAIN_FOLDER)/hmm16/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm17 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm17/macros -H $(TMP_TRAIN_FOLDER)/hmm17/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm18 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm18/macros -H $(TMP_TRAIN_FOLDER)/hmm18/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm19 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	@echo "\n--- mixtures of 17 components ---"
	HHed -H $(TMP_TRAIN_FOLDER)/hmm19/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU17.hed $(TMP_TRAIN_FOLDER)/monophones1
	mkdir $(TMP_TRAIN_FOLDER)/hmm20
	mkdir $(TMP_TRAIN_FOLDER)/hmm21
	mkdir $(TMP_TRAIN_FOLDER)/hmm22
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm19/macros -H $(TMP_TRAIN_FOLDER)/hmm19/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm20 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm20/macros -H $(TMP_TRAIN_FOLDER)/hmm20/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm21 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm21/macros -H $(TMP_TRAIN_FOLDER)/hmm21/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm22 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO

realign: train_monophones
	# TODO check the production of aligned.mlf, and TODO use it for triphones
	@echo "\n>>> re-aligning the training data\n"
	HVite -l '*' -o SWT -b sil -a -H $(TMP_TRAIN_FOLDER)/hmm8/macros -H $(TMP_TRAIN_FOLDER)/hmm8/hmmdefs -i $(TMP_TRAIN_FOLDER)/aligned.mlf -m -t 250.0 -y lab -S $(TMP_TRAIN_FOLDER)/train.scp $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/monophones1
	mkdir $(TMP_TRAIN_FOLDER)/hmm9
	mkdir $(TMP_TRAIN_FOLDER)/hmm10
	HERest -I $(TMP_TRAIN_FOLDER)/aligned.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm8/macros -H $(TMP_TRAIN_FOLDER)/hmm8/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm9 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO
	HERest -I $(TMP_TRAIN_FOLDER)/aligned.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm9/macros -H $(TMP_TRAIN_FOLDER)/hmm9/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm10 $(TMP_TRAIN_FOLDER)/monophones1 # check these -t parameters TODO

train_triphones: train_monophones
	# TODO use aligned.mlf instead of train.mlf?
	@echo "\n>>> make triphones from monophones\n"
	#HLEd -n $(TMP_TRAIN_FOLDER)/triphones1 -l '*' -i $(TMP_TRAIN_FOLDER)/wintri.mlf mktri.led $(TMP_TRAIN_FOLDER)/aligned.mlf
	#mkdir $(TMP_TRAIN_FOLDER)/hmm11
	#mkdir $(TMP_TRAIN_FOLDER)/hmm12
	#mkdir $(TMP_TRAIN_FOLDER)/hmm13
	HLEd -n $(TMP_TRAIN_FOLDER)/triphones1 -l '*' -i $(TMP_TRAIN_FOLDER)/wintri.mlf mktri.led $(TMP_TRAIN_FOLDER)/train.mlf
	mkdir $(TMP_TRAIN_FOLDER)/hmm11
	mkdir $(TMP_TRAIN_FOLDER)/hmm12
	mkdir $(TMP_TRAIN_FOLDER)/hmm13
	maketrihed $(TMP_TRAIN_FOLDER)/monophones1 $(TMP_TRAIN_FOLDER)/triphones1
	HHEd -B -H $(TMP_TRAIN_FOLDER)/hmm10/macros -H $(TMP_TRAIN_FOLDER)/hmm10/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm11 mktri.hed $(TMP_TRAIN_FOLDER)/monophones1
	HERest -B -I tmp_train/wdnet $(TMP_TRAIN_FOLDER)/wdnet # TODO create the wordnet from the grammar "gram"
	HVite -H $(TMP_TRAIN_FOLDER)/hmm11/macros -H $(TMP_TRAIN_FOLDER)/hmm11/hmmdefs -S $(dataset_test_folder)/test.scp -l '*' -i $(TMP_TRAIN_FOLDER)/recout.mlf -w $(TMP_TRAIN_FOLDER)/labels -p 0.0 -s 5.0 $(TMP_TRAIN_FOLDER)/dict 
	#HVite -w tmp_train/wdnet -H tmp_train/hmm4/hmmdefs -i tmp_train/outtrans.mlf -S ~/postdoc/datasets/TIMIT/test/test.scp -T 3 -o ST tmp_train/dict tmp_train/monophones1
	#HResults -I ~/postdoc/datasets/TIMIT/test/test.mlf tmp_train/monophones1 tmp_train/outtrans.mlf

test:
	@echo "*** testing the trained model ***"
	HVite -H $(TMP_TRAIN_FOLDER)/hmm42/macros -H $(TMP_TRAIN_FOLDER)/hmm42/hmmdefs -S $(dataset_test_folder)/test.scp -l '*' -i $(TMP_TRAIN_FOLDER)/recout.mlf -p 0.0 -s 5.0 $(TMP_TRAIN_FOLDER)/dict

clean:
	rm -rf $(TMP_TRAIN_FOLDER)
