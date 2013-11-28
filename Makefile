TMP_TRAIN_FOLDER=tmp_TIMIT_wo_sa

help:
	@echo -e "Usage (in order):"
	@echo -e "make prepare dataset=~/postdoc/datasets/TIMIT"
	@echo -e "make train dataset_train_folder=~/postdoc/datasets/TIMIT/train"
	@echo -e "make test dataset_test_folder=~/postdoc/datasets/TIMIT/test"


prepare_timit: wav_config src/mfcc_and_gammatones.py src/timit_to_htk_labels.py
	@echo -e "*** preparing the dataset for phones recognition ***"
	@echo -e "\n>>> produce MFCC from WAV files\n"
	python src/mfcc_and_gammatones.py --htk-mfcc --forcemfcext $(dataset)/train
	python src/mfcc_and_gammatones.py --htk-mfcc --forcemfcext $(dataset)/test
	@echo -e "\n>>> transform .phn files into .lab files (frames into nanoseconds)\n"
	python src/timit_to_htk_labels.py $(dataset)/train
	python src/timit_to_htk_labels.py $(dataset)/test
	@echo -e "\n>>> substitute phones (61 down to 39 if using timit_foldings.json) \n"
	@echo -e ">>> Here we are just putting the !ENTER and !EXIT symbols\n"
	python src/substitute_phones.py $(dataset)/train --sentences
	python src/substitute_phones.py $(dataset)/test --sentences
	@echo -e "\n>>> creates (train|test).mlf, (train|test).scp listings and labels (dicts)\n"
	python src/create_phonesMLF_list_labels.py $(dataset)/train
	python src/create_phonesMLF_list_labels.py $(dataset)/test


prepare_mocha: wav_config src/mfcc_and_gammatones.py src/timit_to_htk_labels.py
	# first split in train and test with src/train_test_folders.py
	# & see files in src/ and in mocha-timit/ for specifities of MOCHA-TIMIT
	@echo -e "*** preparing the dataset for phones recognition ***"
	@echo -e "\n>>> produce MFCC from WAV files\n"
	python src/mfcc_and_gammatones.py --htk-mfcc --forcemfcext $(dataset)/train
	python src/mfcc_and_gammatones.py --htk-mfcc --forcemfcext $(dataset)/test
	@echo -e "\n>>> transform .phn files into .lab files (frames into nanoseconds)\n"
	python mocha-timit/mocha_timit_to_htk_labels.py $(dataset)/train
	python mocha-timit/mocha_timit_to_htk_labels.py $(dataset)/test
	@echo -e "\n>>> put !ENTER and !EXIT symbols \n"
	python src/substitute_phones.py $(dataset)/train --sentences
	python src/substitute_phones.py $(dataset)/test --sentences
	@echo -e "\n>>> creates (train|test).mlf, (train|test).scp listings and labels (dicts)\n"
	python src/create_phonesMLF_list_labels.py $(dataset)/train
	python src/create_phonesMLF_list_labels.py $(dataset)/test


prepare_CSJ:
	# use src/train_test_folders.py to split in train/test sets
	@echo -e "\n>>> put !ENTER and !EXIT symbols and substitute phones\n"
	python src/substitute_phones.py $(dataset)/train --sentences CSJ_foldings.json
	python src/substitute_phones.py $(dataset)/test --sentences CSJ_foldings.json
	@echo -e "\n>>> creates (train|test).mlf, (train|test).scp listings and labels (dicts)\n"
	python src/create_phonesMLF_list_labels.py $(dataset)/train
	python src/create_phonesMLF_list_labels.py $(dataset)/test


prepare_buckeye:
	@echo -e "\n>>> remove train and test folders if any\n"
	rm -rf $(dataset)/train
	rm -rf $(dataset)/test
	@echo -e "\n>>> convert the Buckeye annotations (.phones) to .lab (HTK format) \n"
	python src/buckeye_to_htk_labels.py $(dataset)
	@echo -e "\n>>> split the dataset into a training and testing set\n"
	mkdir -p $(dataset)/train
	mkdir -p $(dataset)/test
	###python src/train_test_folders.py $(dataset) $(dataset)/../dataset_train_test --wav # <- this is a random split that samples accross all speakers
	cp -r $(dataset)/full/s0* $(dataset)/train/
	cp -r $(dataset)/full/s1* $(dataset)/train/
	cp -r $(dataset)/full/s2* $(dataset)/train/
	cp -r $(dataset)/full/s3* $(dataset)/train/
	cp -r $(dataset)/full/s40 $(dataset)/test/
	#find $(dataset)/train/ -name ._* | xargs rm
	#find $(dataset)/test/ -name ._* | xargs rm
	#rm -rf $(dataset)/train/s39/ # that's a fix because of several corrupted waves
	@echo -e "\ndataset is now $(dataset)"
	@echo -e "\n>>> split the WAV and LAB files on IVER (other interlocutor)\n"
	python src/split_lab_wav.py $(dataset)/train IVER VOCNOISE
	python src/split_lab_wav.py $(dataset)/test IVER VOCNOISE
	@echo -e "\n>>> put !ENTER and !EXIT symbols and substitute phones\n"
	python src/substitute_phones.py $(dataset)/train --sentences buckeye_foldings.json
	python src/substitute_phones.py $(dataset)/test --sentences buckeye_foldings.json
	@echo -e "\n>>> produce MFCC from WAV files\n"
	python src/mfcc_and_gammatones.py --htk-mfcc --forcemfcext $(dataset)/train
	python src/mfcc_and_gammatones.py --htk-mfcc --forcemfcext $(dataset)/test
	@echo -e "\n>>> creates (train|test).mlf, (train|test).scp listings and labels (dicts)\n"
	python src/create_phonesMLF_list_labels.py $(dataset)/train
	python src/create_phonesMLF_list_labels.py $(dataset)/test


train: train_monophones
	@echo -e "\n>>> We will only train monophones, see train_triphones make cmd otherwise\n"


train_monophones_monogauss:
	@echo -e "*** training the HMMs with HTK ***"
	@echo -e "using folder $(dataset_train_folder)"
	@echo -e "\n>>> preparing the HMMs\n"
	mkdir -p $(TMP_TRAIN_FOLDER)
	cp $(dataset_train_folder)/labels $(TMP_TRAIN_FOLDER)/monophones0
	cp $(dataset_train_folder)/train.mlf $(TMP_TRAIN_FOLDER)/
	cp $(dataset_train_folder)/train.scp $(TMP_TRAIN_FOLDER)/
	python -c "import sys;print '( < ' + ' | '.join([line.strip('\n') for line in sys.stdin]) + ' > )'" < $(TMP_TRAIN_FOLDER)/monophones0 > $(TMP_TRAIN_FOLDER)/gram
	HParse $(TMP_TRAIN_FOLDER)/gram $(TMP_TRAIN_FOLDER)/wdnet
	#HBuild $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/wdnet
	#cp proto.hmm $(TMP_TRAIN_FOLDER)/
	python src/proto_hmm.py $(TMP_TRAIN_FOLDER)/
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_simple0
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_simple1
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_simple2
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_simple3
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_final
	# -A -D -T 1 
	HCompV -f 0.0001 -m -S $(TMP_TRAIN_FOLDER)/train.scp -M $(TMP_TRAIN_FOLDER)/hmm_mono_simple0 $(TMP_TRAIN_FOLDER)/proto.hmm
	python src/create_hmmdefs_from_proto.py $(TMP_TRAIN_FOLDER)/hmm_mono_simple0/proto $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/hmm_mono_simple0/ $(TMP_TRAIN_FOLDER)/hmm_mono_simple0/vFloors
	@echo -e "\n>>> training the HMMs (3 times)\n"
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mono_simple0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mono_simple0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mono_simple1 $(TMP_TRAIN_FOLDER)/monophones0 
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mono_simple1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mono_simple1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mono_simple2 $(TMP_TRAIN_FOLDER)/monophones0 
	HERest -s $(TMP_TRAIN_FOLDER)/stats -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mono_simple2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mono_simple2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mono_simple3 $(TMP_TRAIN_FOLDER)/monophones0 
	cp $(TMP_TRAIN_FOLDER)/hmm_mono_simple3/* $(TMP_TRAIN_FOLDER)/hmm_final/
	cp $(dataset_train_folder)/dict $(TMP_TRAIN_FOLDER)/dict
	cp $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/phones


add_short_pauses: train_monophones_monogauss
	# TODO incomplete
	python src/create_short_pause_silence_model.py $(TMP_TRAIN_FOLDER)/hmm3/hmmdefs $(TMP_TRAIN_FOLDER)/hmm4/hmmdefs $(TMP_TRAIN_FOLDER)/monophones1
	#tr "\n" " | " < $(TMP_TRAIN_FOLDER)/monophones1 > $(TMP_TRAIN_FOLDER)/gram
	awk '{if(!$$2) print $$1 " " $$1}' $(TMP_TRAIN_FOLDER)/monophones1 | sort > $(TMP_TRAIN_FOLDER)/dict # TODO replace sort by a script because GNU sort sucks
	echo -e "silence sil" >> $(TMP_TRAIN_FOLDER)/dict # why?
	

tweak_silence_model: train_monophones_monogauss 
	@echo -e "\n>>> tweaking the silence model\n"
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_silence0
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_silence1
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_silence2
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mono_silence3
	cp $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs $(TMP_TRAIN_FOLDER)/hmm_mono_silence0/hmmdefs
	cp $(TMP_TRAIN_FOLDER)/hmm_final/macros $(TMP_TRAIN_FOLDER)/hmm_mono_silence0/macros
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm_mono_silence0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mono_silence0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mono_silence1 sil.hed $(TMP_TRAIN_FOLDER)/monophones0
	@echo -e "\n>>> re-training the HMMs\n"
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mono_silence1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mono_silence1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mono_silence2 $(TMP_TRAIN_FOLDER)/monophones0
	HERest -s $(TMP_TRAIN_FOLDER)/stats -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mono_silence2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mono_silence2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mono_silence3 $(TMP_TRAIN_FOLDER)/monophones0 
	cp $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/phones
	cp $(TMP_TRAIN_FOLDER)/hmm_mono_silence3/* $(TMP_TRAIN_FOLDER)/hmm_final/


train_mixtures:
	@echo -e "\n>>> estimating the number of mixtures\n"
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mix0 # we will loop on these folders as we split
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mix1
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mix2
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_mix3
	#HERest -s $(TMP_TRAIN_FOLDER)/stats -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_final/macros -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix0 $(TMP_TRAIN_FOLDER)/phones 
	cp $(TMP_TRAIN_FOLDER)/hmm_final/macros $(TMP_TRAIN_FOLDER)/hmm_mix0/macros
	cp $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs
	python src/create_mixtures_from_stats.py $(TMP_TRAIN_FOLDER)/stats
	@echo -e "\n--- mixtures of 2 components ---"
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU2.hed $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix1 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix2 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix3 $(TMP_TRAIN_FOLDER)/phones
	cp $(TMP_TRAIN_FOLDER)/hmm_mix3/* $(TMP_TRAIN_FOLDER)/hmm_mix0/
	@echo -e "\n--- mixtures of 3 components ---"
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU3.hed $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix1 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix2 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix3 $(TMP_TRAIN_FOLDER)/phones
	cp $(TMP_TRAIN_FOLDER)/hmm_mix3/* $(TMP_TRAIN_FOLDER)/hmm_mix0/
	@echo -e "\n--- mixtures of 5 components ---"
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU5.hed $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix1 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix2 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix3 $(TMP_TRAIN_FOLDER)/phones
	cp $(TMP_TRAIN_FOLDER)/hmm_mix3/* $(TMP_TRAIN_FOLDER)/hmm_mix0/
	@echo -e "\n--- mixtures of 9 components ---"
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU9.hed $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix1 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix2 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix3 $(TMP_TRAIN_FOLDER)/phones
	cp $(TMP_TRAIN_FOLDER)/hmm_mix3/* $(TMP_TRAIN_FOLDER)/hmm_mix0/
	@echo -e "\n--- mixtures of 17 components ---"
	HHEd -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs $(TMP_TRAIN_FOLDER)/TRMU17.hed $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix0/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix1 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix1/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix2 $(TMP_TRAIN_FOLDER)/phones
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_mix2/macros -H $(TMP_TRAIN_FOLDER)/hmm_mix2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_mix3 $(TMP_TRAIN_FOLDER)/phones
	cp $(TMP_TRAIN_FOLDER)/hmm_mix3/* $(TMP_TRAIN_FOLDER)/hmm_final/


train_monophones: train_monophones_monogauss tweak_silence_model train_mixtures
	@echo -e "\n>>> full training of monophones\n"


realign: tweak_silence_model
	# TODO check the production of aligned.mlf, and TODO use it for triphones
	@echo -e "\n>>> re-aligning the training data\n"
	HVite -l '*' -o SWT -b sil -a -H $(TMP_TRAIN_FOLDER)/hmm8/macros -H $(TMP_TRAIN_FOLDER)/hmm8/hmmdefs -i $(TMP_TRAIN_FOLDER)/aligned.mlf -m -t 250.0 -y lab -S $(TMP_TRAIN_FOLDER)/train.scp $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/monophones0
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm9
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm10
	HERest -I $(TMP_TRAIN_FOLDER)/aligned.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm8/macros -H $(TMP_TRAIN_FOLDER)/hmm8/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm9 $(TMP_TRAIN_FOLDER)/monophones0 
	HERest -I $(TMP_TRAIN_FOLDER)/aligned.mlf -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm9/macros -H $(TMP_TRAIN_FOLDER)/hmm9/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm10 $(TMP_TRAIN_FOLDER)/monophones0 
	cp $(TMP_TRAIN_FOLDER)/hmm9/* $(TMP_TRAIN_FOLDER)/hmm_final/


train_untied_triphones: tweak_silence_model
	# TODO use aligned.mlf instead of train.mlf?
	@echo -e "\n>>> make triphones from monophones\n"
	#HLEd -n $(TMP_TRAIN_FOLDER)/triphones1 -l '*' -i $(TMP_TRAIN_FOLDER)/wintri.mlf mktri.led $(TMP_TRAIN_FOLDER)/aligned.mlf
	HLEd -n $(TMP_TRAIN_FOLDER)/triphones0 -l '*' -i $(TMP_TRAIN_FOLDER)/wintri.mlf mktri.led $(TMP_TRAIN_FOLDER)/train.mlf
	cp $(TMP_TRAIN_FOLDER)/train.mlf $(TMP_TRAIN_FOLDER)/mono_train.mlf
	cp $(TMP_TRAIN_FOLDER)/wintri.mlf $(TMP_TRAIN_FOLDER)/train.mlf
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_simple0
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_simple1
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_simple2
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_simple3
	maketrihed $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/triphones0
	HHEd -B -H $(TMP_TRAIN_FOLDER)/hmm_final/macros -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_simple0 mktri.hed $(TMP_TRAIN_FOLDER)/monophones0
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -s $(TMP_TRAIN_FOLDER)/stats -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_tri_simple0/macros -H $(TMP_TRAIN_FOLDER)/hmm_tri_simple0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_simple1 $(TMP_TRAIN_FOLDER)/triphones0 
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -s $(TMP_TRAIN_FOLDER)/stats -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_tri_simple1/macros -H $(TMP_TRAIN_FOLDER)/hmm_tri_simple1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_simple2 $(TMP_TRAIN_FOLDER)/triphones0 
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -s $(TMP_TRAIN_FOLDER)/stats -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_tri_simple2/macros -H $(TMP_TRAIN_FOLDER)/hmm_tri_simple2/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_simple3 $(TMP_TRAIN_FOLDER)/triphones0 
	cp $(TMP_TRAIN_FOLDER)/hmm_tri_simple3/* $(TMP_TRAIN_FOLDER)/hmm_final/
	cp $(TMP_TRAIN_FOLDER)/triphones0 $(TMP_TRAIN_FOLDER)/phones


train_tied_triphones: train_untied_triphones
	@echo -e "\n>>> tying triphones\n"
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_tied0
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_tied1
	mkdir -p $(TMP_TRAIN_FOLDER)/hmm_tri_tied2
	python src/adapt_quests.py $(TMP_TRAIN_FOLDER)/monophones0 quests_example.hed $(TMP_TRAIN_FOLDER)/quests.hed
	#HDMan -n $(TMP_TRAIN_FOLDER)/fulllist -g global.ded -l flog $(TMP_TRAIN_FOLDER)/tri-dict $(TMP_TRAIN_FOLDER)/dict # this is to generate the full list of phones but we consider that we saw all triphones in the training (anyway there are the monophones)
	mkclscript TB 350.0 $(TMP_TRAIN_FOLDER)/monophones0 > $(TMP_TRAIN_FOLDER)/tb_contexts.hed
	python src/create_contexts_tying.py $(TMP_TRAIN_FOLDER)/quests.hed $(TMP_TRAIN_FOLDER)/tb_contexts.hed $(TMP_TRAIN_FOLDER)/tree.hed $(TMP_TRAIN_FOLDER)
	HHEd -B -H $(TMP_TRAIN_FOLDER)/hmm_final/macros -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_tied0 $(TMP_TRAIN_FOLDER)/tree.hed $(TMP_TRAIN_FOLDER)/triphones0 > $(TMP_TRAIN_FOLDER)/log
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -s $(TMP_TRAIN_FOLDER)/stats -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_tri_tied0/macros -H $(TMP_TRAIN_FOLDER)/hmm_tri_tied0/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_tied1 $(TMP_TRAIN_FOLDER)/tiedlist
	HERest -I $(TMP_TRAIN_FOLDER)/train.mlf -s $(TMP_TRAIN_FOLDER)/stats -S $(TMP_TRAIN_FOLDER)/train.scp -H $(TMP_TRAIN_FOLDER)/hmm_tri_tied1/macros -H $(TMP_TRAIN_FOLDER)/hmm_tri_tied1/hmmdefs -M $(TMP_TRAIN_FOLDER)/hmm_tri_tied2 $(TMP_TRAIN_FOLDER)/tiedlist 
	cp $(TMP_TRAIN_FOLDER)/hmm_tri_tied2/* $(TMP_TRAIN_FOLDER)/hmm_final/
	cp $(TMP_TRAIN_FOLDER)/tiedlist $(TMP_TRAIN_FOLDER)/phones


train_triphones: train_tied_triphones train_mixtures
	@echo -e "\n>>> Training fully tied triphones with GMM acoustic models\n"
	

bigram_LM:
	@echo -e "*** Estimating a bigram language model (only with !ENTER & !EXIT) ***"
	# cp $(dataset_train_folder)/train.mlf $(TMP_TRAIN_FOLDER)/train.mlf
	HLStats -o -b $(TMP_TRAIN_FOLDER)/bigram $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/train.mlf
	HLStats -b $(TMP_TRAIN_FOLDER)/bigram2 $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/train.mlf
	HBuild -n $(TMP_TRAIN_FOLDER)/bigram $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/wdnetbigram
	HBuild -m $(TMP_TRAIN_FOLDER)/bigram2 $(TMP_TRAIN_FOLDER)/monophones0 $(TMP_TRAIN_FOLDER)/wdnetbigram2


test_monophones:
	@echo -e "*** testing the monophone trained model ***"
	HVite -p 2.5 -s 5.0 -w $(TMP_TRAIN_FOLDER)/wdnet -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones
	#HVite -w $(TMP_TRAIN_FOLDER)/wdnet -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones
	HResults -I $(dataset_test_folder)/test.mlf $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/outtrans.mlf


test_monophones_bigram_LM:
	@echo -e "*** testing the monophone trained model (with a bigram LM) ***"
	HVite -p 2.5 -s 5.0 -w $(TMP_TRAIN_FOLDER)/wdnetbigram -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones
	#HVite -w $(TMP_TRAIN_FOLDER)/wdnetbigram -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones
	# -n 10 10 for the 10-bests lattice
	HResults -I $(dataset_test_folder)/test.mlf $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/outtrans.mlf


results_TIMIT_equiv:
	@echo -e "*** interpreting the results with TIMIT classic equivalences ***"
	HResults -e n en -e aa ao -e ah ax-h -e ah ax -e ih ix -e l el -e sh zh -e uw ux -e er axr -e m em -e n nx -e ng eng -e hh hv -e pau pcl -e pau tcl -e pau kcl -e pau q -e pau bcl -e pau dcl -e pau gcl -e pau epi -e pau sil -e pau !ENTER -e pau !EXIT -I $(dataset_test_folder)/test.mlf $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/outtrans.mlf


test_triphones:
	@echo -e "*** testing the triphone trained model ***"
	##HBuild $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/wdnet
	#HLStats -b $(TMP_TRAIN_FOLDER)/bigram  -s "<s>" "</s>" $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/train.mlf
	#HBuild -m $(TMP_TRAIN_FOLDER)/bigram -s "<s>" "</s>" $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/wdnet
	#HVite -w $(TMP_TRAIN_FOLDER)/wdnet -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/tri-dict $(TMP_TRAIN_FOLDER)/phones
	#HVite -w $(TMP_TRAIN_FOLDER)/wdnet -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones
	HVite -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(TMP_TRAIN_FOLDER)/outtrans.mlf -S $(dataset_test_folder)/test.scp -o ST $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones
	HResults -I $(dataset_test_folder)/test.mlf $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/outtrans.mlf


reco_align:
	@echo -e "*** aligning the content of input_scp in output_mlf ***"
	@echo -e ">>> you need to have trained a (monophone) model with sentences start & end."
	@# TODO not only monophones
	@echo -e ">>> Using: $(input_scp) , going to $(output_mlf)"
	HVite -l $(TMP_TRAIN_FOLDER) -a -m -y lab -w $(TMP_TRAIN_FOLDER)/wdnet -H $(TMP_TRAIN_FOLDER)/hmm_final/macros -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(output_mlf) -S $(input_scp) $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones # -f if you want the full states alignment


align:
	@echo -e "*** aligning the content of input_scp in output_mlf ***"
	@echo -e ">>> you need to have trained a (monophone) model with sentences start & end."
	@# TODO not only monophones
	@echo -e ">>> Using: $(input_scp) and $(input_mlf), going to $(output_mlf)"
	HVite -a -f -y lab -H $(TMP_TRAIN_FOLDER)/hmm_final/macros -H $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs -i $(output_mlf) -I $(input_mlf) -S $(input_scp) $(TMP_TRAIN_FOLDER)/dict $(TMP_TRAIN_FOLDER)/phones 
	# -f if you want the full states alignment, -o C for likelihoods by phone, see p.326 in the HTK book
	

train_test_monophones:
	make train_monophones dataset_train_folder=$(dataset)/train
	make test_monophones dataset_test_folder=$(dataset)/test


all_timit:
	make prepare_timit $(dataset)
	make train_monophones dataset_train_folder=$(dataset)/train
	make bigram_LM
	make test_monophones_bigram_LM dataset_test_folder=$(dataset)/test


all_buckeye:
	@echo -e "\n>>> You need a $(dataset)/full folder with the full Buckeye corpus\n"
	make prepare_buckeye $(dataset)
	make train_monophones dataset_train_folder=$(dataset)/train
	make bigram_LM
	make test_monophones_bigram_LM dataset_test_folder=$(dataset)/test
	make align input_scp=$(dataset)/train/train.scp input_mlf=$(dataset)/train/train.mlf output_mlf=$(dataset)/aligned_train.mlf
		make align input_scp=$(dataset)/test/test.scp input_mlf=$(dataset)/test/test.mlf output_mlf=$(dataset)/aligned_test.mlf


all_CSJ:
	make prepare_CSJ $(dataset)
	make train_monophones dataset_train_folder=$(dataset)/train
	make bigram_LM
	make test_monophones_bigram_LM dataset_test_folder=$(dataset)/test
	make align input_scp=$(dataset)/train/train.scp input_mlf=$(dataset)/train/train.mlf output_mlf=$(dataset)/aligned_train.mlf
		make align input_scp=$(dataset)/test/test.scp input_mlf=$(dataset)/test/test.mlf output_mlf=$(dataset)/aligned_test.mlf


test_my_bigram:
	python src/viterbi.py $(TMP_TRAIN_FOLDER)/my_viterbi.mlf ~/datasets/TIMIT/test/test.scp $(TMP_TRAIN_FOLDER)/hmm_final/hmmdefs --ub $(TMP_TRAIN_FOLDER)/bigram.pickle
	HResults -I $(dataset_test_folder)/test.mlf $(TMP_TRAIN_FOLDER)/phones $(TMP_TRAIN_FOLDER)/my_viterbi.mlf


clean:
	rm -rf $(TMP_TRAIN_FOLDER)
