## Preparing the dataset

With the TIMIT dataset (.wav sound files, .wrd words annotations and .phn 
phones annotations):

 1. Encode the wave sound in MFCCs:
run `python mfcc_and_gammatones.py --htk-mfcc $DATASET/train` and
`python mfcc_and_gammatones.py --htk-mfcc $DATASET/test` producing the `.mfc` 
files with HCopy according to `wav_config` (`.mfc_unnorm` is no normalization)

 2. Adapt the annotations given in .phn in frames into nanoseconds in .lab
run `python timit_to_htk_labels.py $DATASET/train` and  
`python timit_to_htk_labels.py $DATASET/test` producing the `.lab` files

 3. Replace phones according to the seminal HMM paper of 1989:
"Speaker-independant phone recognition using hidden Markov models", phones 
number (i.e. number of lines in the future labels dictionary) should go from 
61 to 39.
run `python substitute_phones.py $DATASET/train` and 
`python substitute_phones.py $DATASET/test`

 4. run `python create_phonesMLF_and_labels.py $DATASET/train` and 
`python create_phonesMLF_and_labels.py $DATASET/test`

You can also do that with a `make prepare dataset=DATASET_PATH`.

You're ready for training with HTK (mfc and lab files)!

## Training the HMM models

Train monophones HMM:

    make train_monophones dataset_train_folder=PATH_TO_YOUR_DATASET/train
    make test_monophones dataset_test_folder=PATH_TO_YOUR_DATASET/test

Or, train triphones:

    TODO
    make train_triphones dataset_train_folder=PATH_TO_YOUR_DATASET/train
    make test_triphones dataset_test_folder=PATH_TO_YOUR_DATASET/test

## Replacing the GMM by DBNs

TODO
