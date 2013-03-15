With the TIMIT dataset:
 1) run `python mfcc_and_gammatones.py --htk-mfcc $DATASET/train` and
`python mfcc_and_gammatones.py --htk-mfcc $DATASET/test` producing the `.mfc` 
files with HCopy according to `wav_config` (`.mfc_unnorm` is no normalization)
 2) run `python timit_to_htk_labels.py $DATASET/train` and  
`python timit_to_htk_labels.py $DATASET/test` producing the `.lab` files
