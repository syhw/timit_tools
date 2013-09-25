
First train an HTK HMM-GMM with "make prepare_mocha dataset=YOUR_MOCHA_PATH" 
and "make train dataset_train_folder=YOUR_MOCHA_TRAIN_PATH" (you have to have 
split MOCHA in train/test sets). Force align the train and test MLF, and then 
convert MFCCs and EMAs in numpy arrays with src/mocha_timit_to_numpy.py. Then 
train a DBN with DBN/DBN_Gaussian_mocha.timit.py, and finally test it with 
src/batch_mocha_timit.py


Lets do an example:

make prepare_mocha dataset=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/

make train dataset_train_folder=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/train
[%Corr=71.13, Acc=45.20]

make align input_scp=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/train/train.scp input_mlf=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/train/train.mlf output_mlf=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/aligned_train.mlf

make align input_scp=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/test/test.scp input_mlf=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/test/test.mlf output_mlf=/fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/aligned_test.mlf

python src/mocha_timit_to_numpy.py /fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/aligned_train.mlf

python src/mocha_timit_to_numpy.py /fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/aligned_test.mlf

cd DBN

python DBN_Gaussian_mocha_timit.py (with the right DATASET= param inside)

python ../src/batch_mocha_viterbi.py output_viterbi_mocha.mlf /fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/only_msak0/test/test.scp  ../tmp_train_msak0/hmm_final/hmmdefs --d dbn_mocha_gpu.pickle to_int_and_to_state_dicts_tuple_mocha.pickle

HResults -I /fhgfs/bootphon/scratch/gsynnaeve/MOCHA_TIMIT/test/test.mlf ../tmp_train_msak0/phones output_viterbi_mocha.mlf


Regroup phones eventually.

