import pandas as pd
import numpy as np
import ABXpy.database.database as database
import glob


def threshold_item(db, item_file, upper_threshod, seed=0,
        columns=['phone', 'context', 'talker']):
    np.random.seed(0)
    f = item_file[:-5] + '_upper_threshold_%d' % upper_threshold + '.item'    
    with open(f, 'w') as out:
        out.write('#file onset offset')
        out.write('#' + ' '.join(columns) + '\n')
        for group, df in db.groupby(columns):
            df = df.reindex(np.random.permutation(df.index)) # shuffle dataframe
            m = min(upper_threshold, len(df))
            df = df.iloc[:m]            
            for i in range(m):
                out.write(' '.join([str(e) for e in df.iloc[i]]) + '\n')


for item_file in glob.iglob("*.item"):
    print item_file
    with open(item_file, 'r') as inf:
        header = inf.readline()
    columns = header.split('#')[-1].split()
    print columns
    thresholds = [20]
    db, _, feat_db = database.load(item_file, features_info=True)
    db = pd.concat([feat_db, db], axis=1)
    #print db
    for upper_threshold in thresholds:
        threshold_item(db, item_file, upper_threshold, columns=columns)
