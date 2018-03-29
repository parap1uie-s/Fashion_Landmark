import pandas as pd

fileHandle = pd.read_csv("result.csv")
saveHandle = open("result_new.csv",'w+',encoding='utf-8')
saveHandle.write( ','.join(fileHandle.columns)+"\n")
for row in fileHandle.iterrows():
    r = row[1]
    fileName = r['image_id']
    split_axis = r.ix[2:].str.split(pat='_', expand=True)
    split_axis.columns = ['x', 'y', 'vis']
    split_axis.loc[split_axis['vis'].astype('float32') > -1,'vis'] = 1
    split_axis.loc[split_axis['vis'].astype('float32') < 0,'vis'] = -1
    split_axis.loc[split_axis['vis'].astype('float32') > 0,'x'] = round(split_axis.loc[split_axis['vis'].astype('float32') > 0,'x'].astype('float32'))
    split_axis.loc[split_axis['vis'].astype('float32') > 0,'y'] = round(split_axis.loc[split_axis['vis'].astype('float32') > 0,'y'].astype('float32'))

    try:
        res = [fileName] + [r['image_category']] + ['_'.join(i) for i in split_axis.astype('int').astype('str').values.tolist()]
    except Exception as e:
        print(split_axis)
        raise
    
    saveHandle.write(','.join(res) + "\n")
    saveHandle.flush()
saveHandle.close()
