#coding=utf-8

import json
import pandas as pd
from glob import glob

FileName = glob("predict_few_music*")
results = {}
for fileName in FileName:
    # items = fileName.split("_")
    # corpus = items[3].split(".")[0]
    # model = items[1]
    items = fileName.split("_")[2:]
    corpus = items[0]
    model = items[1].split(".")[0]
    #o_predict_*
    # items = fileName.split("_")[2:]
    # corpus = items[0]
    # model = items[1].split(".")[0]
    if corpus not in results:
        results[corpus] = {model:{}}
    else:
        results[corpus].update({model:{}})
    with open(fileName,"r",encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if "EAVE" in line:
                items = line.split()
                _index = items.index("EAVE")
                items = items[_index:]
                PPL = items[-1]
                BLEU1 = items[-11]
                BLEU2 = items[-9]
                BLEU3 = items[-7]
                BLEU4 = items[-5]
                RL = items[-3]
                dist1 = items[-19]
                dist2 = items[-17]
                EAVE = items[1]
                EGRE = items[3]
                EEXT = items[5]
                results[corpus][model]={"PPL":PPL,
                                        "BLEU1":BLEU1,
                                        "BLEU2":BLEU2,
                                        "BLEU3":BLEU3,
                                        "BLEU4":BLEU4,
                                        "Rouge-L":RL,
                                        "dist1":dist1,
                                        "dist2":dist2,
                                        "EAVE":EAVE,
                                        "EGRE":EGRE,
                                        "EEXT":EEXT}

for cor in results:
    models, ppl,b1,b2,b3,b4,rl,d1,d2,ea,eg,ee=[],[],[],[],[],[],[],[],[],[],[],[]
    for model in results[cor]:
        print(cor)
        print(model)
        models.append(model)
        values = results[cor][model]
        ppl.append(values["PPL"])
        b1.append(values["BLEU1"])
        b2.append(values["BLEU2"])
        b3.append(values["BLEU3"])
        b4.append(values["BLEU4"])
        rl.append(values["Rouge-L"])
        d1.append(values["dist1"])
        d2.append(values["dist2"])
        ea.append(values["EAVE"])
        eg.append(values["EGRE"])
        ee.append(values["EEXT"])
    dfData = {
        "model":models,
        "PPL":ppl,
        "BLEU1": b1,
        "BLEU2":b2,
        "BLEU3":b3,
        "BLEU4":b4,
        "Rouge-L":rl,
        "dist1":d1,
        "dist2":d2,
        "EAVE":ea,
        "EGRE":eg,
        "EEXT":ee}
    df = pd.DataFrame(dfData)
    df.to_excel("%s.xlsx"%(cor), index=False)
            