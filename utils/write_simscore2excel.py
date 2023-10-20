#coding=utf-8

import json
import pandas as pd

sim_score = "sim_score.log"

columnName,uni,bi,tri,qua = [],[],[],[],[]
results = {}
with open(sim_score,"r",encoding="utf-8") as f:
    datasets = json.load(f)
    for item in datasets:
        for _k,_v in item.items():
            fragements = _k.split("-")
            name = fragements[1]+"2"+fragements[0]
            if name not in results:
                results[name]={fragements[2]:_v}
            else:
                results[name].update({fragements[2]:_v})

for name in results:
    columnName.append(name)
    values = results[name]
    uni.append(values["uni"])
    bi.append(values["bi"])
    tri.append(values["tri"])
    qua.append(values["qua"])
dfData = {
        'domain': columnName,
        'Uni': uni,
        "Bi":bi,
        "Tri":tri,
        "Quad":qua}
df = pd.DataFrame(dfData)
df.to_excel("sim_score.xlsx", index=False)


