#coding=utf-8
import json
from jieba import cut
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder,QuadgramCollocationFinder
from nltk.probability import FreqDist
from tqdm import tqdm


dictmedical = {"uni":[],"bi":[],"tri":[],"qua":[]}
dicttravel = {"uni":[],"bi":[],"tri":[],"qua":[]}
dictmusic = {"uni":[],"bi":[],"tri":[],"qua":[]}
dictecomm = {"uni":[],"bi":[],"tri":[],"qua":[]}

def loadFile(fileName):
    documents = []
    with open(fileName,"r",encoding="utf-8") as f:
        for line in f:
            tokens = []
            line = line.rstrip().replace("[MASK]","#")
            for token in line:
                if len(token.strip())>0:
                    tokens.append(token)
            documents.append(tokens)
    return documents

def uniGram(documents):
    results =set()
    for line in documents:
        line = " ".join(line).replace("#","")
        line = line.split()
        line = [token for token in line if len(token.strip())>0]
        for token in line:
            results.add(token)
    return list(results)

def biGram(documents):
    results =set()
    for line in documents:
        length = len(line)
        if length<2:
            continue
        for index in range(length-1):
            bi_elem = line[index:index+2]
            results.add(" ".join(bi_elem))
    return list(results)
def triGram(documents):
    results =set()
    for line in documents:
        length = len(line)
        if length<3:
            continue
        for index in range(length-2):
            bi_elem = line[index:index+3]
            results.add(" ".join(bi_elem))
    return list(results)
def quaGram(documents):
    results =set()
    for line in documents:
        length = len(line)
        if length<4:
            continue
        for index in range(length-3):
            bi_elem = line[index:index+4]
            results.add(" ".join(bi_elem))
    return list(results)

filmDocuments = loadFile("datasets/film/train.txt")
dictFilm = {"uni":uniGram(filmDocuments),
            "bi":biGram(filmDocuments),
            "tri":triGram(filmDocuments),
            "qua":quaGram(filmDocuments)}
musicDocuments = loadFile("datasets/music/train.txt")
dictMusic = {"uni":uniGram(musicDocuments),
            "bi":biGram(musicDocuments),
            "tri":triGram(musicDocuments),
            "qua":quaGram(musicDocuments)}

traveDocuments = loadFile("datasets/travel/train.txt")
dictTravel = {"uni":uniGram(traveDocuments),
            "bi":biGram(traveDocuments),
            "tri":triGram(traveDocuments),
            "qua":quaGram(traveDocuments)}
ecommDocuments = loadFile("datasets/ecomm/train.txt")
dictEcomm = {"uni":uniGram(ecommDocuments),
            "bi":biGram(ecommDocuments),
            "tri":triGram(ecommDocuments),
            "qua":quaGram(ecommDocuments)}
medicalDocuments = loadFile("datasets/medical/train.txt")
dictMedical = {"uni":uniGram(medicalDocuments),
            "bi":biGram(medicalDocuments),
            "tri":triGram(medicalDocuments),
            "qua":quaGram(medicalDocuments)}
finalResults = []
DomainNames = ["Film","Music","Travel","Ecomm","Medical"]
for oname,domainOuter in tqdm(zip(DomainNames,[dictFilm,dictMusic,dictTravel,dictEcomm,dictMedical])):
    for iname,domainInner in zip(DomainNames,[dictFilm,dictMusic,dictTravel,dictEcomm,dictMedical]):
        for keyName in ["uni"]:#"bi","tri","qua"
            O_grams = domainOuter[keyName]
            I_grams = domainInner[keyName]
            if oname==iname:
                finalResults.append({"%s-%s-%s"%(oname,iname,keyName):0})
            else:
                counter = 0
                for phrase in O_grams:
                    if phrase in I_grams:
                        counter +=1
                finalResults.append({"%s-%s-%s"%(oname,iname,keyName):counter*1.0/len(O_grams)})

with open("sim_score.json","w",encoding="utf-8") as f:
    json.dump(finalResults,f,ensure_ascii=False,indent=4)
# print(finalResults)

