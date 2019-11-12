#Program which evaluates semeval 2019 task 6 output
#Example run: python3 evaluate.py Kim-CNN_random_out english/agr_en_dev.csv

import sys
import csv

def evaluate(outputFile, goldFile):
    output = {}
    gold = {}
    total = 0.0
    correct = 0.0
    classCor = {}
    classAtt = {}
    classTot = {}
    
    #Read in/store output 
    with open(outputFile, 'r') as csvfile:
        outputreader = csv.reader(csvfile, delimiter=',')
        for curOut in outputreader:
            output[curOut[0].strip()] = curOut[1].strip()
            if(curOut[1].strip() not in classCor):
                classCor[curOut[1].strip()] = 0.0
                classAtt[curOut[1].strip()] = 0.0
                classTot[curOut[1].strip()] = 0.0

    #Read in/store gold keys
    with open(goldFile, 'r') as csvfile:
        goldreader = csv.reader(csvfile, delimiter='\t')
        for curGold in goldreader:
            gold[curGold[0].strip()] = curGold[4].strip()
            if(curGold[4].strip() not in classCor):
                classCor[curGold[4].strip()] = 0.0
                classAtt[curGold[4].strip()] = 0.0
                classTot[curGold[4].strip()] = 0.0


    #Compares each output to the corresponding gold standard, noting each correct
    for curOut in output:
        total += 1
        classTot[gold[curOut]] += 1
        classAtt[output[curOut]] += 1
    
        if(output[curOut] == gold[curOut]):
            correct += 1
            classCor[output[curOut]] += 1
            
    #calculate macro precision and recall, then macro f1 score
    macroPrList = []
    macroReList = [] 
    macroF1List = []
    for c in classCor:
        curMacroRe = classCor[c]/classTot[c]
        curMacroPr = classCor[c]/classAtt[c]
        curMacroF1 = 2 * (curMacroPr * curMacroRe)/(curMacroPr + curMacroRe)
        macroPrList.append(curMacroPr)
        macroReList.append(curMacroRe)
        macroF1List.append(curMacroF1)

    macroPr = sum(macroPrList)/len(macroPrList)
    macroRe = sum(macroReList)/len(macroReList)
    macroF1 = sum(macroF1List)/len(macroF1List)    

    print("Total correct for", outputFile, "-\n Accuracy =", correct/total, "\n Macro Precision =", macroPr, "\n Macro Recall =", macroRe, "\n Macro F1=", macroF1  )



evaluate(sys.argv[1], sys.argv[2])
