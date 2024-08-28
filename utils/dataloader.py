subtitle_path = '../data/subtitles.txt'
import pandas as pd

def datasetProcessing(path):
    dialogue = []
    with open(path,'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        
    for line in lines: 
        dialogue.append(line.split(',')[2])
    
    for i in range(len(dialogue)):
        dialogue[i] = dialogue[i].replace('"','')
        
    script = " ".join(dialogue)
    
    return script