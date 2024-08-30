import transformers, torch, nltk, os, sys, pathlib
import numpy as np
import pandas as pd

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'/../'))
from utils import datasetProcessing

nltk.download('punkt')
nltk.download('punkt_tab')

class ZeroShotClassifier():
    def __init__(self, theme_list):
        self.model_ckpt = 'facebook/bart-large-mnli'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.theme_list = theme_list
        self.zero_shot = self.load_model('zero-shot-classification',self.model_ckpt,self.device)

    def load_model(self,task,model,device):
        pipe = transformers.pipeline(task=task,
                                model=model,
                                device=device)
        return pipe
        
    def getThemeOutputs(self,script,theme_list,pipeline):
        
        script_sentences = nltk.sent_tokenize(script)
        # process the script in batches of 20 sentences
        batch_size = 20
        script_batches = []
        # create batches of 20 sentences
        for index in range(0, len(script_sentences), batch_size):
            sentences = " ".join(script_sentences[index:index + batch_size])
            script_batches.append(sentences)
            
        themes = {}
        # create per batch zero-shot classification scores
        outputs = pipeline(script_batches,theme_list,multi_label=True)
        for output in outputs:
            for label, score in zip(output['labels'],output['scores']):
                # append scores per batch to dictionary
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
        # return a singular score structure from averaging the scores per class 
        themes = {key : np.mean(np.array(value)) for key,value in themes.items()}
        return themes 
    
    def get_themes(self,dataset_path,save_path=None):
        # load Dataset
        script = datasetProcessing(dataset_path)
        # calculate scores per theme across the entire script
        score_dict = self.getThemeOutputs(script,self.theme_list,self.zero_shot)
        score_df = pd.DataFrame.from_dict(data=score_dict,orient='index').transpose()
        
        if save_path is not None and os.path.exists(save_path):
            score_df.to_csv(save_path,index=False)
        
        return score_df
        
        
        