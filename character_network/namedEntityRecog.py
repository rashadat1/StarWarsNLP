import numpy as np
import pandas as pd
import nltk, torch, os, sys, pathlib
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
# file path for this script
folder_path = pathlib.Path(__file__).parent.resolve()

sys.path.append(os.path.join(folder_path,'/../'))
from utils import load_dataset, reconstructEntities

class NamedEntityRecognition:
    
    def __init__(self):
        self.model_ckpt = 'dslim/bert-large-NER'
        self.task = 'ner'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.ner = self.load_model(task=self.task, model_ckpt=self.model_ckpt, device=self.device)
        
    def load_model(self,task,model_ckpt,device):
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForTokenClassification.from_pretrained(model_ckpt)
        
        task_pipeline = pipeline(task=task,model=model,tokenizer=tokenizer,device=device)
        return task_pipeline
    
    def nerInference(self,script):
        script_sentences = nltk.sent_tokenize(script)
    
        ner_output = []
        for sentence in script_sentences:
            docs = self.ner(sentence)
            ners = set()
            output = reconstructEntities(docs=docs)
            for entity_type, entities in output.items():
                ners.update(entities)
            ner_output.append(ners)
        return ner_output
    
    def generate_characterNetwork(self,df):
        window_size = 10
        entity_relationship = []
        # iterate over each script
        for row in df['ners']:
            previous_entities_in_window = []
            # iterate over the named entities in each sentence 
            for sentence in row:
                # append list of named entities from the current sentence 
                # (now the last element of previous entities in window) and only keep the sentences
                # in the last 'window size' sentences
                previous_entities_in_window.append(list(sentence))
                previous_entities_in_window = previous_entities_in_window[-window_size:]
                # flattens the list of lists to a single list of entries 
                previous_entities_flattened = sum(previous_entities_in_window, [])
                # loops through the entities current sentence 
                for entity in sentence:
                    if len(entity) > 2:
                        for entity_in_window in previous_entities_flattened:
                            if len(entity_in_window) > 2:
                                # for every entity in the current sentence we append the pair
                                # entity, entity_in_window for all entities not equal to entity in window
                                if entity != entity_in_window:
                                    # sort so A,B pairs are the same as B,A oaurs
                                    entity_relationship.append(sorted([entity, entity_in_window]))
        
        relationship_df = pd.DataFrame({'value': entity_relationship})
        # relationship_df initially has value column which is just pairs [entity, entity_in_window]
        # lambda function creates new columns 'source' for entity and 'target' for entity_in_window
        relationship_df['source'] = relationship_df['value'].apply(lambda x: x[0])
        relationship_df['target'] = relationship_df['value'].apply(lambda x: x[1])
        # groupby groups the dataframe by the unique pairs of source, targets
        # the count after this replaces the value column initially containing the pairs
        # to a count of how many times each originally appeared in the data
        relationship_df = relationship_df.groupby(['source','target']).count().reset_index()
        relationship_df = relationship_df.sort_values('value',ascending=False)
        
        return relationship_df

    def generate_nerOutput(self,dataset_path,save_path=None):
        # load Dataset
        df = load_dataset(dataset_path)
        # perform named entity recognition
        df['ners'] = df['script'].apply(self.nerInference)
        relationship_df = self.generate_characterNetwork(df)
        # truncate relationship df to only get most important entities
        relationship_df = relationship_df.head(200)
        if save_path is not None:
            relationship_df.to_csv(save_path,index=False)
        return relationship_df
    
    def draw_Network_Graph(self,relationship_df):
        G = nx.from_pandas_edgelist(
            relationship_df,
            source='source',
            target='target',
            edge_attr='value',
            create_using=nx.Graph()
        )
        node_degree = dict(G.degree())
        net = Network(notebook=True, width = '1000px', height = '700px', bgcolor='#222222', font_color='white', cdn_resources='remote')

        nx.set_node_attributes(G, node_degree, 'size')
        net.from_nx(G)
        html = net.generate_html()
        html = html.replace("'","\"")
        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; display-capture; encrypted-media;" sandbox="allow-modals allow-forms allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        return output_html                