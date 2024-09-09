import os
import numpy as np
import pandas as pd
from glob import glob

def load_dataset(path):
    scripts = []
    episodes_num = []
    clone_wars = glob(path + 'CloneWarsS1/*.srt')
    movies = glob(path + '/*.txt')
    files = list(np.append(clone_wars,movies))
    for path in files:
        dialogue = []
        
        with open(path, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                line = lines[i].replace('\n','').replace('</i>','').replace('<i>','')
                if (line.isnumeric() == False) and (('-->' in line) == False) and (line != ''):
                    
                    dialogue.append(line)
                    
        script = " ".join(dialogue)
        scripts.append(script)
        
        if 'CloneWarsS1' in path:
            episodes_num.append(path.split('CloneWarsS1/')[1][:6])
        else:
            episodes_num.append('Star Wars ' + path[29:37].capitalize())
            
    df = pd.DataFrame.from_dict({"episode": episodes_num, "script": scripts})
    return df

def reconstructEntities(docs):
    reconstructed_entities = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    current_entity = ''
    current_type = None
    previous_end = None
    
    if docs != None:
        for entity in docs:
            word = entity['word']
            entity_tag = entity['entity']
            start = entity['start']
            end = entity['end']
            
            # Determine the type of entity (PER, LOC, ORG, MISC)
            entity_type = entity_tag.split('-')[-1]
            
            if entity_tag.startswith('B-') or (entity_tag.startswith('I-') and (previous_end is None or start > previous_end + 3)):
                # Finalize the previous entity if needed
                if current_entity and current_type:
                    if current_entity in ['Ahka','Soka','Ahka Tano']:
                        current_entity = 'Ahsoka'
                    elif current_entity in ['Skywalk','Skywalker','Ana','Anakin','An']:
                        current_entity = 'Anakin Skywalker'
                    elif current_entity in ['Grievous','Gvous','vous']:
                        current_entity = 'General Grievous'
                    elif current_entity in ['Obi - Wan','Obi','Wan','Kenobi','Obi - Wan Kenobi','Ben','Ben Kenobi']:
                        current_entity = 'Obi-Wan Kenobi'
                    elif current_entity in ['Plo']:
                        current_entity = 'Plo Koon'
                    elif current_entity in ['dooku','tyranus','darth tyranus','Doo','Dooku']:
                        current_entity = 'Count Dooku'
                    elif current_entity in ['Thi - Sen','Thi Sen']:
                        current_entity = 'Thi-Sen'
                    elif current_entity in ['Jar Jar','Binks','J Jar']:
                        current_entity = 'Jar Jar Binks'
                    elif current_entity in ['Hondo','Ohnaka']:
                        current_entity = 'Hondo Ohnaka'
                    elif current_entity in ['Vindi']:
                        current_entity = 'Nuvo Vindi'
                    elif current_entity in ['Pame','Amidala','Pa']:
                        current_entity = 'Padme Amidala'
                    elif current_entity in ['Paltine','Chancellor','Sidious','Darth Sidious']:
                        current_entity = 'Palpatine'
                    elif current_entity in ['Bane']:
                        current_entity = 'Cad Bane'
                    elif current_entity in ['Free Taa','Or Free Taa']:
                        current_entity = 'Orn Free Taa'
                    elif current_entity in ['Organa', 'Senator Organa']:
                        current_entity = 'Bail Organa'
                    elif current_entity in ['He']:
                        current_entity = 'Heavy'
                    elif current_entity in ['Windu','Master Windu']:
                        current_entity = 'Mace Windu'
                        
                    elif current_entity in ['Gunray']:
                        current_entity = 'Nute Gunray'
                    elif current_entity in ['Master Fisto','Fisto']:
                        current_entity = 'Kit Fisto'
                    elif current_entity in ['Nahdar']:
                        current_entity = 'Nahdar Vebb'
                    elif current_entity in ['Aayla','Secura']:
                        current_entity = 'Aayla Secura'
                    elif current_entity in ['Qui - Gon']:
                        current_entity = 'Qui Gon Jinn'
                    elif current_entity in ['Jango','Fett']:
                        current_entity = 'Jango Fett'
                    elif current_entity in ['Lord Vader','Vader']:
                        current_entity = 'Darth Vader'
                    elif current_entity in ['Luke']:
                        current_entity = 'Luke Skywalker'
                    elif current_entity in ['Solo','Han','Sol']:
                        current_entity = 'Han Solo'
                    elif current_entity in ['Land','Lando']:
                        current_entity = 'Lando Calrissian'
                    elif current_entity in ['Chew','Chewie']:
                        current_entity = 'Chewbacca'
                    elif current_entity in ['Yo']:
                        current_entity = 'Yoda'
                    elif current_entity in ['Le']:
                        current_entity = 'Leia'
                    elif current_entity in ['Bob Fe']:
                        current_entity = 'Boba Fett'
                    elif current_entity in ['Sep','Septist']:
                        current_entity = 'Separatist'
                    elif current_entity in ['Federation']:
                        current_entity = 'Trade Federation'
                    elif current_entity in ['Jab']:
                        current_entity = 'Jabba'
                    elif current_entity in ['Jed']:
                        current_entity = 'Jedi'
                    
                        
                    reconstructed_entities[current_type].append(current_entity)
                
                # Start a new entity
                current_entity = word if not word.startswith('##') else word[2:]
                current_type = entity_type
            
            elif entity_tag.startswith('I-') and current_type == entity_type:
                if word.startswith('##'):
                    # Continue the current entity with a subword
                    current_entity += word[2:]
                else:
                    # Continue the current entity with a new word
                    current_entity += f" {word}"
            
            # Update the previous end position
            previous_end = end
        
        # Finalize any entity left at the end of the sentence
        if current_entity and current_type:
            if current_entity in ['Ahka','Soka','Ahka Tano']:
                current_entity = 'Ahsoka'
            elif current_entity in ['Skywalk','Skywalker','Ana','Anakin','An','Sky', 'Ani']:
                current_entity = 'Anakin Skywalker'
            elif current_entity in ['Grievous','Gvous','vous']:
                current_entity = 'General Grievous'
            elif current_entity in ['Obi - Wan','Obi','Wan','Kenobi','Obi - Wan Kenobi','Ben','Ben Kenobi','Obi Wan']:
                current_entity = 'Obi-Wan Kenobi'
            elif current_entity in ['Plo']:
                current_entity = 'Plo Koon'
            elif current_entity in ['dooku','tyranus','darth tyranus','Doo','Dooku']:
                current_entity = 'Count Dooku'
            elif current_entity in ['Thi - Sen','Thi Sen']:
                current_entity = 'Thi-Sen'
            elif current_entity in ['Jar Jar','Binks','J Jar']:
                current_entity = 'Jar Jar Binks'
            elif current_entity in ['Hondo','Ohnaka']:
                current_entity = 'Hondo Ohnaka'
            elif current_entity in ['Vindi']:
                current_entity = 'Nuvo Vindi'
            elif current_entity in ['Pame','Amidala','Pa','Padme','Pad']:
                current_entity = 'Padme Amidala'
            elif current_entity in ['Paltine','Chancellor','Sidious','Darth Sidious']:
                current_entity = 'Palpatine'
            elif current_entity in ['Bane']:
                current_entity = 'Cad Bane'
            elif current_entity in ['Free Taa','Or Free Taa']:
                current_entity = 'Orn Free Taa'
            elif current_entity in ['Organa', 'Senator Organa']:
                current_entity = 'Bail Organa'
            elif current_entity in ['He']:
                current_entity = 'Heavy'
            elif current_entity in ['Windu','Master Windu']:
                current_entity = 'Mace Windu'
            elif current_entity in ['Gunray']:
                current_entity = 'Nute Gunray'
            elif current_entity in ['Master Fisto','Fisto']:
                current_entity = 'Kit Fisto'
            elif current_entity in ['Nahdar']:
                current_entity = 'Nahdar Vebb'
            elif current_entity in ['Aayla','Secura']:
                current_entity = 'Aayla Secura'
            elif current_entity in ['Qui - Gon']:
                current_entity = 'Qui Gon Jinn'
            elif current_entity in ['Jango','Fett']:
                current_entity = 'Jango Fett'
            elif current_entity in ['Lord Vader','Vader']:
                current_entity = 'Darth Vader'
            elif current_entity in ['Luke']:
                current_entity = 'Luke Skywalker'
            elif current_entity in ['Solo','Han','Sol']:
                current_entity = 'Han Solo'
            elif current_entity in ['Land','Lando']:
                current_entity = 'Lando Calrissian'
            elif current_entity in ['Chew','Chewie']:
                current_entity = 'Chewbacca'
            elif current_entity in ['Yo']:
                current_entity = 'Yoda'
            elif current_entity in ['Le']:
                current_entity = 'Leia'
            elif current_entity in ['Bob Fe']:
                current_entity = 'Boba Fett'
            elif current_entity in ['Sep','Septist']:
                current_entity = 'Separatist'
            elif current_entity in ['Federation']:
                current_entity = 'Trade Federation'
            elif current_entity in ['Jab']:
                current_entity = 'Jabba'
            elif current_entity in ['Jed']:
                current_entity = 'Jedi'
            
            reconstructed_entities[current_type].append(current_entity)
    # Output the reconstructed entities
    return reconstructed_entities
    