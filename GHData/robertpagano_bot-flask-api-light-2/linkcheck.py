# %%
import pandas as pd
import numpy as np
import docxpy

def flag_private_urls(doc):
    '''This function takes a word document name as an input, identifies hyperlinks and their urls that do not fit the criteria of being in a newsletter. The function returns a table with columns 'Link_Text','Likely_Personal_URL?', and 'URL' with all the URLs within the document, sorted by the likeliness of being a personal link.'''

    #read in document
    #file = f'{doc}.docx'
    #create DOCReader object
    doc = docxpy.DOCReader(doc)
    #process file
    doc.process() 
    #extract hyperlinks
    hyperlinks = doc.data['links']
    
    # create DataFrame using hyperlinks object 
    df = pd.DataFrame(hyperlinks, columns =['Link_Text', 'URL'])
    #the 'text' column is byte type, convert to string type
    df['Link_Text'] = df['Link_Text'].str.decode("utf-8")
    
    #initiate a list of words to filter the dataframe, and return the result. This list will soon grow once we are able to obtain more examples of private urls.
    words_to_filter = ["personal",':p:/t'] 
    df['Likely_Private_URL?'] = df.iloc[:, 1].str.contains(r'\b(?:{})\b'.format('|'.join(words_to_filter)))
    df = df.sort_values(by='Likely_Private_URL?',ascending=False)
    result = df[['Link_Text','Likely_Private_URL?','URL']]
    result = df.loc[df['Likely_Private_URL?'] == True]
    result = result.reset_index()
    return result

def flag_private_urls_to_dict(filename):
    '''This function takes a word document name as an input, identifies hyperlinks and their urls that do not fit the criteria of being in a newsletter. The function returns a table with columns 'Link_Text','Likely_Personal_URL?', and 'URL' with all the URLs within the document, sorted by the likeliness of being a personal link.'''

    #read in document
    #file = f'{doc}.docx'
    #create DOCReader object
    doc = docxpy.DOCReader(filename)
    #process file
    doc.process() 
    #extract hyperlinks
    hyperlinks = doc.data['links']
    
    # create DataFrame using hyperlinks object 
    df = pd.DataFrame(hyperlinks, columns =['Link_Text', 'URL'])
    #the 'text' column is byte type, convert to string type
    df['Link_Text'] = df['Link_Text'].str.decode("utf-8")
    
    #initiate a list of words to filter the dataframe, and return the result. This list will soon grow once we are able to obtain more examples of private urls.
    words_to_filter = ["personal",':p:/t']
    df['Likely_Private_URL?'] = df.iloc[:, 1].str.contains(r'\b(?:{})\b'.format('|'.join(words_to_filter)))
    df = df.sort_values(by='Likely_Private_URL?',ascending=False)
    #add name of document as a column for the excel file in sharepoint
    df['Article_Name'] = f'{filename}'
    df =  df[['Article_Name','Link_Text','Likely_Private_URL?','URL']]
    df = df.loc[df['Likely_Private_URL?'] == True]
  
    if df.shape[0] == 0:
        result = {"Article_Name":f'{filename}',"Link_Text":'No Bad Links Detected',"Likely_Private_URL":False,'URL':'No Bad Links Detected'}
    elif df.shape[0] > 0:
        for i, row in df.iterrows():
            result = {"Article_Name":row['Article_Name'],"Link_Text":row['Link_Text'],"Likely_Private_URL":row['Likely_Private_URL?'],'URL':row['URL']}
        
    return result
    

def check_links():
    
    data = request.files["link"]
    data.save('linkfile.docx')
    results = flag_private_urls('linkfile.docx')
    
    return jsonify(results.to_json())

# %%
flag_private_urls_to_dict("example_document.docx")

# %%
