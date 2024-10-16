import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network


import pandas as pd
df = pd.read_excel('knowlage.xlsx')
with st.expander("See knowledge basse:"):
    df1= df[['Activity','Activity ID Fixed Part']]
    df2= df[['Activity','Activity ID Fixed Part','Successor (1) Fixed ID','Original Duration (Default Value)']]

    df = pd.merge(df1, df2, left_on='Activity ID Fixed Part', right_on='Successor (1) Fixed ID', how='inner')
    df
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def cosine_similarity(v1, v2):
  dot_product = np.dot(v1, v2)
  norm_v1 = np.linalg.norm(v1)
  norm_v2 = np.linalg.norm(v2)
  return dot_product / (norm_v1 * norm_v2)

def compare_text_cosine_similarity(text1, text2):
  vectorizer = CountVectorizer()
  vectors = vectorizer.fit_transform([text1, text2])
  vector1, vector2 = vectors.toarray()
  similarity = cosine_similarity(vector1, vector2)
  return similarity


if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
    
    
st.title('Mahmoud Donia Demo')
text = st.text_input("Add Activity:")
if text:
    addDF = pd.DataFrame()

    for index, row in df.iterrows():
        similarity = compare_text_cosine_similarity(text, row['Activity_y'])
        
        new_row = {'Activity':row['Activity_y'],'similarity':similarity,'Duration':row['Original Duration (Default Value)'], "Successor":row["Successor (1) Fixed ID"] ,"Fixed ID":row["Activity ID Fixed Part_y"] , "next activity":row['Activity_x'] }
        
        addDF = addDF.append(new_row, ignore_index=True)
        
        print(f"Similarity with row {index}: {similarity} word :{row['Activity_y']}")

    addDF = addDF.sort_values(by='similarity', ascending=False).iloc[0]
    if addDF['similarity']:
    
        addDF
        
        if st.button("Append", type="primary"):
            # st.session_state['df'] = pd.concat([st.session_state['df'], addDF], ignore_index=True)
            
            st.session_state['df'] = st.session_state['df'].append(addDF, ignore_index=True)
            print("=====================")
            print(type(st.session_state['df']))
st.session_state['df']



st.write("generation of visual network graphs  ")
    # Create networkx graph object from pandas dataframe
G =  nx.MultiGraph()
print("---------------")
for index, row in st.session_state['df'].iterrows():
    
    print(index, row['Activity'])
    G.add_edge(row['Activity'],row['next activity'])
    G.add_node(row['Activity'], size=int(row['Duration']), title='rtype',)
                
    

# pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
# nx.draw(G,pos=pos,with_labels=True,node_size=1000)

    # Initiate PyVis network object
drug_net = Network(height='665px', width='665px',directed=True, bgcolor='#222222', font_color='white')

    # Take Networkx graph and translate it to a PyVis graph format
drug_net.from_nx(G)

    # Generate network with specific layout settings
drug_net.repulsion(node_distance=420, central_gravity=1.33,spring_length=110, spring_strength=1.70,damping=3.95,)
drug_net.show_buttons(filter_=['physics'])


    # Save and read graph as HTML file (on Streamlit Sharing)
try:
        path = 'img'
        drug_net.save_graph(f'{path}\\pyvis_graph.html')
        HtmlFile = open(f'{path}\\pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
except:
        path = 'html_files'
        drug_net.save_graph(f'{path}\\pyvis_graph.html')
        HtmlFile = open(f'{path}\\pyvis_graph.html', 'r', encoding='utf-8')

components.html(HtmlFile.read(), height=835 ,width=835,scrolling=True)