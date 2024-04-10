#!/usr/bin/env python3
import pandas as pd 
import json

data = pd.read_pickle('data.pkl')

# iteratve over rows and convert to json
data_json = []
query_to_index = {}
for index, row in data.iterrows():
    '''
    category: Category of search query, used to seed diverse generations
    search_query: The user query to retrieve webpages for.
    search_engine_input: The affirmative and negative statements used to query the search API.
    search_type: Either yes_statement or no_statement. Indicates whether the website was retrieved by searching for an affirmative statement or a negative statement.
    url: The url of the website.
    title: The title of the website.
    text_raw: The raw output from jusText. Contains the text for the entire webpage.
    text_window: The 512-token window deemed most relevant to the search query.
    stance: The stance of the website, determined by an ensemble of claude-instant-v1 and GPT-4-1106-preview.
    '''
    search_query = row['search_query']
    search_engine_input = row['search_engine_input']
    search_type = row['search_type']
    url = row['url']
    title = row['title']
    text_raw = row['text_raw']
    text_window = row['text_window']
    stance = row['stance']
    if search_query not in query_to_index:
        data_json.append({
            'search_query': search_query,
            'search_engine_input': [search_engine_input],
            'search_type': [search_type],
            'urls': [url],
            'titles': [title],
            'text_raw': [text_raw],
            'text_window': [text_window],
            'stance': [stance]
        })
        query_to_index[search_query] = len(data_json) - 1
    else:
        target_idx = query_to_index[search_query]
        data_json[target_idx]['search_engine_input'].append(search_engine_input)
        data_json[target_idx]['search_type'].append(search_type)
        data_json[target_idx]['urls'].append(url)
        data_json[target_idx]['titles'].append(title)
        data_json[target_idx]['text_raw'].append(text_raw)
        data_json[target_idx]['text_window'].append(text_window)
        data_json[target_idx]['stance'].append(stance)
    


remove_index = []
for i, instance in enumerate(data_json):
    if len(set(instance['stance'])) == 1:
        remove_index.append(i)

data_json = [data_json[i] for i in range(len(data_json)) if i not in remove_index]

with open('data.json', 'w') as f:
    json.dump(data_json, f, indent=4)

# Analyze data
same_stance_count = 0
for instance in data_json:
    if len(set(instance['stance'])) == 1:
        same_stance_count += 1

print(f"Number of instances: {len(data_json)}")
print(f'Number of instances with the same stance: {same_stance_count}')
