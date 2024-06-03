class prompt_class():
    input_no_meta='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###
###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###
Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''


    input_all='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
URL: {URL_1}
Website Publication Date: {DATE_1}
Text: {TEXT_1}
"""
###End of Website 1###
###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
URL: {URL_2}
Website Publication Date: {DATE_2}
Text: {TEXT_2}
"""
###End of Website 2###
Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_url='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
URL: {URL_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
URL: {URL_2}
Text: {TEXT_2}
"""
###End of Website 2###

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_url_1='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
URL: {URL_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_url_2='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
URL: {URL_2}
Text: {TEXT_2}
"""
###End of Website 2###

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_date='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Website Publication Date: {DATE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Website Publication Date: {DATE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_date_today='''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Website Publication Date: {DATE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Website Publication Date: {DATE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Today is 2024/04/30.
Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_rank = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Page ranking by Google Search: {RANK_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Page ranking by Google Search: {RANK_2}
Text: {TEXT_2}
"""
###End of Website 2###

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_emphasize_url = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Website 1 is from the URL: {URL_1}.
Website 2 is from the URL: {URL_2}.

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_rank_no_google = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Page ranking: {RANK_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Page ranking: {RANK_2}
Text: {TEXT_2}
"""
###End of Website 2###

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_emphasize_wiki_url = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Website 1 is from the URL: {URL_1}.
Website 2 is from the URL: {URL_2}.

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_emphasize_wiki_url_1 = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Website 1 is from the URL: {URL_1}.

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_emphasize_wiki_url_2 = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Website 2 is from the URL: {URL_2}.

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    input_emphasize_src = '''The following are two websites:
###Begin of Website 1###
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""
###End of Website 1###

###Begin of Website 2###
Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""
###End of Website 2###

Website 1 is from {URL_1}.
Website 2 is from {URL_2}.

Using these two sources, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''

    vision_prompts_with_text = [
        '''Here are the screenshots and the texts of two websites:
Website 1:
"""
Title: {TITLE_1}
Text: {TEXT_1}
"""''',
        '''Website 2:
"""
Title: {TITLE_2}
Text: {TEXT_2}
"""''',
        '''Using these two websites, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''
    ]

    vision_prompts = [
        '''Here are the screenshots of two websites:
Website 1:''',
        '''Website 2:''',
        '''Using these two websites, answer the following question: "{search_query}"
Use only the information in the above text to answer the question. Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".'''
    ]