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