llm_input='''The following are two websites:
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