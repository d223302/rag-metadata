import wikipedia
import torch
from sentence_transformers import SentenceTransformer
import json
from time import sleep
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import spacy

# Set language and user_agent
wikipedia.set_lang("en")
wikipedia.set_user_agent("MyApp/1.0 (myemail@example.com)")
nlp = spacy.load("en_core_web_sm")

def get_nouns(sentence):
    nouns = []
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            nouns.append(token.text.capitalize())
    if len(nouns) == 0:
        if "figurative" in sentence:
            nouns.append('figurative')
        elif "spelunking" in sentence:
            nouns.append('spelunking')
    return nouns

class WikipediaSearch:
    def __init__(self, model = "sentence-transformers/gtr-t5-base"):
        self.num_results = 10
        self.cosine_threshold = 0.6
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_encoder =  SentenceTransformer(model).to(self.device)
        self.sentence_encoder.eval()


    def search(self, query):
        # Search Wikipedia
        search_results = wikipedia.search(query, results=self.num_results)

        # Get summary of each article in the search results
        matches = []
        for article_title in search_results:
            try:
                # Get page summary
                page = wikipedia.page(article_title)
                matches.append({"title": page.title, "url": page.url})
                # print(f"Title: {page.title} - Page URL: {page.url}")
            except wikipedia.exceptions.PageError:
                # print(f"Page not found for {article_title}")
                pass
            except wikipedia.exceptions.DisambiguationError as e:
                # print(f"Disambiguation page found for {article_title}, options include: {e.options}")
                pass

        if len(matches) == 0:
            return []
        else:
            return self.filter_matches(query, matches)

    def filter_matches(self, query, matches):
        '''
        This function checks if the title of the matches is similar to the query.
        We use a sentence encoder to encode the query and the title, and then compute the cosine similarity between the two vectors.
        If the similarity is above a certain threshold, we consider the match relevant and add it to the filtered_matches list.
        Additionally, if the title is a substring of the query, we also consider it a relevant match.
        '''

        # Encode the query and the titles
        with torch.inference_mode():
            query_embedding = self.sentence_encoder.encode(
                [query] + [match["title"] for match in matches], 
                convert_to_tensor = True,
                show_progress_bar = False,
                device = self.device,
                batch_size = 16,
            )
            cosine_scores = torch.matmul(
                query_embedding[0].unsqueeze(0) / query_embedding[0].norm(), 
                (query_embedding[1:] / query_embedding[1:].norm(dim = -1, keepdim = True)).T 
            ).flatten().detach().cpu().numpy()
        
        # Sort matches by cosine similarity from high to low
        sorted_indices = cosine_scores.argsort()[::-1]
        matches = [matches[i] for i in sorted_indices]
        cosine_scores = cosine_scores[sorted_indices]



        print(f"Cosine Similarity Scores: {cosine_scores}")
        print(f"Length of Cosine Scores: {len(cosine_scores)}")
        print(f"Numer of Matches: {len(matches)}")

        filtered_matches = []
        for i, match in enumerate(matches):
            title = match["title"]
            if title.lower() in query.lower():
                filtered_matches.append(match)
            elif cosine_scores[i] >= self.cosine_threshold:
                filtered_matches.append(match)
        if len(filtered_matches) == 0:
            return []
        return [filtered_matches[0]]

# Example query
searcher = WikipediaSearch()

with open('../data.json', 'r') as f:
    data = json.load(f)

queries = [instance['search_query'] for instance in data]
searched_wikis = []

queries = queries

pbar = tqdm(queries, total = len(queries))
for query in pbar:
    print(query)
    results = searcher.search(query)
    if len(results) == 0:
        # Create a dummy Wikipedia page and title
        # Extract a noun from the search query
        try:
            noun = get_nouns(query)[0]
        except IndexError:
            noun = "ERROR_HERE"
        results = {
            "title": noun,
            "url": f"https://en.wikipedia.org/wiki/{noun}"
        }
    searched_wikis.append(results)
    with open('searched_wiki_urls.json', 'w') as f:
        json.dump(searched_wikis, f, indent = 4)
    sleep(0.5)