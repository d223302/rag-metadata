def change_url(fill_in_dict, url_prefix, favor_stance):
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = url_prefix + fill_in_dict["URL_1"].strip().strip('/').split('/')[-1]
    else:
        fill_in_dict["URL_2"] = url_prefix + fill_in_dict["URL_2"].strip().strip('/').split('/')[-1]
    return fill_in_dict

def change_url_to_wiki(fill_in_dict, favor_stance, wiki_title, wiki_url):
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = wiki_url
        fill_in_dict["TITLE_1"] = wiki_title
    else:
        fill_in_dict["URL_2"] = wiki_url
        fill_in_dict["TITLE_2"] = wiki_title
    return fill_in_dict

def change_ranking(fill_in_dict, favor_stance, higher_rank = 1, lower_rank = 5):
    assert higher_rank < lower_rank, "Please provide higher_rank < lower_rank"
    higher_rank, lower_rank = str(higher_rank), str(lower_rank)
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["RANK_1"] = higher_rank
        fill_in_dict["RANK_2"] = lower_rank
    else:
        fill_in_dict["RANK_1"] = lower_rank
        fill_in_dict["RANK_2"] = higher_rank
    return fill_in_dict

def change_date(fill_in_dict, favor_stance, newer_date = None, older_date = None):
    assert newer_date is not None and older_date is not None, "Please provide newer_date and older_date"
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["DATE_1"] = newer_date
        fill_in_dict["DATE_2"] = older_date
    else:
        fill_in_dict["DATE_1"] = older_date
        fill_in_dict["DATE_2"] = newer_date
    return fill_in_dict


def wiki_wordpress_url(fill_in_dict, favor_stance, keyword_list):
    if len(keyword_list) < 1:
        print(fill_in_dict)
        raise ValueError("Please provide two keywords")

    url_keyword = "_".join([word.capitalize() for word in keyword_list[0].split(" ")]).strip()
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = f"https://en.wikipedia.org/wiki/{url_keyword}"
        fill_in_dict["URL_2"] = f"https://{url_keyword}.wordpress.com/"
    else:
        fill_in_dict["URL_1"] = f"https://{url_keyword}.wordpress.com/"
        fill_in_dict["URL_2"] = f"https://en.wikipedia.org/wiki/{url_keyword}"
    return fill_in_dict

def cnn_naturalnews_url(fill_in_dict, favor_stance, keyword_list):
    if len(keyword_list) < 1:
        print(fill_in_dict)
        raise ValueError("Please provide two keywords")

    url_keyword = "-".join([word.lower() for word in keyword_list[0].split(" ")]).strip()
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = f"https://edition.cnn.com/{url_keyword}"
        fill_in_dict["URL_2"] = f"https://www.naturalnews.com/{url_keyword}.html"
    else:
        fill_in_dict["URL_1"] = f"https://www.naturalnews.com/{url_keyword}.html"
        fill_in_dict["URL_2"] = f"https://edition.cnn.com/{url_keyword}"
    return fill_in_dict