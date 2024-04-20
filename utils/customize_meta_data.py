def change_url(fill_in_dict, url_prefix, favor_stance):
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = url_prefix + fill_in_dict["URL_1"].strip().strip('/').split('/')[-1]
    else:
        fill_in_dict["URL_2"] = url_prefix + fill_in_dict["URL_2"].strip().strip('/').split('/')[-1]
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

