simple_html = '''<h1>{TITLE}</h1>  
<span style="font-size: 1.2em">{TEXT}</span>'''

pretty_html = '''<!DOCTYPE HTML>
<!--
	TXT by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
-->
<html>
	<head>
		<title>No Sidebar - TXT by HTML5 UP</title>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
		<link rel="stylesheet" href="assets/css/main.css" />
	</head>
	<body class="is-preload">
		<div id="page-wrapper">

			<!-- Header -->
				<header id="header">
					<div class="logo container">
						<div>
							<p>{TITLE}</p>
						</div>
					</div>
				</header>

			<!-- Nav -->
				<nav id="nav">
					<ul>
						<li><a href="index.html">Home</a></li>
						<li class="current"><a href="no-sidebar.html">More Information</a></li>
						<li><a href="left-sidebar.html">Contact Us</a></li>
					</ul>
				</nav>

			<!-- Main -->
				<section id="main">
					<div class="container">
						<div class="row">
							<div class="col-12">
								<div class="content">

									<!-- Content -->

										<article class="box page-content">


											<section>
												<p>
													{TEXT}
												</p>
											</section>
										</article>

								</div>
							</div>
							
						</div>
					</div>
				</section>

			<!-- Footer -->
		</div>

		<!-- Scripts -->
			<script src="assets/js/jquery.min.js"></script>
			<script src="assets/js/jquery.dropotron.min.js"></script>
			<script src="assets/js/jquery.scrolly.min.js"></script>
			<script src="assets/js/browser.min.js"></script>
			<script src="assets/js/breakpoints.min.js"></script>
			<script src="assets/js/util.js"></script>
			<script src="assets/js/main.js"></script>

	</body>
</html>
'''


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

def wiki_wordpress_src(fill_in_dict, favor_stance, keyword_list):
    if len(keyword_list) < 1:
        print(fill_in_dict)
        raise ValueError("Please provide two keywords")

    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = "Wikipedia"
        fill_in_dict["URL_2"] = "Wordpress"
    else:
        fill_in_dict["URL_1"] = "Wordpress"
        fill_in_dict["URL_2"] = "Wikipedia"
    return fill_in_dict

def cnn_naturalnews_src(fill_in_dict, favor_stance, keyword_list):
    if len(keyword_list) < 1:
        print(fill_in_dict)
        raise ValueError("Please provide two keywords")

    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["URL_1"] = "CNN"
        fill_in_dict["URL_2"] = "Naturalnews"
    else:
        fill_in_dict["URL_1"] = "Naturalnews"
        fill_in_dict["URL_2"] = "CNN"
    return fill_in_dict

def pretty_simple_html(fill_in_dict, favor_stance):
    if fill_in_dict["stance_1"] == favor_stance:
        fill_in_dict["HTML_1"] = pretty_html.format(
            TITLE=fill_in_dict["TITLE_1"],
            TEXT=fill_in_dict["TEXT_1"]
        )
        fill_in_dict["HTML_2"] = simple_html.format(
            TITLE=fill_in_dict["TITLE_2"],
            TEXT=fill_in_dict["TEXT_2"]
        )
    else:
        fill_in_dict["HTML_1"] = simple_html.format(
            TITLE=fill_in_dict["TITLE_1"],
            TEXT=fill_in_dict["TEXT_1"]
        )
        fill_in_dict["HTML_2"] = pretty_html.format(
            TITLE=fill_in_dict["TITLE_2"],
            TEXT=fill_in_dict["TEXT_2"]
        )
    return fill_in_dict
