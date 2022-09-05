# Data Extraction
# Hit https://content.guardianapis.com/search endpoint to retrieve filtered articles
# from a single page from TheGuardian API
import requests
import math

API_KEY = 'f360f7e8-033e-4a24-b25c-81bab8512d58'
FORMAT = 'json'
TYPE = 'article'

def get_the_guardian_articles(
		q,
		section,
		from_date,
		to_date,
		show_blocks,
		page,
		page_size,
		order_by
):
	url = "https://content.guardianapis.com/search"
	params = {
		'api-key': API_KEY,
		'format': FORMAT,
		'type': TYPE,
		'q': q,
		'section': section,
		'from-date': from_date,
		'to-date': to_date,
		'show-blocks': show_blocks,
		'page': page,
		'page-size': page_size,
		'order-by': order_by
	}
	response = requests.get(url=url, params=params)
	return response.json()

# Retrieve articles from TheGuardian using pagination to get more results
def get_the_guardian_articles_list(
		number_of_articles,
		q,
		section,
		from_date,
		to_date,
		show_blocks,
		page_size,
		order_by
):
	total_of_pages = math.ceil(number_of_articles / page_size) + 1
	bodyTextSummaryList = []

	for i in range(1, total_of_pages):
		# noinspection PyBroadException
		try:
			response = get_the_guardian_articles(
				q,
				section,
				from_date,
				to_date,
				show_blocks,
				i,
				page_size,
				order_by
			)
			for result in response['response']['results']:
				article_id = result['id']
				for body in result['blocks']['body']:
					bodyTextSummaryList.append({"id": article_id, "body": body['bodyTextSummary']})
		except:
			break
	return bodyTextSummaryList
