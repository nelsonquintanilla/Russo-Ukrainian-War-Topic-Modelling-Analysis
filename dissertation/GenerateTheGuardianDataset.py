"""
Retrieve TheGuardian articles and save them in a .txt file.
"""
from TheGuardianRepository import get_the_guardian_articles_list
import json

SECTION = "world"
FROM_DATE = "2022-02-24"
TO_DATE = "2022-08-31"
SHOW_BLOCKS = "body"
PAGE_SIZE = 50
ORDER_BY = "oldest"

def save_list_of_dicts_to_file(list_of_dicts, file_name):
    with open(file=file_name, mode='w') as fout:
        json.dump(list_of_dicts, fout, ensure_ascii=False)

if __name__ == '__main__':
    articles_query_ukraine = get_the_guardian_articles_list(
        number_of_articles=2267, # 2267 results are available if we use this query
        q="ukraine",
        section=SECTION,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        show_blocks=SHOW_BLOCKS,
        page_size=PAGE_SIZE,
        order_by=ORDER_BY
    )
    articles_query_russia = get_the_guardian_articles_list(
        number_of_articles=2044, # 2044 results are available if we use this query
        q="russia",
        section=SECTION,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        show_blocks=SHOW_BLOCKS,
        page_size=PAGE_SIZE,
        order_by=ORDER_BY
    )
    articles_query_testing = get_the_guardian_articles_list(
        number_of_articles=50,  # 2044 results are available if we use this query
        q="ukraine",
        section=SECTION,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        show_blocks=SHOW_BLOCKS,
        page_size=PAGE_SIZE,
        order_by=ORDER_BY
    )
    # print(len(articles_query_testing))
    # print(articles_query_ukraine)
    # print(articles_query_russia)
    # save_list_of_dicts_to_file(articles_query_testing, 'TestingDataset')
    save_list_of_dicts_to_file(articles_query_ukraine, 'UkraineDataset')
    save_list_of_dicts_to_file(articles_query_russia, 'RussiaDataset')
