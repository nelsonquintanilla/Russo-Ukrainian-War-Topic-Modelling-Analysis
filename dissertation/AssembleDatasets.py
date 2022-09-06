"""
Combine datasets obtained with different search queries. Remove duplicates. And remove items that contain the 'briefing'
word in their title.
"""
import ast
from GenerateTheGuardianDataset import save_list_of_dicts_to_file

# Merge lists of dictionaries containing articles from a specific query search
def merge_dict_lists(list1, list2):
    for dict_list_2 in list2:
        dict_list_2_id = dict_list_2.get('id')
        if not any(dict_list_2_id == dict_list_1.get('id') for dict_list_1 in list1):
            list1.append(dict_list_2)
    return list1

def read_list_of_dicts_from_file(file_name):
    location = '/Users/nelsonquintanilla/Documents/repos/nfq160/dissertation/' + file_name
    with open(location) as f:
        return ast.literal_eval(f.read())

def get_number_briefing_articles(list_of_dicts):
    counter = 0
    for dictionary in list_of_dicts:
        if 'briefing' in dictionary['id']:
            counter += 1
            # print(dictionary['id'])
    return counter

def remove_briefing_articles(list_of_dicts):
    list_of_dicts = [dictionary for dictionary in list_of_dicts if 'briefing' not in dictionary['id']]
    return list_of_dicts

if __name__ == '__main__':
    # Read files containing list of dicts
    dataset1 = read_list_of_dicts_from_file('UkraineDataset')
    dataset2 = read_list_of_dicts_from_file('RussiaDataset')
    # Print datasets length
    print('Dataset 1 length: %s' % len(dataset1))
    print('Dataset 2 length: %d' % len(dataset2))
    # Print number of "briefing" articles within each dataset
    print('\nNumber of "briefing" articles dataset 1: %d' % get_number_briefing_articles(dataset1))
    print('Number of "briefing" articles dataset 2: %d' % get_number_briefing_articles(dataset2))
    # Remove "briefing" articles within each dataset
    dataset1_no_briefing_articles = remove_briefing_articles(dataset1)
    dataset2_no_briefing_articles = remove_briefing_articles(dataset2)
    # Print datasets length after removing "briefing" articles
    print('\nDataset 1 length after removing briefing articles: %s' % len(dataset1_no_briefing_articles))
    print('Dataset 2 length after removing briefing articles: %d' % len(dataset2_no_briefing_articles))
    # Merge datasets
    merged_datasets = merge_dict_lists(dataset1_no_briefing_articles, dataset2_no_briefing_articles)
    # Compared datasets length before/after removing "briefing" articles
    print('\nSum of original datasets length: %d' % (len(dataset1) + len(dataset2)))
    print('Sum of datasets length after removing duplicates and "briefing" articles: %d' % len(merged_datasets))
    # Save merged datasets list
    save_list_of_dicts_to_file(merged_datasets, '2DatasetsMerged')
