"""
Retrieve TheGuardian articles and save them in a .txt file.
"""
def merge_dict_lists_2(list1, list2):
        for dict_list_2 in list2:
                dict_list_2_id = dict_list_2.get('id')
                if not any(dict_list_2_id == dict_list_1.get('id') for dict_list_1 in list1):
                        list1.append(dict_list_2)
        return list1

if __name__ == '__main__':

    articles_list_1 = [
            {'id': 'world/2022/feb/23/ukraine-urges-its-citizens-to-leave-russia-immediately',
             'body': 'Body text of articles number 1'},
            {'id': 'world/2022/feb/24/denis-pushilin-leonid-pasechnik-donetsk-luhansk-ukraine-crisis',
             'body': 'Body text of articles number 2'},
            {'id': 'world/2022/feb/24/qa-could-putin-use-russian-gas-supplies-to-hurt-europe',
             'body': 'Body text of articles number 3'},
            {'id': 'world/2022/feb/24/western-leaders-decry-vladimir-putin-as-russia-launches-attacks-on-ukraine',
             'body': 'Body text of articles number 4'},
            {'id': 'world/2022/feb/24/thursday-briefing-russia-invades-ukraine',
             'body': 'Body text of articles number 5'},
            {'id': 'world/2022/feb/24/moment-that-putin-thundered-to-war-drowning-out-last-entreaties-for-peace',
             'body': 'Body text of articles number 6'}
    ]

    articles_list_2 = [
            {'id': 'world/2022/feb/24/australia-condemns-russias-brutal-and-unprovoked-invasion-of-ukraine',
             'body': 'Body text of articles number 7'},
            {'id': 'world/2022/jan/26/living-in-ukraine-how-have-you-been-affected-by-the-current-situation',
             'body': 'Body text of articles number 8'},
            {'id': 'world/2022/feb/12/leaving-ukraine-have-you-fled-the-country',
             'body': 'Body text of articles number 9'},
            {'id': 'world/2022/feb/24/kyiv-ukraine-russia-invasion',
             'body': 'Body text of articles number 10'},
            {'id': 'world/2022/feb/24/thursday-briefing-russia-invades-ukraine',
             'body': 'Body text of articles number 11'},
            {'id': 'world/2022/feb/24/moment-that-putin-thundered-to-war-drowning-out-last-entreaties-for-peace',
             'body': 'Body text of articles number 12'}
    ]

    merged_lists = merge_dict_lists_2(articles_list_1, articles_list_2)
    print(merged_lists)
