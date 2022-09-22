import random

from AssembleDatasets import read_list_of_dicts_from_file

def test_f(start_, limit_, step_):
    for x in range(start_, limit_, step_):
        print(x)

if __name__ == '__main__':
    # test_f(
    #     start_=6,
    #     limit_=11,
    #     step_=2
    # )

    # print(random.randint(1, 4))

    # x = [1,2,3,4,5,5,6]
    # print(x[0:5])

    # I have 2275 documents
    # lo = 0
    # hi = 2274
    # size = 5
    # random_numbers = [lo + int(random.random() * (hi - lo)) for _ in range(size)]
    # print(random_numbers)

    # lo = 0
    # hi = 5
    # size = 4
    # random_numbers = [lo + int(random.random() * (hi - lo)) for _ in range(size)]
    # print(random_numbers)
    #
    # random.shuffle(random_numbers)
    # print(random_numbers)

    # to randomise word order in forms
    x = [1,2,3]
    random.shuffle(x)
    print(x)

    x = [0,2,5,7,8,9]
    random.shuffle(x)
    print(x)
    #
    #to select the intruder word
    lo = 0
    hi = 5
    size = 1
    random_numbers = [lo + int(random.random() * (hi - lo)) for _ in range(size)]
    print(random_numbers)

    # lo = 0
    # hi = 2274
    # size = 5
    # random_numbers = [lo + int(random.random() * (hi - lo)) for _ in range(size)]
    # print(random_numbers)
    # articles_list_of_dicts = read_list_of_dicts_from_file('2DatasetsMerged')
    # for number in random_numbers:
    #     print(articles_list_of_dicts[number])

    # random_ids = [1104, 1487, 876]
    #
    # articles_list_of_dicts = read_list_of_dicts_from_file('2DatasetsMerged')
    # for id in random_ids:
    #     print(articles_list_of_dicts[id])