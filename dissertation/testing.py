import random

from AssembleDatasets import read_list_of_dicts_from_file

def test_f(start_, limit_, step_):
    for x in range(start_, limit_, step_):
        print(x)

if __name__ == '__main__':
    # test_f(
    #     start_=2,
    #     limit_=15,
    #     step_=2
    # )

    # print(random.randint(1, 4))

    # x = [1,2,3,4,5,5,6]
    # print(x[0:5])

    # I have 2275 documents
    lo = 0
    hi = 2274
    size = 5
    random_numbers = [lo + int(random.random() * (hi - lo)) for _ in range(size)]
    print(random_numbers)

    articles_list_of_dicts = read_list_of_dicts_from_file('2DatasetsMerged')
    for number in random_numbers:
        print(articles_list_of_dicts[number])