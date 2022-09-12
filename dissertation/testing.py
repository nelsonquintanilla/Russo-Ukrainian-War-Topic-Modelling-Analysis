

def test3(list1, list2):
    return list1, list2


if __name__ == '__main__':
    num_topics_list=['uno','dos','tres','cuatro','cinco']
    num_topics_list2=['seis','siete','ocho','nueve','diez']
    test = [(index + 1) for index, _ in enumerate(num_topics_list)]
    print(test)

    # x = range(2, 30, 2)
    # for each in x:
    #     print(each)

    # var1, var2 = test3(num_topics_list, num_topics_list2)
    # print(var1)
    # print(var2)