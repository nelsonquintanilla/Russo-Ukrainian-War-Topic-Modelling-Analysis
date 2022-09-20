def test_f(start_, limit_, step_):
    for x in range(start_, limit_, step_):
        print(x)

if __name__ == '__main__':
    test_f(
        start_=2,
        limit_=15,
        step_=2
    )