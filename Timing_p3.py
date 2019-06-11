def time_me(fn, *args, **kwargs):
    import time as time

    def get_time_list(time_elapsed):
        run_time = [0, 0, time_elapsed]
        if run_time[2] > 60:
            run_time[1] = int(run_time[2] / 60)
            run_time[2] = run_time[2] % 60
            if run_time[1] > 60:
                run_time[0] = int(run_time[1] / 60)
                run_time[1] = run_time[1] % 60
        return run_time

    if 'ntimes' in kwargs:
        ntimes = kwargs['ntimes']
        del kwargs['ntimes']
    else:
        ntimes = 1

    start = time.time()
    res = fn(*args, **kwargs)
    for i in range(ntimes - 1):
        fn(*args, **kwargs)
    end = time.time()
    time_list = get_time_list(end - start)
    if not isinstance(res, tuple):
        res = (res,)
    return res + (time_list,)


def print_time_list(func, run_time):
    print("{0.__name__} took: {1[0]}:{1[1]}:{1[2]}".format(func, run_time))