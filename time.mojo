from python import Python

fn sleep1ms():
    try: 
        var time = Python.import_module("time")
        time.sleep(.001)
    except e:
        print("An error occurred:", e)

fn measure():
    try:
        var time_function = Python.import_module("time")
        var start = time_function.time()
        sleep1ms()
        var end = time_function.time()
        var nanos = (end - start) * 1e9
        print("sleeper took", nanos, "nanoseconds to run")
    except e:
        print("An error occurred:", e)


fn main():
    measure()