import time
import threading
from multiprocessing import Process

p1 = None


def something():
    time.sleep(4)
    print("terminating")
    global p1
    p1.terminate()



def hang():
    print('hanging1..')
    t3 = threading.Thread(target=hang2)
    t3.daemon = True
    t3.start()
    print('hanging1..')
    # while True:
    #     print('hanging..')
    #     time.sleep(10)


def hang2():
    while True:
        print('hanging2..')
        time.sleep(10)


def main():
    global p1
    t = time.time()
    p1 = Process(target=hang)
    t1 = threading.Thread(target=something)
    print("something !!!")
    t1.start()
    p1.start()
    print("something else !!!")
    t1.join()
    p1.join()
    print("success !!!")


if __name__ == '__main__':
    main()
