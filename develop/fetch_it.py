"""by GeoHotz"""


def fetch(url):
    import requests
    import hashlib
    import os
    import tempfile
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(
        url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()

    else:
        dat = requests.get(url).content
        # important trick here to create a .tmp file
        with open(fp+".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp+".tmp", fp)

    return dat


def mnist(url1="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", url2="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", url3="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", url4="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"):
    import gzip
    import numpy as np

    def parse(dat): return np.frombuffer(
        gzip.decompress(dat), dtype=np.uint8).copy()

    X_train = parse(fetch(url1))[0x10:].reshape((-1, 28, 28))
    Y_train = parse(fetch(url2))[8:]
    X_test = parse(fetch(url3))[0x10:].reshape((-1, 28, 28))
    Y_test = parse(fetch(url4))[8:]
    return X_train, Y_train, X_test, Y_test
