class MAEngine:

    def __init__(self, price):

        self.price = price
        self.cache = {}

    def sma(self, p):

        key = ("SMA", p)

        if key not in self.cache:
            self.cache[key] = self.price.rolling(p).mean()

        return self.cache[key]

    def ema(self, p):

        key = ("EMA", p)

        if key not in self.cache:
            self.cache[key] = self.price.ewm(span=p).mean()

        return self.cache[key]


MA_LIST = ["SMA", "EMA"]