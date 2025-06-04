import numpy as np
class Fraction:
    def __init__(self,numerator,denominator=1):
        if denominator == 0:
            raise ZeroDivisionError
        common = np.gcd(numerator, denominator)

        self.numerator = int(numerator/common)
        self.denominator = int(denominator/common)
    def __repr__(self):
        return str(self.numerator) + "/" + str(self.denominator)
    def __str__(self):
        if self.denominator != 1:
            return str(self.numerator) + "/" + str(self.denominator)
        return str(self.numerator)
    def __add__(self,other):
        if type(other) == int:
            other = Fraction(other)
        return Fraction(self.numerator*other.denominator+other.numerator*self.denominator,self.denominator*other.denominator)
    def __radd__(self,other):
        return self+other
    def __mul__(self,other):
        if type(other) == int:
            other = Fraction(other)
        return Fraction(self.numerator * other.numerator,
                        self.denominator * other.denominator)
    def __rmul__(self,other):
        return self*other
    def __sub__(self,other):
        return self + other *-1
    def __rsub__(self,other):
        return self - other
    def __int__(self):
        return int(self.numerator/self.denominator)
    def __float__(self):
        return self.numerator/self.denominator