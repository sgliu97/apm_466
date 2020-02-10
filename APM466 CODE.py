import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from numpy import interp, ndarray, cov
import numpy as np

class Bond:
    """
    This class is used to store the bonds.

    A bond has several key elements, such as maturity date, coupon rate,
    id...

    === Attributes ===
    _id:
        The id is the maturity date of the bond labeled in the form "M_D_Y"
        eg. "Jan_1_2020"
    mdate:
        The date of maturity
    cr:
        The coupon rate
    price:
        The list of price from Jan 2 to Jan 15
    t_start:
        The list of date corresponding to prices
    """
    _id: str
    mdate: datetime
    cr: float
    price: List[float]
    t_start: List[datetime]

    def __init__(self, id:str, mdate: datetime, cr:float, price:List[float]):
        self._id = id
        self.mdate = mdate
        self.cr = cr
        self.price = price
        self.t_start = [datetime(2020, 1, 2), datetime(2020, 1, 3),
                   datetime(2020, 1, 6), datetime(2020, 1, 7),
                   datetime(2020, 1, 8), datetime(2020, 1, 9),
                   datetime(2020, 1, 10), datetime(2020, 1, 13),
                   datetime(2020, 1, 14), datetime(2020, 1, 15)]

    def get_id(self) -> str:
        """
        Returns the ISIN of the bond
        """
        return self._id

    def n_coupon_payments(self) -> int:
        """
        The method is used to calculate the number of coupon payments until the
        maturity date, including the last one.
        """
        d = timedelta(days=182)
        t1 = datetime(2020, 1, 15)
        count = 0
        while t1 <= self.mdate:
            count += 1
            t1 += d

        return count

    def get_dirty_price(self) -> List[float]:
        """
        this method is used to calculate the corresponding dirty price from the
        quoted prices.
        """
        dirty_price = []
        if self.mdate.month == 6:
            t = datetime(2019, 12, 1)
        else:
            t = datetime(2019, 9, 1)

        for i in range(10):
            day = (self.t_start[i] - t).days
            ac_interest = (day/365)*self.cr/2
            dirty_price.extend([ac_interest + self.price[i]])

        return dirty_price

    def get_ttm(self) -> List[float]:
        """
        This method returns the time to maturity as years.
        i.e. bonds mature in 1 year has ttm == 1
        """
        ttm = []
        for i in range(10):
            ttm.append((self.mdate - self.t_start[i]).days/365)

        return ttm

    def get_ytm(self, day: int) -> float:
        """
        This method can be used to obtain the yield to maturity

        The input should be a integer indicating days between Jan 2 and Jan 15
        where the first work day is 0.

        t_points is a list used to store the time to each coupon payments
        """
        dirty_price = self.get_dirty_price()
        t_points = []
        t1 = datetime(2020, 3, 1)
        if self.mdate.month == 6:
            t1 = datetime(2020, 6, 1)
        t_initial = (t1 - self.t_start[day]).days / 365

        for i in range(self.n_coupon_payments()):
            if i == 0:
                t_points.append(t_initial)
            else:
                t_points.append(t_initial + (i * 0.5))


        def _function(p):
            sum_list = []
            x = p

            for n in range(self.n_coupon_payments()):
                if n == self.n_coupon_payments()-1:
                    sum_list.append((100 + self.cr/2) *
                                    math.exp(-x * t_points[n]))
                else:
                    sum_list.append(self.cr/2 * math.exp(-x * t_points[n]))

            return sum(sum_list) - dirty_price[day]

        return fsolve(_function, 0)[0]


b1 = Bond("Mar_1_2020", datetime(2020, 3, 1), 1.5, [99.85, 99.86, 99.86,
                                                    99.86, 99.86, 99.86,
                                                    99.86, 99.86, 99.86,
                                                    99.86])
b2 = Bond("Sep_1_2020", datetime(2020, 9, 1), 0.75, [99.26, 99.28, 99.28,
                                                     99.27, 99.28, 99.28,
                                                     99.28, 99.27, 99.28,
                                                     99.3])
b3 = Bond("Mar_1_2021", datetime(2021, 3, 1), 0.75, [98.89, 98.93, 98.95,
                                                     98.94, 98.92, 98.92,
                                                     98.88, 98.9, 98.9,
                                                     98.83])
b4 = Bond("Sep_1_2021", datetime(2021, 9, 1), 0.75, [98.41, 98.45, 98.49,
                                                     98.46, 98.46, 98.43,
                                                     98.43, 98.38, 98.41,
                                                     98.42])
b5 = Bond("Mar_1_2022", datetime(2022, 3, 1), 0.5, [97.57, 97.63, 97.66,
                                                    97.65, 97.64, 97.6,
                                                    97.61, 97.57, 97.58,
                                                    97.61])
b6 = Bond("Jun_1_2022", datetime(2022, 6, 1), 2.75, [102.53, 102.59, 102.62,
                                                     102.59, 102.58, 102.52,
                                                     102.52, 102.46, 102.47,
                                                     102.51])
b7 = Bond("Mar_1_2023", datetime(2023, 3, 1), 1.75, [100.31, 100.42, 100.48,
                                                     100.45, 100.44, 100.35,
                                                     100.31, 100.27, 100.31,
                                                     100.38])
b8 = Bond("Jun_1_2023", datetime(2023, 6, 1), 1.5, [99.48, 99.59, 99.65,
                                                    99.61, 99.62, 99.54,
                                                    99.53, 99.44, 99.49,
                                                    99.56])
b9 = Bond("Mar_1_2024", datetime(2024, 3, 1), 2.25, [102.52, 102.65, 102.75,
                                                     102.58, 102.68, 102.53,
                                                     102.47, 102.46, 102.54,
                                                     102.64])
b10 = Bond("Sep_1_2024", datetime(2024, 9, 1), 1.5, [98.72, 98.95, 99.29,
                                                     99.11, 99.25, 98.99,
                                                     99.03, 99.06, 98.99,
                                                     99.1])

t_start = [datetime(2020, 1, 2), datetime(2020, 1, 3),
           datetime(2020, 1, 6), datetime(2020, 1, 7),
           datetime(2020, 1, 8), datetime(2020, 1, 9),
           datetime(2020, 1, 10), datetime(2020, 1, 13),
           datetime(2020, 1, 14), datetime(2020, 1, 15)]

My_Bonds = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]


def problem_1_a(show_plot: bool):
    """
    This is the result of problem 1a, returning the interpolated ytm and ttm

    Use input to indicate whether or not to plot graphs
    """

    x = []
    y = []

    for days in range(10):
        x.append([])
        y.append([])
        for bonds in My_Bonds:
            x[days].append(bonds.get_ttm()[days])
            y[days].append(bonds.get_ytm(days))

    ttm_inter = []
    ytm_inter = []

    for i in range(10):

        x_new = [(k+1)/2 for k in range(10)]
        ttm_inter.append(x_new)
        ytm_inter.append(ndarray.tolist(interp(x_new, x[i], y[i])))



    if show_plot:
        for i in range(10):
            plt.plot(ttm_inter[i],ytm_inter[i])
        plt.xlabel("time to maturity")
        plt.ylabel("yield to maturity")
        plt.show()

    return [ttm_inter, ytm_inter]



def problem_1_b(plot: bool):
    """It is the method for problem 2b

    Assumption: The starting time of all prices are assumed to be 0

    set PV = FV = 100, Coupon rate = yield to maturity
    since we use interpolated ttm, ytm

    Return derived spot rate from year 1 - 5

    """
    ttm = problem_1_a(False)[0]
    ytm = problem_1_a(False)[1]
    spot_rate = []
    coupon_rate = []
    for bonds in My_Bonds:
        coupon_rate.append(bonds.cr)
    for i in range(10):
        r = ytm[i][0]
        spot = [r]
        t = [(k+2)/2 for k in range(9)]
        for time in t:
            def _func(p):
                x = p
                s_list = []
                for n in range(t.index(time)+1):
                    s_list.append(100 * ytm[i][int(time*2 - 1)] *
                                  math.exp(-spot[n] * (n+1)/2))
                s_list.append((100 + 100 * ytm[i][int(time*2 - 1)])
                              * math.exp(-x * time))

                return (-100 + sum(s_list))
            r = fsolve(_func, 0)[0]

            spot.append(r)
        spot_rate.append(spot)

    for i in range(10):
        spot_rate[i].remove(ytm[i][0])

    if plot:
        x = [(k+2)/2 for k in range(9)]
        for i in range(10):
            plt.plot(x, spot_rate[i])
        plt.xlabel("time")
        plt.ylabel("spot rate")
        plt.show()

    return spot_rate



def problem_1_c(plot: bool):
    """
    This method is used to calculate the forward rate

    Note that p0 * e^-t1r1 * e^-(t2-t1)f = p1 * e^-(t2-t1)f = p2 = p0 * e^-r2t2
    => f = (t2r2 - t1r1) t2 - t1

    We can derive our forward rate immediately from the above equation
    """
    forward_rate = []
    spot_rate = problem_1_b(False)
    t = [(k+1) for k in range(5)]
    for i in range(10):
        f = []
        for k in range(4):
            f.append((t[k+1] * spot_rate[i][2 * (k+1)] - t[k]
                      * spot_rate[i][2 * k])/(t[k+1]-t[k]))
        forward_rate.append(f)

    t.remove(5)

    if plot:
        for i in range(10):
            plt.plot(t, forward_rate[i])
        plt.xlabel("starting year")
        plt.ylabel("forward rate")
        plt.show()

    return forward_rate

def problem_2(print_: bool, type: str):
    """
    This method calculates the two covariance matrices

    Use type == 'y' indicating return the yield covariance matrix
        type == 'f' indicating return the forward covariance matrix
        otherwise return None


    Use print_ == True to indicate printing the two matrices
    """
    ytm = problem_1_a(False)[1]
    x = [[], [], [], [], []]
    for i in range(5):
        for j in range(9):
            x[i].append(math.log(ytm[j+1][i*2 + 1]/ytm[j][i*2 + 1]))

    variable_yield = np.array(x)

    cov_yield = cov(variable_yield)

    fwr = problem_1_c(False)
    y = [[], [], [], []]
    for i in range(4):
        for j in range(9):
            y[i].append(math.log(fwr[j+1][i]/fwr[j][i]))

    cov_forward = cov(np.array(y))

    if print_:
        print(cov_yield)
        print(cov_forward)

    if type == 'y':
        return cov_yield
    elif type == 'f':
        return cov_forward
    else:
        return None


def problem_3(print_: bool):
    cov_y = problem_2(False, 'y')
    cov_f = problem_2(False, 'f')

    if print_:
        print(np.linalg.eig(cov_y))
        print(np.linalg.eig(cov_f))


problem_1_a(True)
problem_1_b(True)
problem_1_c(True)
problem_2(True, '')
problem_3(True)
