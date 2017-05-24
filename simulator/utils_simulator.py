from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt


def daterange(start_date, end_date):
    """ Iterable over dates, including end_date """
    for n in range(int((end_date - start_date).days)+1):
        yield start_date + timedelta(n)


def rand_skew_norm(fAlpha, fScale, random_state):
    # adapted from
    # http://stackoverflow.com/questions/36200913/generate-n-random-numbers-from-a-skew-normal-distribution-using-numpy
    """
    Right-skewed normal distribution around zero
    :param fAlpha: 
    :param fLocation: 
    :param fScale: 
    :return: 
    """

    # pick locatiom so that the mean is zero
    fLocation = -fScale * fAlpha / np.sqrt(1 + fAlpha**2) * np.sqrt(2/np.pi)

    sigma = fAlpha / np.sqrt(1.0 + fAlpha**2)

    afRN = random_state.randn(2)
    u0 = afRN[0]
    v = afRN[1]
    u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v

    if u0 >= 0:
        return u1*fScale + fLocation

    return (-u1)*fScale + fLocation


if __name__ == '__main__':

    points = []
    random_state = np.random.RandomState(5)
    for i in range(1000):
        points.append(rand_skew_norm(5, 0.1, random_state))

    print(np.mean(points))

    plt.hist(points, bins=100)
    plt.show()
