import matplotlib.pyplot as plt
from stochastic.continuous import FractionalBrownianMotion
from stochastic.continuous import BrownianMotion
from statistics import mean
import numpy as np

def sqrt(x):

	return x**(1/2.0)

def fit_slope(x, y):

	xs = np.array(x, dtype=np.float64)
	ys = np.array(y, dtype=np.float64)
	return (((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)**2)-mean(xs**2)))

def main():

	# These parameters are changeable for different runs of this experiment
	seconds = 10
	hurst = 0.7
	count = 100

	av100 = []
	av1000 = []
	av10000 = []

	bm_av100 = []
	bm_av1000 = []
	bm_av10000 = []

	for x in range(0,count):

		fbm = FractionalBrownianMotion(t=seconds, hurst=hurst)
		s1 = fbm.sample(100)
		av100.append(sqrt(s1[99]**2))
		times1 = fbm.times(100)
		s2 = fbm.sample(1000)
		av1000.append(sqrt(s2[999]**2))
		times2 = fbm.times(1000)
		s3 = fbm.sample(10000)
		av10000.append(sqrt(s3[9999]**2))
		times3 = fbm.times(10000)


		bm = BrownianMotion(drift=hurst, t=seconds, scale=1)
		s4 = bm.sample(100)
		bm_av100.append(sqrt(s4[99]**2))
		times4 = bm.times(100)
		s5 = bm.sample(1000)
		bm_av1000.append(sqrt(s5[999]**2))
		times5 = bm.times(1000)
		s6 = bm.sample(10000)
		bm_av10000.append(sqrt(s6[9999]**2))
		times6 = bm.times(10000)

	fs=15
	fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,8))

	axes[0,0].plot(times1, s1)
	axes[0,0].set_title("n=100", fontsize=fs)
	axes[1,0].plot(times2, s2)
	axes[1,0].set_title("n=1000", fontsize=fs)
	axes[2,0].plot(times3, s3)
	axes[2,0].set_title("n=10000", fontsize=fs)

	axes[0,1].plot(times4, s4, 'g')
	axes[0,1].set_title("n=100", fontsize=fs)
	axes[1,1].plot(times5, s5, 'g')
	axes[1,1].set_title("n=1000", fontsize=fs)
	axes[2,1].plot(times6, s6, 'g')
	axes[2,1].set_title("n=10000", fontsize=fs)

	n_values = [100, 1000, 10000]
	fbm_values = [mean(av100), mean(av1000), mean(av10000)]
	bm_values = [mean(bm_av100), mean(bm_av1000), mean(bm_av10000)]

	axes[3,0].loglog(n_values, fbm_values)
	axes[3,0].set_title("Log(n) vs. Log(sqrt(<r>^2))")
	fbm_slope = fit_slope(n_values, fbm_values)
	print("The best-fit slope of the fbm line is: " + repr(fbm_slope))

	axes[3,1].loglog(n_values, bm_values, 'g')
	axes[3,1].set_title("Log(n) vs. Log(sqrt(<r>^2))")
	bm_slope = fit_slope(n_values, bm_values)
	print("The best-fit slope of the bm line is: " + repr(bm_slope))

	fig.suptitle("Fractional/Fractal Brownian Motion        ||        Regular Brownian Motion")
	fig.subplots_adjust(hspace=0.7, wspace=0.4)
	plt.show()

if __name__ == '__main__':
	main()