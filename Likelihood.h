// Library for computing the SAY likelihood

#include <cmath>
#include <vector>
#include <numeric>
#include <sstream>

namespace SAY {
namespace detail {

template <class InIt>
typename std::iterator_traits<InIt>::value_type accumulate(InIt begin, InIt end) {
	typedef typename std::iterator_traits<InIt>::value_type real;
	real sum = real(0);
	real running_error = real(0);
	real temp;
	real difference;

	for (; begin != end; ++begin) {
		difference = *begin;
		difference -= running_error;
		temp = sum;
		temp += difference;
		running_error = temp;
		running_error -= sum;
		running_error -= difference;
		sum = std::move(temp);
	}
	return sum;
};

// compute log(1+x) without losing precision for small values of x
template<typename T>
T LogOnePlusX(T x)
{
	if (x <= -1.0)
	{
		std::stringstream os;
		os << "Invalid input argument (" << x
		   << "); must be greater than -1.0";
		throw std::invalid_argument( os.str() );
	}

	if (fabs(x) > 1e-4)
	{
		// x is large enough that the obvious evaluation is OK
		return log(1.0 + x);
	}

	// Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3
	// Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
	T x2 = x*x;
	T x3 = x2*x;
	T x4 = x3*x;
	return x-x2/2.0+x3/3.0-x4/4.0;
};

struct poissonLikelihood{
	template <typename T>
	T operator()(double dataCount, T const & lambda, T const & w2_sum) const{
		if(lambda==0)
			return(dataCount==0?0:-std::numeric_limits<T>::infinity());
		T sum(lambda);
		sum+=lgamma(dataCount+1);
		return(dataCount*log(lambda)-sum);
	}
};

struct gammaPriorPoissonLikelihood {
	template<typename T>
	T operator()(double k, T const & alpha, T const & beta) {
		std::vector<T> items(5);
		items[0] = alpha*log(beta);
		items[1] = lgamma(k+alpha);
		items[2] = -lgamma(k+1);
		items[3] = -(k+alpha)*LogOnePlusX(beta);
		items[4] = -lgamma(alpha);
		return accumulate(items.begin(), items.end());
	}
};

struct SAYMeanLikelihood {
    template<typename T>
    T operator()(double k, T const & w_sum, T const & w2_sum) const {
        if(w_sum <= 0 || w2_sum < 0) {
            return(k==0?0:-std::numeric_limits<T>::infinity());
        }

        if(w2_sum == 0) {
            return poissonLikelihood()(k, w_sum, w2_sum);
        }

        T one(1);
        T zero(0);
        if(w_sum == zero) {
            if(k == 0) {
                return zero;
            }
            else {
                return T(-std::numeric_limits<double>::infinity());
            }
        }

        T alpha = w_sum*w_sum/w2_sum;
        T beta = w_sum/w2_sum;
        T L = gammaPriorPoissonLikelihood()(k, alpha, beta);

        return L;
    }
};

struct SAYModeLikelihood {
	template<typename T>
	T operator()(double k, T const & w_sum, T const & w2_sum) const {
		if(w_sum <= 0 || w2_sum < 0) {
			return(k==0?0:-std::numeric_limits<T>::infinity());
		}

		if(w2_sum == 0) {
			return poissonLikelihood()(k, w_sum, w2_sum);
		}

		T one(1);
		T zero(0);
		if(w_sum == zero) {
			if(k == 0) {
				return zero;
			}
			else {
				return T(-std::numeric_limits<double>::infinity());
			}
		}

		const T & mu = w_sum;
		T mu2 = mu*mu;
		const T & sigma2 = w2_sum;

		T beta = (mu + sqrt(mu2+sigma2*4.0))/(sigma2*2);
		T alpha = (mu*sqrt(mu2+sigma2*4.0)/sigma2 + mu2/sigma2 + 2.0) / 2.0;
		T L = gammaPriorPoissonLikelihood()(k, alpha, beta);

		return L;
	}
};

struct SAYUninformedLikelihood {
    template<typename T>
    T operator()(double k, T const & w_sum, T const & w2_sum) const {
        if(w_sum <= 0 || w2_sum < 0) {
            return(k==0?0:-std::numeric_limits<T>::infinity());
        }

        if(w2_sum == 0) {
            return poissonLikelihood()(k, w_sum, w2_sum);
        }

        T one(1);
        T zero(0);
        if(w_sum == zero) {
            if(k == 0) {
                return zero;
            }
            else {
                return T(-std::numeric_limits<double>::infinity());
            }
        }

        T alpha = w_sum*w_sum/w2_sum + 1.0;
        T beta = w_sum/w2_sum;
        T L = gammaPriorPoissonLikelihood()(k, alpha, beta);

        return L;
    }
};

template<typename DataType>
DataType get_mu(std::vector<DataType> const & wi) {
    return detail::accumulate(wi.begin(), wi.end());
};

template<typename DataType>
DataType get_mu2(std::vector<DataType> const & wi) {
    std::vector<double> w2i(wi.size());
    std::transform(wi.begin(), wi.end(), w2i.begin(), [](double w)->double{return w*w;});
    return detail::accumulate(w2i.begin(), w2i.end());
};

struct chi2Likelihood {
    template<typename T>
    T operator()(unsigned int k, const std::vector<T>& wi) const{
        T mu = detail::get_mu(wi);
        if(mu > 0){
            T diff = mu - k;
            return -(diff * diff) / mu / 2;
        }
        return(0.0);
    }
};

struct pearsonChi2Likelihood {
    template<typename T>
    T operator()(unsigned int k, const std::vector<T>& wi) const{
        T mu = detail::get_mu(wi);
        T mu2 = detail::get_mu2(wi);
        if(mu > 0){
            T diff = mu - k;
            return -(diff * diff) / (mu + mu2) / 2;
        }
        return(0.0);
    }
};

} // namespace detail

} // namespace SAY
