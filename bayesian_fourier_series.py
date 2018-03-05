import numpy as np
import pymc
import matplotlib.pyplot as plt
from scipy import stats
import emcee
def fourier_series_x_intercepts(C,QW,x):
    v = fourier_series2(C,QW,x)

    foldrotm  = np.ma.masked_where(v > 0, v)
    b = np.roll(foldrotm.mask,1).astype(int)-foldrotm.mask.astype(int)
    c = np.roll(foldrotm.mask,-1).astype(int)-foldrotm.mask.astype(int)
    x_int = x[b!=0]
    x_int2 = x[c!=0]
    x_intr = x_int+x_int2
    x_intr /=2
    return x_intr
def fourier_series2(C,QW,x):
    N = (len(C)-1) / 2
    N = 1
    wavelength_number = len(QW)
    v = np.array(x.astype(float))
    v.fill(C[0])
    for w in range(wavelength_number):
        for i in range(1,N+1):
            v = v + C[(2*i-1)+2*N*w]*np.cos(2*np.pi/QW[w] * i * (x)) + \
            C[(2*i)+2*N*w]*np.sin(2*np.pi/QW[w] * i * (x))
    return v
class bayesian_fourier_series_model():
    def __init__(self,xdata,ydata,N):
        self.xdata = xdata
        self.ydata = np.tan(ydata*np.pi/180.)
        self.wavelength = 0.0
        self.setup = False
        self.N = N
        self.use_mixture = False
        self.xmin = min(self.xdata)
        self.xmax = max(self.xdata)
        self.wavelength_sd_defined = False
        self.axial_trace_likelihoods = []
        self.interlimb_likelihoods = []
        self.vergence = []
        self.asymmetry_likelihoods = []
    def add_likelihood_axial_trace(self,x,s):
        self.axial_trace_sigma = s
        self.axial_trace_likelihoods.append(x)
    def add_likelihood_interlimb(self,v,s):
        self.interlimb_sigma = s        
        self.interlimb_likelihoods.append(v)
    def add_likelihood_asymmetry(self,v,s):
        self.asymmetry_sigma = s        
        self.asymmetry_likelihoods.append(v)
        
    def add_likelihood_vergence(self, s, v):
        self.vergence.append([s,v]) 
    def add_reference_foldshape(self,x,y):
        self.reference_fold_x = x
        self.reference_fold_y = y
    def add_reference_foldlocations(self,x_,foldpts):
        self.reference_fold_points_x = x_
        self.reference_fold_points_y = foldpts
    def add_reference_foldprofile(self,x_,foldrot):
        self.reference_fold_profile_x = x_
        self.reference_fold_profile_y = foldrot
        
    def set_wavelength_sd(wavelength_sd):
        self.wavelength_sd = wavelength_sd
        self.wavelength_sd_defined = True
    def find_wavelength(self,step=0,nlags=0):
        self.semivariogram = s_variogram(self.xdata,np.arctan(self.ydata)*180./np.pi)
        self.wavelengths = []
        self.semivariogram.setup()
        wl1, wl2 = self.semivariogram.find_wavelengths(step,nlags)
        print wl1, wl2
        if np.abs(wl1 - wl2) < wl1*.2:
            self.wavelengths.append((wl1+wl2)/2.)
            return
        if wl1 > 0.:
            self.wavelengths.append(wl1)
        if wl2 > 0.:
            self.wavelengths.append(wl2)
    def setup_inference(self):
        #depending on the number of wavelengths
        wavelength_number = len(self.wavelengths)
        l = []
        i = 0
        #add c0
        t = 1. / 5.** 2
        #mu_ = np.mean(self.ydata)
        l.append(pymc.Normal("c_%i" % (i), mu=0, tau = t))
        i+=1
        for x in range(wavelength_number):
            for _ in range(2*self.N):
                t = 1. / 5.**2
                mu_ = 0
                l.append(pymc.Normal("c_%i" % (i), mu=mu_, tau = t))
                i+=1
        C = pymc.Container(l)#\
                            #for i in range(1+2*self.N) for x in range(wavelength_number)])
        #C[0]
        @pymc.stochastic(observed=False)
        def sigma(value=1):
            return -np.log(abs(value))
        @pymc.stochastic(observed=False)
        def sigma3(value=1):
            return -np.log(abs(value))
        
        qw_sigs =  pymc.Container([pymc.HalfCauchy("qw_sigs_%i" % x, beta = 10, alpha=1) \
                                 for x in range(wavelength_number)])
        if self.wavelength_sd_defined:
            qw = pymc.Container([pymc.distributions.Lognormal('qw_%i' %x,mu=self.wavelengths[x], \
                                                       tau = 1. / self.wavelength_sd[x] ** 2) \
                                 for x in range(wavelength_number)])
        else:
            qw = pymc.Container([pymc.distributions.Normal('qw_%i' %x,mu=self.wavelengths[x],\
                                                           tau = 1. / self.wavelengths[x]/3.) \
                                                       for x in range(wavelength_number)])





        def fourier_series(C,N,QW,x,wavelength_number):
            v = np.array(x)
            v.fill(0.0)
            v = v.astype('float')
           

            for ii in range(len(x)):
                v[ii] += C[0]
                for w in range(wavelength_number):
                    for i in range(1,N+1):
                        v[ii] = v[ii] + C[(2*i-1)+2*N*w]*np.cos(2*np.pi/QW[w] * i * (x[ii])) + \
                        C[(2*i)+2*N*w]*np.sin(2*np.pi/QW[w] * i * (x[ii]))
            return v
        self.vector_fourier_series = np.vectorize(fourier_series)
        # Define the form of the model and likelihood
        @pymc.deterministic
        def y_model(C=C,x=self.xdata,qw=qw,nn=self.N,wavelength_number=wavelength_number):
            return fourier_series(C,nn,qw,x,wavelength_number) 
        y = pymc.Normal('y', mu=y_model, tau=1. / sigma ** 2, observed=True, value=self.ydata)
        # package the full model in a dictionary
        self.model1 = dict(C=C, qw=qw, sigma=sigma,qw_sigs=qw_sigs,
                      y_model=y_model, y=y,x_values=self.xdata,y_values=self.ydata)
        self.model_e = pymc.Model([C,qw,sigma,y])
        if len(self.vergence) > 0:
            @pymc.deterministic
            def vergence_values(c=C,qw=qw,y=np.array(self.vergence)[:,0]):
                return np.sign(fourier_series2(c,qw,y))
            @pymc.stochastic(observed=True)
            def vergence(value=np.array(self.vergence)[:,1],mu=vergence_values):
                loglike = 0.
                loglike+=pymc.distributions.normal_like((mu[value==1]),mu=1,tau=1.)
                loglike+=pymc.distributions.normal_like((mu[value==-1]),mu=-1,tau=1.)
                if loglike < float(-1.7876931348623157e+308):
                    return float(-1.7876931348623157e+308)
                return loglike 
            self.model1.update({'vergence':vergence})
        if len(self.asymmetry_likelihoods) > 0:
            @pymc.deterministic
            def y_model_asym(c=C,qw=qw):
                x = np.linspace(-np.max(qw),np.max(qw))
                v = np.rad2deg(np.arctan(fourier_series2(c,qw,x)))
                m =  np.median(v)#-np.min(v)
                return m#np.max(v)-np.min(v)
            @pymc.stochastic(observed=True)
            def y_asym(mu=y_model_asym,value=self.asymmetry_likelihoods[0],tau=1. / self.asymmetry_sigma**2):
                loglike = pymc.distributions.normal_like(x=value,mu=mu,tau=tau)
                return loglike*10
            #y_interlimb = pymc.Normal('y_interlimb',mu=y_model_interlimb,value=self.interlimb_likelihoods[0],
            #tau = 1. / self.interlimb_sigma**2 )
            self.model1.update({'y_asym':y_asym})                 
        if len(self.interlimb_likelihoods) > 0:
            @pymc.deterministic
            def y_model_interlimb(c=C,qw=qw):
                x = np.linspace(-np.max(qw),np.max(qw))
                v = np.rad2deg(np.arctan(fourier_series2(c,qw,x)))
                d =  np.max(v)-np.min(v)
                return d#np.max(v)-np.min(v)
            @pymc.stochastic(observed=True)
            def y_interlimb(mu=y_model_interlimb,value=self.interlimb_likelihoods[0],tau=1. / self.interlimb_sigma**2):
                loglike = pymc.distributions.normal_like(x=value,mu=y_model_interlimb,tau=tau)
                return loglike*10
            #y_interlimb = pymc.Normal('y_interlimb',mu=y_model_interlimb,value=self.interlimb_likelihoods[0],
            #tau = 1. / self.interlimb_sigma**2 )
            self.model1.update({'y_interlimb':y_interlimb})
        if len(self.axial_trace_likelihoods) > 0:
            d = self.wavelengths[0]#np.max(self.axial_trace_likelihoods_limb) - np.min(self.axial_trace_likelihoods_limb)
            x_at = np.linspace(np.min(self.axial_trace_likelihoods)-d,np.max(self.axial_trace_likelihoods)+d,300)
            @pymc.stochastic(observed=False)
            def at_sigma(value=1):
                return -np.log(abs(value))
            @pymc.deterministic
            def z_model_axial_t(c=C,wl=qw,z_at=x_at):
                return np.array(fourier_series_x_intercepts(c,wl,z_at))

            @pymc.stochastic(observed=True)
            def z_at(mu=z_model_axial_t,sigma = at_sigma,value=self.axial_trace_likelihoods):
                loglike = 0.
                mu = np.array(mu)
                #print mu
                if not np.array(mu).size:
                    return float(-1.7876931348623157e+308)#-99999#-np.2inf
                for v in value:
                    m = 0.
                    if mu.shape:
                        dif = np.sort(np.abs(mu-v))
                        #if there are two hinges for the same axial trace penalise this!
                        if dif[1] < sigma:
                            loglike += -99999
                        m = mu[(np.abs(mu-v)).argmin()]
                        
                        #m = mu[(np.abs(mu-v)).argmin()]
                    else:
                        m = mu
                    #print 'm', m
                    loglike+=pymc.distributions.normal_like(x=v,mu=m,tau=1./sigma**2)
                loglike
                if loglike < float(-1.7876931348623157e+308):
                    return float(-1.7876931348623157e+308)
                return loglike
            #z_at = pymc.Normal('z_at',mu=z_model_axial_t,tau = 1. / self.axial_trace_limb_sigma,value=self.axial_trace_likelihoods_limb)
            self.model1.update({'z_at':z_at,'at_sigma':at_sigma})
        self.setup = True
        self.mcmc_uptodate = False
        return True
    def setup_inference_mixture(self):
        #depending on the number of wavelengths
        wavelength_number = len(self.wavelengths)
        l = []
        i = 0
        #add c0
        t = 1. / 5.** 2
        mu_ = np.mean(self.ydata)
        l.append(pymc.Normal("c_%i" % (i), mu=mu_, tau = t))
        i+=1
        for x in range(wavelength_number):
            for _ in range(2*self.N):
                t = 1. / 5.**2
                mu_ = 0
                l.append(pymc.Normal("c_%i" % (i), mu=mu_, tau = t))
                i+=1
        C = pymc.Container(l)#\
                            #for i in range(1+2*self.N) for x in range(wavelength_number)])
        #C[0]
        i_ = pymc.Container([pymc.DiscreteUniform('i_%i' %i,lower=0,upper=1) for i in range(len(self.xdata))])

        @pymc.stochastic(observed=False)
        def sigma(value=1):
            return -np.log(abs(value))
        @pymc.stochastic(observed=False)
        def sigma3(value=1):
            return -np.log(abs(value))
        
        qw_sigs =  pymc.Container([pymc.HalfCauchy("qw_sigs_%i" % x, beta = 10, alpha=1) \
                                 for x in range(wavelength_number)])
        if self.wavelength_sd_defined:
            qw = pymc.Container([pymc.distributions.Lognormal('qw_%i' %x,mu=self.wavelengths[x], \
                                                       tau = 1. / self.wavelength_sd[x] ** 2) \
                                 for x in range(wavelength_number)])
        else:
            qw = pymc.Container([pymc.distributions.Normal('qw_%i' %x,mu=self.wavelengths[x],\
                                                           tau = 1. / self.wavelengths[x]/3.) \
                                                       for x in range(wavelength_number)])



        def fourier_series(C,N,QW,x,wavelength_number,i_):
            v = np.array(x)
            v.fill(0.0)
            v = v.astype('float')
           

            for ii in range(len(x)):
                v[ii] += C[0]
                for w in range(wavelength_number):
                    for i in range(1,N+1):
                        v[ii] = v[ii] + C[(2*i-1)+2*N*w]*np.cos(2*np.pi/QW[w] * i * (x[ii])) + \
                        C[(2*i)+2*N*w]*np.sin(2*np.pi/QW[w] * i * (x[ii]))
                if i_[ii] == 0:
                    v[ii] = -v[ii]
            return v
        self.vector_fourier_series = np.vectorize(fourier_series)
        # Define the form of the model and likelihood
        @pymc.deterministic
        def y_model(C=C,x=self.xdata,qw=qw,nn=self.N,wavelength_number=wavelength_number,i_=i_):
            return fourier_series(C,nn,qw,x,wavelength_number,i_) 
        y = pymc.Normal('y', mu=y_model, tau=1. / sigma ** 2, observed=True, value=self.ydata)
        # package the full model in a dictionary
        self.model1 = dict(C=C, qw=qw, sigma=sigma,qw_sigs=qw_sigs,
                      y_model=y_model, y=y,x_values=self.xdata,y_values=self.ydata,i_=i_)
        self.model_e = pymc.Model([C,qw,sigma,y])

        self.setup = True
        self.mcmc_uptodate = False
        return True
        
    def run_sampler(self, samples = 10000, burnin = 5000):
        self.S = pymc.MCMC(self.model1)
        self.S.sample(iter=samples, burn=burnin)
        self.mcmc_uptodate = True
        return True
    def sample_using_emcee(self):
        m = self.model_e
        # This is the likelihood function for emcee
        def lnprob(vals): # vals is a vector of parameter values to try
            # Set each random variable of the pymc model to the value
            # suggested by emcee
            for val, var in zip(vals, m.stochastics):
                var.value = val
    
            # Calculate logp
            return m.logp
    
        # emcee parameters
        ndim = len(m.stochastics)
        nwalkers = 1000
    
        # Find MAP
        pymc.MAP(m).fit()
        start = np.empty(ndim)
        for i, var in enumerate(m.stochastics):
            start[i] = var.value
    
        # sample starting points for walkers around the MAP
        p0 = np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim)) + start
    
        # instantiate sampler passing in the pymc likelihood function
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    
        # burn-in
        pos, prob, state = sampler.run_mcmc(p0, 10)
        sampler.reset()
    
        # sample 10 * 500 = 5000
        sampler.run_mcmc(pos, 50)
    
        # Save samples back to pymc model
        self.S_e = pymc.MCMC(m)
        self.S_e.sample(1) # This call is to set up the chains
        for i, var in enumerate(self.S_e.stochastics):
            var.trace._trace[0] = sampler.flatchain[:, i]
    
        pymc.Matplot.plot(self.S_e)
    
    def find_map(self):
        self.map = pymc.MAP(self.model1)
        self.map.fit()

##define some global functions that are vectorised for numpy speed
@np.vectorize
def distance(p1,p2):
    return abs(p1-p2)
@np.vectorize
def covar(p1,p2):
    return ((p1-p2)**2)
@np.vectorize
def inside(a,l,u):
    if a > l and a<u:
        b= 0
    else:
        b = 1
    return b
class s_variogram():
    def __init__(self,xdata,ydata):
        self.xdata = xdata
        self.ydata = ydata
    def setup(self):
        x1 = np.array([self.xdata,]*len(self.xdata))
        y1 = np.array([self.ydata,]*len(self.ydata))
        #x2 = np.tile(xdata,len(xdata))
        y2 = y1.transpose()
        x2 = x1.transpose()
        self.distance_m = distance(x1,x2)
        self.covariance_m = covar(y1,y2)#x1 = np.array(np.)    
    def calc_semivariogram(self, step, nlags, tol):
        self.lags = np.arange(step,nlags*step,step)
        self.variance,self.npairs = self.semivariogram(self.lags,step,self.distance_m,self.covariance_m)
        return self.lags, self.variance, self.npairs
    def semivariogram(self,lags,tol,distance,covariance):
        variance = np.zeros(len(lags))
        npairs = np.zeros(len(lags))
        for i in range(len(lags)):
            ma = np.ma.array(data = covariance, mask =inside(distance,lags[i]-tol/2.,lags[i]+tol/2.))
            variance[i] = np.mean(ma) / 2.
            npairs[i] = ma.count()
        return variance,npairs
    def find_wavelengths(self,step=0,nlags=0):
        if step==0:
            #print np.amin(self.distance_m,axis=0)\
            #step = np.mean(np.amin(self.distance_m[np.nonzero(self.distance_m)],axis=0))
            minxx = min(self.xdata)
            maxx = max(self.xdata)
            step = abs((float((maxx - minxx)) / float(len(self.xdata))))
            step*=1.2
            #print step
        if nlags == 0:
            distance = abs(min(self.xdata)-max(self.xdata))
            nlags = (distance*1.2 / step)
        h, var, npairs = self.calc_semivariogram(step,nlags,step)
        self.px, self.py = self.find_peaks_and_troughs(h,var)
        
        self.averagex = []
        self.averagey = []

        for i in range(len(self.px)-1):
            self.averagex.append((self.px[i]+self.px[i+1])/2.)
            self.averagey.append((self.py[i]+self.py[i+1])/2.)
            i+=1 #iterate twice
        #find the extrema of the average curve
        self.px2, self.py2 = self.find_peaks_and_troughs(self.averagex,self.averagey)
        self.wl1 = 0.
        wl1py = 0.
        for i in range(len(self.px)):
            if i > 0 and i < len(self.px)-1:
                if self.py[i] > 10:
                    
                    if self.py[i-1] < self.py[i]*.70:
                        if self.py[i+1] < self.py[i]*.70:
                            self.wl1 = self.px[i]
                            if self.wl1 > 0.:
                                wl1py = self.py[i]
                                break
        self.wl2 = 0.
        for i in range(len(self.px2)):
            if i > 0 and i < len(self.px2)-1:
                if self.py2[i-1] < self.py2[i]*.90:
                    if self.py2[i+1] < self.py2[i]*.90:
                        self.wl2 = self.px2[i]
                        if self.wl2 > 0. and self.wl2 > self.wl1*2 and wl1py < self.py2[i]:
                            
                            break
        if self.wl1 == 0.0 and self.wl2 == 0.0:
            return 0.0, 2*(maxx-minxx)
        return self.wl1*2., self.wl2*2.
    def find_peaks_and_troughs(self,x,y):
        if len(x) != len(y):
            return False
        pairsx = []
        pairsy = []
        for i in range(0,len(x)):
            if i < 1:
                pairsx.append(x[i])
                pairsy.append(y[i])

                continue
            if i > len(x)-2:
                pairsx.append(x[i])
                pairsy.append(y[i])
                continue
            left_grad = (y[i-1]-y[i]) / (x[i-1]-x[i])
            right_grad = (y[i]-y[i+1]) / (x[i]-x[i+1])
            if np.sign(left_grad) != np.sign(right_grad):
                pairsx.append(x[i])
                pairsy.append(y[i])
        return pairsx,pairsy
        
class bayesian_fourier_series_figure():
    def __init__(self,bayesian_fourier_series,markersize=5,alpha=0.8):
        self.fourier_series_model = bayesian_fourier_series
        self.S = bayesian_fourier_series.S
        self.fig, self.ax = plt.subplots(2,3,figsize=(20,10))
        self.markersize = markersize
        self.alpha = alpha
    def plot_reference_fold_shape(self,col = 'k-',i = 0,j =0):
        model = self.fourier_series_model
        self.ax[j][i].plot(model.reference_fold_x,model.reference_fold_y,col,alpha=self.alpha)
        self.ax[j][i].set_ylim(min(model.reference_fold_y)*1.3,max(model.reference_fold_y)*1.3)

    def plot_reference_fold_points(self, col = 'ko', i = 0, j= 0):
        model = self.fourier_series_model
        
        self.ax[j][i].scatter(model.reference_fold_points_x, model.reference_fold_points_y,marker="o",edgecolor="black",facecolor="white")

    def plot_reference_fold_profile(self,col = 'k-', i = 1, j= 0):
        model = self.fourier_series_model
        self.ax[j][i].plot(model.reference_fold_profile_x,model.reference_fold_profile_y,col,alpha=0.5)
    def plot_reference_fold_profile_points(self,col = 'ko', i = 1, j= 0):
        model = self.fourier_series_model
        self.ax[j][i].scatter(model.xdata,np.arctan(model.ydata)*180./np.pi,marker="o",edgecolor="black",facecolor="white")#col,alpha=0.8,markersize=self.markersize)
        #self.ax[j][i].plot(model.xdata,(model.ydata),col,alpha=0.8,markersize=self.markersize)
        self.ax[j][i].set_ylim(-90,90)
    def plot_variogram(self):
        plt.sca(self.ax[0][2])
        model = self.fourier_series_model
        plt.plot(model.semivariogram.lags,model.semivariogram.variance,'bo',markersize=self.markersize)
        plt.plot(model.semivariogram.px,model.semivariogram.py,'ro',markersize=self.markersize+1)
        for i in range(len(model.wavelengths)):
            plt.axvline(model.wavelengths[i]/2.,color='k',linestyle='--')
            #plt.annotate('wl_%i_est'%i, xy=(model.wavelengths[i]/2., max(model.semivariogram.variance)*.8), xytext=(model.wavelengths[i]+model.semivariogram.lags[3], max(model.semivariogram.variance)*.9),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            #)
        #plt.axvline(model.wavelengths[1],color='k',linestyle='--')
        
        self.black_white = False
        #tax = self.ax[0][2].twinx()
        #tax.plot(model.semivariogram.lags,model.semivariogram.npairs,'ko',markersize=self.markersize)
        #self.semivariogram.plot()
    def plot_curve_density(self,xmin,xmax):
        
        model = self.fourier_series_model
        if model.mcmc_uptodate == False:
            model.run_sampler()
        wavelength_number = len(model.wavelengths)
        self.C_ =  []
        i = 0
        self.C_.append(model.S.trace('c_%i' %(i))[:])
        i+=1
        for _ in range (2*model.N*wavelength_number):
            self.C_.append(model.S.trace('c_%i' %(i))[:])
            i+=1
        self.qw = []
        for x in range(wavelength_number):
            self.qw.append(model.S.trace('qw_%i' %x)[:])
        ends_ = (model.xmax-model.xmin)*2
        x = np.linspace(xmin,xmax,600)
        v = np.array((self.C_[0][:,None]))
        #v.fill(0.0)
        for w in range(wavelength_number):
            for i in range(1,model.N+1):
                v = v + self.C_[(2*i-1)+2*model.N*w][:,None]*\
                np.cos(2*np.pi/self.qw[w][:,None] * i * x) + self.C_[(2*i)+2*model.N*w][:,None]\
                *np.sin(2*np.pi/self.qw[w][:,None] * i * x)
        self.v = np.arctan(v)*180.0/np.pi
        self.x = x
        #ymin = -90
        #ymax = 90
        x_f = np.tile(x,len(self.qw[0]))
        y_f = self.v.flatten()
        vv = np.linspace(-90,90,180)
        H = np.zeros((len(x),len(vv)))
        for i in range(len(x)):
            for j in range(len(self.v)):
                vind = np.nonzero(np.abs(vv-self.v[j][i]) == np.min(np.abs(vv-self.v[j][i])))[0]
                H[i,vind[0]]+=1
        self.xmin = xmin
        self.xmax = xmax
        H/=len(self.qw[0])
        H[H==0.] = np.nan
        # = np.abs(np.log(H))
        im= self.ax[1][1].imshow((np.rot90(H)), cmap='viridis', extent=[xmin, xmax, -90, 90],aspect='auto')
        #self.fig.subplots_adjust(right=0.8)
        
        #cbar_ax = self.fig.add_axes([0.82, 0.60, 0.02, 0.28])
        #cbar_ax2 = self.fig.add_axes([0.82, 0.16, 0.02, 0.28])
        #self.fig.colorbar(im,cbar_ax)
    def plot_map(self,xmin,xmax,colour='r-'):
        model = self.fourier_series_model

        model.find_map()
        wavelength_number = len(model.wavelengths)
        x = np.linspace(xmin,xmax,600)

        map_v = np.array(x)
        map_v.fill(0.0)
        map_v += model.map.C[0].value
        for w in range(wavelength_number):
            for i in range(1,model.N+1):
                map_v = map_v + model.map.C[(2*i-1)+2*model.N*w].value * np.cos(2*np.pi/ \
                model.map.qw[w].value* i * x) + model.map.C[(2*i)+2*model.N*w].value \
                *np.sin(2*np.pi/model.map.qw[w].value*i*x)
        map_v = np.arctan(map_v)*180. / np.pi
        self.ax[1][1].plot(x,map_v,colour,alpha=0.8)   
    def plot_map_fold_shape(self,xmin,xmax,intercept,colour='r-'):
        gradient = np.tan(map_v*np.pi/180.)
        #start all points at xmin = 0
        step = x[1] - x[0]
        p = []
        for i in range(len(self.x)):
            if not p:
                p.append(intercept)
                continue
            else:
                if i == (len(self.x) - 1):
                    p_ = p[len(p)-1] + ((gradient[i-1]+gradient[i]) / 2.) * step
                else:
                    p_ = p[len(p)-1] + ((gradient[i-1]+gradient[i+1]) / 2.) * step
                p.append(p_)
        self.ax[1][0].plot(x,p,colour,alpha=0.8)
    def plot_random_fold_shape(self,i,x=1,y=1,colour='b-'):
        y = self.v[i]
        
        gradient = np.tan(y*np.pi/180.)
        #start all points at xmin = 0
        step = self.x[1] - self.x[0]
        p = []
        for i in range(len(self.x)):
            if not p:
                p.append(intercept)
                continue
            else:
                if i == (len(self.x) - 1):
                    p_ = p[len(p)-1] + ((gradient[i-1]+gradient[i]) / 2.) * step
                else:
                    p_ = p[len(p)-1] + ((gradient[i-1]+gradient[i+1]) / 2.) * step
                p.append(p_)
        self.ax[x][y].plot(self.x,p,colour,alpha=0.8)
            
    def plot_fold_heat_map(self, intercept,ii=0,jj=1):
        pr_ = []
        model = self.fourier_series_model

        for i in range(len(self.qw[0])):            
            y = self.v[i]
            gradient = np.tan(y*np.pi/180.)
            #start all points at xmin = 0
            step = self.x[1] - self.x[0]
            p = []
            for i in range(len(self.x)):
                if not p:
                    p.append(intercept)
                    continue
                else:
                    if i == (len(self.x) - 1):
                        p_ = p[len(p)-1] + ((gradient[i-1]+gradient[i]) / 2.) * step
                    else:
                        p_ = p[len(p)-1] + ((gradient[i-1]+gradient[i+1]) / 2.) * step
                    p.append(p_)
            pr_.append(p)
                      
        #plt.plot(self.x,p)
        
        
        x_f = np.tile(self.x,len(self.qw[0]))
        y_f = np.array(pr_).flatten()
        miny = min(y_f)
        maxy = max(y_f)
        miny*=1.2
        maxy*=1.2
        vv = np.linspace(miny,maxy,120)

        H = np.zeros((len(self.x),len(vv)))
        for i in range(len(self.x)):
            for j in range(len(pr_)):
                vind = np.nonzero(np.abs(vv-pr_[j][i]) == np.min(np.abs(vv-pr_[j][i])))[0]
                H[i,vind[0]]+=1
        H/=len(self.qw[0])
        H[H==0]=np.nan
        self.ax[jj][ii].imshow((np.rot90(H)), extent=[self.xmin,self.xmax, miny, maxy],cmap='viridis',aspect='auto')
        self.ax[jj][ii].set_ylim(min(model.reference_fold_y)*1.3,max(model.reference_fold_y)*1.3)
        #reference_fold_profile_x,
    def plot_traces(self):
        pymc.Matplot.plot(self.S)
    def plot_trace(self,name):
        pymc.Matplot.plot(name)
    def plot_violin(self,x,y,labels,pos,arrays):
        violin_parts = self.ax[x][y].violinplot(arrays,pos,points=80, vert=False, widths=0.7,
                      showmeans=True, showextrema=False, showmedians=False)
        rrred = '#ff2222'
        bluuu = '#2222ff'
        c = 0
        for vp in violin_parts['bodies']:
            if c > 0 and c % 2 == 1:    
                vp.set_facecolor(bluuu)
                vp.set_edgecolor(bluuu)
                vp.set_linewidth(1)
                vp.set_alpha(0.5)
            else:
                vp.set_facecolor(rrred)
                vp.set_edgecolor(rrred)
                vp.set_linewidth(1)
                vp.set_alpha(0.5)
            c+=1      
        self.ax[x][y].set_yticks(pos)
        self.ax[x][y].set_yticklabels(labels)
    def plot_kde(self,name,x,y,label='kde_pdf',colour='b-'):
        model = self.fourier_series_model

        d = model.S.trace(name)[:][:]
        
        minn = min(d)
        maxx = max(d)
        diff = (maxx-minn)
        minn = minn - diff
        maxx = maxx + diff
        ind = np.linspace(minn,maxx,100)
        kde = stats.gaussian_kde(d)
        kdepdf = kde.evaluate(ind)
        #kdepdf /= len(d)
        #kdepdf = np.log(kdepdf)
        if self.ax[x][y].has_data():
            temp = self.ax[x][y].twinx()
            temp.plot(ind,kdepdf,colour,label=label,alpha=0.8)
            #temp.legend(loc=0)
            return temp
        self.ax[x][y].plot(ind,kdepdf,colour,label=label,alpha=0.8)
        return self.ax[x][y]#self.ax[x][y].legend(loc=0)
    def plot_normal_pdf(self,x,y,mu,sigma,l='pdf',colour='b-'):
        xx = np.linspace(mu-4.*sigma,mu+4.*sigma,1000)
        self.ax[x][y].plot(xx,plt.mlab.normpdf(xx,mu,sigma),colour,alpha=0.8,label=l)
        
    def plot_normal_pdf2(self,ax,mu,sigma,l='pdf',colour='b-'):
        xx = np.linspace(mu-4.*sigma,mu+4.*sigma,1000)
        ax.plot(xx,plt.mlab.normpdf(xx,mu,sigma),colour,label=l,alpha=0.8)    
    def plot_random_curve(self,ii):
        i = np.random.randint(0,len(self.v))
        y = self.v[i]
        #plt.figure()
        x = np.linspace(0,600,600)

        self.ax[1][2].plot(self.x,y,'r-')
        C = []
        C.append(self.C_[0][i,None][0])
        wavelength_number = len(self.fourier_series_model.wavelengths)
        for _ in range(1,(self.fourier_series_model.N*2)*wavelength_number+1):
            C.append(self.C_[_][i,None][0])

        qw = []
        for w in range(len(self.fourier_series_model.wavelengths)):
            qw.append(self.qw[w][i,None][0])
        #self.ax[1][2].plot(x,v,'b-')
        return C,qw

    def set_xlims(self,minx,maxx):
        self.ax[0][0].set_xlim(minx,maxx)
        self.ax[0][1].set_xlim(minx,maxx)
        self.ax[1][0].set_xlim(minx,maxx)
