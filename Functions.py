def logder_cubic(x, y, wl=5, xrange = None):
    '''
    Function that takes x, y (linear values),
    smooths the y values using Savitsky-Golay,
    fits a cubic spline to the profile,
    and returns the log derivative dlny/dlnx
    '''

    # Remove all nan values from input
    # Shouldn't get inf values, but will get nan in the inner
    # part of profile due to binning being finer than map resolution
    Mask = np.invert(np.isnan(y) | np.isinf(y))
    x, y = x[Mask], y[Mask]

    # Set xrange if none provided
    # This just gives us the range to determine
    # the derivative over.
    if xrange == None:
        xrange = (np.min(x), np.max(x))

    bins = np.geomspace(xrange[0], xrange[1], 1000)

    # Smooth the log profile
    smoothed_y = signal.savgol_filter(np.log(y), window_length=wl, polyorder=2)

    # Interpolate via Cubic Spline and get the analytic
    # derivative within each interval. Smoothed_y is already
    # in log, so don't take log of it here.
    Derivative = interpolate.CubicSpline(np.log(x), smoothed_y, extrapolate=None).derivative()

    #Return the bins and the the derivative in each bin
    return bins, Derivative(np.log(bins))

def logder_linear(x, data, N=100, xrange = None, window_length=5, polyorder=3):
    '''
    Chihway's code.
    Function that takes x, y (linear values),
    smooths the y values using Savitsky-Golay,
    fits a linear spline to the profile,
    and returns the log derivative dlny/dlnx
    '''
    data_sm = signal.savgol_filter(np.log10(data), window_length=window_length, polyorder=polyorder)

    f = interpolate.interp1d(np.log10(x), data_sm, kind='linear') #'cubic'

    if xrange == None:
        xrange = (np.min(x), np.max(x))

    # Evaluate spline across a very fine grid or radii
    lnrad_fine = np.linspace(np.log10(xrange[0]), np.log10(xrange[1]), num=N)

    lnsigma_fine = f(lnrad_fine)
    # Calculate derivative using finite differencing

    dlnsig_dlnr_fine = (lnsigma_fine[1:] - lnsigma_fine[:-1])/(lnrad_fine[1:] - lnrad_fine[:-1])

    return 10**((lnrad_fine[1:]+lnrad_fine[:-1])/2), dlnsig_dlnr_fine
