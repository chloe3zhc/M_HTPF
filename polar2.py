import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def polar2(*args):
    """
    POLAR  Polar coordinate plot.
    POLAR(THETA, RHO) makes a plot using polar coordinates of
    the angle THETA (in radians) versus the radius RHO.
    POLAR(THETA,RHO,R) uses radial limits specified by R.
    POLAR(THETA,RHO,S) uses linestyle specified in S.
    POLAR(THETA,RHO,R,S) combines radial limits and linestyle.
    POLAR(AX,...) plots into AX instead of GCA.
    H = POLAR(...) returns a handle to the plotted object.
    """
    # Parse possible Axes input
    cax = plt.gca()
    args = list(args)
    nargs = len(args)

    if nargs < 1 or nargs > 4:
        raise ValueError('Requires 2 to 4 data arguments.')

    if nargs == 2:
        theta, rho = args
        if isinstance(rho, str):
            line_style = rho
            rho = theta
            mr, nr = rho.shape
            if mr == 1:
                theta = np.arange(1, nr + 1)
            else:
                th = np.arange(1, mr + 1).reshape(-1, 1)
                theta = np.tile(th, (1, nr))
        else:
            line_style = 'auto'
        radial_limits = []
    elif nargs == 1:
        theta = args[0]
        line_style = 'auto'
        rho = theta
        mr, nr = rho.shape
        if mr == 1:
            theta = np.arange(1, nr + 1)
        else:
            th = np.arange(1, mr + 1).reshape(-1, 1)
            theta = np.tile(th, (1, nr))
        radial_limits = []
    elif nargs == 3:
        if isinstance(args[2], str):
            theta, rho, line_style = args
            radial_limits = []
        else:
            theta, rho, radial_limits = args
            line_style = 'auto'
            if len(radial_limits) == 2:
                radial_limits = np.concatenate([radial_limits, [0, 2 * np.pi]])
            elif len(radial_limits) != 4:
                raise ValueError('R must be a 2 or 4 element vector')
    else:  # nargs == 4
        theta, rho, radial_limits, line_style = args
        if len(radial_limits) == 2:
            radial_limits = np.concatenate([radial_limits, [0, 2 * np.pi]])
        elif len(radial_limits) != 4:
            raise ValueError('R must be a 2 or 4 element vector')

    if isinstance(theta, str) or isinstance(rho, str):
        raise TypeError('Input arguments must be numeric.')

    if theta.shape != rho.shape:
        raise ValueError('THETA and RHO must be the same size.')

    # Get hold state
    next_plot = plt.rcParams['axes.hold']
    hold_state = plt.isinteractive()

    # Get x-axis text color so grid is in same color
    tc = plt.rcParams['axes.edgecolor']
    ls = plt.rcParams['grid.linestyle']

    # Hold on to current Text defaults
    fAngle = plt.rcParams['font.angle']
    fName = plt.rcParams['font.family']
    fSize = plt.rcParams['font.size']
    fWeight = plt.rcParams['font.weight']
    fUnits = plt.rcParams['font.units']

    plt.rcParams['font.angle'] = 'normal'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['font.units'] = 'points'

    # Make a radial grid
    if not hold_state:
        plt.hold(True)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        # Calculate radial limits
        if len(radial_limits) == 0:
            arho = np.abs(rho.flatten())
            arho = arvo[np.isfinite(arho)]
            maxrho = np.max(arho) if len(arho) > 0 else 1
            minrho = 0
        else:
            rmin, rmax, thmin, thmax = radial_limits

        # Draw radial circles
        th = np.linspace(0, 2 * np.pi, 100)
        xunit = np.cos(th)
        yunit = np.sin(th)

        # Plot background
        if plt.rcParams['axes.facecolor'] != 'none':
            plt.fill(xunit * (rmax - rmin), yunit * (rmax - rmin), color=plt.rcParams['axes.facecolor'])

        # Draw radial lines
        for i in np.arange(rmin + rinc, rmax + rinc, rinc):
            plt.plot(xunit * (i - rmin), yunit * (i - rmin), ls=ls, color=tc, lw=1)
            plt.text((i - rmin + rinc / 20) * np.cos(np.pi / 180 * 82),
                     (i - rmin + rinc / 20) * np.sin(np.pi / 180 * 82),
                     f'  {i}', va='bottom')

        # Plot spokes
        th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        cst = np.cos(th)
        snt = np.sin(th)
        plt.plot((rmax - rmin) * np.r_[-cst, cst], (rmax - rmin) * np.r_[-snt, snt],
                 ls=ls, color=tc, lw=1)

        # Annotate spokes
        rt = 1.1 * (rmax - rmin)
        for i, angle in enumerate(th):
            plt.text(rt * np.cos(angle), rt * np.sin(angle), f'{int(np.degrees(angle))}', ha='center')
            plt.text(-rt * np.cos(angle), -rt * np.sin(angle), f'{int(np.degrees(angle + np.pi))}', ha='center')

        plt.xlim(-(rmax - rmin), (rmax - rmin))
        plt.ylim(-1.15 * (rmax - rmin), 1.15 * (rmax - rmin))
    else:
        rmin = plt.gca().get_xlim()[0] if hasattr(plt.gca(), '_rMin') else 0

    # Reset font defaults
    plt.rcParams['font.angle'] = fAngle
    plt.rcParams['font.family'] = fName
    plt.rcParams['font.size'] = fSize
    plt.rcParams['font.weight'] = fWeight
    plt.rcParams['font.units'] = fUnits

    # Transform data to Cartesian coordinates
    xx = (rho - rmin) * np.cos(theta)
    yy = (rho - rmin) * np.sin(theta)

    # Plot data
    if line_style == 'auto':
        q, = plt.plot(xx, yy)
    else:
        q, = plt.plot(xx, yy, line_style)

    if not hold_state:
        plt.axis('off')
        plt.xlabel('')
        plt.ylabel('')

    return q