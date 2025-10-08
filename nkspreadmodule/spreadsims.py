
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from scipy.stats import ncx2, skew, kurtosis
#  First we define the spread term structures. These are deterministic, as we had no access to free data for term-specific spreads. 

#  the below is a set of spreads at 31/12/2024 as taken from a website: https://www.cambonds.com/wp-content/uploads/2025/06/HY-IG-Spreads-to-Treasuries-May-2025.pdf
#  some are inter/extrapolated from the data on the website


MIN_SPREAD_DISTANCE_IG = 1.02
MIN_SPREAD_DISTANCE_SUBIG = 1.1
MIN_IG_SUBIG_SEPARATION = 1.1

terms = np.array([2, 5, 10, 15, 20, 30, 50])
#  IG
AAA = np.array([0.62, 0.65, 0.70, 0.74, 0.75, 0.76, 0.76])
AA  = np.array([0.73, 0.76, 0.83, 0.89, 0.94, 0.95, 0.96])
A   = np.array([0.82, 0.82, 0.89, 1.02, 1.28, 1.35, 1.38])
BBB = np.array([1.19, 1.22, 1.44, 1.71, 1.88, 1.95, 1.98])
#  Sub-IG
BB  = np.array([2.04, 2.27, 2.57, 2.95, 3.31, 3.60, 3.80])
B = np.array([3.15, 3.54, 4, 4.36, 4.88, 5.34, 5.8])
CCC = np.array([8.31, 7.95, 8.66, 8.48, 8.95, 9.3, 9.65])

n_sims = 10
days = 253
simulations = {}

ratings = {'AAA': AAA, 'AA': AA, 'A': A, 'BBB': BBB,'BB': BB, 'B': B, 'CCC': CCC}
ratings_IG = {'AAA': AAA, 'AA': AA, 'A': A, 'BBB': BBB}
ratings_subIG = {'BB': BB, 'B': B, 'CCC': CCC}

rating_styles = {'AAA':'-', 'AA': '-.','A': ':','BBB':'--', 'BB':  (0, (1, 1)), 'B':   (0, (5, 2, 1, 2)), 'CCC': (0, (3, 5, 1, 5))} 
colors = ['blue', 'green', 'orange', 'red','magenta', 'black','grey','pink']

STARTING_POINT = 7
poly = {'AAA':None, 'AA':None,'A':None,'BBB':None}
poly_sig = {'BB':None, 'B':None, 'CCC':None}

def monotonic_poly_fit(x, y, deg=3):
#  monotonic used for IG - 
    def poly_eval(c, x):
        return sum(c[i] * x**i for i in range(len(c)))
    
    def loss(c):
        return np.mean((y - poly_eval(c, x))**2)
    
    def constraint_deriv(c):
        x_fine = np.linspace(x.min(), x.max(), 200)
        deriv = sum(i * c[i] * x_fine**(i - 1) for i in range(1, len(c)))
        return deriv  # ≥ 0

    init = np.polyfit(x, y, deg)[::-1]
    cons = {'type': 'ineq', 'fun': constraint_deriv}
    result = minimize(loss, init, constraints=cons, method='SLSQP')
    return Polynomial(result.x)


for rating, values in ratings_IG.items():
    poly[rating] = monotonic_poly_fit(terms, values, deg=3)

AAA_spread_y0 = poly['AAA'](STARTING_POINT)
AA_spread_y0 = poly['AA'](STARTING_POINT)
A_spread_y0 = poly['A'](STARTING_POINT)
BBB_spread_y0 = poly['BBB'](STARTING_POINT)

weighted_spread_y0 = AAA_spread_y0 * 0.1 + AA_spread_y0 * 0.2 + A_spread_y0 * 0.3 + BBB_spread_y0 * 0.4

AAA_ratio_y0 = AA_spread_y0 / AAA_spread_y0
AA_ratio_y0 = A_spread_y0 / AA_spread_y0
A_ratio_y0 = weighted_spread_y0 / A_spread_y0
BBB_ratio_y0 = BBB_spread_y0 / weighted_spread_y0

def normal_poly_fit(terms, values, degree=3):
#  "unimodal" used for subIG - 
    coefs = Polynomial.fit(terms, values, deg=degree).convert().coef
    return Polynomial(coefs)


for rating, values in ratings_subIG.items():
    poly_sig[rating] = normal_poly_fit(terms, values, degree=3)


BB_spread_y0 = poly_sig['BB'](STARTING_POINT)
B_spread_y0 = poly_sig['B'](STARTING_POINT)
CCC_spread_y0 = poly_sig['CCC'](STARTING_POINT)

BB_ratio_y0 = B_spread_y0 / BB_spread_y0
CCC_ratio_y0 = CCC_spread_y0 / B_spread_y0

# print(BB_spread_y0,B_spread_y0,CCC_spread_y0)
# print(weighted_spread_y0,AAA_ratio_y0,AA_ratio_y0,A_ratio_y0,BBB_ratio_y0,BB_ratio_y0,B_spread_y0,CCC_ratio_y0)

params = {
    "weighted_spread": (0.1531107, 1.1966083, 0.2765605, weighted_spread_y0),
    "AAA_ratio": (1.956686, 1.2422919, 0.5208816, AAA_ratio_y0),
    "AA_ratio": (2.5019013, 1.3657295, 0.2166543, AA_ratio_y0),
    "A_ratio": (1.0750765, 1.1391079, 0.1217743, A_ratio_y0),
    "BBB_ratio": (1.124496, 1.3776287, 0.1055011, BBB_ratio_y0),
    "BB_ratio": (1.04960971295, 1.55139557205, 0.2430319919429 , BB_ratio_y0),
    "B_spread": ( 0.292823846133333, 5.21278894172, 0.715626880103274, B_spread_y0),
    "CCC_ratio": ( 0.7558164024, 2.1603218979, 0.3334164446, CCC_ratio_y0)
}



def cir_simulate(kappa, theta, sigma, y0, days=40, n_sims=2, dt=1):
    c = (sigma**2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
    df = 4 * kappa * theta / (sigma**2)
    paths = np.zeros((n_sims, days + 1))
    paths[:, 0] = y0

    for t in range(1, days + 1):
        nc = (4 * kappa * np.exp(-kappa * dt) * paths[:, t-1]) / (sigma**2 * (1 - np.exp(-kappa * dt)))
        paths[:, t] = c * ncx2.rvs(df, nc, size=n_sims)

    return paths

def apply_tilt(poly, x_vals, tilt_pct=0.05, tilt_center=30, y0_anchor = 7):
    slope = -tilt_pct / (x_vals[-1] - tilt_center)
    tilt_factors = 1 + slope * (x_vals - tilt_center)
    idx = np.argmin(np.abs(x_vals - y0_anchor))
    tilt_number = poly(y0_anchor) - poly(y0_anchor) * tilt_factors[idx]
    return poly(x_vals) * tilt_factors, tilt_number


def simulate_spreads(n_sims = n_sims):
#  THIS BLOCK PRODUCES THE STRESSED SPREADS TO BE USED BY THE CAPITAL MODEL
#  FOR A SMALL NUMBER OF SIMULATIONS, YOU CAN ALSO ENABLE THE PLOTTING CODE - BUT PROBABLY TOO BUSY FOR ANYTHING MORE THAN 2 SIMS. 

    np.random.seed(910874002)
    for series, (kappa, theta, sigma, y0) in params.items():
        simulations[series] = cir_simulate(kappa, theta, sigma, y0, days, n_sims)


    ws = simulations["weighted_spread"][:,-1]
    bbb_r = np.maximum(MIN_SPREAD_DISTANCE_IG, simulations["BBB_ratio"][:, -1])
    a_r   = np.maximum(MIN_SPREAD_DISTANCE_IG, simulations["A_ratio"][:, -1])
    aa_r  = np.maximum(MIN_SPREAD_DISTANCE_IG, simulations["AA_ratio"][:, -1])
    aaa_r = np.maximum(MIN_SPREAD_DISTANCE_IG, simulations["AAA_ratio"][:, -1])

    bs = simulations["B_spread"][:,-1]
    bb_r = np.maximum(MIN_SPREAD_DISTANCE_SUBIG, simulations["BB_ratio"][:, -1])
    ccc_r   = np.maximum(MIN_SPREAD_DISTANCE_SUBIG, simulations["CCC_ratio"][:, -1])

    bbb_spread = ws * bbb_r
    a_spread = ws / a_r
    aa_spread = a_spread / aa_r
    aaa_spread = aa_spread / aaa_r

    b_spread = bs
    bb_spread = bs / bb_r

    #  separation routine: to ensure lowest sub-IG spread is greater than highest IG spread
    separation_vector = np.maximum(MIN_IG_SUBIG_SEPARATION, bb_spread / bbb_spread)
    bb_spread = bbb_spread * separation_vector
    b_spread = bb_spread * bb_r
    ccc_spread = b_spread * ccc_r

    # spread array
    draft_spreads = np.column_stack((aaa_spread, aa_spread, a_spread, bbb_spread, bb_spread, b_spread, ccc_spread))

    # note that the draft spreads do not attain the thetas of the original series. we therefore do some messing-up to bring them to the correct point.
    # this will generate sufficient spread widening scenarios for us to play with. it is not entirely correct, but will work with it for now. 
    means = np.mean(draft_spreads, axis=0)
    spread_thetas = np.array([0.75897, 0.94711, 1.296353, 1.980514, 3.44979, 5.212789, 11.190462])
    temp_spread_adjustment = spread_thetas - means
#    final_spreads = draft_spreads + temp_spread_adjustment
    final_spread_deltas = np.column_stack((aaa_spread - AAA_spread_y0, aa_spread - AA_spread_y0, a_spread - A_spread_y0, bbb_spread - BBB_spread_y0, 
                                        bb_spread - BB_spread_y0, b_spread - B_spread_y0, ccc_spread - CCC_spread_y0)) + temp_spread_adjustment

# some of the commented out code below is for plotting, in case one models veeeeeery few sims (2 to 4) and want to see what they look like. 
    tilts = np.random.uniform(-0.1, 0.1, n_sims) # same tilt for all ratings, but only half applied to sub-IGs due to greater gradient/steepness
    tilts = np.zeros(n_sims)
#    legend_entries = []
    terms_fine = np.linspace(min(terms), max(terms), 50)
    tilt_pivot_point = 12 # the point of the curve which is used to pivot the term-structure
    final_curve =  np.empty((n_sims, 7), dtype=object)
    # plt.figure(figsize=(16, 6))

    for s in range(n_sims):
        for CQS, rating in enumerate(ratings_IG):
            poly2plot, tilt_delta = apply_tilt(poly[rating], terms_fine, tilt_pct=tilts[s], tilt_center=tilt_pivot_point)
            shift = tilt_delta + final_spread_deltas[s, CQS]
            final_curve[s,CQS] = poly2plot + shift
    #        plt.plot(terms_fine, final_curve[s,CQS], color=colors[s] , linestyle = rating_styles[rating], alpha=0.3)
    #        legend_entry = Line2D([], [], color=colors[s], linestyle=rating_styles[rating], label=f'Sim{s} - {rating} Fit')
    #        legend_entries.append(legend_entry)
        for CQS, rating in enumerate(ratings_subIG):
            poly2plot, tilt_delta = apply_tilt(poly_sig[rating], terms_fine, tilt_pct=0.5 * tilts[s], tilt_center=tilt_pivot_point)
            shift = tilt_delta + final_spread_deltas[s, CQS + 4]
            final_curve[s,CQS + 4] = poly2plot + shift
    #        plt.plot(terms_fine, final_curve[s,CQS], color=colors[s] , linestyle = rating_styles[rating], alpha=0.3)
    #        legend_entry = Line2D([], [], color=colors[s], linestyle=rating_styles[rating], label=f'Sim{s} - {rating} Fit')
    #        legend_entries.append(legend_entry)

        
    # plt.title("Simulated Spread Term Structures (Tilted + Shifted)")
    # plt.xlabel('Term (years)')
    # plt.ylabel('Spread')
    # plt.legend(handles=legend_entries)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    return final_curve, ws


def printalittle():    # if it is to be used, need to be relinked to the right variable names.
# optional block for checking - 
    print(final_spreads.shape)
    print(final_spread_deltas.shape)

    ratiosused = np.column_stack((ws,aaa_r,aa_r,a_r,bbb_r,bb_r,b_spread,ccc_r))
    print_stats(ratiosused,colnames = ['ws','aaa_r','aa_r','a_r','bbb_r','bb_r','b_spread','ccc_r'])
    print_stats(final_spreads)

    print(AAA_spread_y0,AA_spread_y0,A_spread_y0,BBB_spread_y0,BB_spread_y0,B_spread_y0,CCC_spread_y0)
    return 1

