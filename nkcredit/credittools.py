import numpy as np
from scipy.optimize import root_scalar

def ForwardsFromSpots(rfr_values):
    """
    Computes 1-period forward rates from a list vector of spot rates.
    Requires rates to exist in each year. 
    Assumes input is annual spot rates, compounding annually
    """
    rfr_values = np.asarray(rfr_values).flatten()
    n = len(rfr_values)

    t = np.arange(1, n)

    # Spot rates: s_i = rfr_values[1:], s_prev = rfr_values[:-1]
    s_i = rfr_values[1:]
    s_prev = rfr_values[:-1]

    numer = (1 + s_i) ** (t + 1)
    denom = (1 + s_prev) ** t
    fwd_rates = numer / denom - 1

    fwd_rates = np.insert(fwd_rates, 0, rfr_values[0])

    return fwd_rates


def pv_diff(spread, RFRs, m, PVi,CFi):
    rates = RFRs + spread
    disc_factors = 1 / (1 + rates) ** (np.arange(1, m + 1) - 0.5)
    return np.sum(CFi * disc_factors) - PVi


def calc_spreads(CFs,PVs,RFR_ts = None):
    # assume dimensions as follows: CFs: n x m; RFR_ts: 1xm ; PVs: nx1;
    CFs = np.atleast_2d(CFs)  # Ensure 2D
    n, m = CFs.shape
    PVs = np.asarray(PVs).flatten()
    if RFR_ts is None:
        RFR_ts = np.zeros(m)
    else:
        RFR_ts = np.asarray(RFR_ts).flatten()

    spreads = np.zeros(n)

    for i in range(n):
        fa = pv_diff(-1.0, RFR_ts, m, PVs[i], CFs[i])
        fb = pv_diff(1.0, RFR_ts, m, PVs[i], CFs[i])
        if fa * fb > 0: 
            spreads[i] = -30 
        else:
            sol = root_scalar(pv_diff, bracket=[-1.0, 1.0], args = (RFR_ts, m, PVs[i], CFs[i]), method='brentq')
            spreads[i] = sol.root if sol.converged else -30

    return spreads  # shape: (n,)

def calc_RA_cashflows(CFs, Ratings, PD_prob, DEFAULT_IND, RECOVERY_RATE,SpotY1):
# calculates the Risk-adjusted cashflows by applying the PD Probability methodology, 
# i.e. RACF = CF * (1 - PD + PD * RR)

    n, m = CFs.shape
    o, p = PD_prob.shape
    t_idx = np.minimum(np.arange(m), o - 1)
    rating_idx = np.array(Ratings)[:, None]
    RA_CFs = np.zeros((n,m))
    
    # Compute FS factors
    default_mask = (Ratings == DEFAULT_IND)

# Compute FS factors only for non-defaulted assets
    FS_factors = np.ones_like(CFs)
    FS_factors[~default_mask] = 1 - PD_prob[t_idx, rating_idx[~default_mask]] + PD_prob[t_idx, rating_idx[~default_mask]] * RECOVERY_RATE


#    FS_factors = (1 - FS_PD[t_idx, rating_idx + 1]) ** (t_idx + 1)
    RA_CFs[~default_mask] = CFs[~default_mask] * FS_factors[~default_mask]

    # Apply default adjustments
    RA_CFs[default_mask] = 0
    RA_CFs[default_mask, 0] = 100 * RECOVERY_RATE * ((1+SpotY1) ** 0.5)

    return RA_CFs



def calc_MA_4assets(CFs_RA,Ratings, FS_OTPD,RAspreads,RFRs,DEFAULT_INDEX = 999):
    # assume CFs_RA is n x m, FS_OTPD is 8 x 30, Ratings and RAspreads is nx1 and RFRs is 1xm
    #  calculates the MA at an Asset Level - and the FS_CoDG component at an asset level - essentially the FS by weight of the cashflows sort of. 

    n, m = CFs_RA.shape
    o, p = FS_OTPD.shape
    t_idx = np.minimum(np.arange(m), o - 1) # risk that user may provide the transpose of the "correct" matrix, in which case the value of o will be wrong, and the whole thing will give rubbish...
    Ratings = np.asarray(Ratings).flatten()
    RFRs = np.asarray(RFRs).flatten()
        
    weighted_FSs = np.zeros(n)

    b_defaults = (Ratings == DEFAULT_INDEX)
    b_sillyspreads = (RAspreads == -30)
    b_ignore = b_defaults | b_sillyspreads

    discount_factors = (1 + RFRs) ** (-(np.arange(1, m+1)))  # shape (m,)
    discounted_CFs = CFs_RA * discount_factors  # (n, m)

    fs_indices = Ratings.astype(int)
    fs_indices[b_ignore] = 0
    fs_values = FS_OTPD[t_idx[:, None], fs_indices].T

    # Calculate numerator and denominator for weighting FS by discounted cashflow: 
    num = discounted_CFs * fs_values  # (n, m)
    d = discounted_CFs  # (n, m)

    # Compute weighted_FSs:
    weighted_FSs = np.sum(num, axis=1) / np.sum(d, axis=1)
    weighted_FSs[b_ignore] = 0.0
  
    MAs = RAspreads - weighted_FSs
    MAs[b_ignore] = 0.0

    return weighted_FSs, MAs


def calc_MA_portfolio(CF_Assets, AssetPVs, LiabCFs, RFRs, Asset_FSOTPD, Cash = 0.0):
#  calculates the MA at portfolio level. Assumes we know the RA spread from each asset. 
#  it also assumes that cashflows and PVs are for Comp A assets only. if not, obvioulsy will produce garbage. 
#  It does not check if T1 or T3 pass - that needs to be done separately.
    RFRs = RFRs.flatten()
    AssetPVs = AssetPVs.flatten()
    cash_CF = Cash * (1+float(RFRs[0]))**0.5
    CompA_CFs = np.sum(CF_Assets,axis=0)
    CompA_CFs[0] += cash_CF
    CompA_MV = np.sum(AssetPVs) + Cash
 
    MA_B4_CoD = calc_spreads(CompA_CFs,np.array([CompA_MV]),RFRs)
    FS_CoD_weighted = np.sum(Asset_FSOTPD*AssetPVs)/CompA_MV # could also weigh by duration - skip for now . . . 
    MA_port = MA_B4_CoD - FS_CoD_weighted
    BEL = PV(LiabCFs,RFRs,MA_port)
    CompB_MV =  BEL - CompA_MV
    # t1 = T1stat(CompA_CFs,LiabCFs,RFRs,BEL)
    # if t1 > 0.03:   # 0.03 needs to become T1_THRESHOLD
    #     return 0, BEL_RF, BEL_RF - CompA_MV
    return MA_port, FS_CoD_weighted, BEL, CompB_MV

def calc_PVs_assets(asset_cashflows,rfrs,spreads,defaulted_spread = -30, recovery_rate = 0.35):
    # dimensionality: asset_cashflows: n x m; rfrs: 1 x m; spreads: n x s
    # allows update of all PVs, including ones for defaulted assets, assuming the parameters above have been set up corrctly. 

    n, m = asset_cashflows.shape
    s = spreads.shape[0]
    is_defaulted = (spreads == defaulted_spread)
    solid_spreads = spreads.copy()
    solid_spreads[is_defaulted] = 0.0
    solid_spreads_expanded = solid_spreads[:, :, np.newaxis]

    times = np.arange(1, m+1).reshape(1,1, m)  # shape: (1 x m)
    rfrs_expanded = rfrs.reshape(1, 1, m)

    discount_factors = (1 + rfrs_expanded + solid_spreads_expanded ) ** (-times+0.5)  # (n x s x m)
    cashflows_expanded = asset_cashflows[np.newaxis,:, :]    # (n x 1 x m)
    PV_tensor = cashflows_expanded * discount_factors   
    PV_vector = np.sum(PV_tensor, axis=2)  # (n x s)
    PV_vector[is_defaulted] = recovery_rate * 100
    return PV_vector

def PV(CFs, RFR, spr):
    CFs = np.asarray(CFs).reshape(-1)
    t = np.arange(1, len(CFs) + 1)  
    RFR = np.asarray(RFR).flatten()
    discount_factors = (1 + RFR + spr) ** -(t-0.5) # assumes all cashflows are mid-year; minor. 
    return np.sum(discount_factors * CFs)

def T1stat(ACFs,liabCFs,rfr, PVliabs):
#  calculate T1 statistic
#  calculates the greatest accumulated shortfall over a set of assets, liabilities and risk free rates
#  it is assumed that the three vectors are same length - else an error will be generated.
#  assumes mid-period cashflows; hence accumulation is for the half of first and half of second rfrate  
#  Asset CFs need to be PD-adjusted

    ACFs = np.asarray(ACFs).flatten()  
    liabCFs = np.asarray(liabCFs).flatten()  
    fwd_rates = ForwardsFromSpots(rfr)

    deltaCFs = ACFs - liabCFs
    accum = deltaCFs[0] * (1 + fwd_rates[0])**0.5
    tortn = accum

    for i in range(1, len(ACFs)):  
        accum = accum * (1 + fwd_rates[i-1])**0.5 * (1 + fwd_rates[i])**0.5 + deltaCFs[i]
        tortn = min(tortn, accum)

    return - tortn / PVliabs

def T3stat(assetCFs,liabCFs,rfr):
#  calculate T3 statistic; asset CFs need to be Risk-adjusted for PD
    assetCFs = np.asarray(assetCFs).flatten()  
    liabCFs = np.asarray(liabCFs).flatten()  
    rfr = np.asarray(rfr).flatten()

    return PV(liabCFs,rfr,0) / PV(assetCFs,rfr,0)



def stress_downgrades(trans_matrix, X):
    """
    Apply X% stress uplift to downgrade/default probabilities 
    and adjust 'no change' probability to preserve row sums.
    Upgrade probabilities remain unchanged.
    
    Parameters:
        trans_matrix (np.ndarray): Square transition matrix (rows = from-rating, cols = to-rating)
        X (float): Stress uplift factor (e.g. 0.2 for 20%)
        
    Returns:
        np.ndarray: Stressed transition matrix
    """
    stressed = trans_matrix.copy()
    n = stressed.shape[0]
    
    for i in range(n):
        row = stressed[i]
        downgrade_mask = np.zeros(n, dtype=bool)
        downgrade_mask[i+1:] = True  # entries below the diagonal (downgrades/defaults)
        
        P_downgrade = row[downgrade_mask].sum()
        uplift = P_downgrade * X
        
        # Scale downgrades by (1 + X)
        row[downgrade_mask] *= (1 + X)
        
        # Reduce the 'no change' entry proportionally
        row[i] -= uplift
        
        # Optional: Clip negatives from numerical drift
        row = np.clip(row, 0.0, 1.0)
        
        # Renormalize row if needed (for numerical stability)
        stressed[i] = row / row.sum()

    return stressed

def Alloc4CompB(asset_array, asset_dict, optimCompA, CompB):

    # this function calculates the component B and creates an allocation of assets for it. 
    # IT ASSUMES THAT ANY NON-ZERO ALLOCATIONS REFER TO COMPONENT A. 
    # it is agnostic of liability shape or anything like that. 
    # its almost random at this point, in that it allocates money to the first 10 bonds (or less) that it finds. 
    # rest of the assets that still have a zero allocation after that, are assumed to be outside the MAP. 

    # identify zero allocations in the provided asset data
    FundColumn = np.full(len(optimCompA), "NotMAP", dtype=object)
    b_zeroallocs = (optimCompA == 0)
    FundColumn[~b_zeroallocs] = "CompA"
    l = min(10,sum(b_zeroallocs))
    compB_inds = np.where(optimCompA==0)[0][:l]

    # identify the MV of the unit for each of the zero allocations
    col_index = asset_dict['PV Unit']
    base_unit_PVs = asset_array[:, col_index]
    zeroMVs = base_unit_PVs[b_zeroallocs]

    # sum up one of each for comp B and identify how much to allocate in each (it'll be the same)
    alloc_rate = CompB / np.sum(zeroMVs[:l])
