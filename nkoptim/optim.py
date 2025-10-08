import numpy as np
from nkutils import forward_rate
from nkcredit import T1stat, T3stat

def identify_columns(asset_dict):
    '''
    This function identifies the cashflow columns (labelled 1-n, so first column should be '1') 
    and the column labelled PV. It is not written to allow for errors (i.e. if your columns have 
    different naming conventions, it wont work) 
    '''
    pv_index = asset_dict.get("MV Base", None)  # Get PV column index
    cf_start_index = asset_dict.get(1, None)  # Find where '1' appears
    cf_end_index = max(v for v in asset_dict.values() if isinstance(v, (int, float)))
    n = cf_end_index - cf_start_index + 1  # Cashflow column count

    cols2return = [pv_index] + list(range(cf_start_index, cf_start_index + n))
    # Sum across rows for the selected columns
    cols2return = np.array(cols2return)  # Ensure it's a NumPy array    
    return cols2return


def SumCFs(asset_array, mydict):
# function to summarise the Market Values (PVs) and Cashflows at each time from a portfolio of assets  
    selected_cols = identify_columns(mydict)
    return np.sum(asset_array[:, selected_cols], axis=0)


def scale_array(asset_array, asset_dict, A):
    """
    Scales the cashflow columns (1 to 55), Quantity, and PV columns by the column vector A.

    Parameters:
    asset_array (np.ndarray): NumPy array containing asset data.
    A (np.ndarray): Column vector of scaling factors (should match the number of rows in asset_array).

    Returns:
    np.ndarray: Scaled NumPy array.
    """
    # Ensure A is a 1D NumPy array
    A = np.asarray(A).flatten()

    # Ensure A matches the number of rows in asset_array
    if A.shape[0] != asset_array.shape[0]:
        raise ValueError(f"Mismatch: A has {A.shape[0]} elements, but asset_array has {asset_array.shape[0]} rows.")

    # Perform element-wise multiplication (broadcasting across columns)
    scaled_array = asset_array.copy()
    selected_columns = identify_columns(asset_dict)
    scaled_array[:,selected_columns] *= A[:, np.newaxis]

    return scaled_array


def allocate_assets(asset_array, asset_dict, liabCFs, rfr,method="Equal"):
    ''' allocates assets in a way such as to minimise PRA test 1; it starts from longest duration and goes towards today.  
    method can take values "Equal" (use all bonds equally at each maturity)
    "MaxMA" (only use the bond with the greatest MA at each maturity)
    "MaxCoupon" (only highest coupon)
    "MinCoupon" (self-explanatory)
    While previously thought was given to work with only a section of the assets of the array supplied (e.g. Comp A), 
    in the end I've decided to let it work over all assets; if less are needed, that should be done before calling this 
    function. 
    '''
    num_assets = asset_array.shape[0]  # Number of assets
    liabCFs = np.asarray(liabCFs).flatten()
    liabcfs = liabCFs.copy()
    rfr = np.asarray(rfr).flatten()

    num_periods = len(liabCFs)  # Number of time periods

    # Initialize allocation vector (all assets start at 0)
    A = np.zeros(num_assets)
    MatCol = asset_dict.get("Maturity (years)", None)
    MACol = asset_dict.get("MA Base", None)
    CouponCol = asset_dict.get("Coupon rate", None)
    
    years_to_maturity = asset_array[:, MatCol]


    # Iterate from the latest liability backward
    for t in range(num_periods - 1 , -1, -1):
        # Compute total asset cashflows at time t
        currAsset_arr = scale_array(asset_array,asset_dict, A)
        assetCFs = SumCFs(currAsset_arr,asset_dict)[1:]
        curMatcol = asset_dict.get(t+1, None)
        # Compute remaining liability cashflow
        remaining_liability = liabcfs[t] - assetCFs[t]

        if remaining_liability > 0:
            # Find assets that mature at time t
            assets_with_maturity = np.where(years_to_maturity==t+1)[0]

            if len(assets_with_maturity) > 0:
                if method == "Equal":
                # Determine how many units of each asset are needed
                    allocation_per_asset = remaining_liability / np.sum(asset_array[assets_with_maturity, curMatcol])
                    for asset_idx in assets_with_maturity:
                        A[asset_idx] += allocation_per_asset

                else: 
                    if method == "MaxMA":
                        asset_idx = assets_with_maturity[np.argmax(asset_array[assets_with_maturity, MACol])]
                    elif method == "MaxCoupon":
                        asset_idx = assets_with_maturity[np.argmax(asset_array[assets_with_maturity, CouponCol])]
                    elif method == "MinCoupon":
                        asset_idx = assets_with_maturity[np.argmin(asset_array[assets_with_maturity, CouponCol])]

                        
                    allocation_asset = remaining_liability / asset_array[asset_idx, curMatcol]
                    A[asset_idx] += allocation_asset


                # Assign allocations

            else:
                # No assets mature at this time, discount liability to previous timepoint
                if t > 0:
                    fwd_rate = forward_rate(rfr, t-1, t)
                    liabcfs[t-1] += remaining_liability / (1 + fwd_rate)

    return A

def finetune4T3(asset_array, asset_dict, liabs_array, rfr, optimised_A, T3_Target=0.995, Cash = 0):
    '''  
        this function attempts to fix T3 to a target; it is used usually after an attempt 
        is done to match cashflows, as best as possible within a portfolio, based on some criteria

        optimised_A is the starting attempt to find how much to allocate to each asset

    '''

    vectorB = optimised_A.copy()
    #  first fine-tune T3; by amending whole scaling vector
    start_array = scale_array(asset_array,asset_dict,vectorB)
    ACFs = SumCFs(start_array,asset_dict)[1:]

    T3 = T3stat(ACFs,liabs_array,rfr)

    i=1
    # the below probably does not even need a loop - an exact solution should be possible...
    while (abs(T3-T3_Target) > 0.00001) and (i<20) :
        dummy = T3/T3_Target 
        vectorB = vectorB * dummy
        i+=1
        start_array = scale_array(asset_array,asset_dict,vectorB)
        ACFs = SumCFs(start_array,asset_dict)[1:]
        T3 = T3stat(ACFs,liabs_array,rfr)

    return vectorB
