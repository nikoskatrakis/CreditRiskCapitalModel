## INTRODUCTION

A model has been created in Python to calculate the Solvency UK credit risk capital (SCR) within a Matching adjustment portfolio based on the most recent (September 2025) PRA requirements 
(apart from rating granularity), including the 5-step process for calculating the Matching adjustment in Stress. 

The ultimate goal is to enable users to run different portfolios and investigate different optimisation strategies, or calibrate the risk in different ways. 
It can also be used in the future for capital attribution to individual asset holdings in an SCR critical window. 
Many other uses are possible with further coding, for example: 
- Impact of more granular ratings for assets
- Capital implications for new assets or asset classes
- Adding components to allow modelling of HP assets

At the current stage, the model is probably not ready for use. It has not been peer reviewed, or tested properly, although significant effort has been put to ensure results are reasonable. Indeed the current version is a "dump" from my C drive in case I lose my laptop or other catastrophe happens. When I have a better version, I'll update this message. But feel free to "steal" any code you find helpful at your own risk. 
Depending on priorities and demand, the model may be enriched over time. Or it may be covered by spider webs, who knows :D. 

## OPERATION AND MAIN FEATURES
The model assumes a Matching-adjustment compliant Portfolio (MAP) exists at the start which has 3 funds; a CompA fund (Component A assets), a CompB fund and a NotMAP fund. 

The model will calculate the base position characteristics based on the asset portfolio data provided by the user. 

The model then runs a large number of simulations. How large depends on computing capacity. 
- For each simulation, the model calculates for each rating a spread widening. It also allows asset migration. 
- The final spread of an asset is then defined as a function of its original spread and rating and the spread widening for the post-migration rating for the appropriate duration.
- Some assets may have defaulted, in which case a recovery rate is applied to the market value of the asset at the outset.
- Fundamental spreads are also assume to stress in line with the spread widening, but its a harder stress than allowed by EIOPA rules. 
- Stressed Market values and stressed risk-adjusted cashflows are then calculated. 
- The model checks if the portfolio still satisfies PRA's Test 1 and Test 3 (T1 and T3) eligibility criteria. 
  - If so, it recalculates the MA in stress, the stressed BEL and the resulting Credit risk SCR as the sum of the change in market value of assets and the change in present value of liabilities.
  - If not, it optimises the portfolio first only by using cash, subject to a trading threshold; if not successful, then using available assets in components B and the NotMAP fund. 
  - If a list of assets available to buy in the stress scenario is available, this can be utilised as well.
  - After all optimisations have been applied, the model calculates the final SCR, by also allowing for the purchase of assets not held and the trading limits.
  - If the matching adjustment is lost (T1 fail T3 fail), the risk-free BEL is used to calculate the SCR. 

Results are produced and show the value of the portfolio held a/immediately after the stress, b/after the cash injection, if needed, c/after the asset injection if needed.

User can also extract critical window summaries for particular variables (e.g. Final SCR, DeltaBEL, DeltaMV etc). 


## KEY MODEL COMPONENTS
  - Spread Module
  - Fundamental spread module
  - Migration/Default module
  - Data input module
  - Simulation and optimisation in stress module
  - Statistics/results summary module

## KEY LIMITATIONS

The code is not meant to be ultra-efficient, although vectorisation where possible has been implemented. It is not built for cloud deployment.

There is no correlation between the spread and the migration module. A copula structure to correlate the two is high on the development agenda.

An attribution module is also quite key to develop. 

A single asset class is currently assumed. However it is probably trivial to modify the code to allow for more asset classes, e.g. financials, non-financials, infrastructure, HP assets, etc.

The duration of the assets and liabilities are not changing in the stress (deemed minor)

Trading costs for buying/selling assets in stress are ignored.

A user interface does not currently exist. Unless you know python (and in some cases, even if you do), you'll need help to run the model. but I'm sure chatGPT can explain any aspect of it, if you ask nicely. 

## CONTACT 
Feel free to reach me at nikosandthepython@gmail.com for any bugs/comments/thoughts or suggestions. I aspire to communicate very clearly, 
so if you don't understand something (but you are familiar with credit risk and matching adjustment in the UK, and AI has not been particularly helpful) its probably my fault and I'd like to put it right. 

