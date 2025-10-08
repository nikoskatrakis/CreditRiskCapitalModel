
how to stress an asset: An asset has a base spread, before risk adjustment. There are two things happening in exactly the following order: 
 1. The asset migrates (or not) to another rating/default. 
 2. The spread stress applies. 

 The stressed spread is equal to: spread of new rating@stress + delta (spread asset at original rating, y0 at original rating);  
 these deltas are fixed and can be calculated for each asset at the start - non-simulation dependent  

 forgot that need curves for the sub-IG, since downgrading means that some bonds will transition downwards...  
 ok, lets do that...

 Turns out that my methodology was not robust and certainly not consistent with SDE theory. So I'll have to build another/correct the model for  
 projecting spreads. For the time being I managed to "force" the stressed spreads to have means that correspond to the thetas of the  
 underlying CIR process. Variance is left alone, to largely preserve rating monotonicity. The whole process of fitting and simulating  
 spreads will be revised at a later day. Its parked for now. 

 I still have trouble with having sensible stressed spreads. Need to keep working to address this. hopefully I'll manage. 



 STARTING_POINT (the variable defining the pivot around which spreads are "tilted" as will be defined later) could/should be different for each rating, and potentially between IG and Sub-IG; we park this for now - it can be further researched.  
 (as it will result in further complexity down below, when base and stress and tilting have various impacts...)
 for simplicity we will assume that sub-IG's are also equally weighted between themselves and therefore the 
 equivalent of the weighted_spread is the B spread (and ratios are calculated around that)
 we also assume that there is a minimum separation (as a ratio) between lowest IG and highest subIG rating 
 arbitrarily we assume that the above spreads are at least 5% apart. Other floors to prevent overlap also exist.
 note that the floor will only apply at the starting point - for other points see below!!!!
 it is also worth noting that subIG curves are not monotonic - beyond a certain term, spreads reduce. 
 this probably reflects the fact that bonds that are really close to default are less likely to default if they 
 manage to survive X years. 

 this  could result in overlaps at certain terms between IG and subIG spreads, which means the sub-IG spreads
 could end up being lower than the IG ones. for the time being we will allow this to happen and it is not expected
 to have a material impact on capital in reality, provided a company holds very little in sub-IG assets and the 
 chance of an IG asset becoming sub-IG and its spread overall reduces is fairly low. 
 
 in general, we will attempt to ensure that the IG modelling is robust, but the sub-IG maybe less so, but 
 generally sufficiently prudent.


  the parameters below have been estimated in R. Here are parts of the code (assumes you've already read and built the CCC_ratio as CCC_spread / B_spread ). 

for example: 
```r
cir_euler_loglik <- function(params, y_data) {
  kappa <- params[1]
  theta <- params[2]
  sigma <- params[3]
  dt <- 1/252  # daily steps, assuming 252 trading days/year, as per data source - modify as appropriate. 
  
  if (kappa <= 0 || theta <= 0 || sigma <= 0) return(1e6)
  
  y_t <- y_data[-length(y_data)]
  y_tp1 <- y_data[-1]
  
  mu <- y_t + kappa * (theta - y_t) * dt
  var <- sigma^2 * y_t * dt
  
  # Guard against numerical issues
  if (any(var <= 0 | !is.finite(var))) return(1e6)
  
  log_lik <- -0.5 * sum(log(2 * pi * var) + ((y_tp1 - mu)^2) / var)
  return(-log_lik)  

> fit <- optim(
+     par = start_params,
+     fn = cir_euler_loglik,
+     y_data = CCC_ratio/1.5,    #  this can be BBB_ratio, weighted_spread, etc; ensure the series has no NA values in it. 
+     method = "L-BFGS-B",
+     lower = c(kappa = 1e-6, theta =  1e-6, sigma = 1e-6),
+     upper = c(kappa = 5, theta = 20, sigma = 2),
+     control = list(fnscale = 1)
+ )
> fit$par
       kappa        theta        sigma 
0.7536062168 1.4403803454 0.2722340055 
```

The hashed-out statements are used to produce a graph that shows for each simulation the full term-structures for a particular simulation. 

Visual inspection of the above graph for single simulation runs indicates that the spread separation code (which ensures that a spread for a rating should always be greater to that of a higher rating for the same term) *generally* works, but it is perhaps not robust enough. This is something to be further investigated and potentially improved at the first validation cycle. 

Also need to validate that the $\delta$'s are correct - they have the right sign and are applied in the right manner vs what they are supposed to do. 

by now we have simulated spread stresses at the deemed maturity anchor point. We now need to expand this for the whole curve. 
next we will be tilting the term-structures (for added noise that we have no data about) and producing simulated term structures. 

We will now proceed in our plan with 100 simulations to build a model that considers downgrade and optimisation under stress.


Good news is that 1 million sims took about 3 minutes to produce and 10 seconds to save in a python-based format that was 1.73GB big - unfortunately it only included IG curves, which means that it probably takes 6 minutes and 3GB if we also include the 3 subIG curves! But this suggests that the approach generates spread structures very quickly and quite efficiently for use in capital modelling or optimisation. Also the code is not fully vectorised (there is a nested for loop for producing the "final curve" object, by rating and simulation) which means the runtime could be reduced even further, but lets park this for now. 
