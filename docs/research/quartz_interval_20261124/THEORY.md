# Outward EVPI and plan reuse
## R01
Anchor 87c15dc828d6eea519c219fd8a412fa21dfc464d. Import the native model from a108cec049f786c3f521bcc1bdfb7cf60b945bd2. Separate an exact-model STOP certificate from replay equivalence to the existing rounded planner.

## R02
Hay et al. UAI 2012 Theorem 5 supplies the perfect-information ceiling. Rump, Acta Numerica 2010 §1.1 motivates verification by enclosing arithmetic. Here the enclosure needs only nonnegative products, sums, division and two subtractions. https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf

## R03
Origin watchdog: 15f3a19 QUARTZ_THEORY.md defines VOC = expected improvement minus cost; archived paper 3bcaa435 §5 uses measured time. Preserve downstream decision and paid search. No new P_flip, Gibbs, Gomoku or game efficacy claim. Inputs are binary64 nonnegative masses interpreted as exact reals and renormalized row by row.

## R04
Compare a fixed epsilon, direct outward interval arithmetic, and only the universal ceiling 1. The direct enclosure can exploit a smaller model ceiling. The universal ceiling is useful when STOP is exactly 1 and the cost is positive.

## R05
For each component normalize its two or three masses mathematically. Lower-bound each local CDF by downward numerator divided by upward denominator. Multiply positive lower bounds and mix with lower normalized weights. For b∈[0,1], M=1−(1/2−b)+ F0−(1−max(b,1/2)) Fhalf, with the positive part applied only to the coefficient. Downward coefficients and CDFs yield an upper bound via upward subtraction. The maximum of lower posterior means and b is a lower bound B. If upper(M−cmin) < lower(B), every paid policy is dominated in the represented normalized model.

## R06
A positive first cost and nonnegative later costs give Q≤M−c_i regardless of the future observation law. Disabled queries are excluded from cmin, never from deploy-any terminal support. Known-only deployment uses certified known labels plus b. Exact zero numerators are retained when possible. Lack of separation falls through. This is a represented-posterior certificate, not a forward error theorem for the inherited floating likelihood recursion.
