# ADV-3: Survey Intensity Defense — Revision Support Material for P2

## Anticipated Critique
"Your settlement model predicts site-suitable areas near volcanoes as under-surveyed. But perhaps the model simply reflects survey bias — sites are found where archaeologists look, which is near roads and institutions."

## Defense

ADV-3 robustness test directly addresses this. After controlling for road distance, BPCB office proximity, and university proximity in a Poisson regression over 703 East Java grid cells:

- Volcanic proximity adds significant explanatory power (quasi-Poisson LR p = 0.0015)
- Survey-only R2 = 0.382, full model R2 = 0.398
- Volcanic coefficient is NEGATIVE (beta = -0.477): fewer sites near volcanoes even after survey control

This validates P2's tautology defense (E013 Challenge 1): the settlement model is not simply selecting areas far from volcanoes. The volcanic effect is real and independent of survey intensity.

**For P2 specifically:** The road_dist feature in E013 (our strongest predictor) already captures the main survey intensity signal. ADV-3 shows that even when this is fully controlled, volcanic proximity adds an independent contribution.

## Citation-ready text
"An critical regression test (ADV-3) confirmed that volcanic proximity contributes significant additional explanatory power beyond survey intensity proxies including road accessibility (quasi-Poisson LR p = 0.0015), validating our model's identification of volcanic-zone archaeological gaps as substantive rather than artifactual."
