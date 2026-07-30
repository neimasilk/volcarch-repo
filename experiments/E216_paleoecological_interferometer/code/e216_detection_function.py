#!/usr/bin/env python3
"""
E216 — Paleo-Ecological Interferometer
S3-S6: Forward Model + Detection Function + Two-Mode Separation

Pre-registered parameters (PREREG.md, 2026-06-25):
  C = 0.90 (90% detection threshold)
  N_floor = 631,059 (E196 p5, comparative island scaling)
  N_central = 1,270,000 (E196 median, comparative island scaling)
  Diagnostic = charcoal + Cerealia/Oryza-type co-occurrence

Simplified REVEALS forward model:
  Expected %NAP at core = (RPP_NAP * A_cleared) / (RPP_NAP * A_cleared + RPP_AP * A_forested)
  RPP_NAP (Poaceae/herbs relative to closed-canopy AP) = 2-4 (tropical range, Sugita 2007)
  Background %NAP (closed tropical forest) = 5-8%
  Detection threshold: %NAP rise >= 15 percentage points above background = 2-sigma for tropical montane records

Data provenance:
  Core network: E214 (palynology SLR, 2026-06-08)
  Population model: E196 (Monte Carlo, 2026-04-13)
  REVEALS parameterisation: Sugita 2007 + published tropical RPP ranges
  Heartland coordinates: Kedu Plain ~-7.5S/110E, Brantas ~-7.8S/112E (from inscription spatial studies)
"""

import sys, json, csv, math
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)

# ── E196 population parameters (read from results JSON) ───────────────────
E196_FLOOR   = 631_059    # p5, comparative island scaling — most conservative
E196_CENTRAL = 1_270_000  # median, comparative island scaling

JAVA_AREA_KM2     = 129_000
ARABLE_FRAC_LO    = 0.65   # E196 method2 bounds
ARABLE_FRAC_HI    = 0.80
CULT_FRAC_A_LO    = 0.10   # Mode A: wet-rice / large-swidden (landscape clearing)
CULT_FRAC_A_HI    = 0.40
CULT_FRAC_B_LO    = 0.01   # Mode B: dispersed forest-garden / arboriculture
CULT_FRAC_B_HI    = 0.05

# ── REVEALS parameters ────────────────────────────────────────────────────
# RPP (Relative Pollen Productivity) for tropical SE Asia, Sugita 2007 framework
# Poaceae RPP relative to closed-canopy AP sum
RPP_NAP_LO  = 2.0   # conservative (dense closed canopy dampens signal)
RPP_NAP_HI  = 4.0   # optimistic (open to semi-open landscape)
RPP_NAP_MID = 3.0   # central estimate

# Cerealia/Oryza-type RPP is lower (large pollen, low dispersal)
RPP_ORYZA_LO  = 0.05  # very low dispersal
RPP_ORYZA_HI  = 0.15  # short-range only
RPP_ORYZA_MID = 0.10

# Background NAP fraction in closed tropical forest (montane and lowland)
NAP_BACKGROUND_LO  = 0.04   # 4% in dense rainforest
NAP_BACKGROUND_HI  = 0.10   # 10% in more open montane
NAP_BACKGROUND_MID = 0.06   # 6% central estimate

# Detection threshold: rise in NAP above background that is "substantial"
# Based on: (a) Dieng qualitative = "substantial nearly continuous clearance"
# (b) Rawa Danau food crops appear in Zone 5 (last ~200-400 yr)
# (c) SE Asian tropical literature: >15% NAP rise = clear anthropogenic
# We use 15 pp as the conservative threshold, 20 pp as stringent
NAP_THRESHOLD_LO    = 0.15   # 15 pp rise — conservative
NAP_THRESHOLD_HI    = 0.20   # 20 pp rise — stringent
NAP_THRESHOLD_MID   = 0.175  # central

# ── Core network geometry ─────────────────────────────────────────────────
# RSAP radius from REVEALS/LOVE modelling for each archive type:
#   crater lake (r ~1 km): RSAP ~5-10 km   (Sugita 2007 Fig.3)
#   highland lake (r ~0.5-2 km): RSAP ~8-15 km
#   lowland swamp/bog: RSAP ~20-30 km
#   marine near-coast: effective pollen catchment ~100-500 km (diluted)
# Crucially: distance_to_heartland vs RSAP determines if core can see heartland

KEDU_LAT,   KEDU_LON   = -7.50, 110.00  # Kedu Plain (Candi Borobudur / Prambanan zone)
BRANTAS_LAT, BRANTAS_LON = -7.80, 112.00  # Brantas headwaters (Kediri / Singosari zone)

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

CORES = [
    dict(id='J1', name='Dieng/Telaga Balekambang', archive='crater_lake',
         lat=-7.20, lon=109.90, elevation_m=2000, lake_radius_km=1.0,
         rsap_lo=5, rsap_hi=12, rsap_mid=8,
         has_0_500ce=True, positive_control=True,
         control_event='~600 CE Hindu-Javanese centre', control_signal='substantial Plantago + herb rise',
         data_access='paywall', citation='Pudjoarinto & Cushing 2001, RPP 116:13-45'),
    dict(id='J2', name='Rawa Danau', archive='lowland_swamp',
         lat=-6.20, lon=105.90, elevation_m=0, lake_radius_km=5.0,
         rsap_lo=20, rsap_hi=35, rsap_mid=25,
         has_0_500ce=True, positive_control=True,
         control_event='~AD 1770 recent clearance', control_signal='food crops, open vegetation',
         data_access='open_jstage', citation='Yulianto et al. 2005, Tropics 14:271'),
    dict(id='J3', name='Teluk Banten', archive='marine_estuarine',
         lat=-6.00, lon=106.10, elevation_m=0, lake_radius_km=None,
         rsap_lo=100, rsap_hi=300, rsap_mid=200,
         has_0_500ce=True, positive_control=False,
         control_event=None, control_signal='historic times, qualitative',
         data_access='paywall', citation='van der Kaars & van den Bergh 2004, JQS 19:229'),
    dict(id='J4', name='Bandung Basin', archive='lacustrine',
         lat=-6.90, lon=107.60, elevation_m=665, lake_radius_km=15.0,
         rsap_lo=25, rsap_hi=45, rsap_mid=35,
         has_0_500ce=True, positive_control=False,
         control_event=None, control_signal='human only late Holocene',
         data_access='paywall', citation='van der Kaars & Dam 1995, P3 117:55'),
    dict(id='J5', name='Situ Bayongbong', archive='highland_lake',
         lat=-7.10, lon=107.80, elevation_m=1300, lake_radius_km=0.5,
         rsap_lo=5, rsap_hi=10, rsap_mid=7,
         has_0_500ce=True, positive_control=False,
         control_event=None, control_signal='no human signal detected',
         data_access='unclear', citation='Stuijts 1993, MQRSEA 12'),
    dict(id='J6', name='Marine Solo River', archive='marine',
         lat=-6.50, lon=112.00, elevation_m=0, lake_radius_km=None,
         rsap_lo=200, rsap_hi=600, rsap_mid=400,
         has_0_500ce=True, positive_control=True,
         control_event='~2950 cal BP canopy decline (hedged, climate-confounded)',
         control_signal='canopy decline, charcoal, Solo drainage catchment',
         data_access='open_phd', citation='Poliakova/Zonneveld et al. 2017, RPP 244'),
    dict(id='J7', name='Song Gupuh (Gunung Sewu karst)', archive='cave_alluvial',
         lat=-8.00, lon=110.50, elevation_m=200, lake_radius_km=1.0,
         rsap_lo=3, rsap_hi=6, rsap_mid=4,
         has_0_500ce=True, positive_control=False,
         control_event=None, control_signal='Neolithic ~2.6 ka (hunter-gatherer/karst context)',
         data_access='paywall', citation='Song Gupuh studies S2352226722000782'),
]

# Add distances to heartlands
for c in CORES:
    c['dist_kedu_km']   = haversine_km(c['lat'], c['lon'], KEDU_LAT,    KEDU_LON)
    c['dist_brantas_km'] = haversine_km(c['lat'], c['lon'], BRANTAS_LAT, BRANTAS_LON)

# ── S3: Population → cleared area (Mode A and Mode B) ─────────────────────

def pop_to_cleared_km2(N, mode='A'):
    """
    Convert population N to a range of cleared/cultivated area in km².
    Uses E196 coefficients: arable_frac, cultivation_frac.

    Mode A (wet-rice/large-swidden): contiguous cleared patches
    Mode B (dispersed/forest-garden): diffuse mosaic, 1/5 the clearing intensity

    Returns (lo, mid, hi) in km²
    """
    if mode == 'A':
        cf_lo, cf_hi = CULT_FRAC_A_LO, CULT_FRAC_A_HI
    else:
        cf_lo, cf_hi = CULT_FRAC_B_LO, CULT_FRAC_B_HI

    cf_mid = (cf_lo + cf_hi) / 2
    af_mid = (ARABLE_FRAC_LO + ARABLE_FRAC_HI) / 2

    # Total cultivated area in Java = JAVA_AREA * arable_frac * cultivation_frac
    total_cult_lo  = JAVA_AREA_KM2 * ARABLE_FRAC_LO  * cf_lo
    total_cult_mid = JAVA_AREA_KM2 * af_mid            * cf_mid
    total_cult_hi  = JAVA_AREA_KM2 * ARABLE_FRAC_HI   * cf_hi

    # Scale by (N / E196_central) to get area for population N
    # (linear: more people = more land)
    scale = N / E196_CENTRAL
    return (total_cult_lo * scale, total_cult_mid * scale, total_cult_hi * scale)


# ── S4: Forward model (simplified REVEALS) ────────────────────────────────

def nap_rise(cleared_km2, rsap_km2, rpp_nap=RPP_NAP_MID, nap_bg=NAP_BACKGROUND_MID, alpha=0.55):
    """
    Rise in NAP above background at the core due to local clearing within RSAP.

    Derivation (REVEALS-consistent):
      NAP_observed = alpha * NAP_landscape(f) + nap_bg
      where NAP_landscape(f) = RPP_NAP * f / (RPP_NAP * f + (1 - f))
      f = cleared_km2 / rsap_km2 (fraction of RSAP cleared)
      alpha = local RSAP contribution weight (0.4-0.7 for small highland lakes)
      nap_bg = background from long-distance regional transport (constant)

    When f=0: rise = 0 (correct baseline).
    When f=0.1: rise = alpha * RPP_NAP * 0.1 / (RPP_NAP*0.1 + 0.9) ≈ alpha * 0.25 * RPP_NAP
    """
    if rsap_km2 <= 0:
        return 0.0
    f = min(cleared_km2 / rsap_km2, 1.0)
    denom = rpp_nap * f + (1 - f)
    nap_landscape = (rpp_nap * f / denom) if denom > 0 else 0.0
    return max(alpha * nap_landscape, 0.0)  # rise above background (nap_bg already in baseline)


def detect_prob(cleared_within_rsap_km2, rsap_km2, rpp_nap=RPP_NAP_MID,
                nap_bg=NAP_BACKGROUND_MID, threshold=NAP_THRESHOLD_MID, alpha=0.55):
    """
    Binary detection: P = 1.0 if signal > threshold, else 0.0.
    In practice, add uncertainty from count statistics (±5pp for 300-grain count).
    P(detect) = probability that observed NAP rise >= threshold, given Poisson count uncertainty.
    Approximated as: P = Phi((rise - threshold) / count_sigma)
    where count_sigma = sqrt(rise/300) (binomial approx for pollen count)

    NOTE (Opus review 2026-06-25, Defect 2): this count-statistics term is a MINOR
    source of uncertainty (~5pp) relative to the PARAMETER uncertainty in rpp_nap,
    threshold, and alpha (each spanning a 2-4x range in the literature). A single
    call to this function at MID parameter values is a point estimate, not a
    calibrated probability. Use e216_sensitivity_sweep.py to get the parameter-space
    interval before reporting "P(detect)" as a probability in any write-up.
    """
    import math
    rise = nap_rise(cleared_within_rsap_km2, rsap_km2, rpp_nap, nap_bg, alpha)
    count_sigma = math.sqrt(max(rise, 0.01) * (1 - max(rise, 0.01)) / 300)
    # Standard normal CDF
    z = (rise - threshold) / count_sigma if count_sigma > 0 else float('inf')
    # Simple erf-based CDF
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── S5: Per-core and network detection function ───────────────────────────

def run_detection_analysis(N_pop, mode='A', rpp_nap=RPP_NAP_MID, threshold=NAP_THRESHOLD_MID, alpha=0.55):
    """
    For population N with mode A or B:
    1. Compute total cleared area in Java
    2. For each core: estimate cleared area within its RSAP
       (assuming clearing proportional to RSAP fraction of Java, or spatially concentrated)
    3. Compute P(detect) per core
    4. Compute P(network detects) = 1 - prod(1 - P_i)
    """
    total_lo, total_mid, total_hi = pop_to_cleared_km2(N_pop, mode)

    results = []
    for c in CORES:
        rsap_mid_km = c['rsap_mid']
        rsap_km2    = math.pi * rsap_mid_km**2

        # For LAKE/SWAMP cores: only detect clearing within RSAP radius.
        # The clearing is concentrated in the heartland (Kedu/Brantas), which is
        # AT LEAST dist_kedu_km away from the core.
        # If dist_kedu > rsap, then the heartland clearing is OUTSIDE the RSAP → 0 contribution.
        # For marine cores: pollen is catchment-integrated, so we use the fraction of
        # Java within the marine catchment.

        dist_kedu   = c['dist_kedu_km']
        dist_brantas = c['dist_brantas_km']
        min_dist    = min(dist_kedu, dist_brantas)

        if c['archive'] == 'marine':
            # Marine cores see Java-wide signal (diluted over full catchment)
            # For Solo marine (J6): Solo River drains ~16,000 km² of Java
            # Pollen dilution factor: ~16,000 km² / (pi * rsap_mid^2 km²)
            if c['id'] == 'J6':
                # Solo River drainage ~16,000 km², covers parts of C+E Java including Brantas
                solo_drainage_km2 = 16_000
                brantas_in_solo   = 0.35  # Brantas headwaters partially in Solo catchment
                # Fraction of total Java clearing that falls in Solo drainage
                frac_in_drainage = (solo_drainage_km2 / JAVA_AREA_KM2) * brantas_in_solo
                cleared_in_rsap = total_mid * frac_in_drainage
                # Additional dilution by marine deposition area
                marine_dilution = 0.3
                cleared_effective = cleared_in_rsap * marine_dilution
            else:
                # Other marine cores (J3): west Java orientation, far from heartland
                cleared_effective = 0.0
        else:
            # Terrestrial/lacustrine core: RSAP-based detection
            # If heartland is outside RSAP, clearing contribution is minimal
            if min_dist > rsap_mid_km:
                # Heartland is OUTSIDE RSAP: no heartland signal detectable
                # Only very weak long-distance signal from regional background pollen
                # This is below detection threshold — assign essentially 0
                frac_in_rsap = rsap_km2 / (JAVA_AREA_KM2 * 100)  # RSAP is tiny slice of Java
                cleared_in_rsap = total_mid * frac_in_rsap
            else:
                # Heartland is within RSAP (or nearby): gets full clearance signal
                # This only applies if the heartland itself is within RSAP
                frac_in_rsap = min(rsap_km2 / JAVA_AREA_KM2, 0.5)
                cleared_in_rsap = total_mid * frac_in_rsap
            cleared_effective = cleared_in_rsap

        # P(detect)
        p_detect = detect_prob(cleared_effective, rsap_km2, rpp_nap, NAP_BACKGROUND_MID, threshold, alpha)
        nap_signal = nap_rise(cleared_effective, rsap_km2, rpp_nap, NAP_BACKGROUND_MID, alpha)

        results.append({
            'core_id': c['id'],
            'core_name': c['name'],
            'archive': c['archive'],
            'rsap_km': rsap_mid_km,
            'dist_kedu_km': round(dist_kedu, 1),
            'dist_brantas_km': round(dist_brantas, 1),
            'heartland_in_rsap': min_dist <= rsap_mid_km,
            'cleared_in_rsap_km2': round(cleared_effective, 2),
            'expected_nap_rise': round(nap_signal, 4),
            'threshold_nap_rise': threshold,
            'p_detect': round(p_detect, 4),
            'positive_control': c['positive_control'],
            'mode': mode,
            'N_pop': N_pop,
        })

    # Network-level: P(at least one core detects)
    p_miss_network = 1.0
    for r in results:
        p_miss_network *= (1 - r['p_detect'])
    p_network = 1 - p_miss_network

    return results, p_network


# ── S6: Two-mode separation ───────────────────────────────────────────────

def run_both_modes(N_pop, rpp_nap=RPP_NAP_MID, threshold=NAP_THRESHOLD_MID, alpha=0.55):
    """Run Mode A (clearing) and Mode B (dispersed) and compare."""
    res_a, p_net_a = run_detection_analysis(N_pop, mode='A', rpp_nap=rpp_nap, threshold=threshold, alpha=alpha)
    res_b, p_net_b = run_detection_analysis(N_pop, mode='B', rpp_nap=rpp_nap, threshold=threshold, alpha=alpha)
    return res_a, p_net_a, res_b, p_net_b


# ── S7: Confound controls ─────────────────────────────────────────────────

CONFOUND_ANALYSIS = {
    'climate_fire_confound': {
        'control': 'Require charcoal + Cerealia/Oryza co-occurrence (climate fires lack cultigen pollen)',
        'applies_to': 'J6 Solo marine core (~2950 BP hedged signal)',
        'status': 'OPEN — charcoal + Oryza co-occurrence not verified from available data',
        'mitigation': 'Poliakova 2017 thesis data needed; 403 paywall at present',
    },
    'natural_variance_band': {
        'control': 'Late-Holocene NAP variance from non-anthropogenic fluctuations',
        'estimated_from': 'Bandung Basin (J4) LGM grass signal classified as climatic; Bayongbong (J5) = null',
        'noise_estimate_pp': '±5-8 percentage points (tropical montane)',
        'threshold_chosen': f'{NAP_THRESHOLD_MID*100:.0f} pp rise = ~2sigma above noise',
        'status': 'BOUNDED from available literature',
    },
    'marine_catchment_ambiguity': {
        'control': 'Solo ~2950 BP is run as a worked sensitivity case',
        'conclusion': 'Cannot exclude climate signal; large catchment cannot attribute to specific Java subregion',
        'status': 'SUPPRESSED by pre-registration: not counted toward OUTCOME-1 unless charcoal+Oryza confirmed',
    },
    'highland_vs_lowland_signal': {
        'control': 'Dieng positive control is HIGHLAND forest-garden / religious centre clearing',
        'caveat': 'Not the same geometry as lowland wet-rice intensive clearing',
        'mitigation': 'Mode A forward model parameterised separately for lowland clearing; threshold NOT transferred directly',
        'status': 'BOUNDED',
    },
    'grl_2025_molecular_markers': {
        'note': 'Ruan 2025 GRL finds fire/erosion signal ~3500 BP from E Java marine core',
        'proxy': 'brGDGTs + levoglucosan (NOT pollen + cultigen)',
        'relevance': 'Consistent with Solo marine core; does NOT satisfy pre-registered charcoal+Cerealia diagnostic',
        'treatment': 'Supporting evidence only; does not trigger OUTCOME-2 (wrong proxy)',
        'status': 'NOTED, NOT counted toward outcome',
    },
}

# ── S8: Apply pre-registered rule ─────────────────────────────────────────

def apply_prereg_rule(n_floor, n_central, threshold=0.90, resolve_threshold=0.5):
    """
    Apply the pre-registered 3-outcome rule.
    Returns the outcome label and key supporting data.

    DEFECT 1 FIX (Opus review 2026-06-25): OUTCOME.json previously reported
    n_cores_covering_heartland=1 (J6, geometric RSAP overlap) alongside a
    key_finding claiming "no core covers the heartland" -- a direct self-
    contradiction. The reconciliation is that GEOMETRIC COVERAGE (does the
    core's RSAP polygon reach the heartland?) and RESOLUTION (does the
    resulting diluted signal actually clear the detection threshold?) are
    different questions. J6 (marine Solo) covers but does not resolve: its
    400km RSAP geometrically reaches Brantas (144.6 km away) but catchment
    dilution drives the expected signal ~3 orders of magnitude below
    threshold. We now compute and report BOTH counts explicitly.
    """
    # Run for floor and central estimates, both modes -- capture per-core results
    res_floor_A, p_net_floor_A, res_floor_B, p_net_floor_B   = run_both_modes(n_floor)
    res_central_A, p_net_central_A, res_central_B, p_net_central_B = run_both_modes(n_central)

    # Geometric coverage: does the core's RSAP polygon reach the heartland at all?
    cores_covering_heartland = [c for c in CORES
                                 if min(haversine_km(c['lat'], c['lon'], KEDU_LAT, KEDU_LON),
                                        haversine_km(c['lat'], c['lon'], BRANTAS_LAT, BRANTAS_LON))
                                 <= c['rsap_mid']]

    # Resolution: of the covering cores, which ones actually clear the detection
    # threshold at floor population under Mode A (the more detectable clearing mode)?
    # p_detect >= resolve_threshold means the expected signal is not just nonzero but
    # crosses the calibrated NAP-rise threshold with meaningful confidence.
    covering_ids = {c['id'] for c in cores_covering_heartland}
    cores_resolving_heartland = [r['core_id'] for r in res_floor_A
                                  if r['core_id'] in covering_ids and r['p_detect'] >= resolve_threshold]

    print(f"\nCores with heartland (Kedu/Brantas) geometrically within RSAP: {[c['id'] for c in cores_covering_heartland]}")
    print(f"Of those, cores that actually RESOLVE heartland clearing (p_detect>={resolve_threshold}): {cores_resolving_heartland}")
    print(f"P(network detects | N_floor={n_floor:,}, Mode A): {p_net_floor_A:.4f}")
    print(f"P(network detects | N_floor={n_floor:,}, Mode B): {p_net_floor_B:.4f}")
    print(f"P(network detects | N_central={n_central:,}, Mode A): {p_net_central_A:.4f}")
    print(f"P(network detects | N_central={n_central:,}, Mode B): {p_net_central_B:.4f}")

    # OUTCOME-1 requires P >= 0.90 for Mode A clearing at floor estimate
    if p_net_floor_A >= threshold:
        outcome = 'OUTCOME-1'
        rationale = f'P(detect | N={n_floor:,}, Mode A) = {p_net_floor_A:.3f} >= C={threshold}'
    # OUTCOME-2: positive signal found (not applicable in forward-model analysis)
    elif False:
        outcome = 'OUTCOME-2'
        rationale = 'Pre-400 CE cultigen/charcoal signal detected above threshold'
    else:
        outcome = 'OUTCOME-3'
        rationale = (
            f'P(detect | N={n_floor:,}, Mode A) = {p_net_floor_A:.3f} < C={threshold}. '
            f'{len(cores_covering_heartland)} core(s) geometrically overlap the Kedu/Brantas heartland '
            f'(RSAP polygon reaches it) but {len(cores_resolving_heartland)} core(s) can actually RESOLVE '
            f'heartland clearing above the detection threshold. Coverage is not resolution. '
            f'Instrument is sensitive (Dieng +ctrl, qualitative -- see positive_control_status) '
            f'but the network has a resolution gap at the heartland.'
        )

    return outcome, rationale, {
        'p_net_floor_A': p_net_floor_A,
        'p_net_floor_B': p_net_floor_B,
        'p_net_central_A': p_net_central_A,
        'p_net_central_B': p_net_central_B,
        'n_cores_covering_heartland': len(cores_covering_heartland),
        'cores_covering_heartland': [c['id'] for c in cores_covering_heartland],
        'n_cores_resolving_heartland': len(cores_resolving_heartland),
        'cores_resolving_heartland': cores_resolving_heartland,
        'coverage_vs_resolution_note': (
            "Coverage != resolution. A core's RSAP can geometrically reach the heartland "
            "while still failing to resolve clearing there, if the catchment is large enough "
            "to dilute the signal below the detection threshold (this is exactly J6/marine Solo: "
            "geometric coverage=yes, resolution=no)."
        ),
    }


# ── S8b: Missing-core specification ──────────────────────────────────────

def compute_missing_core_corner_table(N_floor=E196_FLOOR, N_central=E196_CENTRAL, mode='A',
                                       rpp_nap=RPP_NAP_MID, threshold=NAP_THRESHOLD_MID, alpha=0.55):
    """
    DEFECT 4 FIX (Opus review 2026-06-25): the original compute_missing_core_spec()
    reported a single p_detect=1.0 headline built on a HARDCODED, UNCITED
    CONCENTRATION_FACTOR=4.0 (heartland assumed 4x Java-average clearing density).
    That hid a conservative corner where the claim reverses: at FLOOR population +
    UNIFORM (unclustered) clearing, even a perfectly co-located core does NOT clear
    the detection threshold (NAP rise 12.6pp < 17.5pp).

    This function reports the full 2x2 corner table (population floor/central x
    clustering uniform/clustered) instead of cherry-picking the favourable corner.
    concentration_factor=1.0 means clearing is spread uniformly across Java (no
    preferential heartland concentration); concentration_factor=4.0 assumes the
    heartland has ~4x Java-average density (lowland agricultural preference --
    plausible but NOT independently sourced, hence 1.0 is retained as the honest
    lower bound rather than omitted).
    """
    def rise_at_density(density, rpp=rpp_nap, a=alpha):
        f = density
        return a * rpp * f / (rpp * f + (1 - f))

    rows = []
    for pop_label, N in [('floor', N_floor), ('central', N_central)]:
        _, cleared_mid, _ = pop_to_cleared_km2(N, mode)
        for cf_label, cf in [('uniform', 1.0), ('clustered_4x', 4.0)]:
            density = min((cleared_mid / JAVA_AREA_KM2) * cf, 1.0)
            rise = rise_at_density(density)
            p_detect = detect_prob(density * 314, 314, rpp_nap=rpp_nap, threshold=threshold, alpha=alpha)
            rows.append({
                'population_label': pop_label,
                'population_n': N,
                'clustering_label': cf_label,
                'concentration_factor': cf,
                'heartland_clearing_density_pct': round(density * 100, 1),
                'nap_rise_pp': round(rise * 100, 1),
                'threshold_pp': round(threshold * 100, 1),
                'detects': bool(rise >= threshold),
                'p_detect': round(p_detect, 3),
            })
    return rows


def compute_missing_core_spec(N_floor=E196_FLOOR, N_central=E196_CENTRAL, mode='A'):
    """
    Specify the decisive missing core.

    Key geometric insight (retained -- this is real and is the paper's contribution):
    the barrier is NOT lake size but LOCATION. A core placed AT the Kedu/Brantas
    heartland will see local clearing directly, because clearing IS at the heartland,
    regardless of the core's RSAP radius.

    DEFECT 4 FIX: the "decisive" claim is NOT uniform across the honest parameter
    corners (see compute_missing_core_corner_table). This function now reports the
    corner table plus an explicit caveat instead of a single overclaimed p_detect=1.0.
    """
    corners = compute_missing_core_corner_table(N_floor, N_central, mode)
    conservative_corner = next(r for r in corners
                                if r['population_label'] == 'floor' and r['clustering_label'] == 'uniform')
    favourable_corner = next(r for r in corners
                              if r['population_label'] == 'central' and r['clustering_label'] == 'clustered_4x')
    conservative_fails = not conservative_corner['detects']

    return {
        'target_population_floor':    N_floor,
        'target_population_central':  N_central,
        'mode':                        mode,
        'corner_table':                corners,
        'conservative_corner':         'floor population + uniform (unclustered) clearing',
        'conservative_corner_detects': conservative_corner['detects'],
        'conservative_corner_nap_rise_pp': conservative_corner['nap_rise_pp'],
        'favourable_corner':           'central population + 4x clustered clearing',
        'favourable_corner_detects':   favourable_corner['detects'],
        'favourable_corner_nap_rise_pp': favourable_corner['nap_rise_pp'],
        'detection_threshold_pp':      round(NAP_THRESHOLD_MID * 100, 1),
        'KEY_CONSTRAINT':              'LOCATION not lake size — any lake/swamp within 20 km of Kedu/Brantas',
        'required_location':           'Kedu Plain (~-7.5S, 110.0E) OR Brantas headwaters (~-7.8S, 112.0E)',
        'required_max_dist_from_heartland_km': 20,
        'required_lake_radius_km':    '≥1 km (RSAP ≥5 km) — even small lakes suffice if placed at heartland',
        'required_archive_type':      'Closed lowland lake or ox-bow swamp (NOT marine, NOT highland)',
        'required_age_span':          '0-2000 CE (covers 0-500 CE window)',
        'required_resolution_yr':     '≤50 years per sample',
        'required_14c_dates':         '~20 AMS dates for robust 50-yr resolution age model',
        'target_taxa_diagnostic':     'Oryza-type / Cerealia-type / Poaceae + microcharcoal CO-OCCURRENCE',
        'secondary_taxa':             'Trema / Macaranga (pioneer disturbance indicators)',
        'existing_core_nearest':      'J7 Song Gupuh (karst cave, 60 km from Kedu, wrong archive type)',
        'existing_core_gap_km':       60,
        'estimated_cost_usd':         '8,000-15,000 (vibrocore + AMS dating of 20 levels)',
        'why_decisive_CAVEATED': (
            f"A core at Kedu/Brantas would settle the question at CENTRAL population "
            f"({N_central:,}) under either clustering assumption, and at FLOOR population "
            f"({N_floor:,}) IF clearing was spatially clustered (4x concentration). "
            f"It does NOT settle the question at floor population + uniform clearing "
            f"(NAP rise {conservative_corner['nap_rise_pp']}pp < {round(NAP_THRESHOLD_MID*100,1)}pp threshold) "
            f"-- that residual stays open and passes to the dispersed-mode channel (E215). "
            f"This caveat must appear in the abstract, not only here."
        ),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("E216: Paleo-Ecological Interferometer")
    print("S3-S6: Forward Model + Detection Function + Two-Mode Separation")
    print("=" * 70)

    # ── S3: Population → cleared area table ──────────────────────────────
    print("\n--- S3: Population → Cleared Area (E196 coefficients) ---")
    for N in [E196_FLOOR, E196_CENTRAL]:
        for mode in ['A', 'B']:
            lo, mid, hi = pop_to_cleared_km2(N, mode)
            print(f"  N={N:,}, Mode {mode}: cleared area = {lo:.0f}-{hi:.0f} km² (mid: {mid:.0f})")

    # ── S4/S5: Detection function — all cores ─────────────────────────────
    print("\n--- S4/S5: Detection function (Mode A, N=floor) ---")
    res_a, p_net_a = run_detection_analysis(E196_FLOOR, mode='A')
    for r in res_a:
        ctrl = "[+CTRL]" if r['positive_control'] else ""
        heartland = "[HEARTLAND IN RSAP!]" if r['heartland_in_rsap'] else "[OUTSIDE RSAP]"
        print(f"  {r['core_id']:3s} {r['core_name']:30s} {heartland:25s} "
              f"NAP_rise={r['expected_nap_rise']:.3f} P={r['p_detect']:.3f} {ctrl}")
    print(f"  Network P(detect | Mode A, N={E196_FLOOR:,}): {p_net_a:.4f}")

    print("\n--- S6: Two-mode separation ---")
    res_a, p_net_a, res_b, p_net_b = run_both_modes(E196_FLOOR)
    print(f"  Mode A (landscape clearing): P(network) = {p_net_a:.4f}")
    print(f"  Mode B (dispersed forest-garden): P(network) = {p_net_b:.4f}")
    print(f"  Mode B residual → explicitly handed to E215 (phytolith/starch channel)")

    # ── S7: Confound controls (print) ─────────────────────────────────────
    print("\n--- S7: Confound controls ---")
    for k, v in CONFOUND_ANALYSIS.items():
        print(f"  [{v['status']}] {k}")

    # ── S8: Apply pre-registered rule ─────────────────────────────────────
    print("\n--- S8: Apply pre-registered decision rule ---")
    outcome, rationale, stats = apply_prereg_rule(E196_FLOOR, E196_CENTRAL, threshold=0.90)
    print(f"\n  *** OUTCOME: {outcome} ***")
    print(f"  Rationale: {rationale}")

    # ── Missing-core specification ─────────────────────────────────────────
    print("\n--- Missing-Core Specification (OUTCOME-3 deliverable) ---")
    spec = compute_missing_core_spec(E196_FLOOR, E196_CENTRAL, mode='A')
    for k, v in spec.items():
        print(f"  {k}: {v}")

    # ── Save outputs ──────────────────────────────────────────────────────
    # Detection probability table
    rows_a, _ = run_detection_analysis(E196_FLOOR, 'A')
    rows_b, _ = run_detection_analysis(E196_FLOOR, 'B')
    rows_a_c, _ = run_detection_analysis(E196_CENTRAL, 'A')
    rows_b_c, _ = run_detection_analysis(E196_CENTRAL, 'B')
    all_rows = rows_a + rows_b + rows_a_c + rows_b_c

    with open(OUT / "detection_probability_table.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Missing-core corner table (Defect 4 fix) — save as its own CSV for transparency
    corner_rows = compute_missing_core_corner_table(E196_FLOOR, E196_CENTRAL, mode='A')
    with open(OUT / "missing_core_corner_table.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=corner_rows[0].keys())
        writer.writeheader()
        writer.writerows(corner_rows)

    # Outcome record (Defects 1 and 3 fixed: coverage-vs-resolution distinction;
    # positive control downgraded from overstated "CONFIRMED" to honest "qualitative import")
    outcome_data = {
        'experiment': 'E216',
        'date': '2026-06-25',
        'defects_fixed_date': '2026-07-07 (Opus review 4 defects — Fable/Sonnet hardening pass)',
        'outcome': outcome,
        'rationale': rationale,
        'stats': stats,
        'positive_control_status': (
            'QUALITATIVE ONLY — NOT re-derived from raw data. The 15-20pp NAP-rise threshold '
            'is imported from the original authors\' interpretation of Dieng ~600 CE clearance '
            '(Pudjoarinto & Cushing 2001) and Rawa Danau ~1770 CE (Yulianto et al. 2005); the raw '
            'pollen count series behind both is paywalled (HTTP 403) and was never independently '
            'extracted. This does not invalidate OUTCOME-3 (see go_no_go_branch below) but means the '
            'threshold is a literature import, not a blind re-derivation (SIG G1).'
        ),
        'go_no_go_branch': (
            'NO-GO, technically. PREREG.md S2 GO/NO-GO gate specifies: GO only if the Dieng signal '
            'magnitude is extractable from published data; NO-GO/direct-OUTCOME-3 if S2 thresholds '
            'cannot be quantified from available sources. Raw data was inaccessible, so this run '
            'hit the NO-GO branch. OUTCOME-3 is therefore independently supported by TWO reasons: '
            '(a) the heartland coverage/resolution gap (geometric+forward-model result) and '
            '(b) the calibration threshold being unextractable from raw data (NO-GO branch). '
            'Report both in any write-up -- do not rely on (a) alone.'
        ),
        'data_access_limitation': 'Raw pollen % from Pudjoarinto 2001 + Yulianto 2005 inaccessible (403); calibration threshold from literature consensus (15-20 pp NAP rise = substantial), not re-derived.',
        'key_finding': (
            f"{stats['n_cores_covering_heartland']} core(s) geometrically overlap the Kedu/Brantas "
            f"heartland (RSAP reaches it) -- core J6 (marine Solo) is the case: its 400km RSAP "
            f"reaches Brantas (144.6 km away). But {stats['n_cores_resolving_heartland']} core(s) can "
            f"actually RESOLVE heartland clearing above the detection threshold: catchment dilution "
            f"drives J6's expected signal ~3 orders of magnitude below threshold. Coverage is not "
            f"resolution. The instrument IS sensitive at the cores that DO have direct local RSAP "
            f"(Dieng qualitative positive control), but no existing core combines heartland proximity "
            f"with a resolving (non-diluted) archive type."
        ),
        'mode_b_residual': 'Dispersed forest-garden/arboriculture population is outside detection range of ALL cores including marine (Solo). This residual is the explicitly-defined E215 target.',
        'grl_2025_note': 'Ruan et al. 2025 GRL finds fire/erosion markers ~3500 BP in E Java marine core — consistent with pre-400 CE human activity, but uses molecular markers (brGDGTs/levoglucosan), NOT charcoal+Cerealia diagnostic; therefore does not trigger OUTCOME-2 per pre-registration.',
    }
    with open(OUT / "OUTCOME.json", 'w', encoding='utf-8') as f:
        json.dump(outcome_data, f, indent=2)

    # Missing-core spec (Defect 4 fixed: corner table + caveat, no single overclaimed p=1.0)
    spec = compute_missing_core_spec(E196_FLOOR, E196_CENTRAL, mode='A')
    with open(OUT / "missing_core_spec.json", 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2)

    print("\nOutputs saved to results/")
    print(f"\n{'='*70}")
    print(f"VERDICT: {outcome}")
    print(f"The Java palaeoecological network has cores whose RSAP GEOMETRICALLY reaches")
    print(f"the heartland (n={stats['n_cores_covering_heartland']}) but none that RESOLVES it")
    print(f"(n={stats['n_cores_resolving_heartland']}) -- coverage != resolution. This is the paper's headline.")
    print(f"Missing-core claim is CAVEATED: fails at floor population + uniform clearing.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
