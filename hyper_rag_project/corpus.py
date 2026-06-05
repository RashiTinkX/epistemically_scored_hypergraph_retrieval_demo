"""
corpus.py
─────────
Curated neuroscience passages used to build the demo hypergraph.

Each passage is labelled with:
  - evidence_type  (from the EvidenceType taxonomy)
  - entities       (concepts that become hypergraph nodes)
  - year / source  (for recency + source_authority scoring)
  - sample_n       (for sample_adequacy scoring)

Topic clusters:
  A.  BDNF → synaptic plasticity → learning & memory
  B.  Dopamine → reward prediction error → decision making
  C.  Neuroinflammation → microglia → Alzheimer's disease
  D.  Sleep → hippocampal memory consolidation
  E.  Contradictory / weak evidence (for contrast)
"""

from hypergraph import Passage, EvidenceType

PASSAGES = [

    # ── Cluster A: BDNF / synaptic plasticity / LTP ───────────────────────────

    Passage(
        id="p01",
        text=(
            "Brain-derived neurotrophic factor (BDNF) activates TrkB receptors "
            "at hippocampal synapses, triggering downstream MAPK/ERK signalling "
            "that enhances long-term potentiation (LTP). In a randomised controlled "
            "experiment with 240 adult rats, bilateral infusion of a TrkB antagonist "
            "completely abolished LTP in CA1 and impaired spatial memory acquisition "
            "in the Morris water maze (p < 0.001)."
        ),
        source="Nature Neuroscience",
        evidence_type=EvidenceType.CONTROLLED_EXPERIMENT,
        year=2022,
        sample_n=240,
        entities=["BDNF", "TrkB", "LTP", "hippocampus", "spatial_memory"],
    ),

    Passage(
        id="p02",
        text=(
            "A meta-analysis of 34 studies (total n = 8,412 participants) found "
            "that aerobic exercise consistently elevated serum BDNF levels by "
            "an average of 18 % (95 % CI: 14–22 %) and was associated with "
            "improved episodic memory scores across age groups. Effect sizes were "
            "larger in studies lasting ≥ 12 weeks."
        ),
        source="Neuroscience & Biobehavioral Reviews",
        evidence_type=EvidenceType.META_ANALYSIS,
        year=2023,
        sample_n=8412,
        entities=["BDNF", "aerobic_exercise", "episodic_memory", "hippocampus"],
    ),

    Passage(
        id="p03",
        text=(
            "A cross-sectional survey of 120 elderly volunteers found a weak positive "
            "correlation (r = 0.21, p = 0.04) between self-reported physical activity "
            "and plasma BDNF. The study relied on questionnaire data and did not "
            "measure LTP or hippocampal volume directly."
        ),
        source="Frontiers in Aging Neuroscience",
        evidence_type=EvidenceType.CROSS_SECTIONAL,
        year=2019,
        sample_n=120,
        entities=["BDNF", "aerobic_exercise", "hippocampus"],
    ),

    Passage(
        id="p04",
        text=(
            "Editorial commentary: It is widely believed that BDNF plays a central "
            "role in synaptic plasticity and memory. Several researchers have proposed "
            "that targeting BDNF pathways could be therapeutic for cognitive decline, "
            "though clinical translation remains challenging."
        ),
        source="Trends in Neurosciences (editorial)",
        evidence_type=EvidenceType.SECONDARY_COMMENTARY,
        year=2021,
        sample_n=None,
        entities=["BDNF", "synaptic_plasticity", "cognitive_decline"],
    ),

    # ── Cluster B: Dopamine / reward / decision-making ─────────────────────────

    Passage(
        id="p05",
        text=(
            "Midbrain dopamine neurons in the ventral tegmental area (VTA) encode "
            "reward prediction errors (RPE): they fire above baseline when reward "
            "exceeds expectation and are suppressed when expected reward is omitted. "
            "Single-unit recordings in 18 macaques across 6,000 trials replicated "
            "the RPE signal with high fidelity (R² = 0.91)."
        ),
        source="Science",
        evidence_type=EvidenceType.CONTROLLED_EXPERIMENT,
        year=2021,
        sample_n=18,
        entities=["dopamine", "VTA", "reward_prediction_error", "decision_making"],
    ),

    Passage(
        id="p06",
        text=(
            "Longitudinal fMRI study of 95 healthy adults over 18 months showed "
            "that striatal dopamine release (indexed via [11C]raclopride PET) "
            "predicted individual differences in reinforcement learning rate "
            "(β = 0.43, p < 0.001) and real-world financial decision quality."
        ),
        source="Neuron",
        evidence_type=EvidenceType.LONGITUDINAL,
        year=2022,
        sample_n=95,
        entities=["dopamine", "striatum", "reinforcement_learning", "decision_making"],
    ),

    Passage(
        id="p07",
        text=(
            "A case report describes a patient with a focal VTA lesion who showed "
            "complete insensitivity to monetary rewards on a gambling task. "
            "Neuropsychological testing confirmed intact general cognition. "
            "The authors conclude dopamine is necessary for reward-based choice."
        ),
        source="Neuropsychologia",
        evidence_type=EvidenceType.CASE_REPORT,
        year=2020,
        sample_n=1,
        entities=["dopamine", "VTA", "reward_prediction_error", "decision_making"],
    ),

    # ── Cluster C: Neuroinflammation / microglia / Alzheimer's ────────────────

    Passage(
        id="p08",
        text=(
            "Single-cell RNA sequencing of post-mortem human cortex (n = 48 donors, "
            "24 Alzheimer's / 24 controls) identified a disease-associated microglia "
            "(DAM) subtype upregulating TREM2 and downregulating homeostatic genes. "
            "DAM density correlated with amyloid plaque load (Spearman ρ = 0.74)."
        ),
        source="Cell",
        evidence_type=EvidenceType.OBSERVATIONAL,
        year=2023,
        sample_n=48,
        entities=["microglia", "TREM2", "amyloid", "Alzheimers_disease", "neuroinflammation"],
    ),

    Passage(
        id="p09",
        text=(
            "In a transgenic mouse model of Alzheimer's disease (5xFAD, n = 60 mice), "
            "pharmacological depletion of microglia via CSF1R inhibitor PLX5622 "
            "reduced amyloid plaque burden by 35 % at 6 months but did not improve "
            "spatial memory in the Barnes maze. This suggests microglial removal "
            "has plaque-clearing but not cognitive benefits."
        ),
        source="Journal of Neuroinflammation",
        evidence_type=EvidenceType.CONTROLLED_EXPERIMENT,
        year=2022,
        sample_n=60,
        entities=["microglia", "amyloid", "Alzheimers_disease", "neuroinflammation", "spatial_memory"],
    ),

    Passage(
        id="p10",
        text=(
            "A secondary analysis of data from the ADNI cohort (n = 312) found that "
            "elevated CSF levels of sTREM2, a microglial activation marker, were "
            "associated with slower cognitive decline over 3 years in early-stage "
            "Alzheimer's, suggesting a neuroprotective phase of microglial activation."
        ),
        source="Alzheimer's & Dementia",
        evidence_type=EvidenceType.OBSERVATIONAL,
        year=2023,
        sample_n=312,
        entities=["microglia", "TREM2", "Alzheimers_disease", "cognitive_decline"],
    ),

    # ── Cluster D: Sleep / hippocampal consolidation ──────────────────────────

    Passage(
        id="p11",
        text=(
            "Sharp-wave ripples (SWRs) recorded from hippocampal CA1 in sleeping rats "
            "(n = 30) co-occurred with cortical slow oscillations and spindles in 78 % "
            "of events, consistent with the active systems consolidation hypothesis. "
            "Optogenetic disruption of SWRs during slow-wave sleep impaired novel "
            "object recognition 24 h later (p < 0.001)."
        ),
        source="Nature",
        evidence_type=EvidenceType.CONTROLLED_EXPERIMENT,
        year=2023,
        sample_n=30,
        entities=["sleep", "hippocampus", "sharp_wave_ripples", "memory_consolidation"],
    ),

    Passage(
        id="p12",
        text=(
            "Polysomnographic study of 200 university students found that total slow-wave "
            "sleep duration positively predicted next-day declarative memory retention "
            "(r = 0.38, p < 0.001), while REM sleep predicted procedural memory. "
            "Both findings were replicated in an independent cohort of 185 participants."
        ),
        source="Sleep",
        evidence_type=EvidenceType.OBSERVATIONAL,
        year=2022,
        sample_n=385,
        entities=["sleep", "hippocampus", "memory_consolidation", "declarative_memory"],
    ),

    Passage(
        id="p13",
        text=(
            "It has long been thought that sleep is important for memory. Popular science "
            "accounts often claim that a full 8 hours is necessary for optimal brain "
            "function, though the scientific evidence for a specific duration threshold "
            "is less clear than commonly portrayed."
        ),
        source="Scientific American (blog post)",
        evidence_type=EvidenceType.SECONDARY_COMMENTARY,
        year=2020,
        sample_n=None,
        entities=["sleep", "memory_consolidation"],
    ),

    # ── Cluster E: BDNF contradictory signal ──────────────────────────────────

    Passage(
        id="p14",
        text=(
            "A pre-registered randomised trial (n = 180 older adults, 12-week aerobic "
            "training vs. stretching control) found no significant difference in serum "
            "BDNF between groups (p = 0.41) and no correlation between BDNF change "
            "and memory performance. Authors note possible ceiling effects in healthy "
            "older adults."
        ),
        source="JAMA Neurology",
        evidence_type=EvidenceType.CONTROLLED_EXPERIMENT,
        year=2024,
        sample_n=180,
        entities=["BDNF", "aerobic_exercise", "episodic_memory"],
    ),
]
