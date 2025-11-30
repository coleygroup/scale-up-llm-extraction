#!/usr/bin/env python
"""
Scale-up problem/solution classifier + verifier with high-fidelity prompts.

- Defines a maximal-resolution ontology of process-chemistry scale-up fixes.
- Uses GPT responses.parse + Pydantic for structured classification.
- Includes a second QA agent (verifier) that can override misclassifications.
- Provides an optional CLI for batch processing JSONL files with checkpointing.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
from openai import APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Category ontology
# ---------------------------------------------------------------------------

CategoryLiteral = Literal[
    # Reagent, catalyst, oxidant, reductant
    "change_of_primary_reagent",
    "change_of_oxidant_or_oxidation_protocol",
    "change_of_reductant_or_reduction_protocol",
    "change_of_catalyst_or_ligand_identity",
    "optimization_of_catalyst_loading",
    "replacement_of_unstable_or_impure_reagent",
    "replacement_of_costly_or_low_purity_reagent",
    "reagent_change_to_reduce_waste_or_increase_green_score",
    # Base / acid / pH
    "change_of_base",
    "change_of_acid",
    "optimization_of_pH_or_buffer_conditions",
    "optimization_of_acid_or_base_stoichiometry",
    # Solvent / solubility / phase
    "solvent_change_for_reactivity",
    "solvent_change_for_solubility",
    "solvent_change_for_safety_or_green_chemistry",
    "solvent_change_to_reduce_side_products",
    "solvent_system_optimization_or_cosolvent_addition",
    "phase_behavior_or_biphasic_system_optimization",
    # Additives
    "introduction_of_additive_to_enable_reaction",
    "introduction_of_additive_to_improve_selectivity",
    "quench_or_scavenger_addition_optimization",
    "removal_or_simplification_of_additives",
    # Stoichiometry / T / P / time
    "optimization_of_reagent_stoichiometry",
    "change_of_addition_order_or_feed_profile",
    "temperature_setpoint_or_temperature_ramp_optimization",
    "pressure_or_gas_headspace_optimization",
    "hold_time_or_residence_time_optimization",
    # Protecting groups / substrate / FG management
    "protecting_group_change_or_removal",
    "substrate_scaffold_change",
    "functional_group_change_to_avoid_instability",
    "substrate_change_to_improve_purification",
    "route_change_to_rescue_selectivity_or_stability",
    # Workup / extraction / purification
    "improved_extraction_or_phase_separation",
    "improved_quench_or_neutralization",
    "improved_impurity_purge_or_byproduct_control",
    "filtration_or_solid_liquid_separation_improvement",
    "centrifugation_or_clarification_optimization",
    "drying_or_isolation_process_optimization",
    # Crystallization / solid-form
    "crystallization_solvent_or_antisolvent_optimization",
    "crystallization_nucleation_or_seed_control",
    "polymorph_or_solid_form_management",
    # Impurities, feedstock, QC
    "feedstock_or_supplier_variability_management",
    "tolerance_of_reagent_or_feed_impurities",
    "improved_in_process_control_or_analytical_method",
    "adjustment_of_release_criteria_or_quality_specifications",
    # Special chemistries
    "optimization_of_halogenation_protocol",
    "optimization_of_fluorination_protocol",
    "optimization_of_chlorination_or_activation_protocol",
    "optimization_of_peptide_coupling_or_amidation",
    "optimization_of_reductive_or_hydrogenolysis_protocol",
    "optimization_of_formylation_or_acylation_protocol",
    "biocatalytic_or_enzymatic_route_change_or_optimization",
    # Reactor / equipment / mixing
    "reactor_scale_change_without_changing_chemistry",
    "reactor_geometry_or_hold_up_optimization",
    "improved_mixing_or_impeller_design",
    "improved_heat_transfer_or_temperature_control_hardware",
    "improved_photochemical_reactor_or_light_delivery",
    "improved_electrochemical_reactor_or_electrode_system",
    "custom_or_novel_reactor_design_to_reduce_cost_or_risk",
    # Flow / solids / clogging / gas handling
    "batch_to_flow_or_flow_process_implementation",
    "packed_bed_or_fixed_bed_optimization",
    "clogging_or_fouling_mitigation",
    "slurry_handling_or_particle_size_optimization",
    "gas_evolution_or_mass_transfer_management",
    "continuous_process_intensification",
    # Atmosphere / gas environment / safety
    "optimization_of_inert_atmosphere_or_deoxygenation",
    "optimization_of_oxygen_or_gas_enriched_environment",
    "condition_change_to_reduce_or_remove_process_hazard",
    "thermal_runaway_mitigation_or_quench_safety_modification",
    # Screening, HTE, modeling
    "reaction_condition_screening_or_HTE",
    "design_of_experiments_or_multivariate_optimization",
    "kinetic_or_mechanistic_modeling_guided_change",
    "process_modeling_or_digital_twin_guided_change",
    # Route / step restructuring
    "step_reordering_or_telescope_strategy",
    "entirely_new_route_or_step_replacement",
    # Final catchall
    "insufficient_information_to_classify",
    "other",
]

# Deterministic order for prompt construction
CATEGORY_ORDER: List[CategoryLiteral] = [
    "change_of_primary_reagent",
    "change_of_oxidant_or_oxidation_protocol",
    "change_of_reductant_or_reduction_protocol",
    "change_of_catalyst_or_ligand_identity",
    "optimization_of_catalyst_loading",
    "replacement_of_unstable_or_impure_reagent",
    "replacement_of_costly_or_low_purity_reagent",
    "reagent_change_to_reduce_waste_or_increase_green_score",
    "change_of_base",
    "change_of_acid",
    "optimization_of_pH_or_buffer_conditions",
    "optimization_of_acid_or_base_stoichiometry",
    "solvent_change_for_reactivity",
    "solvent_change_for_solubility",
    "solvent_change_for_safety_or_green_chemistry",
    "solvent_change_to_reduce_side_products",
    "solvent_system_optimization_or_cosolvent_addition",
    "phase_behavior_or_biphasic_system_optimization",
    "introduction_of_additive_to_enable_reaction",
    "introduction_of_additive_to_improve_selectivity",
    "quench_or_scavenger_addition_optimization",
    "removal_or_simplification_of_additives",
    "optimization_of_reagent_stoichiometry",
    "change_of_addition_order_or_feed_profile",
    "temperature_setpoint_or_temperature_ramp_optimization",
    "pressure_or_gas_headspace_optimization",
    "hold_time_or_residence_time_optimization",
    "protecting_group_change_or_removal",
    "substrate_scaffold_change",
    "functional_group_change_to_avoid_instability",
    "substrate_change_to_improve_purification",
    "route_change_to_rescue_selectivity_or_stability",
    "improved_extraction_or_phase_separation",
    "improved_quench_or_neutralization",
    "improved_impurity_purge_or_byproduct_control",
    "filtration_or_solid_liquid_separation_improvement",
    "centrifugation_or_clarification_optimization",
    "drying_or_isolation_process_optimization",
    "crystallization_solvent_or_antisolvent_optimization",
    "crystallization_nucleation_or_seed_control",
    "polymorph_or_solid_form_management",
    "feedstock_or_supplier_variability_management",
    "tolerance_of_reagent_or_feed_impurities",
    "improved_in_process_control_or_analytical_method",
    "adjustment_of_release_criteria_or_quality_specifications",
    "optimization_of_halogenation_protocol",
    "optimization_of_fluorination_protocol",
    "optimization_of_chlorination_or_activation_protocol",
    "optimization_of_peptide_coupling_or_amidation",
    "optimization_of_reductive_or_hydrogenolysis_protocol",
    "optimization_of_formylation_or_acylation_protocol",
    "biocatalytic_or_enzymatic_route_change_or_optimization",
    "reactor_scale_change_without_changing_chemistry",
    "reactor_geometry_or_hold_up_optimization",
    "improved_mixing_or_impeller_design",
    "improved_heat_transfer_or_temperature_control_hardware",
    "improved_photochemical_reactor_or_light_delivery",
    "improved_electrochemical_reactor_or_electrode_system",
    "custom_or_novel_reactor_design_to_reduce_cost_or_risk",
    "batch_to_flow_or_flow_process_implementation",
    "packed_bed_or_fixed_bed_optimization",
    "clogging_or_fouling_mitigation",
    "slurry_handling_or_particle_size_optimization",
    "gas_evolution_or_mass_transfer_management",
    "continuous_process_intensification",
    "optimization_of_inert_atmosphere_or_deoxygenation",
    "optimization_of_oxygen_or_gas_enriched_environment",
    "condition_change_to_reduce_or_remove_process_hazard",
    "thermal_runaway_mitigation_or_quench_safety_modification",
    "reaction_condition_screening_or_HTE",
    "design_of_experiments_or_multivariate_optimization",
    "kinetic_or_mechanistic_modeling_guided_change",
    "process_modeling_or_digital_twin_guided_change",
    "step_reordering_or_telescope_strategy",
    "entirely_new_route_or_step_replacement",
    "insufficient_information_to_classify",
    "other",
]

# Short definitions for the ontology used in prompts.
CATEGORY_DEFINITIONS: Dict[str, str] = {
    "change_of_primary_reagent": "Replacement or redesign of the main electrophile, nucleophile, or coupling partner while keeping the overall transformation conceptually similar.",
    "change_of_oxidant_or_oxidation_protocol": "Change in oxidant identity, stoichiometry, or oxidation protocol (e.g., one-pot vs stepwise) aimed at improving yield, robustness, or impurity profile.",
    "change_of_reductant_or_reduction_protocol": "Change in reductant identity or reduction protocol used to improve conversion, selectivity, or safety.",
    "change_of_catalyst_or_ligand_identity": "Switch to a different catalyst or ligand family to address activity, selectivity, or robustness issues.",
    "optimization_of_catalyst_loading": "Adjusting catalyst loading (up or down) while keeping catalyst identity fixed to reach robust performance or reduce cost.",
    "replacement_of_unstable_or_impure_reagent": "Replacement of a reagent that is chemically unstable, decomposes in storage, or is supplied with unacceptable impurity levels.",
    "replacement_of_costly_or_low_purity_reagent": "Replacement of a very expensive or low-purity reagent by a more economical or higher quality alternative without fundamentally changing the step.",
    "reagent_change_to_reduce_waste_or_increase_green_score": "Reagent change primarily motivated by EHS, waste, or green chemistry considerations rather than yield alone.",
    "change_of_base": "Change in base identity (e.g., organic vs inorganic, stronger vs weaker) to address conversion, side reactions, or stability.",
    "change_of_acid": "Change in acid identity (Brønsted or Lewis) used to catalyze or quench the transformation.",
    "optimization_of_pH_or_buffer_conditions": "Adjusting pH or buffer composition as a continuous variable, rather than simply swapping discrete acids or bases.",
    "optimization_of_acid_or_base_stoichiometry": "Tuning the equivalents of acid or base used while keeping identity fixed.",
    "solvent_change_for_reactivity": "Switch in solvent driven primarily by observed changes to rate, conversion, or selectivity of the reaction.",
    "solvent_change_for_solubility": "Switch in solvent driven primarily by solubility/phase issues (slurries, crystallization in reactor, etc.) rather than intrinsic reactivity.",
    "solvent_change_for_safety_or_green_chemistry": "Solvent change driven mainly by hazard, environmental, or regulatory considerations.",
    "solvent_change_to_reduce_side_products": "Solvent change that specifically reduces formation of a particular impurity or side product.",
    "solvent_system_optimization_or_cosolvent_addition": "Fine-tuning of mixed solvent systems or addition of minor cosolvents to balance solubility, reactivity, or workup.",
    "phase_behavior_or_biphasic_system_optimization": "Design or optimization of biphasic (e.g., aqueous/organic) systems, including salting-out, phase ratio, and mixing.",
    "introduction_of_additive_to_enable_reaction": "Addition of a new additive required to make the reaction proceed at all or avoid complete failure.",
    "introduction_of_additive_to_improve_selectivity": "Addition of a new additive primarily intended to improve chemo-, regio-, or stereoselectivity.",
    "quench_or_scavenger_addition_optimization": "Use or optimization of quench reagents or scavengers that selectively remove reactive intermediates or metal residues.",
    "removal_or_simplification_of_additives": "Removing unnecessary additives, consolidating multiple additives, or simplifying the formulation without loss of performance.",
    "optimization_of_reagent_stoichiometry": "General tuning of equivalents of one or more reagents (not acids/bases specifically) to improve performance or robustness.",
    "change_of_addition_order_or_feed_profile": "Changing the order of additions, addition mode, or feed profile (e.g., semi-batch, controlled feed, split charges).",
    "temperature_setpoint_or_temperature_ramp_optimization": "Adjusting temperature setpoints or ramps (e.g., cooling or heating profiles) to manage rate, selectivity, or safety.",
    "pressure_or_gas_headspace_optimization": "Changing total pressure, gas type, or headspace to manage gas solubility, mass transfer, or safety.",
    "hold_time_or_residence_time_optimization": "Tuning batch hold times or continuous-flow residence times to achieve robust completion or avoid overreaction.",
    "protecting_group_change_or_removal": "Changing or removing protecting groups to simplify synthesis or avoid instability without redesigning the core route.",
    "substrate_scaffold_change": "Changing the core scaffold or backbone of the substrate while maintaining overall functional intent.",
    "functional_group_change_to_avoid_instability": "Swapping or relocating functional groups to avoid instability, decomposition, or incompatibility.",
    "substrate_change_to_improve_purification": "Changing substrate identity to enable easier isolation or purification (e.g., better crystallization behavior).",
    "route_change_to_rescue_selectivity_or_stability": "Redesign of the route or step sequence to eliminate problematic intermediates or selectivity failures.",
    "improved_extraction_or_phase_separation": "Changes to extraction or phase separation operations to reduce emulsion, improve phase split, or simplify workup.",
    "improved_quench_or_neutralization": "Optimization of the quench or neutralization protocol to improve safety or downstream processing.",
    "improved_impurity_purge_or_byproduct_control": "Specific changes aimed at improving purge of an identified impurity or suppressing a known byproduct.",
    "filtration_or_solid_liquid_separation_improvement": "Improvements to filtration, cake washing, or other solid/liquid separation steps.",
    "centrifugation_or_clarification_optimization": "Process changes involving centrifugation, decantation, or clarification hardware or conditions.",
    "drying_or_isolation_process_optimization": "Changes to drying method, drying conditions, or isolation protocol (e.g., spin-drying, tray vs vacuum).",
    "crystallization_solvent_or_antisolvent_optimization": "Tuning crystallization solvent, antisolvent, or volumes to control yield and impurity levels.",
    "crystallization_nucleation_or_seed_control": "Changes in seeding, nucleation temperature, or seeding protocol to control crystal habit or size.",
    "polymorph_or_solid_form_management": "Interventions specifically targeting polymorph control, solvate form, or amorphous vs crystalline states.",
    "feedstock_or_supplier_variability_management": "Strategies to handle variability in raw material quality from different suppliers or lots.",
    "tolerance_of_reagent_or_feed_impurities": "Deliberate redesign to tolerate higher levels of known impurities in feeds without additional purification.",
    "improved_in_process_control_or_analytical_method": "Development or improvement of analytical methods or IPCs to track and control the process.",
    "adjustment_of_release_criteria_or_quality_specifications": "Changing or tightening quality specifications, including analytical acceptance criteria.",
    "optimization_of_halogenation_protocol": "Specific improvements to halogenation steps (not just solvent or reagent changes) to control regioselectivity or safety.",
    "optimization_of_fluorination_protocol": "Specific improvements to fluorination steps to improve selectivity, yield, or robustness.",
    "optimization_of_chlorination_or_activation_protocol": "Optimization of chlorination or other activation steps (e.g., acid chlorides, sulfonyl chlorides).",
    "optimization_of_peptide_coupling_or_amidation": "Improvements to peptide coupling, amide formation, or related condensation chemistries.",
    "optimization_of_reductive_or_hydrogenolysis_protocol": "Improvements to hydrogenolysis or reductive transformations, often involving H2 gas or catalysts.",
    "optimization_of_formylation_or_acylation_protocol": "Optimization of formylation or acylation steps (e.g., Vilsmeier, acyl chloride, anhydride conditions).",
    "biocatalytic_or_enzymatic_route_change_or_optimization": "Introduction or optimization of a biocatalytic or enzymatic transformation as part of the route.",
    "reactor_scale_change_without_changing_chemistry": "Scaling up reactor volume or throughput without major changes to chemistry or conditions.",
    "reactor_geometry_or_hold_up_optimization": "Changes to reactor geometry, internal volume, or hold-up that materially affect performance.",
    "improved_mixing_or_impeller_design": "Changes to mixing hardware or agitation strategy to address mixing-limited behavior.",
    "improved_heat_transfer_or_temperature_control_hardware": "Hardware-level improvements to heat transfer or temperature control (e.g., jackets, coils).",
    "improved_photochemical_reactor_or_light_delivery": "Improved light delivery, reactor design, or LED configuration for photochemical steps.",
    "improved_electrochemical_reactor_or_electrode_system": "Improvements to electrochemical cell design, electrode materials, or current distribution.",
    "custom_or_novel_reactor_design_to_reduce_cost_or_risk": "Use of custom reactor hardware specifically to reduce cost, risk, or complexity vs standard equipment.",
    "batch_to_flow_or_flow_process_implementation": "Transition from batch to continuous flow operation, or significant redesign within flow.",
    "packed_bed_or_fixed_bed_optimization": "Optimization of packed-bed or fixed-bed reactors including packing material, size, or configuration.",
    "clogging_or_fouling_mitigation": "Changes specifically designed to mitigate clogging, fouling, or plugging phenomena.",
    "slurry_handling_or_particle_size_optimization": "Changes focusing on particle size, slurry stability, or handling of solids in suspension.",
    "gas_evolution_or_mass_transfer_management": "Changes aimed at managing gas evolution, gas–liquid mass transfer, or venting.",
    "continuous_process_intensification": "Process changes that significantly intensify throughput or space-time-yield while maintaining control.",
    "optimization_of_inert_atmosphere_or_deoxygenation": "Improvements to inert gas control, deoxygenation, or oxygen exclusion.",
    "optimization_of_oxygen_or_gas_enriched_environment": "Improvements involving controlled oxygen or other reactive gas environments.",
    "condition_change_to_reduce_or_remove_process_hazard": "Changes to conditions specifically targeting reduction of explosion, fire, or acute hazard risk.",
    "thermal_runaway_mitigation_or_quench_safety_modification": "Changes aimed at controlling exotherms or mitigating thermal runaway risk during quench or reaction.",
    "reaction_condition_screening_or_HTE": "Use of systematic condition screening (e.g., HTE, plates) to identify better conditions.",
    "design_of_experiments_or_multivariate_optimization": "Use of DOE or multivariate analysis to optimize several parameters simultaneously.",
    "kinetic_or_mechanistic_modeling_guided_change": "Changes guided primarily by kinetic or mechanistic modeling insight.",
    "process_modeling_or_digital_twin_guided_change": "Changes guided by process models, digital twins, or flowsheet simulations.",
    "step_reordering_or_telescope_strategy": "Reordering steps or telescoping operations without fundamentally changing their chemistry.",
    "entirely_new_route_or_step_replacement": "Introduction of a new synthetic route or complete replacement of a problematic step.",
    "insufficient_information_to_classify": "Cases where the text does not provide enough information to assign a more specific category.",
    "other": "Cases that are clearly out of scope of all defined categories.",
}


def build_taxonomy_block() -> str:
    """Construct a human-readable taxonomy block for use in system prompts."""
    lines: List[str] = []
    lines.append("Here is the complete list of allowed categories.\n")
    for cat in CATEGORY_ORDER:
        desc = CATEGORY_DEFINITIONS.get(cat, "")
        lines.append(f"- {cat}\n  - Definition: {desc}\n")
    return "\n".join(lines)


TAXONOMY_BLOCK = build_taxonomy_block()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ScaleupClassification(BaseModel):
    """
    Single-category classification of a scale-up problem/solution pair.
    """

    category: CategoryLiteral
    description: str  # concise summary of problem + solution together
    confidence: Literal["low", "medium", "high"]
    reasoning: str  # 1–2 sentence justification for the category choice


class VerificationResult(BaseModel):
    """
    QA checker output for a scale-up classification.
    """

    agree: bool
    corrected_category: CategoryLiteral
    corrected_description: str
    confidence: Literal["low", "medium", "high"]
    comment: Optional[str] = None  # short explanation if disagreeing


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_shared_client: Optional[OpenAI] = None


def get_shared_client() -> OpenAI:
    """
    Shared OpenAI client, configured via OPENAI_API_KEY.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key)
    return _client

# ---------------------------------------------------------------------------
# Classifier agent
# ---------------------------------------------------------------------------


def classify_scaleup_problem_solution(
    problem: str,
    solution: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-5.1",
) -> ScaleupClassification:
    """
    Primary classifier: assign a single category + description to a scale-up problem/solution pair.
    """
    if client is None:
        client = get_shared_client()

    system_msg = (
        "You are an expert process chemist and text annotator.\n"
        "You receive two short passages from a process scale-up manuscript:\n"
        "  - problem: describes the scale-up problem.\n"
        "  - solution: describes how the chemists solved it.\n\n"
        "Your task is to assign a SINGLE best category from a fixed ontology and to produce "
        "a concise English description summarizing problem + solution.\n\n"
        "IMPORTANT PRINCIPLES:\n"
        "1) Choose EXACTLY ONE category from the provided ontology.\n"
        "2) Do NOT invent, rename, or combine categories.\n"
        "3) Classify based on the PRIMARY modification described in the solution.\n"
        "4) If multiple changes are mentioned, select the category that best reflects the main "
        "   chemical or process lever that solved the problem.\n"
        "5) If the text does not provide enough detail to choose a specific category, use "
        "   'insufficient_information_to_classify'.\n"
        "6) Only use 'other' if the case is clearly out of scope of ALL defined categories.\n\n"
        "DECISION STEPS:\n"
        "1) Identify the root issue being addressed (e.g., low yield, impurity, safety, clogging).\n"
        "2) Identify the dominant change in the solution (reagent, solvent, base, temperature, "
        "   stoichiometry, crystallization, reactor hardware, flow vs batch, etc.).\n"
        "3) Map that dominant change to the category whose definition best matches.\n"
        "4) If more than one category could apply, prefer the most specific category and avoid generic buckets.\n\n"
        f"{TAXONOMY_BLOCK}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "You MUST output a JSON object matching the following schema:\n"
        "  - category: string, one of the category names above.\n"
        "  - description: 1–3 sentences summarizing the problem and solution together.\n"
        "  - confidence: one of 'low', 'medium', 'high'.\n"
        "  - reasoning: 1–2 sentences explaining why the chosen category fits the text.\n"
        "The JSON must contain ONLY these four fields.\n"
    )

    user_payload = {
        "problem": problem,
        "solution": solution,
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False)

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        reasoning={"effort": "medium"},
        text={"verbosity": "low"},
        text_format=ScaleupClassification,
    )

    return response.output_parsed


# ---------------------------------------------------------------------------
# Verifier agent
# ---------------------------------------------------------------------------


def verify_scaleup_classification(
    problem: str,
    solution: str,
    primary: ScaleupClassification,
    client: Optional[OpenAI] = None,
    model: str = "gpt-5.1",
) -> VerificationResult:
    """
    QA checker: confirms or corrects a classification from the primary agent.
    """
    if client is None:
        client = get_shared_client()

    system_msg = (
        "You are an expert process chemist acting as a strict QA checker for a classification model.\n"
        "You are given:\n"
        "  - problem: a scale-up problem.\n"
        "  - solution: how the authors fixed it.\n"
        "  - primary_category: the category chosen by another model.\n"
        "  - primary_description: that model's concise summary of problem + solution.\n"
        "  - primary_reasoning: that model's explanation of its choice.\n\n"
        "Your job is to decide whether the primary_category and primary_description are correct "
        "and consistent with the ontology and the text. If they are not, you MUST correct them.\n\n"
        "RULES:\n"
        "1) You MUST choose corrected_category from the SAME ontology as the classifier.\n"
        "2) If you fully agree with the classifier, set agree=true and copy:\n"
        "     corrected_category     = primary_category\n"
        "     corrected_description  = primary_description\n"
        "3) If you partially or fully disagree, set agree=false and choose your own\n"
        "   corrected_category and corrected_description based on the ontology.\n"
        "4) Be strict: if the chosen category does NOT clearly match the definition, override it.\n"
        "5) Use comment (one sentence) to explain why you disagreed, if agree=false.\n"
        "6) If you cannot choose a more specific category with reasonable confidence, you may use "
        "   'insufficient_information_to_classify' (for missing detail) or 'other' (for out-of-scope).\n\n"
        "ONTOLOGY (same as classifier):\n"
        f"{TAXONOMY_BLOCK}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "You MUST output a JSON object matching the following schema:\n"
        "  - agree: boolean, true if you accept the primary classification, false otherwise.\n"
        "  - corrected_category: string, one of the category names above.\n"
        "  - corrected_description: 1–3 sentences summarizing problem + solution in your own words.\n"
        "  - confidence: one of 'low', 'medium', 'high' reflecting your confidence in the corrected_category.\n"
        "  - comment: optional string (or null) with at most one sentence explaining your judgment.\n"
        "The JSON must contain ONLY these five fields.\n"
    )

    user_payload = {
        "problem": problem,
        "solution": solution,
        "primary_category": primary.category,
        "primary_description": primary.description,
        "primary_reasoning": primary.reasoning,
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False)

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        reasoning={"effort": "medium"},
        text={"verbosity": "low"},
        text_format=VerificationResult,
    )

    return response.output_parsed


# ---------------------------------------------------------------------------
# Parallel runner with checkpointing (optional CLI-style usage)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    row_index: int
    primary: ScaleupClassification
    verification: VerificationResult


@dataclass
class ClassificationError:
    row_index: int
    error_type: str
    error_message: str


def classify_row(
    idx: int,
    row: pd.Series,
    max_retries: int = 5,
    backoff_initial: float = 1.0,
    model_classifier: str = "gpt-5.1",
    model_verifier: str = "gpt-5.1",
) -> Tuple[int, Optional[ClassificationResult], Optional[ClassificationError]]:
    """
    Worker function for ThreadPoolExecutor: classify + verify a single row.
    Returns either a ClassificationResult or a ClassificationError.
    """
    client = get_shared_client()
    backoff = backoff_initial

    problem = str(row["Problem_Description"])
    solution = str(row["Solution_Description"])

    for attempt in range(max_retries):
        try:
            primary = classify_scaleup_problem_solution(
                problem=problem,
                solution=solution,
                client=client,
                model=model_classifier,
            )
            verification = verify_scaleup_classification(
                problem=problem,
                solution=solution,
                primary=primary,
                client=client,
                model=model_verifier,
            )
            return idx, ClassificationResult(idx, primary, verification), None

        except (RateLimitError, APIStatusError) as e:
            if attempt == max_retries - 1:
                return idx, None, ClassificationError(
                    row_index=idx,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            time.sleep(backoff)
            backoff *= 2.0
        except Exception as e:
            return idx, None, ClassificationError(
                row_index=idx,
                error_type=type(e).__name__,
                error_message=str(e),
            )

    # Should not get here
    return idx, None, ClassificationError(
        row_index=idx,
        error_type="UnknownError",
        error_message="Exhausted retries without returning.",
    )


def run_batch_classification(
    df: pd.DataFrame,
    mask_column: Optional[str] = None,
    mask_substring: Optional[str] = None,
    max_workers: int = 6,
    output_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Run classification + verification over a DataFrame with optional filtering.

    Args:
        df: DataFrame containing at least 'Problem_Description' and 'Solution_Description'.
        mask_column: optional column to filter rows on.
        mask_substring: if provided, only rows where mask_column contains this substring are processed.
        max_workers: thread pool size.
        output_prefix: prefix for output JSONL files; if None, a timestamped prefix is used.

    Returns:
        (results_path, errors_path)
    """
    if mask_column and mask_substring is not None:
        mask = df[mask_column].astype(str).str.contains(mask_substring, regex=False)
        rows_to_process = list(df[mask].iterrows())
    else:
        rows_to_process = list(df.iterrows())

    print(f"Will process {len(rows_to_process)} rows")

    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"scaleup_condition_changes_{timestamp}"

    results_path = f"{output_prefix}_results.jsonl"
    errors_path = f"{output_prefix}_errors.jsonl"

    results_fh = open(results_path, "a", encoding="utf-8")
    errors_fh = open(errors_path, "a", encoding="utf-8")

    n_success = 0
    n_error = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(classify_row, idx, row)
            for idx, row in rows_to_process
        ]

        with tqdm(total=len(futures), desc="Classifying scale-up cases") as pbar:
            for fut in as_completed(futures):
                idx, ok, err = fut.result()

                if ok is not None:
                    primary = ok.primary
                    verify = ok.verification

                    results_fh.write(
                        json.dumps(
                            {
                                "row_index": int(idx),
                                "primary_category": primary.category,
                                "primary_description": primary.description,
                                "primary_confidence": primary.confidence,
                                "primary_reasoning": primary.reasoning,
                                "verifier_agree": verify.agree,
                                "verifier_category": verify.corrected_category,
                                "verifier_description": verify.corrected_description,
                                "verifier_confidence": verify.confidence,
                                "verifier_comment": verify.comment,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    results_fh.flush()
                    n_success += 1

                elif err is not None:
                    errors_fh.write(
                        json.dumps(
                            {
                                "row_index": int(idx),
                                "error_type": err.error_type,
                                "error_message": err.error_message,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    errors_fh.flush()
                    n_error += 1

                pbar.update(1)
                pbar.set_postfix(success=n_success, errors=n_error)

    results_fh.close()
    errors_fh.close()

    print(f"\nFinished. Success: {n_success}, Errors: {n_error}")
    print(f"Results written live to: {results_path}")
    print(f"Errors written live to:  {errors_path}")

    return results_path, errors_path


# ---------------------------------------------------------------------------
# CLI entry point (edit paths as needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scale-up problem/solution classifier + verifier."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL or JSON file with Problem_Description and Solution_Description columns.",
    )
    parser.add_argument(
        "--mask-column",
        type=str,
        default=None,
        help="Optional column name to filter on (e.g., 'Solutions').",
    )
    parser.add_argument(
        "--mask-substring",
        type=str,
        default=None,
        help="Substring to require in mask-column for a row to be processed.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Number of concurrent worker threads.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output JSONL files; if omitted, uses a timestamp.",
    )

    args = parser.parse_args()

    # Load input
    if args.input.endswith(".jsonl"):
        df_in = pd.read_json(args.input, lines=True)
    else:
        df_in = pd.read_json(args.input)

    run_batch_classification(
        df=df_in,
        mask_column=args.mask_column,
        mask_substring=args.mask_substring,
        max_workers=args.max_workers,
        output_prefix=args.output_prefix,
    )