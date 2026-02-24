#!/usr/bin/env python3
"""
Streamlined GPT-5.2 Paper Classification Script

Downloads a process chemistry paper, parses the XML, and uses GPT-5.2 to:
1. Identify all reactions in the paper
2. Classify each reaction using the NameRXN taxonomy
3. Extract scale-up problems and solutions for each reaction
"""

import argparse
import enum
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from lxml import etree # pyright: ignore[reportAttributeAccessIssue]
from openai import OpenAI
from pydantic import BaseModel, field_serializer, field_validator


# ============================================================================
# Configuration
# ============================================================================

SERVER_BASE_URL = "http://molgpu02.mit.edu:8999/download"


# ============================================================================
# Problem Taxonomy
# ============================================================================

class Problems(str, enum.Enum):
    SOLUBILITY = "[SOLUBILITY]"
    PURIFICATION = "[PURIFICATION]"
    MIXABILITY = "[MIXABILITY]"
    COST = "[COST]"
    LONG_REACTIONS = "[LONG_REACTIONS]"
    EXOTHERMIC = "[EXOTHERMIC]"
    SAFETY = "[SAFETY]"
    AVAILABILITY = "[AVAILABILITY]"
    VARIABILITY = "[VARIABILITY]"
    LOW_YIELD = "[LOW YIELD]"
    DECOMPOSITION = "[DECOMPOSITION]"


# ============================================================================
# Solution Taxonomy
# ============================================================================

class CategoryLiteral(str, enum.Enum):
    CHANGE_OF_PRIMARY_REAGENT = "change_of_primary_reagent"
    CHANGE_OF_OXIDANT_OR_OXIDATION_PROTOCOL = "change_of_oxidant_or_oxidation_protocol"
    CHANGE_OF_REDUCTANT_OR_REDUCTION_PROTOCOL = "change_of_reductant_or_reduction_protocol"
    CHANGE_OF_CATALYST_OR_LIGAND_IDENTITY = "change_of_catalyst_or_ligand_identity"
    OPTIMIZATION_OF_CATALYST_LOADING = "optimization_of_catalyst_loading"
    REPLACEMENT_OF_UNSTABLE_OR_IMPURE_REAGENT = "replacement_of_unstable_or_impure_reagent"
    REPLACEMENT_OF_COSTLY_OR_LOW_PURITY_REAGENT = "replacement_of_costly_or_low_purity_reagent"
    REAGENT_CHANGE_TO_REDUCE_WASTE_OR_INCREASE_GREEN_SCORE = "reagent_change_to_reduce_waste_or_increase_green_score"
    CHANGE_OF_BASE = "change_of_base"
    CHANGE_OF_ACID = "change_of_acid"
    OPTIMIZATION_OF_ACID_OR_BASE_STOICHIOMETRY = "optimization_of_acid_or_base_stoichiometry"
    SOLVENT_CHANGE_FOR_REACTIVITY = "solvent_change_for_reactivity"
    SOLVENT_CHANGE_FOR_SOLUBILITY = "solvent_change_for_solubility"
    SOLVENT_CHANGE_FOR_SAFETY_OR_GREEN_CHEMISTRY = "solvent_change_for_safety_or_green_chemistry"
    SOLVENT_CHANGE_TO_REDUCE_SIDE_PRODUCTS = "solvent_change_to_reduce_side_products"
    SOLVENT_SYSTEM_OPTIMIZATION_OR_COSOLVENT_ADDITION = "solvent_system_optimization_or_cosolvent_addition"
    PHASE_BEHAVIOR_OR_BIPHASIC_SYSTEM_OPTIMIZATION = "phase_behavior_or_biphasic_system_optimization"
    CHANGE_OF_ADDITIVE = "change_of_additive"
    OPTIMIZATION_OF_REAGENT_STOICHIOMETRY = "optimization_of_reagent_stoichiometry"
    CHANGE_OF_ADDITION_ORDER_OR_FEED_PROFILE = "change_of_addition_order_or_feed_profile"
    TEMPERATURE_SETPOINT_OR_TEMPERATURE_RAMP_OPTIMIZATION = "temperature_setpoint_or_temperature_ramp_optimization"
    PRESSURE_OR_GAS_HEADSPACE_OPTIMIZATION = "pressure_or_gas_headspace_optimization"
    HOLD_TIME_OR_RESIDENCE_TIME_OPTIMIZATION = "hold_time_or_residence_time_optimization"
    PROTECTING_GROUP_CHANGE_OR_REMOVAL = "protecting_group_change_or_removal"
    IMPROVED_QUENCH_OR_NEUTRALIZATION = "improved_quench_or_neutralization"
    IMPROVED_IMPURITY_PURGE_OR_BYPRODUCT_CONTROL = "improved_impurity_purge_or_byproduct_control"
    DRYING_OR_ISOLATION_PROCESS_OPTIMIZATION = "drying_or_isolation_process_optimization"
    CRYSTALLIZATION_SOLVENT_OR_ANTISOLVENT_OPTIMIZATION = "crystallization_solvent_or_antisolvent_optimization"
    CRYSTALLIZATION_NUCLEATION_OR_SEED_CONTROL = "crystallization_nucleation_or_seed_control"
    POLYMORPH_OR_SOLID_FORM_MANAGEMENT = "polymorph_or_solid_form_management"
    FEEDSTOCK_OR_SUPPLIER_VARIABILITY_MANAGEMENT = "feedstock_or_supplier_variability_management"
    TOLERANCE_OF_REAGENT_OR_FEED_IMPURITIES = "tolerance_of_reagent_or_feed_impurities"
    IMPROVED_IN_PROCESS_CONTROL_OR_ANALYTICAL_METHOD = "improved_in_process_control_or_analytical_method"
    OPTIMIZATION_OF_HALOGENATION_PROTOCOL = "optimization_of_halogenation_protocol"
    BIOCATALYTIC_OR_ENZYMATIC_ROUTE_CHANGE_OR_OPTIMIZATION = "biocatalytic_or_enzymatic_route_change_or_optimization"
    CUSTOM_OR_NOVEL_REACTOR_DESIGN = "custom_or_novel_reactor_design"
    IMPROVED_MIXING_OR_IMPELLER_DESIGN = "improved_mixing_or_impeller_design"
    IMPROVED_HEAT_TRANSFER_OR_TEMPERATURE_CONTROL_HARDWARE = "improved_heat_transfer_or_temperature_control_hardware"
    IMPROVED_PHOTOCHEMICAL_REACTOR_OR_LIGHT_DELIVERY = "improved_photochemical_reactor_or_light_delivery"
    IMPROVED_ELECTROCHEMICAL_REACTOR_OR_ELECTRODE_SYSTEM = "improved_electrochemical_reactor_or_electrode_system"
    BATCH_TO_FLOW_OR_FLOW_PROCESS_IMPLEMENTATION = "batch_to_flow_or_flow_process_implementation"
    PACKED_BED_OR_FIXED_BED_OPTIMIZATION = "packed_bed_or_fixed_bed_optimization"
    CLOGGING_OR_FOULING_MITIGATION = "clogging_or_fouling_mitigation"
    SLURRY_HANDLING_OR_PARTICLE_SIZE_OPTIMIZATION = "slurry_handling_or_particle_size_optimization"
    GAS_EVOLUTION_OR_MASS_TRANSFER_MANAGEMENT = "gas_evolution_or_mass_transfer_management"
    CONTINUOUS_PROCESS_INTENSIFICATION = "continuous_process_intensification"
    OPTIMIZATION_OF_INERT_ATMOSPHERE_OR_DEOXYGENATION = "optimization_of_inert_atmosphere_or_deoxygenation"
    OPTIMIZATION_OF_OXYGEN_OR_GAS_ENRICHED_ENVIRONMENT = "optimization_of_oxygen_or_gas_enriched_environment"
    CONDITION_CHANGE_TO_REDUCE_OR_REMOVE_PROCESS_HAZARD = "condition_change_to_reduce_or_remove_process_hazard"
    THERMAL_RUNAWAY_MITIGATION_OR_QUENCH_SAFETY_MODIFICATION = "thermal_runaway_mitigation_or_quench_safety_modification"
    PROCESS_MODELING_OR_DIGITAL_TWIN_GUIDED_CHANGE = "process_modeling_or_digital_twin_guided_change"
    STEP_REORDERING_OR_TELESCOPE_STRATEGY = "step_reordering_or_telescope_strategy"
    ENTIRELY_NEW_ROUTE_OR_STEP_REPLACEMENT = "entirely_new_route_or_step_replacement"
    INSUFFICIENT_INFORMATION_TO_CLASSIFY = "insufficient_information_to_classify"
    OTHER = "other"

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
    "optimization_of_acid_or_base_stoichiometry": "Tuning the equivalents of acid or base used while keeping identity fixed.",
    "solvent_change_for_reactivity": "Switch in solvent driven primarily by observed changes to rate, conversion, or selectivity of the reaction.",
    "solvent_change_for_solubility": "Switch in solvent driven primarily by solubility/phase issues (slurries, crystallization in reactor, etc.) rather than intrinsic reactivity.",
    "solvent_change_for_safety_or_green_chemistry": "Solvent change driven mainly by hazard, environmental, or regulatory considerations.",
    "solvent_change_to_reduce_side_products": "Solvent change that specifically reduces formation of a particular impurity or side product.",
    "solvent_system_optimization_or_cosolvent_addition": "Fine-tuning of mixed solvent systems or addition of minor cosolvents to balance solubility, reactivity, or workup.",
    "phase_behavior_or_biphasic_system_optimization": "Design or optimization of biphasic (e.g., aqueous/organic) systems, including salting-out, phase ratio, and mixing.",
    "change_of_additive": "Introduction, removal, or substitution of an additive (e.g., scavenger, promoter, stabilizer) to improve reaction performance or selectivity.",
    "optimization_of_reagent_stoichiometry": "General tuning of equivalents of one or more reagents to improve performance, robustness, or optimize pH/buffer conditions.",
    "change_of_addition_order_or_feed_profile": "Changing the order of additions, addition mode, or feed profile (e.g., semi-batch, controlled feed, split charges).",
    "temperature_setpoint_or_temperature_ramp_optimization": "Adjusting temperature setpoints or ramps (e.g., cooling or heating profiles) to manage rate, selectivity, or safety.",
    "pressure_or_gas_headspace_optimization": "Changing total pressure, gas type, or headspace to manage gas solubility, mass transfer, or safety.",
    "hold_time_or_residence_time_optimization": "Tuning batch hold times or continuous-flow residence times to achieve robust completion or avoid overreaction.",
    "protecting_group_change_or_removal": "Changing or removing protecting groups to simplify synthesis or avoid instability without redesigning the core route.",
    "improved_quench_or_neutralization": "Optimization of the quench or neutralization protocol to improve safety or downstream processing.",
    "improved_impurity_purge_or_byproduct_control": "Specific changes aimed at improving purge of an identified impurity or suppressing a known byproduct.",
    "drying_or_isolation_process_optimization": "Changes to drying method, drying conditions, or isolation protocol (e.g., spin-drying, tray vs vacuum).",
    "crystallization_solvent_or_antisolvent_optimization": "Tuning crystallization solvent, antisolvent, or volumes to control yield and impurity levels.",
    "crystallization_nucleation_or_seed_control": "Changes in seeding, nucleation temperature, or seeding protocol to control crystal habit or size.",
    "polymorph_or_solid_form_management": "Interventions specifically targeting polymorph control, solvate form, or amorphous vs crystalline states.",
    "feedstock_or_supplier_variability_management": "Strategies to handle variability in raw material quality from different suppliers or lots.",
    "tolerance_of_reagent_or_feed_impurities": "Deliberate redesign to tolerate higher levels of known impurities in feeds without additional purification.",
    "improved_in_process_control_or_analytical_method": "Development or improvement of analytical methods or IPCs to track and control the process.",
    "optimization_of_halogenation_protocol": "Specific improvements to halogenation steps (not just solvent or reagent changes) to control regioselectivity or safety.",
    "biocatalytic_or_enzymatic_route_change_or_optimization": "Introduction or optimization of a biocatalytic or enzymatic transformation as part of the route.",
    "custom_or_novel_reactor_design": "Implementation of a custom-designed reactor or novel reactor configuration to address specific process challenges.",
    "improved_mixing_or_impeller_design": "Changes to mixing hardware or agitation strategy to address mixing-limited behavior.",
    "improved_heat_transfer_or_temperature_control_hardware": "Hardware-level improvements to heat transfer or temperature control (e.g., jackets, coils).",
    "improved_photochemical_reactor_or_light_delivery": "Improved light delivery, reactor design, or LED configuration for photochemical steps.",
    "improved_electrochemical_reactor_or_electrode_system": "Improvements to electrochemical cell design, electrode materials, or current distribution.",
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
    "process_modeling_or_digital_twin_guided_change": "Changes guided by process models, digital twins, or flowsheet simulations.",
    "step_reordering_or_telescope_strategy": "Reordering steps or telescoping operations without fundamentally changing their chemistry.",
    "entirely_new_route_or_step_replacement": "Introduction of a new synthetic route or complete replacement of a problematic step.",
    "insufficient_information_to_classify": "Cases where the text does not provide enough information to assign a more specific category.",
    "other": "Cases that are clearly out of scope of all defined categories.",
}

# ============================================================================
# Named Reaction Taxonomy
# ============================================================================

def return_rxn_class():
    import pandas as pd
    rxnc = pd.read_csv("scripts/namerxn_map.csv")  

    out =[]
    for i,k in rxnc.iterrows():
        out.append(k["Reaction Type"])
    return Literal[out]

rxn_classes = return_rxn_class()


def load_reaction_types() -> List[str]:
    """Load reaction types from namerxn_map.csv as a list of strings for the prompt."""
    rxnc = pd.read_csv("scripts/namerxn_map.csv")
    return rxnc["Reaction Type"].tolist()


def build_taxonomy_block() -> str:
    """Build the solution taxonomy block for the prompt."""
    lines = ["Solution Categories:"]
    for cat, desc in CATEGORY_DEFINITIONS.items():
        lines.append(f"- {cat}: {desc}")
    return "\n".join(lines)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

from class_enum import RxnClass

class ProblemSolutionPair(BaseModel):
    problemCategory: Problems
    problemDescription: str
    solutionCategory: CategoryLiteral
    solutionDescription: str

class ScaleupClassification(BaseModel):
    reaction_class: RxnClass
    description: str
    reagents_used: List[str]
    problemSolutionPairs: List[ProblemSolutionPair]
    confidence: Literal["low", "medium", "high"]
    reasoning: str

    @field_validator("reaction_class", mode="before")
    def normalize_reaction_class(cls, v):
        if not v:
            return "-" + v
        # accept exact valid enum values; otherwise fallback
        try:
            return RxnClass(v).value  # keep canonical value
        except Exception:
            return "failed"


class InitialResult(BaseModel):
    reactions: List[ScaleupClassification]


# ============================================================================
# Paper Download Functions
# ============================================================================

def download_folder(folder_name_to_download, target_download_location) -> Path:
    """Download a paper folder from the remote server and return the zip path."""
    response = requests.get(
        f"http://molgpu02.mit.edu:8999/download",
        params={"path": folder_name_to_download},
        stream=True,
    )

    if response.status_code == 200:
        # Save the downloaded zip file locally
        zip_filename = Path(f"{target_download_location}.zip")
        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Folder downloaded and saved as: {zip_filename}")
        return zip_filename
    else:
        raise RuntimeError(f"Error downloading {folder_name_to_download}: {response.status_code} - {response.text}")


def unzip_folder(zip_path: Path, extract_to: Path, folder_name: str) -> Path:
    """Unzip a downloaded folder into a dedicated subdirectory."""
    paper_dir = extract_to / folder_name
    paper_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(paper_dir)
    
    zip_path.unlink()
    print(f"Extracted to: {paper_dir}")
    return paper_dir


# ============================================================================
# XML Parsing
# ============================================================================

def parse_xml_text(xml_path: Path) -> str:
    """Parse XML file and extract all text content."""
    parser = etree.XMLParser(recover=True, huge_tree=True)
    
    with open(xml_path, 'rb') as f:
        root = etree.parse(f, parser=parser).getroot()
    
    # Extract all text
    text_parts = []
    for elem in root.iter():
        if elem.text:
            text_parts.append(elem.text.strip())
        if elem.tail:
            text_parts.append(elem.tail.strip())
    
    raw_text = " ".join(text_parts)
    # Clean up whitespace
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()
    return raw_text


def find_xml_file(paper_dir: Path) -> Optional[Path]:
    """Find the XML file in the paper directory."""
    # Try .txt files first (common for JATS)
    txt_files = list(paper_dir.glob("*.txt"))
    if txt_files:
        return txt_files[0]
    
    # Then try .xml files
    xml_files = list(paper_dir.glob("*.xml"))
    if xml_files:
        return xml_files[0]
    
    # Search recursively
    for pattern in ["**/*.txt", "**/*.xml"]:
        files = list(paper_dir.glob(pattern))
        if files:
            return files[0]
    
    return None


def extract_paper_metadata(xml_path: Path) -> Dict[str, str]:
    """Extract metadata from the XML file."""
    parser = etree.XMLParser(recover=True, huge_tree=True)
    
    with open(xml_path, 'rb') as f:
        root = etree.parse(f, parser=parser).getroot()
    
    metadata = {"title": "", "doi": "", "journal": ""}
    
    # Try to find title
    for title_elem in root.xpath("//*[local-name()='article-title' or local-name()='title']"):
        if title_elem.text:
            metadata["title"] = "".join(title_elem.itertext()).strip()
            break
    
    # Try to find DOI
    for doi_elem in root.xpath("//*[local-name()='article-id'][@pub-id-type='doi']"):
        if doi_elem.text:
            metadata["doi"] = doi_elem.text.strip()
            break
    
    return metadata


# ============================================================================
# GPT-5.2 API Call
# ============================================================================

def build_system_prompt(reaction_types: List[str]) -> str:
    """Build the system prompt with taxonomies."""
    problem_taxonomy = """
Problem Categories:
- [SOLUBILITY]: Low solubility of reactant, reagent, or product leading to low conversion.
- [PURIFICATION]: Difficulty isolating product due to high solubility or impurities.
- [MIXABILITY]: Mixture too viscous to stir or forms multiple phases.
- [COST]: Reagent too expensive or impractical amount needed.
- [LONG_REACTIONS]: Excessive time for conversion or purification.
- [EXOTHERMIC]: Excessive heat release posing safety risks or hindering conversion.
- [SAFETY]: Toxic reagents, unsafe conditions, or environmental risk.
- [AVAILABILITY]: Needed reactant or reagent not readily available.
- [VARIABILITY]: Inconsistent conversions under similar conditions.
- [LOW YIELD]: Suboptimal conversion rate.
- [DECOMPOSITION]: Reactant, reagent, or product decomposes under conditions.
"""
    
    solution_taxonomy = build_taxonomy_block()
    
    # Limit reaction types to avoid token overflow (just show categories)
    reaction_sample = reaction_types[:100]
    reaction_list = "\n".join(f"- {r}" for r in reaction_sample)
    
    return f"""You are an expert process chemist. Analyze the given paper text to:

1. Identify all chemical reactions described in the paper
2. Classify each reaction using the NameRXN ontology (reaction types listed below) and store a list of reagents used in the reaction
3. For each reaction, extract scale-up problems and their solutions as described in the text

{problem_taxonomy}

{solution_taxonomy}

Named Reaction Types (sample - use closest match):
{reaction_list}
... and more reaction types from NameRXN taxonomy.

RULES:
- Only extract problems/solutions explicitly described in the text
- Do NOT invent problems or solutions not supported by the text
- Use the exact category names from the taxonomies
- Limit to 5 problem/solution pairs per reaction
- Structure your response according to the Initial_Result BaseModel
"""


def classify_paper(
    raw_text: str,
    paper_id: str,
    metadata: Dict[str, str],
    reaction_types: List[str],
    model: str = "gpt-5.2",
    base_url: str = SERVER_BASE_URL,
):
    """Call GPT API to classify reactions and extract problems/solutions."""
    load_dotenv()
    # api_key = os.getenv("OPENAI_API_KEY")
    
    # if not api_key:
        # raise ValueError("OPENAI_API_KEY not found in environment")
    
    client = OpenAI(base_url=base_url, api_key="")
    
    system_prompt = build_system_prompt(reaction_types)
    
    user_prompt = f"""Analyze the following paper and extract all reactions with their scale-up problems and solutions.

PAPER_ID: {paper_id}
PAPER_TITLE: {metadata.get('title', 'Unknown')}
DOI: {metadata.get('doi', 'Unknown')}

TEXT:
{raw_text}  # Limit text length to avoid token limits
"""
    
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=InitialResult,
        temperature=0.2,
    )
    
    return response.output_parsed

# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify reactions and extract scale-up problems from process chemistry papers using GPT-5.2"
    )
    parser.add_argument(
        "paper_id",
        type=str,
        help="Paper ID in format 'opxxxxxxx'"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: {paper_id}_classification.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="OpenAI model to use (default: gpt-5.2)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("VLLM_BASE_URL", "http://localhost:9001/v1"),
        help="Base URL for the VLLM API (default: http://localhost:9001/v1)"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/nfs/ccoleylab001/bmahjour/corpus/documents"),
        help="Working directory for downloads (default: ./downloads)"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Use local paper directory instead of downloading"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    load_dotenv()

    # Get paper directory
    if args.local_dir:
        paper_dir = args.local_dir
        print(f"Using local directory: {paper_dir}")
    else:
        # Download and extract paper
        args.work_dir.mkdir(parents=True, exist_ok=True)
        zip_path = download_folder(args.paper_id, args.work_dir / args.paper_id)
        paper_dir = unzip_folder(zip_path, args.work_dir, args.paper_id)
    
    # Find and parse XML
    xml_path = find_xml_file(paper_dir)
    if not xml_path:
        raise FileNotFoundError(f"No XML file found in {paper_dir}")
    
    print(f"Parsing XML: {xml_path}")
    raw_text = parse_xml_text(xml_path)
    metadata = extract_paper_metadata(xml_path)
    print(f"Extracted {len(raw_text)} characters of text")
    print(f"Paper title: {metadata.get('title', 'Unknown')[:80].encode('ascii', 'replace').decode()}...")
    
    # Classify with GPT
    print(f"Calling {args.model} for classification...")
    reaction_types = load_reaction_types()
    result = classify_paper(
        raw_text=raw_text,
        paper_id=args.paper_id,
        metadata=metadata,
        reaction_types=reaction_types,
        model=args.model,
        base_url=args.base_url,     
    )
    if not result or not result.reactions:
        print("No reactions found or classification failed.")
        return
    
    # Save output
    output_path = args.output or Path(f"{args.paper_id}_classification.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Found {len(result.reactions)} reactions with scale-up analysis")


if __name__ == "__main__":
    main()
