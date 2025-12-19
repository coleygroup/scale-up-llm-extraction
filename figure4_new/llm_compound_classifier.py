"""
Helpers for extracting problematic/replacement compounds from
scale-up problem/solution text and resolving their SMILES via PubChem.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import requests
from openai import OpenAI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class NamedCompound(BaseModel):
    """
    A single compound mention.

    name   : Human-readable name as written in the text.
    smiles : ONLY if explicitly given in the text, or filled later by
             resolve_smiles_from_name.
    """
    name: str
    smiles: Optional[str] = None


class ReplacementPair(BaseModel):
    """
    Optional pairing between a problematic compound and its replacement.
    Any side of the pair may be null if the mapping is ambiguous.
    """
    problematic: Optional[NamedCompound] = None
    replacement: Optional[NamedCompound] = None


class CompoundReplacementResult(BaseModel):
    """
    Top-level result for a problem/solution pair.
    """
    problematic_compounds: List[NamedCompound]
    replacement_compounds: List[NamedCompound]
    pairs: List[ReplacementPair]


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
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
# PubChem SMILES resolver (tool)
# ---------------------------------------------------------------------------


def resolve_smiles_from_name(
    name: str,
    *,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """
    Resolve a compound name (or synonym) to a canonical SMILES string
    using PubChem PUG REST.

    Args:
        name: Chemical name or synonym (e.g. "4CzIPN", "benzene").

    Returns:
        Canonical SMILES string or None if not resolvable.
    """
    if not name:
        return None

    close_session = False
    if session is None:
        session = requests.Session()
        close_session = True

    try:
        # 1) Look up CID from name
        cid_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{requests.utils.quote(name)}/cids/JSON"
        )
        cid_resp = session.get(cid_url, timeout=10)

        if cid_resp.status_code != 200:
            return None

        cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None

        cid = cids[0]

        # 2) Fetch canonical SMILES for that CID
        prop_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/CanonicalSMILES/JSON"
        )
        prop_resp = session.get(prop_url, timeout=10)

        if prop_resp.status_code != 200:
            return None

        props = prop_resp.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None

        if "CanonicalSMILES" in props[0]:
            return props[0].get("CanonicalSMILES")
        if "ConnectivitySMILES" in props[0]:
            return props[0].get("ConnectivitySMILES")

    except Exception as e:
        print(f"Error resolving SMILES for '{name}': {e}")
        return None

    finally:
        if close_session:
            session.close()


# ---------------------------------------------------------------------------
# Core extraction agent
# ---------------------------------------------------------------------------


def extract_compound_replacements(
    problem: str,
    solution: str,
    *,
    client: Optional[OpenAI] = None,
    auto_resolve_smiles: bool = True,
) -> CompoundReplacementResult:
    """
    Extract problematic and replacement compounds from a problem/solution
    description and optionally resolve their SMILES via PubChem.

    Args:
        problem: Text describing the scale-up problem.
        solution: Text describing the associated solution.
        client: Optional OpenAI client; if None, uses get_client().
        auto_resolve_smiles: If True, call resolve_smiles_from_name for any
            compounds whose 'smiles' field is still None.

    Returns:
        CompoundReplacementResult with lists of problematic and replacement
        compounds and optional pairings.
    """

    if client is None:
        client = get_client()

    system_msg = (
        "You are an expert process chemist and information extractor.\n"
        "You receive two passages from a scale-up manuscript:\n"
        "  - problem: describes a problematic situation, including any compounds.\n"
        "  - solution: describes how the authors fixed it, including any replacements.\n\n"
        "Your job is to extract:\n"
        "  1) All compounds that are clearly described as PROBLEMATIC in the "
        "     problem text (e.g. too expensive, unstable, low-yielding, impure, "
        "     hazardous, etc.).\n"
        "  2) All compounds that are clearly described as REPLACEMENTS or solutions "
        "     in the solution text (e.g. chosen instead, selected for further "
        "     optimisation, switched to, etc.).\n"
        "  3) Whenever possible, pair a problematic compound with its replacement.\n\n"
        "Important rules:\n"
        "  - Use short, chemistry-style names exactly as written (e.g. "
        "    '4CzIPN', 'Ru(bpy)3Cl2', 'benzene').\n"
        "  - Do NOT treat generic words like 'solvent', 'oxidant', 'base', "
        "    'catalyst' as compounds unless a specific chemical name is given.\n"
        "  - A compound is 'problematic' if the text clearly associates it with "
        "    unwanted cost, safety, yield, stability, impurity, etc.\n"
        "  - A compound is a 'replacement' if the text clearly says it was "
        "    selected or introduced instead of another compound.\n\n"
        "SMILES handling:\n"
        "  - Do NOT guess or invent SMILES.\n"
        "  - Only fill the 'smiles' field if a SMILES or other explicit line "
        "    notation appears in the input text.\n"
        "  - If no explicit line notation is present, set 'smiles' to null.\n"
        "  - An external tool may later populate SMILES from the compound name, "
        "    so the name field is the primary identifier.\n\n"
        "Pairing rules:\n"
        "  - If the text clearly implies 'X was replaced by Y', pair X (problematic) "
        "    with Y (replacement) as one ReplacementPair.\n"
        "  - If multiple problematic compounds are replaced by a single new "
        "    compound, you may create multiple pairs sharing the same replacement.\n"
        "  - If the mapping is unclear, leave the pair fields as null or partial.\n\n"
        "Return a JSON object that matches the CompoundReplacementResult schema."
    )

    user_payload = {
        "problem": problem,
        "solution": solution,
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False)

    response = client.responses.parse(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        reasoning={"effort": "medium"},
        text={"verbosity": "low"},
        text_format=CompoundReplacementResult,
    )

    result: CompoundReplacementResult = response.output_parsed

    # Optionally resolve SMILES for any compounds that are missing them
    if auto_resolve_smiles:
        session = requests.Session()
        try:
            for comp_list in (result.problematic_compounds, result.replacement_compounds):
                for comp in comp_list:
                    if comp.smiles is None and comp.name:
                        smi = resolve_smiles_from_name(comp.name, session=session)
                        comp.smiles = smi
        finally:
            session.close()

    return result