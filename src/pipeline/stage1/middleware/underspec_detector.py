from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.pipeline.stage1.models.raw_fact import RawFact


@dataclass(frozen=True)
class UnderspecReport:
    is_underspecified: bool
    fact_count: int
    entity_count: int
    relationship_count: int
    constraint_count: int
    ambiguity_count: int
    suggested_domain_searches: List[str]
    reasoning: str


def detect_underspecification(
    facts: List[RawFact],
    nl_description: str,
    domain: str = "Unknown",
) -> UnderspecReport:
    """
    Heuristic detection of underspecified NL descriptions.
    Returns report with recommended domain pattern searches.
    """
    fact_count = len(facts)
    
    # Count unique entities (simple heuristic: capitalized words in facts)
    entities = set()
    relationships = 0
    constraints = 0
    
    for fact in facts:
        if fact.is_external:
            continue
        text = fact.fact.lower()
        
        # Count relationships
        rel_keywords = ["belongs to", "assigned to", "routes to", "maps to", 
                       "links to", "points to", "references", "runs on",
                       "associated with", "connected to", "per "]
        if any(kw in text for kw in rel_keywords):
            relationships += 1
            
        # Count constraints
        constraint_keywords = ["must", "should", "cannot", "required", 
                              "if ", "then ", "constraint", "limit",
                              "at least", "at most", "between", "range"]
        if any(kw in text for kw in constraint_keywords):
            constraints += 1
            
        # Extract entity-like words (simple heuristic)
        words = fact.fact.split()
        for w in words:
            if w and w[0].isupper() and len(w) > 2:
                entities.add(w.rstrip('.,;:'))
    
    entity_count = len(entities)
    
    # Heuristics for underspecification
    is_underspecified = False
    reasons = []
    searches = []
    
    # Too few facts for a complex domain
    if fact_count < 8:
        is_underspecified = True
        reasons.append(f"Only {fact_count} facts extracted (expected 10+ for complex domain)")
        searches.append(f"{domain} database schema standard tables")
    
    # Few entities mentioned
    if entity_count < 4:
        is_underspecified = True
        reasons.append(f"Only {entity_count} entities identified")
        searches.append(f"{domain} core entities and relationships")
    
    # No relationships found
    if relationships == 0 and fact_count > 3:
        is_underspecified = True
        reasons.append("No explicit relationships found in extracted facts")
        searches.append(f"{domain} foreign key relationships cardinality")
    
    # No constraints found
    if constraints == 0 and fact_count > 5:
        is_underspecified = True
        reasons.append("No constraints/rules extracted")
        searches.append(f"{domain} typical data constraints and validation rules")
    
    # Domain-specific searches
    domain_lower = domain.lower()
    if "bank" in domain_lower or "loan" in domain_lower or "credit" in domain_lower:
        searches.extend([
            "banking database schema loan origination",
            "credit score tiers interest rate calculation",
        ])
    elif "hospital" in domain_lower or "clinic" in domain_lower or "medical" in domain_lower:
        searches.extend([
            "hospital database schema patient encounter",
            "clinical data model HL7 FHIR",
        ])
    elif "ecommerce" in domain_lower or "retail" in domain_lower or "shop" in domain_lower:
        searches.extend([
            "ecommerce database schema order product",
            "retail inventory management schema",
        ])
    elif "saas" in domain_lower or "b2b" in domain_lower:
        searches.extend([
            "SaaS billing subscription database schema",
            "multi-tenant database architecture",
        ])
    
    return UnderspecReport(
        is_underspecified=is_underspecified,
        fact_count=fact_count,
        entity_count=entity_count,
        relationship_count=relationships,
        constraint_count=constraints,
        ambiguity_count=0,  # Will be updated by verifier
        suggested_domain_searches=list(dict.fromkeys(searches))[:8],  # dedupe, limit
        reasoning="; ".join(reasons) if reasons else "Sufficient specification detected",
    )