"""
Extract, serialize, perturb, and apply Slipnet link weight configurations.

A weight config is a plain dict mapping (source_name, dest_name) → fixedLength.
Only links with fixedLength > 0 are considered "learnable"; zero-length links
are structural (they derive strength from their label node) and are excluded
from perturbation.
"""

import json
import random as _random_mod


def extract_weights(slipnet):
    """Return a dict of all Sliplink fixedLength values keyed by (source_name, dest_name).

    Extracted immediately after Slipnet.__init__() so it captures the
    original Mitchell weights before any learning.
    """
    weights = {}
    for link in slipnet.sliplinks:
        key = (link.source.name, link.destination.name)
        weights[key] = link.fixedLength
    return weights


def apply_weights(slipnet, weight_dict):
    """Set fixedLength on every matching Sliplink in slipnet.

    intrinsicDegreeOfAssociation() is computed from fixedLength at call
    time, so no separate recomputation step is needed.
    """
    for link in slipnet.sliplinks:
        key = (link.source.name, link.destination.name)
        if key in weight_dict:
            link.fixedLength = weight_dict[key]


def perturb(weight_dict, delta, rng):
    """Return (new_weight_dict, perturbed_key) with one weight nudged by N(0, delta).

    Only links with fixedLength > 0 are eligible — zero-length links are
    structural and must not be touched.  The new value is clamped to [0, 100].
    If no eligible links exist the dict is returned unchanged and key is None.
    """
    candidates = [k for k, v in weight_dict.items() if v > 0]
    if not candidates:
        return dict(weight_dict), None

    key = rng.choice(candidates)
    new_dict = dict(weight_dict)
    noise = rng.gauss(0, delta)
    new_dict[key] = max(0.0, min(100.0, new_dict[key] + noise))
    return new_dict, key


def save(weight_dict, path):
    """Serialize weight_dict to JSON at path, encoding tuple keys as 'src->dst'."""
    serialized = {
        "{}->{}" .format(k[0], k[1]): v
        for k, v in weight_dict.items()
    }
    with open(path, "w") as fh:
        json.dump(serialized, fh, indent=2, sort_keys=True)


def load(path):
    """Load a weight dict from JSON, restoring tuple keys from 'src->dst' strings."""
    with open(path) as fh:
        serialized = json.load(fh)
    return {tuple(k.split("->", 1)): v for k, v in serialized.items()}
