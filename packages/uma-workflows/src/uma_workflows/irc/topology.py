from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


Bond = tuple[int, int]

# Endpoint topology is a covalent graph.  Alkali/alkaline-earth, transition,
# lanthanide, and actinide contacts are coordination geometry and must not be
# allowed to create or remove a covalent endpoint bond.  Keeping this list
# explicit also avoids treating a short Li--C/O/Cl contact as a bond.
_METAL_ATOMIC_NUMBERS = frozenset(
    {
        3, 11, 19, 37, 55, 87,  # alkali metals
        4, 12, 20, 38, 56, 88,  # alkaline-earth metals
        *range(21, 31), *range(39, 49), *range(72, 81),
        *range(89, 104),         # transition metals and actinides
        *range(57, 72),          # lanthanides
    }
)


@dataclass(frozen=True)
class IrcEndpointTopologyMatch:
    endpoint_a_role: str
    endpoint_b_role: str
    endpoint_a_mismatch_count: int
    endpoint_b_mismatch_count: int
    endpoint_a_missing_bonds: list[Bond]
    endpoint_a_extra_bonds: list[Bond]
    endpoint_b_missing_bonds: list[Bond]
    endpoint_b_extra_bonds: list[Bond]
    matches_expected_endpoints: bool


def perceived_bonds(
    atoms: Atoms,
    *,
    scale: float = 1.20,
) -> set[Bond]:
    atomic_numbers = atoms.get_atomic_numbers()
    distances = np.asarray(atoms.get_all_distances(mic=False), dtype=float)
    bonds: set[Bond] = set()
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            cutoff = scale * (
                covalent_radii[atomic_numbers[first]]
                + covalent_radii[atomic_numbers[second]]
            )
            if distances[first, second] <= cutoff:
                bonds.add((first, second))
    return bonds


def perceived_heavy_atom_bonds(
    atoms: Atoms,
    *,
    scale: float = 1.20,
) -> set[Bond]:
    """Return the legacy heavy-atom-only bond graph.

    Endpoint validation now uses :func:`perceived_bonds` so hydrogen transfer is
    visible, but this public helper remains available for existing consumers.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    return {
        (first, second)
        for first, second in perceived_bonds(atoms, scale=scale)
        if atomic_numbers[first] != 1 and atomic_numbers[second] != 1
    }


def perceived_covalent_bonds(
    atoms: Atoms,
    *,
    scale: float = 1.20,
) -> set[Bond]:
    """Return distance-perceived bonds after removing metal coordination.

    This is the graph used for IRC endpoint identity.  Metal contacts remain
    available from the coordinates for display and coordination analysis, but
    they are not covalent topology changes.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    return {
        bond
        for bond in perceived_bonds(atoms, scale=scale)
        if atomic_numbers[bond[0]] not in _METAL_ATOMIC_NUMBERS
        and atomic_numbers[bond[1]] not in _METAL_ATOMIC_NUMBERS
    }


def match_irc_endpoint_topologies(
    endpoint_a: Atoms,
    endpoint_b: Atoms,
    expected_reactant: Atoms,
    expected_product: Atoms,
    *,
    scale: float = 1.20,
    include_metal_bonds: bool = False,
) -> IrcEndpointTopologyMatch:
    structures = (endpoint_a, endpoint_b, expected_reactant, expected_product)
    symbols = structures[0].get_chemical_symbols()
    if any(value.get_chemical_symbols() != symbols for value in structures[1:]):
        raise ValueError(
            "IRC endpoints and expected structures must use identical atom ordering"
        )

    perceive = perceived_bonds if include_metal_bonds else perceived_covalent_bonds
    endpoint_bonds = [
        perceive(value, scale=scale)
        for value in (endpoint_a, endpoint_b)
    ]
    expected_bonds = {
        "reactant": perceive(expected_reactant, scale=scale),
        "product": perceive(expected_product, scale=scale),
    }

    scored: list[
        tuple[
            int,
            tuple[str, str],
            tuple[set[Bond], set[Bond], set[Bond], set[Bond]],
        ]
    ] = []
    for endpoint_a_role, endpoint_b_role in (
        ("reactant", "product"),
        ("product", "reactant"),
    ):
        a_missing = expected_bonds[endpoint_a_role] - endpoint_bonds[0]
        a_extra = endpoint_bonds[0] - expected_bonds[endpoint_a_role]
        b_missing = expected_bonds[endpoint_b_role] - endpoint_bonds[1]
        b_extra = endpoint_bonds[1] - expected_bonds[endpoint_b_role]
        scored.append(
            (
                len(a_missing) + len(a_extra) + len(b_missing) + len(b_extra),
                (endpoint_a_role, endpoint_b_role),
                (a_missing, a_extra, b_missing, b_extra),
            )
        )

    score, roles, differences = min(scored, key=lambda item: item[0])
    a_missing, a_extra, b_missing, b_extra = differences
    return IrcEndpointTopologyMatch(
        endpoint_a_role=roles[0],
        endpoint_b_role=roles[1],
        endpoint_a_mismatch_count=len(a_missing) + len(a_extra),
        endpoint_b_mismatch_count=len(b_missing) + len(b_extra),
        endpoint_a_missing_bonds=sorted(a_missing),
        endpoint_a_extra_bonds=sorted(a_extra),
        endpoint_b_missing_bonds=sorted(b_missing),
        endpoint_b_extra_bonds=sorted(b_extra),
        matches_expected_endpoints=score == 0,
    )
