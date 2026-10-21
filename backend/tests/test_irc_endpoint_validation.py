from __future__ import annotations

from ase import Atoms

from app.services.irc_endpoint_validation import match_irc_endpoint_topologies


def test_matches_opposite_reactant_and_product_heavy_atom_graphs() -> None:
    reactant = Atoms(
        "CCCC",
        positions=[
            (0.0, 0.0, 0.0),
            (1.4, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.4, 0.0, 0.0),
        ],
    )
    product = reactant.copy()
    product.positions[2:] -= (3.6, 0.0, 0.0)

    result = match_irc_endpoint_topologies(
        product,
        reactant,
        reactant,
        product,
    )

    assert result.matches_expected_endpoints
    assert result.endpoint_a_role == "product"
    assert result.endpoint_b_role == "reactant"
    assert result.endpoint_a_mismatch_count == 0
    assert result.endpoint_b_mismatch_count == 0


def test_rejects_two_endpoints_in_the_same_bonding_basin() -> None:
    reactant = Atoms(
        "CCCC",
        positions=[
            (0.0, 0.0, 0.0),
            (1.4, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.4, 0.0, 0.0),
        ],
    )
    product = reactant.copy()
    product.positions[2:] -= (3.6, 0.0, 0.0)

    result = match_irc_endpoint_topologies(
        reactant,
        reactant,
        reactant,
        product,
    )

    assert not result.matches_expected_endpoints
    assert result.endpoint_a_mismatch_count + result.endpoint_b_mismatch_count > 0
