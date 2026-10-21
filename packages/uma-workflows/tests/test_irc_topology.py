from ase import Atoms

from uma_workflows.irc import match_irc_endpoint_topologies, perceived_covalent_bonds
from uma_workflows.irc.topology import perceived_bonds


def structures() -> tuple[Atoms, Atoms]:
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
    return reactant, product


def test_matches_opposite_reactant_and_product_graphs() -> None:
    reactant, product = structures()
    result = match_irc_endpoint_topologies(
        product,
        reactant,
        reactant,
        product,
    )
    assert result.matches_expected_endpoints
    assert result.endpoint_a_role == "product"
    assert result.endpoint_b_role == "reactant"


def test_rejects_two_endpoints_in_the_same_basin() -> None:
    reactant, product = structures()
    result = match_irc_endpoint_topologies(
        reactant,
        reactant,
        reactant,
        product,
    )
    assert not result.matches_expected_endpoints


def test_metal_contacts_are_not_endpoint_covalent_bonds() -> None:
    reactant = Atoms(
        "CLiO",
        positions=[
            (0.0, 0.0, 0.0),
            (1.9, 0.0, 0.0),
            (0.0, 2.0, 0.0),
        ],
    )
    assert perceived_covalent_bonds(reactant) == set()
    # The explicit opt-in remains available for diagnostics and legacy users.
    assert (0, 1) in perceived_bonds(reactant)


def test_endpoint_match_ignores_li_coordination() -> None:
    reactant = Atoms("CLi", positions=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    product = Atoms("CLi", positions=[(0.0, 0.0, 0.0), (3.2, 0.0, 0.0)])
    result = match_irc_endpoint_topologies(product, reactant, reactant, product)
    assert result.matches_expected_endpoints
