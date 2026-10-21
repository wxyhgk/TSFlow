"""Run a CPU water single-point calculation with the recovered UMA calculator."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/uma-workflows/src'))

import numpy as np
from ase import Atoms
from uma_workflows.calculators.uma import UmaCalculatorFactory


def main():
    assets = ROOT / 'model-assets/UMA'
    checkpoint = assets / 'checkpoints/uma-s-1p2p1.pt'
    started = time.monotonic()
    print('Loading UMA 1.2.1 on CPU...', flush=True)
    factory = UmaCalculatorFactory(assets / 'references', 'cpu')
    atoms = Atoms('H2O', positions=[[0.9572, 0, 0], [-0.2390, 0.9270, 0], [0, 0, 0]], pbc=False)
    atoms.info.update(charge=0, spin=1)
    atoms.calc = factory.calculator(checkpoint, 'cpu', 'omol')
    print('Evaluating water energy and forces...', flush=True)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    if not np.isfinite(energy) or forces.shape != (3, 3) or not np.isfinite(forces).all():
        raise RuntimeError('Invalid energy or force output')
    result = {
        'model': 'uma-s-1p2p1', 'task': 'omol', 'device': 'cpu',
        'charge': 0, 'spin': 1, 'symbols': atoms.get_chemical_symbols(),
        'positions_a': atoms.positions.tolist(), 'energy_ev': energy,
        'forces_ev_per_a': forces.tolist(),
        'max_force_ev_per_a': float(np.linalg.norm(forces, axis=1).max()),
        'elapsed_seconds': time.monotonic() - started,
        'validation': 'finite single-point outputs only; not geometry optimization or TS validation',
    }
    output = ROOT / 'data/validation/uma-water-singlepoint.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
