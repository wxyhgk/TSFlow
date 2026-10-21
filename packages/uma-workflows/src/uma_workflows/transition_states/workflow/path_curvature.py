from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled


@dataclass(frozen=True)
class PathCurvatureProbe:
    frame_index: int
    path_coordinate: float
    positions_a: np.ndarray
    tangent: np.ndarray
    directional_curvature_ev_per_a2: float
    gradient_projection_ev_per_a: float
    max_force_ev_per_a: float
    has_negative_directional_curvature: bool


@dataclass(frozen=True)
class PathCurvatureProbeResult:
    probes: tuple[PathCurvatureProbe, ...]
    selected_probe: PathCurvatureProbe | None
    peak_frame_index: int
    force_evaluations: int
    warnings: tuple[str, ...] = ()


def probe_path_curvature(
    atoms: Atoms,
    positions_a: np.ndarray,
    coordinates: np.ndarray,
    energies_ev: np.ndarray,
    *,
    calculator,
    cancel_event: CancellationToken,
    displacement_a: float = 0.01,
    negative_threshold_ev_per_a2: float = 0.05,
    max_frames: int = 7,
) -> PathCurvatureProbeResult:
    """Probe Cartesian curvature along an optimized path tangent.

    This is a cheap, non-stationary seed screen. A negative directional
    curvature is not a frequency calculation and does not prove that a frame
    is an index-one saddle.
    """
    positions = np.asarray(positions_a, dtype=float)
    path_coordinates = np.asarray(coordinates, dtype=float)
    energies = np.asarray(energies_ev, dtype=float)
    _validate_inputs(
        atoms,
        positions,
        path_coordinates,
        energies,
        displacement_a=displacement_a,
        negative_threshold_ev_per_a2=negative_threshold_ev_per_a2,
        max_frames=max_frames,
    )
    peak_index = int(np.argmax(energies[1:-1])) + 1
    interior = sorted(
        range(1, len(path_coordinates) - 1),
        key=lambda index: (
            abs(path_coordinates[index] - path_coordinates[peak_index]),
            index,
        ),
    )[:max_frames]
    probes: list[PathCurvatureProbe] = []
    warnings: list[str] = []
    force_evaluations = 0
    for index in interior:
        raise_if_cancelled(cancel_event)
        tangent = _normalized_tangent(positions, path_coordinates, index)
        if tangent is None:
            warnings.append(
                f"Skipped path frame {index}: its Cartesian tangent is zero."
            )
            continue
        center_forces = _forces_at(atoms, positions[index], calculator)
        plus_forces = _forces_at(
            atoms,
            positions[index] + displacement_a * tangent,
            calculator,
        )
        minus_forces = _forces_at(
            atoms,
            positions[index] - displacement_a * tangent,
            calculator,
        )
        force_evaluations += 3
        hessian_tangent = -(
            plus_forces - minus_forces
        ) / (2.0 * displacement_a)
        curvature = float(np.vdot(tangent, hessian_tangent))
        gradient_projection = float(np.vdot(-center_forces, tangent))
        probes.append(
            PathCurvatureProbe(
                frame_index=index,
                path_coordinate=float(path_coordinates[index]),
                positions_a=positions[index].copy(),
                tangent=tangent.copy(),
                directional_curvature_ev_per_a2=curvature,
                gradient_projection_ev_per_a=gradient_projection,
                max_force_ev_per_a=float(
                    np.linalg.norm(center_forces, axis=1).max(initial=0.0)
                ),
                has_negative_directional_curvature=(
                    curvature < -negative_threshold_ev_per_a2
                ),
            )
        )
    negative = [
        probe for probe in probes if probe.has_negative_directional_curvature
    ]
    selected = min(
        negative,
        key=lambda probe: (
            abs(probe.path_coordinate - path_coordinates[peak_index]),
            abs(probe.gradient_projection_ev_per_a),
            probe.directional_curvature_ev_per_a2,
        ),
        default=None,
    )
    if selected is None:
        warnings.append(
            "No probed frame has negative curvature along the local path "
            "tangent; saddle refinement should not be started from this screen."
        )
    return PathCurvatureProbeResult(
        probes=tuple(probes),
        selected_probe=selected,
        peak_frame_index=peak_index,
        force_evaluations=force_evaluations,
        warnings=tuple(warnings),
    )


def _normalized_tangent(
    positions: np.ndarray,
    coordinates: np.ndarray,
    index: int,
) -> np.ndarray | None:
    tangent = (positions[index + 1] - positions[index - 1]) / (
        coordinates[index + 1] - coordinates[index - 1]
    )
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        return None
    return tangent / norm


def _forces_at(
    template: Atoms,
    positions_a: np.ndarray,
    calculator,
) -> np.ndarray:
    probe = template.copy()
    probe.set_positions(positions_a)
    probe.calc = calculator
    return np.asarray(probe.get_forces(), dtype=float)


def _validate_inputs(
    atoms: Atoms,
    positions: np.ndarray,
    coordinates: np.ndarray,
    energies: np.ndarray,
    *,
    displacement_a: float,
    negative_threshold_ev_per_a2: float,
    max_frames: int,
) -> None:
    if positions.ndim != 3 or positions.shape[1:] != (len(atoms), 3):
        raise ValueError("path positions must have shape (frames, atoms, 3)")
    if len(positions) < 3:
        raise ValueError("path curvature probing requires at least three frames")
    if coordinates.shape != (len(positions),):
        raise ValueError("path coordinates must match the frame count")
    if energies.shape != (len(positions),):
        raise ValueError("path energies must match the frame count")
    if not np.all(np.isfinite(positions)):
        raise ValueError("path positions must be finite")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.diff(coordinates) > 0):
        raise ValueError("path coordinates must be finite and strictly increasing")
    if not np.all(np.isfinite(energies)):
        raise ValueError("path energies must be finite")
    if not np.isfinite(displacement_a) or displacement_a <= 0.0:
        raise ValueError("curvature displacement must be positive and finite")
    if (
        not np.isfinite(negative_threshold_ev_per_a2)
        or negative_threshold_ev_per_a2 <= 0.0
    ):
        raise ValueError("negative-curvature threshold must be positive and finite")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
