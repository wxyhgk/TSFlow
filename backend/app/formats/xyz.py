import math
from pathlib import Path

from app.contracts.jobs import Atom, AtomicStructure


MAX_XYZ_BYTES = 5 * 1024 * 1024
MAX_XYZ_ATOMS = 2_000


class XYZParseError(ValueError):
    pass


def parse_xyz(data: bytes, filename: str, charge: int, spin: int) -> AtomicStructure:
    if len(data) > MAX_XYZ_BYTES:
        raise XYZParseError("XYZ file exceeds the 5 MB upload limit")
    if Path(filename).suffix.lower() != ".xyz":
        raise XYZParseError("file must use the .xyz extension")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise XYZParseError("XYZ file must be UTF-8 text") from exc

    lines = text.splitlines()
    if len(lines) < 2:
        raise XYZParseError("XYZ file must contain an atom count and comment line")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise XYZParseError("first XYZ line must be an integer atom count") from exc
    if not 1 <= atom_count <= MAX_XYZ_ATOMS:
        raise XYZParseError(f"XYZ atom count must be between 1 and {MAX_XYZ_ATOMS}")
    if len(lines) < atom_count + 2:
        raise XYZParseError(
            f"XYZ declares {atom_count} atoms but contains only {max(0, len(lines) - 2)} atom lines"
        )
    if any(line.strip() for line in lines[atom_count + 2 :]):
        raise XYZParseError("only one XYZ frame can be submitted per job")

    atoms: list[Atom] = []
    for index, line in enumerate(lines[2 : atom_count + 2], start=1):
        fields = line.split()
        if len(fields) < 4:
            raise XYZParseError(f"atom line {index} must contain symbol, x, y and z")
        symbol = fields[0]
        try:
            position = tuple(float(value) for value in fields[1:4])
        except ValueError as exc:
            raise XYZParseError(f"atom line {index} contains an invalid coordinate") from exc
        if not all(math.isfinite(value) for value in position):
            raise XYZParseError(f"atom line {index} contains a non-finite coordinate")
        try:
            atoms.append(Atom(symbol=symbol, position=position))
        except ValueError as exc:
            raise XYZParseError(f"atom line {index} contains an invalid element symbol") from exc

    comment = lines[1].strip()
    if len(comment) > 1_000:
        raise XYZParseError("XYZ comment line exceeds 1000 characters")
    return AtomicStructure(
        atoms=atoms,
        charge=charge,
        spin=spin,
        name=Path(filename).stem[:255],
        comment=comment or None,
        source_format="xyz",
    )
