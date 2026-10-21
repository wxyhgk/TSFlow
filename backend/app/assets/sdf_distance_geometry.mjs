import { readFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'

function fail(message) {
  process.stderr.write(`${message}\n`)
  process.exit(1)
}

function normalizeMolHeader(text) {
  const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  let countsIndex = -1
  for (let index = 0; index < Math.min(lines.length, 20); index += 1) {
    if (/\bV[23]000\b/i.test(lines[index] ?? '')) {
      countsIndex = index
      break
    }
  }
  if (countsIndex < 0 || countsIndex === 3) return lines.join('\n')
  const header = lines.slice(0, countsIndex).filter(line => line.trim().length > 0)
  return [header[0] ?? '', header[1] ?? '  UMA CLI', header[2] ?? '', ...lines.slice(countsIndex)].join('\n')
}

const [, , openChemLibPath, sdfPath] = process.argv
if (!openChemLibPath || !sdfPath) fail('usage: node sdf_distance_geometry.mjs OPENCL_JS INPUT.sdf')

let OCL
try {
  OCL = await import(pathToFileURL(openChemLibPath).href)
  OCL.Resources.registerFromNodejs()
} catch (error) {
  fail(`cannot load OpenChemLib: ${error instanceof Error ? error.message : String(error)}`)
}

const text = await readFile(sdfPath, 'utf8')
const records = text.split(/\$\$\$\$/).filter(block => block.trim().length > 10)
if (records.length !== 1) fail(`SDF must contain exactly one molecule; found ${records.length}`)

let molecule
try {
  molecule = OCL.Molecule.fromMolfile(normalizeMolHeader(records[0]))
} catch (error) {
  fail(`cannot parse SDF: ${error instanceof Error ? error.message : String(error)}`)
}
if (molecule.getAllAtoms() < 1) fail('SDF contains no atoms')
if (molecule.getAllAtoms() > 1 && molecule.getAllBonds() < 1) {
  fail('SDF contains no bonds; distance geometry requires molecular connectivity')
}

let conformer
try {
  conformer = new OCL.ConformerGenerator(0x5eed).getOneConformerAsMolecule(molecule)
} catch (error) {
  fail(`distance geometry failed: ${error instanceof Error ? error.message : String(error)}`)
}
if (!conformer) fail('distance geometry could not generate a 3D conformer')

const atoms = []
for (let index = 0; index < conformer.getAllAtoms(); index += 1) {
  atoms.push({
    symbol: conformer.getAtomLabel(index),
    position: [
      conformer.getAtomX(index),
      -conformer.getAtomY(index),
      -conformer.getAtomZ(index),
    ],
  })
}
const getName = typeof conformer.getName === 'function' ? conformer.getName.bind(conformer) : null
process.stdout.write(JSON.stringify({ atoms, name: getName?.() || null }))
