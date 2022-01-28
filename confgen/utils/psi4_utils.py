import psi4
from rdkit import Chem


psi4.set_memory("64 GB")
psi4.set_num_threads(36)
psi4.core.set_output_file("/tmp/output.dat", False)


def mol2xyz(mol, addhs=False):
    if addhs:
        mol = Chem.AddHs(mol)

    atoms = mol.GetAtoms()
    string = "\n"
    for i, atom in enumerate(atoms):
        pos = mol.GetConformer(0).GetAtomPosition(atom.GetIdx())
        string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)

    string += "units angstrom\n"
    return string, mol


def get_energy(mol, addhs=False):
    const = 27.211386245988
    xyz, mol = mol2xyz(mol, addhs=addhs)
    benz = psi4.geometry(xyz)
    scf_e, scf_wfn = psi4.energy("B3LYP/cc-pVDZ", return_wfn=True)
    HOMO = scf_wfn.epsilon_a_subset("AO", "ALL").np[scf_wfn.nalpha() - 1]
    LUMO = scf_wfn.epsilon_a_subset("AO", "ALL").np[scf_wfn.nalpha()]
    energy = scf_e * const
    gap = (LUMO - HOMO) * const
    return energy, gap
