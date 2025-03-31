class AtomicLine:
    def __init__(self, label: str, wavelength: float):
        self.label = label
        self.wavelength = wavelength

# -------------------------------------------------------------------------------------
# -------------------------------------- SOURCES --------------------------------------
# -------------------------------------------------------------------------------------
# Na I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/sodiumtable3.htm
# K  I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/potassiumtable3.htm
# Ca I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable3.htm
# Ca II: https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable4.htm

NA_I_LINES = [
    AtomicLine('Na I', 5889.950),
    AtomicLine('Na I', 5895.924)
]

K_I_LINES = [
    AtomicLine('K I', 7664.8991),
    AtomicLine('K I', 7698.9645),
    AtomicLine('K I', 5801.75),
    AtomicLine('K I', 5812.15)
]

CA_I_LINES = [
    AtomicLine('Ca I', 6102.722),
    AtomicLine('Ca I', 6122.219)
]

CA_II_LINES = [
    # TODO
]
