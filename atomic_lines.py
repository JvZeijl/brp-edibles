type AtomicLine = tuple[str, float]

# -------------------------------------------------------------------------------------
# -------------------------------------- SOURCES --------------------------------------
# -------------------------------------------------------------------------------------
# Na I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/sodiumtable3.htm
# K  I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/potassiumtable3.htm
# Ca I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable3.htm
# Ca II: https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable4.htm

NA_I_LINES: list[AtomicLine] = [
    ('Na I', 5889.950),
    ('Na I', 5895.924)
]

K_I_LINES: list[AtomicLine] = [
    ('K I', 7664.8991),
    ('K I', 7698.9645)
]

CA_I_LINES: list[AtomicLine] = [
    # TODO
]

CA_II_LINES: list[AtomicLine] = [
    # TODO
]
