"""Mapping class for AA dictionaries

"""

# Sentinel used as the default for ``SymMap(exclude_syms=...)``. Identity
# comparison (``exclude_syms is NONCANONICAL``) flips SymMap into "exclude
# everything not in sym_list" mode, in which the explicit ``exclude_syms``
# list is left empty. Passing any real iterable (including ``[]``) instead
# switches to "exclude only these listed symbols" mode.
NONCANONICAL = "noncanonical"


class SymMap:

    def __init__(
        self,
        aa_syms: str,
        gapsym: str,
        exclude_syms=NONCANONICAL,
        gap_value: int = 0,
    ):
        self.aa_list = list(aa_syms)
        self.gapsym = gapsym
        if "*" in self.aa_list:
            raise ValueError(
                "'*' (stop codon) cannot be a member of the alphabet. "
                "Stop codons are handled by mysca.io.load_msa: trailing "
                "'*' is replaced with gap, internal '*' causes the "
                "sequence to be dropped."
            )
        if not (0 <= gap_value <= len(self.aa_list)):
            raise ValueError(
                f"gap_value must be in [0, {len(self.aa_list)}]; "
                f"got {gap_value}"
            )
        self.sym_list = (
            self.aa_list[:gap_value] + [gapsym] + self.aa_list[gap_value:]
        )
        self._valid_syms = set(self.sym_list)
        if exclude_syms is NONCANONICAL:
            self._exclude_noncanonical = True
            self.exclude_syms = []
        else:
            self._exclude_noncanonical = False
            self.exclude_syms = list(exclude_syms)
        self.sym2int = {sym: i for i, sym in enumerate(self.sym_list)}
        self.aa2int = {
            k: v for k, v in self.sym2int.items() if k in self.aa_list
        }
        self.gapint = self.sym2int[self.gapsym]

    def is_excluded(self, sym: str) -> bool:
        """Return True if the symbol should be excluded."""
        if self._exclude_noncanonical:
            return sym not in self._valid_syms
        return sym in self.exclude_syms

    def __getitem__(self, key):
        return self.sym2int[key]

    def __len__(self):
        return len(self.sym2int)


DEFAULT_MAP = SymMap(
    "ACDEFGHIKLMNPQRSTVWY", "-"
)

