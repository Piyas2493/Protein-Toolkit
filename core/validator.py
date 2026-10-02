from config import (
    SUPPORTED_AMINO_ACIDS,
    AMBIGUOUS_AMINO_ACIDS,
)


class SequenceValidator:

    @staticmethod
    def clean(sequence: str):

        return "".join(
            i.upper()
            for i in sequence
            if i.isalpha()
        )

    @staticmethod
    def validate(sequence: str):

        sequence = SequenceValidator.clean(sequence)

        unsupported = []

        ambiguous = []

        for aa in sequence:

            if aa in SUPPORTED_AMINO_ACIDS:
                continue

            elif aa in AMBIGUOUS_AMINO_ACIDS:
                ambiguous.append(aa)

            else:
                unsupported.append(aa)

        return {
            "length": len(sequence),
            "ambiguous": sorted(set(ambiguous)),
            "unsupported": sorted(set(unsupported)),
            "valid":
                len(unsupported) == 0
        }