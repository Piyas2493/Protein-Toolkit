from dataclasses import dataclass
from pathlib import Path


@dataclass
class FASTARecord:
    header: str
    sequence: str


class FASTAParser:

    @staticmethod
    def load(filepath: str):

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(filepath)

        records = []

        header = ""
        seq = []

        with open(path) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):

                    if header:

                        records.append(
                            FASTARecord(
                                header,
                                "".join(seq)
                            )
                        )

                    header = line[1:]
                    seq = []

                else:

                    seq.append(line)

        if header:

            records.append(
                FASTARecord(
                    header,
                    "".join(seq)
                )
            )

        return records