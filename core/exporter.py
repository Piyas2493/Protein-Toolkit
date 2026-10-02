"""
ProteinToolkit Export Engine
"""

import csv
import json
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _stringify(value):
    """List-valued fields (e.g. evidence/interpretation summaries) would
    otherwise render as a raw Python list repr in CSV/text/HTML output."""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)


class ExportEngine:

    def export_json(
        self,
        filename,
        data
    ):

        path = OUTPUT_DIR / f"{filename}.json"

        with open(path, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=4
            )

        return path

    def export_csv(
        self,
        filename,
        data
    ):

        path = OUTPUT_DIR / f"{filename}.csv"

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["Category", "Value"]
            )

            for key, value in data.items():

                writer.writerow(
                    [key, _stringify(value)]
                )

        return path

    def export_text(
        self,
        filename,
        data
    ):

        path = OUTPUT_DIR / f"{filename}.txt"

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            for key, value in data.items():

                f.write(
                    f"{key}: {_stringify(value)}\n"
                )

        return path

    def export_html(
        self,
        filename,
        data
    ):

        path = OUTPUT_DIR / f"{filename}.html"

        html = """

<html>

<head>

<title>ProteinToolkit Report</title>

<style>

body{
font-family:Arial;
margin:40px;
background:#f8f8f8;
}

table{
border-collapse:collapse;
width:100%;
}

th,td{
border:1px solid #ddd;
padding:8px;
}

th{
background:#1e88e5;
color:white;
}

</style>

</head>

<body>

<h1>ProteinToolkit Report</h1>

<table>

<tr>

<th>Category</th>

<th>Value</th>

</tr>

"""

        for key, value in data.items():

            html += f"""

<tr>

<td>{key}</td>

<td>{_stringify(value)}</td>

</tr>

"""

        html += """

</table>

</body>

</html>

"""

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(html)

        return path