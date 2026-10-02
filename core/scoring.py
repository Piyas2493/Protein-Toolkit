"""
Confidence scoring engine

Combines evidence from multiple modules
into a single confidence score.
"""


class ConfidenceScorer:

    def __init__(self):

        self.weights = {

            "motif":40,

            "catalytic":25,

            "enzyme":20,

            "statistics":15

        }

    def calculate(

        self,

        motif_hits,

        catalytic_hits,

        enzyme_predictions,

        statistics

    ):

        score = 0

        details = {}

        # ----------

        distinct_motifs = {hit["name"] for hit in motif_hits}

        motif_score = min(

            len(distinct_motifs) * 10,

            self.weights["motif"]

        )

        score += motif_score

        details["Motif"] = motif_score

        # ----------

        catalytic_score = min(

            len(catalytic_hits) * 12,

            self.weights["catalytic"]

        )

        score += catalytic_score

        details["Catalytic"] = catalytic_score

        # ----------

        enzyme_score = 0

        if enzyme_predictions:

            top = enzyme_predictions[0]

            strength = (
                top["score"] / top["max_score"]
                if top.get("max_score") else 0
            )

            enzyme_score = round(self.weights["enzyme"] * strength)

        score += enzyme_score

        details["Enzyme"] = enzyme_score

        # ----------

        stat_score = 0

        gravy = statistics.gravy()

        instability = statistics.instability_index()

        if -1.5 <= gravy <= 1.5:

            stat_score += 8

        if instability < 40:

            stat_score += 7

        stat_score = min(

            stat_score,

            self.weights["statistics"]

        )

        score += stat_score

        details["Statistics"] = stat_score

        score = min(score,100)

        return {

            "confidence":score,

            "breakdown":details

        }