"""Guards on the two joins that silently corrupt the selection/outcome analysis.

Both of these have been got wrong in analysis code already, and neither fails loudly: they
produce plausible numbers that are simply about something else.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neighborhood_judge as NJ  # noqa: E402

SUBSET = Path("analysis/scaffold_subset.csv")
csv.field_size_limit(10 ** 7)


def test_form_size_gates_inflection_only_groups():
    """`form_n` is authoritative; `len(form_group)` is not.

    The judge reports its membership, and `form_size` then enforces the definitional
    constraints -- connectivity, and the rule that a group whose only relation is inflection
    is not a group. A verdict can therefore carry a two-member `form_group` and still measure
    zero. 14% of the corpus records look like this, so a join on `len(form_group) >= 2`
    silently counts them as paradigms.
    """
    verdict = {"form_group": ["system", "systems"], "form_basis": "system",
               "form_kind": "same_lemma_inflection", "form_inflection_discounted": True}
    words = ["system", "systems", "method", "device", "plan",
             "scheme", "order", "way", "means", "form"]
    assert len(verdict["form_group"]) == 2
    assert NJ.form_size(verdict, words) == 0

    real = {"form_group": ["hepton", "proton"], "form_basis": "-on",
            "form_kind": "shared_affix", "form_inflection_discounted": False}
    assert NJ.form_size(real, ["hepton", "proton", "atom", "ion", "mass",
                               "unit", "charge", "field", "wave", "ray"]) == 2


@pytest.mark.skipif(not SUBSET.exists(), reason="scaffold_subset.csv not built")
def test_subset_hit_column_is_recall_at_100_not_top_10():
    """`hit` is recall@100. RNL@10 needs `rank <= 10`.

    The two differ by roughly a factor of three on the post-cutoff arm, and `hit` is the more
    inviting column name, so anything reporting an @10 quantity must derive it from `rank`.
    """
    rows, mismatched = 0, 0
    with open(SUBSET) as f:
        for r in csv.DictReader(f):
            rank = int(r["rank"] or -1)
            assert int(r["hit"]) == int(0 < rank <= 100)
            if int(r["hit"]) != int(0 < rank <= 10):
                mismatched += 1
            rows += 1
            if rows >= 20000:
                break
    assert mismatched > 0, "hit and correct@10 must not be treated as interchangeable"
