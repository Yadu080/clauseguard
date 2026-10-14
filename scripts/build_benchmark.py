"""Build data/benchmark.csv from the hand-annotated pairs below.

The pairs are written by hand; this script only flattens them into a CSV and
assigns the dev/test split. The split is made per *base sentence* (all variants
of one original go to the same split) with a fixed random seed, so it is
reproducible and variants of the same sentence never leak across splits.

Annotation guideline
--------------------
expected_drift = 1 when a careful reader would say the rewrite changes who,
what, when or whether the rule applies. expected_categories lists every
category whose constraint changed, separated by '|':

    negation    polarity of a clause (not, no, never, without, un-/in- words)
    modality    obligation / permission / recommendation / prohibition
    condition   conditions and exceptions (if, only if, unless, except ...)
    quantifier  scope words and bounds (all, only, some, at least, before ...)
    numeric     numbers, percentages, amounts, dates, weekdays and times
    other       a real change outside the five categories (e.g. 'and' -> 'or');
                kept on purpose to measure what the detector cannot see

Benign paraphrases (expected_drift = 0) keep the meaning while changing words,
voice or order. Some deliberately use expressions outside the lexicon.

Run:  python scripts/build_benchmark.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark.csv"
SEED = 42
TEST_FRACTION = 0.4

# (rewritten_text, expected_drift, expected_categories, mutation_type)
BENCHMARK: dict[str, tuple[str, list[tuple[str, int, str, str]]]] = {
    "B01": ("Students below 75% attendance cannot appear for the examination unless the HOD approves.", [
        ("Students below 75% attendance cannot appear for the examination.", 1, "condition", "exception_removed"),
        ("Students below 70% attendance cannot appear for the examination unless the HOD approves.", 1, "numeric", "numeric_changed"),
        ("Unless the HOD approves, students with attendance below 75% cannot sit for the examination.", 0, "", "benign_paraphrase"),
    ]),
    "B02": ("Applicants must have at least 75% attendance.", [
        ("Applicants must have at least 70% attendance.", 1, "numeric", "numeric_changed"),
        ("Applicants must have at most 75% attendance.", 1, "quantifier", "quantifier_changed"),
        ("A minimum attendance of 75% is required for all applicants.", 0, "", "benign_paraphrase"),
    ]),
    "B03": ("Candidates must submit the form before Friday.", [
        ("Candidates may submit the form before Friday.", 1, "modality", "modality_weakened"),
        ("Candidates must submit the form after Friday.", 1, "quantifier", "quantifier_changed"),
        ("The form has to be submitted by candidates before Friday.", 0, "", "benign_paraphrase"),
    ]),
    "B04": ("Students who have not paid the fee are ineligible.", [
        ("Students who have paid the fee are ineligible.", 1, "negation", "negation_removed"),
        ("Students who haven't paid the fee are not eligible.", 0, "", "benign_paraphrase"),
    ]),
    "B05": ("Submit the application before 15 November.", [
        ("The application must be submitted before 15 November.", 0, "", "benign_paraphrase"),
        ("Submit the application before 25 November.", 1, "numeric", "numeric_changed"),
        ("Submit the application after 15 November.", 1, "quantifier", "quantifier_changed"),
        ("Submit the application before 15 November.", 0, "", "unchanged"),
    ]),
    "B06": ("Mobile phones are not permitted inside the examination hall.", [
        ("Mobile phones are permitted inside the examination hall.", 1, "negation|modality", "negation_removed"),
        ("Mobile phones are prohibited inside the examination hall.", 0, "", "benign_paraphrase"),
        ("Candidates may not carry mobile phones into the examination hall.", 0, "", "benign_paraphrase"),
    ]),
    "B07": ("Late submissions will be accepted only if a medical certificate is attached.", [
        ("Late submissions will be accepted if a medical certificate is attached.", 1, "condition", "condition_weakened"),
        ("Late submissions will be accepted.", 1, "condition", "condition_removed"),
        ("A late submission is accepted only when a medical certificate is attached to it.", 0, "", "benign_paraphrase"),
    ]),
    "B08": ("Hostel residents must return by 9:30 pm.", [
        ("Hostel residents must return by 10:30 pm.", 1, "numeric", "numeric_changed"),
        ("Hostel residents should return by 9:30 pm.", 1, "modality", "modality_weakened"),
        ("Residents of the hostel are required to be back by 9:30 pm.", 0, "", "benign_paraphrase"),
    ]),
    "B09": ("A maximum of two books can be borrowed at a time.", [
        ("A maximum of three books can be borrowed at a time.", 1, "numeric", "numeric_changed"),
        ("A minimum of two books can be borrowed at a time.", 1, "quantifier", "quantifier_changed"),
        ("Members can borrow up to 2 books at a time.", 0, "", "benign_paraphrase"),
    ]),
    "B10": ("All students must carry their identity cards on campus.", [
        ("Some students must carry their identity cards on campus.", 1, "quantifier", "quantifier_changed"),
        ("All students may carry their identity cards on campus.", 1, "modality", "modality_weakened"),
        ("Every student must carry an identity card on campus.", 0, "", "benign_paraphrase"),
        ("All students must carry their identity cards on campus.", 0, "", "unchanged"),
    ]),
    "B11": ("The scholarship is available only to students with a CGPA of 8.0 or above.", [
        ("The scholarship is available to students with a CGPA of 8.0 or above.", 1, "quantifier", "quantifier_removed"),
        ("The scholarship is available only to students with a CGPA of 7.5 or above.", 1, "numeric", "numeric_changed"),
        ("Only students whose CGPA is at least 8.0 are eligible for the scholarship.", 0, "", "benign_paraphrase"),
    ]),
    "B12": ("Fees once paid will not be refunded under any circumstances.", [
        ("Fees once paid will be refunded under any circumstances.", 1, "negation", "negation_removed"),
        ("Fees once paid will not be refunded in most circumstances.", 1, "quantifier", "quantifier_changed"),
        ("Paid fees are non-refundable in all circumstances.", 0, "", "benign_paraphrase"),
    ]),
    "B13": ("Students must not use calculators during the test unless the invigilator permits it.", [
        ("Students must use calculators during the test unless the invigilator permits it.", 1, "negation|modality", "negation_removed"),
        ("Students must not use calculators during the test.", 1, "condition", "exception_removed"),
        ("Calculators may not be used by students during the test without the invigilator's permission.", 0, "", "benign_paraphrase"),
    ]),
    "B14": ("Attendance in laboratory sessions is mandatory.", [
        ("Attendance in laboratory sessions is optional.", 1, "modality", "modality_weakened"),
        ("Attendance in laboratory sessions is recommended.", 1, "modality", "modality_weakened"),
        ("Students are required to attend all laboratory sessions.", 0, "", "benign_paraphrase"),
    ]),
    "B15": ("Re-evaluation requests must be filed within 7 days of the result announcement.", [
        ("Re-evaluation requests must be filed within 10 days of the result announcement.", 1, "numeric", "numeric_changed"),
        ("Re-evaluation requests must be filed within 7 weeks of the result announcement.", 1, "numeric", "numeric_changed"),
        ("Requests for re-evaluation have to be filed within seven days of the announcement of results.", 0, "", "benign_paraphrase"),
    ]),
    "B16": ("The library is closed on Sundays and public holidays.", [
        ("The library is closed on Saturdays and public holidays.", 1, "numeric", "numeric_changed"),
        ("The library is not open on Sundays and public holidays.", 0, "", "benign_paraphrase"),
    ]),
    "B17": ("Students may use the computer lab after 6 pm if they have written permission.", [
        ("Students may use the computer lab after 6 pm.", 1, "condition", "condition_removed"),
        ("Students may use the computer lab before 6 pm if they have written permission.", 1, "quantifier", "quantifier_changed"),
        ("Provided they have written permission, students can use the computer lab after 6 pm.", 0, "", "benign_paraphrase"),
    ]),
    "B18": ("The registration fee is Rs. 500 per course.", [
        ("The registration fee is Rs. 800 per course.", 1, "numeric", "numeric_changed"),
        ("The registration fee is ₹500 for each course.", 0, "", "benign_paraphrase"),
    ]),
    "B19": ("A student can register for no more than 6 credits in the summer term.", [
        ("A student can register for no more than 8 credits in the summer term.", 1, "numeric", "numeric_changed"),
        ("A student can register for at least 6 credits in the summer term.", 1, "quantifier", "quantifier_changed"),
        ("Students may register for a maximum of 6 credits in the summer term.", 0, "", "benign_paraphrase"),
    ]),
    "B20": ("Laptops are allowed in the library except during examination week.", [
        ("Laptops are allowed in the library during examination week.", 1, "condition", "exception_removed"),
        ("Laptops are not allowed in the library except during examination week.", 1, "negation|modality", "negation_added"),
        ("Except during examination week, laptops are permitted in the library.", 0, "", "benign_paraphrase"),
        ("Laptops are allowed in the library except during examination week.", 0, "", "unchanged"),
    ]),
    "B21": ("Every assignment must be submitted through the portal.", [
        ("Some assignments must be submitted through the portal.", 1, "quantifier", "quantifier_changed"),
        ("Every assignment should be submitted through the portal.", 1, "modality", "modality_weakened"),
        ("All assignments have to be submitted through the portal.", 0, "", "benign_paraphrase"),
    ]),
    "B22": ("Students with more than three backlogs cannot register for the internship.", [
        ("Students with more than five backlogs cannot register for the internship.", 1, "numeric", "numeric_changed"),
        ("Students with fewer than three backlogs cannot register for the internship.", 1, "quantifier", "quantifier_changed"),
        ("Students having over 3 backlogs are not allowed to register for the internship.", 0, "", "benign_paraphrase"),
    ]),
    "B23": ("The canteen remains open until 8 pm on weekdays.", [
        ("The canteen remains open until 6 pm on weekdays.", 1, "numeric", "numeric_changed"),
        ("On weekdays, the canteen is open till 8 pm.", 0, "", "benign_paraphrase"),
    ]),
    "B24": ("Research scholars need not attend the orientation programme.", [
        ("Research scholars need to attend the orientation programme.", 1, "negation|modality", "negation_removed"),
        ("Research scholars are not required to attend the orientation programme.", 0, "", "benign_paraphrase"),
        ("Attending the orientation programme is optional for research scholars.", 0, "", "benign_paraphrase"),
    ]),
    "B25": ("Students must obtain a no-dues certificate before collecting their degree.", [
        ("Students may obtain a no-dues certificate before collecting their degree.", 1, "modality", "modality_weakened"),
        ("Students must obtain a no-dues certificate after collecting their degree.", 1, "quantifier", "quantifier_changed"),
        ("Before collecting their degree, students are required to obtain a no-dues certificate.", 0, "", "benign_paraphrase"),
    ]),
    "B26": ("If a student is absent for more than 10 consecutive days, the parents will be informed.", [
        ("If a student is absent for more than 15 consecutive days, the parents will be informed.", 1, "numeric", "numeric_changed"),
        ("The parents of a student will be informed.", 1, "condition|quantifier|numeric", "condition_removed"),
        ("Parents will be informed when a student is absent for over 10 consecutive days.", 0, "", "benign_paraphrase"),
    ]),
    "B27": ("The project report should not exceed 40 pages.", [
        ("The project report should not exceed 60 pages.", 1, "numeric", "numeric_changed"),
        ("The project report should be at least 40 pages.", 1, "quantifier", "quantifier_changed"),
        ("The project report should be at most 40 pages long.", 0, "", "benign_paraphrase"),
    ]),
    "B28": ("Students are permitted to bring one A4 sheet of notes into the exam.", [
        ("Students are not permitted to bring one A4 sheet of notes into the exam.", 1, "negation|modality", "negation_added"),
        ("Students are required to bring one A4 sheet of notes into the exam.", 1, "modality", "modality_strengthened"),
        ("Students may bring a single A4 sheet of notes into the exam.", 0, "", "benign_paraphrase"),
    ]),
    "B29": ("The deadline for fee payment is 30 June 2025.", [
        ("The deadline for fee payment is 30 July 2025.", 1, "numeric", "numeric_changed"),
        ("Fee payment is due by 30/06/2025.", 0, "", "benign_paraphrase"),
    ]),
    "B30": ("Hostel fees are refundable if the student withdraws before the semester begins.", [
        ("Hostel fees are refundable if the student withdraws after the semester begins.", 1, "quantifier", "quantifier_changed"),
        ("Hostel fees are non-refundable if the student withdraws before the semester begins.", 1, "negation", "negation_added"),
        ("If a student withdraws before the start of the semester, hostel fees can be refunded.", 0, "", "benign_paraphrase"),
    ]),
    "B31": ("No more than 30 students may enrol in an elective.", [
        ("No fewer than 30 students may enrol in an elective.", 1, "quantifier", "quantifier_changed"),
        ("No more than 40 students may enrol in an elective.", 1, "numeric", "numeric_changed"),
        ("An elective can have at most 30 students enrolled.", 0, "", "benign_paraphrase"),
    ]),
    "B32": ("Students should consult their mentor before dropping a course.", [
        ("Students must consult their mentor before dropping a course.", 1, "modality", "modality_strengthened"),
        ("Students should consult their mentor after dropping a course.", 1, "quantifier", "quantifier_changed"),
        ("Before dropping a course, students ought to consult their mentor.", 0, "", "benign_paraphrase"),
    ]),
    "B33": ("Calculators are not allowed unless specified in the question paper.", [
        ("Calculators are allowed unless specified in the question paper.", 1, "negation|modality", "negation_removed"),
        ("Calculators are not allowed.", 1, "condition", "exception_removed"),
        ("Calculators are prohibited except where the question paper specifies otherwise.", 0, "", "benign_paraphrase"),
        ("Calculators are not allowed unless specified in the question paper.", 0, "", "unchanged"),
    ]),
    "B34": ("Each team must have at least three and at most five members.", [
        ("Each team must have at least two and at most five members.", 1, "numeric", "numeric_changed"),
        ("Each team must have at least three and at most six members.", 1, "numeric", "numeric_changed"),
        ("Every team must consist of three to five members.", 0, "", "benign_paraphrase"),
    ]),
    "B35": ("The Wi-Fi password must not be shared with anyone.", [
        ("The Wi-Fi password may be shared with anyone.", 1, "negation|modality", "negation_removed"),
        ("Do not share the Wi-Fi password with anyone.", 0, "", "benign_paraphrase"),
    ]),
    "B36": ("Candidates must report to the examination centre at least 30 minutes before the exam starts.", [
        ("Candidates must report to the examination centre at least 15 minutes before the exam starts.", 1, "numeric", "numeric_changed"),
        ("Candidates must report to the examination centre at most 30 minutes before the exam starts.", 1, "quantifier", "quantifier_changed"),
        ("Candidates have to reach the exam centre no later than 30 minutes before the start of the exam.", 0, "", "benign_paraphrase"),
    ]),
    "B37": ("Only final-year students are eligible for campus placements.", [
        ("All students are eligible for campus placements.", 1, "quantifier", "quantifier_changed"),
        ("Final-year students are eligible for campus placements.", 1, "quantifier", "quantifier_removed"),
        ("Campus placements are open only to final-year students.", 0, "", "benign_paraphrase"),
    ]),
    "B38": ("Medical leave is granted only if a certificate from a registered doctor is submitted within 3 days.", [
        ("Medical leave is granted only if a certificate from a registered doctor is submitted within 5 days.", 1, "numeric", "numeric_changed"),
        ("Medical leave is granted if a certificate from a registered doctor is submitted within 3 days.", 1, "condition", "condition_weakened"),
        ("Medical leave will be granted only when a registered doctor's certificate is submitted within three days.", 0, "", "benign_paraphrase"),
    ]),
    "B39": ("Visitors are not allowed in the hostel after 8 pm.", [
        ("Visitors are allowed in the hostel after 8 pm.", 1, "negation|modality", "negation_removed"),
        ("Visitors are not allowed in the hostel before 8 pm.", 1, "quantifier", "quantifier_changed"),
        ("Visitors may not enter the hostel after 8 pm.", 0, "", "benign_paraphrase"),
    ]),
    "B40": ("Students may appeal the decision within 15 days.", [
        ("Students must appeal the decision within 15 days.", 1, "modality", "modality_strengthened"),
        ("Students may appeal the decision after 15 days.", 1, "quantifier", "quantifier_changed"),
        ("An appeal against the decision can be filed within fifteen days.", 0, "", "benign_paraphrase"),
        ("Students may appeal the decision within 15 days.", 0, "", "unchanged"),
    ]),
    "B41": ("The penalty for late return of books is Rs. 10 per day.", [
        ("The penalty for late return of books is Rs. 20 per day.", 1, "numeric", "numeric_changed"),
        ("The penalty for late return of books is Rs. 10 per week.", 1, "numeric", "numeric_changed"),
        ("A fine of ₹10 per day is charged for returning books late.", 0, "", "benign_paraphrase"),
    ]),
    "B42": ("Students must wear lab coats in the chemistry laboratory at all times.", [
        ("Students should wear lab coats in the chemistry laboratory at all times.", 1, "modality", "modality_weakened"),
        ("Students need not wear lab coats in the chemistry laboratory.", 1, "negation|modality", "negation_added"),
        ("Lab coats must be worn in the chemistry laboratory at all times.", 0, "", "benign_paraphrase"),
    ]),
    "B43": ("Admission will be cancelled if the documents are found to be fake.", [
        ("Admission will not be cancelled if the documents are found to be fake.", 1, "negation", "negation_added"),
        ("Admission will be cancelled.", 1, "condition", "condition_removed"),
        ("If the documents turn out to be fake, the admission will be cancelled.", 0, "", "benign_paraphrase"),
    ]),
    "B44": ("A minimum CGPA of 7.0 is required to apply for the exchange programme.", [
        ("A minimum CGPA of 6.0 is required to apply for the exchange programme.", 1, "numeric", "numeric_changed"),
        ("A CGPA of 7.0 is recommended to apply for the exchange programme.", 1, "modality|quantifier", "modality_weakened"),
        ("Applicants for the exchange programme need a CGPA of at least 7.0.", 0, "", "benign_paraphrase"),
    ]),
    "B45": ("Leave applications must be approved by the class advisor and the HOD.", [
        ("Leave applications must be approved by the class advisor or the HOD.", 1, "other", "out_of_scope"),
        ("Leave applications may be approved by the class advisor and the HOD.", 1, "modality", "modality_weakened"),
        ("The class advisor and the HOD must both approve leave applications.", 0, "", "benign_paraphrase"),
    ]),
    "B46": ("Students are encouraged to participate in at least one technical event per semester.", [
        ("Students are required to participate in at least one technical event per semester.", 1, "modality", "modality_strengthened"),
        ("Students are encouraged to participate in at least two technical events per semester.", 1, "numeric", "numeric_changed"),
        ("Participation in a minimum of one technical event each semester is encouraged.", 0, "", "benign_paraphrase"),
    ]),
    "B47": ("Internal marks will be awarded only to students who submit all assignments.", [
        ("Internal marks will be awarded to students who submit all assignments.", 1, "quantifier", "quantifier_removed"),
        ("Internal marks will be awarded only to students who submit some assignments.", 1, "quantifier", "quantifier_changed"),
        ("Only students who submit every assignment will be awarded internal marks.", 0, "", "benign_paraphrase"),
    ]),
    "B48": ("The exam will start at 9:30 am and candidates will not be allowed to enter after 10 am.", [
        ("The exam will start at 9:30 am and candidates will be allowed to enter after 10 am.", 1, "negation|modality", "negation_removed"),
        ("The exam will start at 10 am and candidates will not be allowed to enter after 10 am.", 1, "numeric", "numeric_changed"),
        ("The exam begins at 9:30 am; entry is not permitted after 10 am.", 0, "", "benign_paraphrase"),
    ]),
    "B49": ("Students may not use mobile phones in the library.", [
        ("Students may use mobile phones in the library.", 1, "negation|modality", "negation_removed"),
        ("The use of mobile phones in the library is prohibited for students.", 0, "", "benign_paraphrase"),
    ]),
    "B50": ("Students who fail to register by 31 July will be charged a late fee of Rs. 1,000.", [
        ("Students who register by 31 July will be charged a late fee of Rs. 1,000.", 1, "negation", "negation_removed"),
        ("Students who fail to register by 31 August will be charged a late fee of Rs. 1,000.", 1, "numeric", "numeric_changed"),
        ("A late fee of ₹1000 will be charged to students who do not register by 31 July.", 0, "", "benign_paraphrase"),
    ]),
    "B51": ("Students must submit both the assignment and the lab record.", [
        ("Students must submit either the assignment or the lab record.", 1, "other", "out_of_scope"),
        ("Both the assignment and the lab record have to be submitted by students.", 0, "", "benign_paraphrase"),
    ]),
    "B52": ("Only teaching staff may access the server room.", [
        ("Only non-teaching staff may access the server room.", 1, "negation", "negation_added"),
        ("Access to the server room is restricted to teaching staff.", 0, "", "benign_paraphrase"),
    ]),
}


def build_rows() -> list[dict]:
    base_ids = sorted(BENCHMARK)
    shuffled = base_ids[:]
    random.Random(SEED).shuffle(shuffled)
    n_test = round(len(shuffled) * TEST_FRACTION)
    test_ids = set(shuffled[:n_test])

    rows = []
    for base_id in base_ids:
        original, variants = BENCHMARK[base_id]
        for i, (rewritten, drift, categories, mutation) in enumerate(variants, start=1):
            if drift == 0 and categories:
                raise ValueError(f"{base_id}-{i}: benign pair must not have categories")
            if drift == 1 and not categories:
                raise ValueError(f"{base_id}-{i}: drift pair needs at least one category")
            rows.append({
                "example_id": f"{base_id}-{i}",
                "base_id": base_id,
                "original_text": original,
                "rewritten_text": rewritten,
                "expected_drift": drift,
                "expected_categories": categories,
                "mutation_type": mutation,
                "split": "test" if base_id in test_ids else "dev",
            })
    return rows


def main() -> None:
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_dev = sum(r["split"] == "dev" for r in rows)
    n_pos = sum(r["expected_drift"] for r in rows)
    print(f"Wrote {len(rows)} pairs from {len(BENCHMARK)} base sentences to {OUT.relative_to(ROOT)}")
    print(f"  dev: {n_dev}  test: {len(rows) - n_dev}  drift: {n_pos}  no-drift: {len(rows) - n_pos}")


if __name__ == "__main__":
    main()
