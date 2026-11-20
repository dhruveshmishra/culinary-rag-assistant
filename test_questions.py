import sys
import re
from typing import Tuple
from assistant import ask

# Acceptance Criteria Verification Functions

def check_q1(res: dict) -> Tuple[bool, str]:
    ans = res.get("answer", "")
    sources = res.get("sources", [])
    has_850 = "850" in ans
    has_620 = "620" in ans
    cites_p10 = 10 in sources or "Page 10" in ans or "page 10" in ans.lower()
    passed = (not res.get("refused", True)) and has_850 and has_620 and cites_p10
    detail = f"850={has_850}, 620={has_620}, cites_p10={cites_p10}"
    return passed, detail

def check_q2(res: dict) -> Tuple[bool, str]:
    ans = res.get("answer", "")
    sources = res.get("sources", [])
    has_terrine = "roasted fig & hazelnut duck terrine" in ans.lower() or "duck terrine" in ans.lower()
    has_lamb = "pistachio crusted lamb rack" in ans.lower() or "lamb rack" in ans.lower()
    has_veloute = "charred eggplant & tahini veloute" in ans.lower() or "tahini veloute" in ans.lower()
    cites_p25 = 25 in sources or "Page 25" in ans or "page 25" in ans.lower()
    
    # Must NOT list shellfish or lobster bisque as tree nut or sesame dish
    lower_ans = ans.lower()
    shellfish_misclassified = False
    if "lobster" in lower_ans or "shellfish" in lower_ans:
        # Check if lobster is misclassified under tree nuts or sesame
        lines = lower_ans.split("\n")
        for line in lines:
            if ("tree nut" in line or "sesame" in line) and ("lobster" in line or "shellfish" in line):
                shellfish_misclassified = True
    
    passed = (
        (not res.get("refused", True)) and
        has_terrine and has_lamb and has_veloute and
        cites_p25 and (not shellfish_misclassified)
    )
    detail = f"terrine={has_terrine}, lamb={has_lamb}, veloute={has_veloute}, cites_p25={cites_p25}, no_misclass={not shellfish_misclassified}"
    return passed, detail

def check_q3(res: dict) -> Tuple[bool, str]:
    ans = res.get("answer", "").strip()
    refused = res.get("refused", False)
    exact_ans = (ans == "I cannot find this information in the manual.")
    passed = refused and exact_ans
    detail = f"refused={refused}, exact_refusal={exact_ans}"
    return passed, detail

def check_q4(res: dict) -> Tuple[bool, str]:
    ans = res.get("answer", "")
    sources = res.get("sources", [])
    has_temp = "57.5" in ans
    has_time = "110" in ans and "minute" in ans.lower()
    cites_p40 = 40 in sources or "Page 40" in ans or "page 40" in ans.lower()
    passed = (not res.get("refused", True)) and has_temp and has_time and cites_p40
    detail = f"57.5={has_temp}, 110_mins={has_time}, cites_p40={cites_p40}"
    return passed, detail

def check_q5(res: dict) -> Tuple[bool, str]:
    ans = res.get("answer", "").strip()
    refused = res.get("refused", False)
    exact_ans = (ans == "I cannot find this information in the manual.")
    passed = refused and exact_ans
    detail = f"refused={refused}, exact_refusal={exact_ans}"
    return passed, detail

# Test definitions
ACCEPTANCE_TESTS = [
    {
        "id": "Q1",
        "question": "What exact quantities, in grams, of dark couverture chocolate and heavy cream are required for a standard 20-portion batch of Signature Truffle Ganache?",
        "check": check_q1,
        "desc": "Contains 850, 620 and cites Page 10"
    },
    {
        "id": "Q2",
        "question": "According to the Allergen Matrix, which dishes on the Autumn Tasting Menu contain tree nuts or sesame?",
        "check": check_q2,
        "desc": "Lists Duck Terrine, Lamb Rack (tree nuts), Tahini Veloute (sesame), cites Page 25, no shellfish misclassification"
    },
    {
        "id": "Q3",
        "question": "At what time does the Eiffel Tower in Paris close to visitors during the winter season?",
        "check": check_q3,
        "desc": "refused=True and exact answer 'I cannot find this information in the manual.'"
    },
    {
        "id": "Q4",
        "question": "What is the mandatory minimum internal cooking temperature, in degrees Celsius, and the required holding time for sous-vide duck breast, as specified in the Food Safety section?",
        "check": check_q4,
        "desc": "Contains 57.5 and 110 minutes and cites Page 40"
    },
    {
        "id": "Q5",
        "question": "What is the current Michelin-star rating of Noma in Copenhagen?",
        "check": check_q5,
        "desc": "refused=True and exact answer 'I cannot find this information in the manual.'"
    }
]

PARAPHRASE_TESTS = [
    {
        "id": "P1",
        "question": "how much cream goes in the truffle ganache?",
        "expected_page": 10,
        "must_contain": ["620"]
    },
    {
        "id": "P2",
        "question": "What is the quantity of Valrhona dark couverture chocolate for the signature ganache?",
        "expected_page": 10,
        "must_contain": ["850"]
    },
    {
        "id": "P3",
        "question": "Which autumn dishes have sesame or tahini?",
        "expected_page": 25,
        "must_contain": ["veloute", "eggplant"]
    },
    {
        "id": "P4",
        "question": "How long and at what temperature should sous-vide duck breast be pasteurized?",
        "expected_page": 40,
        "must_contain": ["57.5", "110"]
    },
    {
        "id": "P5",
        "question": "Are there any tree nut allergens in the pistachio crusted lamb rack or duck terrine?",
        "expected_page": 25,
        "must_contain": ["tree nut"]
    }
]

OUT_OF_SCOPE_TESTS = [
    ("OOS1", "What is the weather forecast for Paris this weekend?"),
    ("OOS2", "Who won the FIFA World Cup in 2022?"),
    ("OOS3", "What is the secret recipe for Kentucky Fried Chicken original chicken?"),
    ("OOS4", "What are the opening hours of the Louvre Museum in Paris?"),
    ("OOS5", "How many calories are in a Big Mac hamburger from McDonald's?"),
    ("OOS6", "What is the capital city of Australia?"),
    ("OOS7", "How do you calibrate an espresso machine at Starbucks?"),
    ("OOS8", "What was the stock price of Apple on January 1st?"),
    ("OOS9", "Who is the head chef at The French Laundry in California?"),
    ("OOS10", "What temperature should a standard New York style pizza be baked at?")
]

def run_all_tests():
    print("=" * 85)
    print("RUNNING ACCEPTANCE TEST SUITE (Culinary Operations Assistant)")
    print("=" * 85)

    all_passed = True
    results_summary = []

    # 1. Five Main Acceptance Tests
    print("\n--- STAGE 1: CORE ACCEPTANCE TESTS (Q1 - Q5) ---")
    for t in ACCEPTANCE_TESTS:
        res = ask(t["question"])
        passed, detail = t["check"](res)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        results_summary.append((t["id"], status, detail, res.get("answer", "").replace("\n", " ")[:70]))
        print(f"\n[{status}] {t['id']}: {t['question'][:65]}...")
        print(f"       Result Detail : {detail}")
        print(f"       Sources Cited : {res.get('sources')}")
        print(f"       Refused       : {res.get('refused')}")
        print(f"       Actual Answer :\n       {res.get('answer')}")

    # 2. Five Paraphrase Tests
    print("\n--- STAGE 2: IN-SCOPE PARAPHRASE TESTS (P1 - P5) ---")
    for p in PARAPHRASE_TESTS:
        res = ask(p["question"])
        ans_lower = res.get("answer", "").lower()
        has_keywords = all(k.lower() in ans_lower for k in p["must_contain"])
        cites_page = p["expected_page"] in res.get("sources", []) or f"page {p['expected_page']}" in ans_lower
        passed = (not res.get("refused", True)) and has_keywords and cites_page
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        detail = f"keywords={has_keywords}, page_{p['expected_page']}={cites_page}"
        results_summary.append((p["id"], status, detail, res.get("answer", "").replace("\n", " ")[:70]))
        print(f"[{status}] {p['id']}: {p['question']} -> Page {res.get('sources')}, Refused={res.get('refused')}")

    # 3. Ten Out-Of-Scope Tests
    print("\n--- STAGE 3: OUT-OF-SCOPE REFUSAL TESTS (OOS1 - OOS10) ---")
    for oos_id, q in OUT_OF_SCOPE_TESTS:
        res = ask(q)
        ans = res.get("answer", "").strip()
        refused = res.get("refused", False)
        exact_refusal = (ans == "I cannot find this information in the manual.")
        passed = refused and exact_refusal
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        detail = f"refused={refused}, exact_refusal={exact_refusal}"
        results_summary.append((oos_id, status, detail, ans[:70]))
        print(f"[{status}] {oos_id}: {q[:50]}... -> Refused={refused}")

    # Summary Table
    print("\n" + "=" * 85)
    print("FINAL TEST RESULTS TABLE")
    print("=" * 85)
    print(f"{'Test ID':<8} | {'Status':<6} | {'Verification Detail':<40} | {'Answer Snippet'}")
    print("-" * 85)
    for tid, stat, det, ans_snip in results_summary:
        print(f"{tid:<8} | {stat:<6} | {det:<40} | {ans_snip}...")
    print("=" * 85)

    if all_passed:
        print("\nALL 20 ACCEPTANCE & GENERALIZATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("\nSOME TESTS FAILED. PLEASE REVIEW LOGS.")

    return all_passed

# Pytest compatibility
def test_q1():
    res = ask(ACCEPTANCE_TESTS[0]["question"])
    passed, detail = check_q1(res)
    assert passed, f"Q1 Failed: {detail} | Answer: {res.get('answer')}"

def test_q2():
    res = ask(ACCEPTANCE_TESTS[1]["question"])
    passed, detail = check_q2(res)
    assert passed, f"Q2 Failed: {detail} | Answer: {res.get('answer')}"

def test_q3():
    res = ask(ACCEPTANCE_TESTS[2]["question"])
    passed, detail = check_q3(res)
    assert passed, f"Q3 Failed: {detail} | Answer: {res.get('answer')}"

def test_q4():
    res = ask(ACCEPTANCE_TESTS[3]["question"])
    passed, detail = check_q4(res)
    assert passed, f"Q4 Failed: {detail} | Answer: {res.get('answer')}"

def test_q5():
    res = ask(ACCEPTANCE_TESTS[4]["question"])
    passed, detail = check_q5(res)
    assert passed, f"Q5 Failed: {detail} | Answer: {res.get('answer')}"

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
