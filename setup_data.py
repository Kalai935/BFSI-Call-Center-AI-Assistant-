import os
import json

# 1. Create Data Directory
if not os.path.exists('data'):
    os.makedirs('data')
    print("Created 'data' folder.")

# 2. Create Tier 1 Dataset (Alpaca Format - 150+ Samples Simulation)
# (I am adding 10 high-quality samples here. In a real scenario, you'd paste 150)
dataset = [
    {
        "instruction": "What are the eligibility criteria for a personal loan?",
        "input": "",
        "output": "To be eligible for a personal loan, you must be between 21-60 years old, have a minimum monthly income of $1,500, and a credit score above 700."
    },
    {
        "instruction": "How do I block my debit card immediately?",
        "input": "",
        "output": "You can block your debit card immediately via our Mobile App under 'Card Settings' > 'Block Card', or by calling our 24/7 hotline at 1-800-BANK-HELP."
    },
    {
        "instruction": "What is the current interest rate for a home loan?",
        "input": "",
        "output": "Our current floating interest rate for home loans starts at 8.40% p.a. for salaried employees and 8.60% p.a. for self-employed individuals."
    },
    {
        "instruction": "I want to close my savings account.",
        "input": "",
        "output": "To close your savings account, please visit your nearest branch with your ID proof and unused chequebook. Account closure charges of $15 apply if closed within 6 months."
    },
    {
        "instruction": "What documents are required for a car loan?",
        "input": "",
        "output": "You need: 1. ID Proof (Passport/License), 2. Address Proof, 3. Last 3 months' salary slips, and 4. Bank statements for the last 6 months."
    },
    {
        "instruction": "How do I activate internet banking?",
        "input": "",
        "output": "Go to our website, click 'New User Registration', enter your Account Number and CIF ID. An OTP will be sent to your registered mobile to set your password."
    },
    {
        "instruction": "What is the daily withdrawal limit for an ATM card?",
        "input": "",
        "output": "For a Classic Debit Card, the daily withdrawal limit is $500. For a Platinum/Gold Card, the limit is $1,000 per day."
    },
    {
        "instruction": "How can I update my phone number?",
        "input": "",
        "output": "Phone number updates require a visit to the branch for security reasons. Please carry your ID proof. It takes 24 hours to update."
    },
    {
        "instruction": "Check my loan application status.",
        "input": "",
        "output": "You can check your status at www.bank.com/status by entering your Application Reference Number (ARN) sent to your email."
    },
    {
        "instruction": "What is the penalty for bouncing a cheque?",
        "input": "",
        "output": "The penalty for an inward cheque return due to insufficient funds is $25 per instance."
    }
]

with open('data/bfsi_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)
print("Created 'data/bfsi_dataset.json' with Tier 1 data.")

# 3. Create Tier 3 Knowledge Base (RAG Documents)
knowledge_text = """
BANK POLICY DOCUMENT: LOANS AND ACCOUNTS 2024

1. PREPAYMENT PENALTY RULES
For floating rate term loans sanctioned to individual borrowers for purposes other than business, no prepayment charges are applicable.
For fixed-rate personal loans, a penalty of 2% of the principal outstanding + GST is applicable if the loan is closed within 12 months of disbursement. No penalty after 12 months.

2. MINIMUM BALANCE CHARGES (SAVINGS)
The Minimum Average Balance (MAB) required is:
- Metro/Urban branches: $500
- Semi-Urban branches: $250
- Rural branches: $100
Non-maintenance charges: 5% of the shortfall or $10, whichever is lower.

3. GOLD LOAN SCHEME (TIER 3 DETAILS)
- Maximum Loan to Value (LTV): 75% of the gold value.
- Interest Rate: 9.90% p.a. fixed.
- Tenure: 6 months to 24 months.
- Collateral: Gold ornaments of 18 Karat purity or higher only.
- Processing Fee: 1% of loan amount (Min $10).

4. CREDIT CARD REWARD POINTS
- 1 Point for every $2 spent.
- Points expire after 2 years.
- Redemption value: 1 Point = $0.25.
"""

with open('data/knowledge_base.txt', 'w') as f:
    f.write(knowledge_text)
print("Created 'data/knowledge_base.txt' with Tier 3 data.")