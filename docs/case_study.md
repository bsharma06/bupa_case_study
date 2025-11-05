# ABC HealthCare Case Study

**Internal Audit – Payment Process at ABC HealthCare**  
**Leveraging Data Science and AI to Strengthen Payment Oversight**

---

## Background

ABC HealthCare is a global pharmaceutical company specializing in cancer research. It operates with:

- **206 independent research teams** across universities and research organizations worldwide
- **Centralized legal and finance services** managed by ABC HealthCare
- **Payments team based in Thailand**, handling over 200,000 payment requests annually

### Payment Oversight Structure

| Invoice Amount | Submission Authoriser | Payment Authoriser |
| -------------- | --------------------- | ------------------ |
| < $1,000       | Submitter             | Payment Analyst    |
| < $5,000       | Submitter's Manager   | Payment Manager    |
| > $5,000       | Submitter's Director  | Payment Director   |

### Whistleblower Report Highlights

- Fraudulent invoices submitted by research teams
- Invoices not matching claims
- Employee fraud within the finance team

A preliminary investigation recommends a **comprehensive audit using advanced data analytics**.

---

## Dataset Overview

You are provided with a `.zip` file containing 4 datasets:

### 1. payments_master.csv

Includes 2000 sample payment records with fields:

- Date received: When the invoice was received.
- Date of invoice: When the invoice was issued.
- Date of authorisation: When the invoice was approved.
- Payment due date: When the payment is due.
- Date of payment: When the payment was made.
- Research team: Team responsible for the expense.
- Submitted by: Person who submitted the invoice.
- Authorised by: Person who approved the invoice.
- Payment authoriser: Person who approved the payment.
- Invoice number: Unique identifier for the invoice.
- Description of spend: What the invoice is for.
- Invoice value: Amount on the invoice.
- Payment amount: Amount paid.
- Payment Status: e.g., Paid, Pending.
- Type of expense: e.g., Lab suppliers, Consulting, etc.
- Company: Vendor or service provider.
- phone_number: Contact number for the company or submitter.
- email: Contact email.

### 2. research_team_master.csv

Includes:

- Research team: Name or identifier of the research group.
- Director: Person leading the team.
- Location: Geographic or institutional location.
- Affiliation: Associated institution or department.
- Research type: Field of research (e.g., Basic Research', 'Translational', etc.)
- Annual budget: Total yearly funding allocated to the team.
- Item type: Specific budget item (e.g., Lab suppliers, Consulting, etc.).
- Item budget: budget associated with the item.
- Comments: Additional notes or context.

### 3. research_team_member_master.csv

Includes:

- Team: The name or identifier of the team.
- Location: Where the team or individual is based.
- Name: The name of the team member.
- Role: The position or function of the person within the team.

### 4. fraud_cases_master.csv

Same structure as Payments Master, with an added `Fraud_flag` column:

- `1` = Fraudulent
- `0` = Not fraudulent

---

## Case Study Tasks

### Section 1: Descriptive & Diagnostic Analytics

#### A. Visual Analysis (Power BI or similar)

- Identify teams with highest spend
- Track volume/time trends by category and location
- Analyze average payment time by expense type, value, and location
- Compare team spend vs. budget

#### B. Regression Analysis

**Hypothesis**: All invoices take ~same time ±1 day  
**Task**: Predict time to payment using regression or similar modeling

#### C. Outlier Detection

**Hypothesis**: Invoices are random; no fraud pattern  
**Task**: Use Z-scores or similar to detect statistical outliers for audit sampling

---

### Section 2: Machine Learning & AI

#### A. Machine Learning

Build and explain:

- **Supervised model** to predict fraudulent payments
- **Unsupervised model** to detect anomalies

Include evaluation metrics like:

- Confusion matrix
- Precision, recall, F1-score
- ROC-AUC or perplexity

#### B. AI-Generated Audit Report

Demonstrate how an AI tool (e.g., GPT) can:

- Summarize findings from Sections 1 & 2
- Generate audit insights and recommendations

---

## Submission Requirements

- **Presentation**: Max 10 slides for Group Internal Audit Leadership Team
- **Files**: Include scripts, notebooks, apps, or tools used
- **Documentation**: Explain your process, assumptions, and conclusions

Identify the research teams with the highest spend
Track volume and time trends by category and location
average payment time by expense type, value and location;
comparing each team spend against their budget.
