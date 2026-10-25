# Customer Complaint Handling Procedure

Source ID: FM-PROC-001
Version: 1.0
Effective date: 2026-01-01
Company: FreshMart Australia (fictional)

This document is a fictional demonstration procedure, not the procedure of any
real Australian retailer.

## Quality complaint intake

1. Explain relevant policy using the current fictional source documents.
2. Offer the fresh produce quality complaint form.
3. Collect store, product, purchase date, issue description, and an optional
   yes/no proof-of-purchase indicator. Use synthetic information only.
4. Show the summary and require explicit confirmation before submission.
5. Application code validates the details and creates the case transactionally.
6. Return only the actual Case ID from the persisted record.

Do not collect customer names, contact details, payment details, or attachments.
Never invent a Case ID or claim success after a failed database write.

## Case states

New cases have status "Submitted". Authorised staff may later mark a case
"Under review" or "Closed". These statuses do not mean that a refund was
approved or paid. Staff case-management actions are outside this prototype.

## Store review

Store managers review complaint patterns alongside recorded quality inspections
and sales. They may recommend investigating stock handling, suppliers, or
inspection practices. Correlation does not establish causation.

## Service limitations

Do not promise a response deadline: this fictional procedure specifies none.
If policy evidence is missing or a service is unavailable, explain the
limitation. The language model must never write to the complaint database.
