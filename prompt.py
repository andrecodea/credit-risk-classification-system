def create_prompt(
    age,
    sex,
    job,
    housing,
    savings,
    checking,
    credit_amount,
    duration,
    purpose,
    risk_label,
    confidence,
):
    return f"""
You are a credit risk analyst. Based on the following applicant profile, 
provide a strategic recommendation for a loan approval committee:

Applicant Profile:
- Age: {age}
- Sex: {sex}
- Job Type: {job} (0=unskilled, 1=skilled, 2/3=highly skilled)
- Housing: {housing}
- Savings Account: {savings}
- Checking Account: {checking}
- Credit Amount: ${credit_amount}
- Duration: {duration} months
- Purpose: {purpose}

Model Assessment: {risk_label} (Confidence: {confidence})

Please provide:
1. Risk assessment summary (2-3 sentences)
2. Approval/rejection recommendation
3. Suggested conditions if approved (collateral, interest rate, etc.)
4. Additional checks recommended

Keep your response concise and professional.
"""
