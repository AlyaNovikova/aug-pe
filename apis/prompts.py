# LABELS = [
#     "[[NAME:Medical_personnel]]", "[[NAME:patient]]", "[[NAME:other]]",
#     "[[ADDRESS]]", "[[DATE]]",
#     "[[CONTACT: Telephone]]", "[[CONTACT: Fax]]", "[[CONTACT: Email]]",
#     "[[ID: SocialID]]", "[[ID: MedicalID]]", "[[ID: InsuranceID]]",
#     "[[NUMBER: Account]]", "[[NUMBER: License]]", "[[NUMBER: VehicleID]]", "[[NUMBER: DeviceID]]",
#     "[[URL]]", "[[IPAdress]]",
#     "[[DEMOGRAPHIC: Age]]", "[[DEMOGRAPHIC: CivilStatus]]", "[[DEMOGRAPHIC: Nationality]]", "[[DEMOGRAPHIC: Profession]]",
#     "[[HOSPITAL: Service]]", "[[HOSPITAL: Building]]", "[[HOSPITAL: Room-Bed]]",
#     "[[PersonalRelation]]"
# ]

LABELS = [
    "[[NAME-1A]]", "[[NAME-1M]]", "[[NAME-1F]]", "[[NAME-2A]]",
    "[[NAME-2M]]", "[[NAME-2F]]", "[[NAME-3A]]", "[[NAME-3M]]",
    "[[NAME-3F]]", "[[AGE]]", "[[CONTACT]]", "[[DATE]]",
    "[[HOSPITAL]]", "[[ID]]", "[[LANGUAGE]]", "[[LOCATION]]",
    "[[PROFESSION]]", "[[OTHER]]"
]

STYLES = [
    "in a professional way", "in a professional tone", "in a professional style",
    "in a professional clinical tone", "using concise medical terminology",
    "with thorough clinical details", "in a structured but natural clinical narrative",
    "with precise medical observations", "including relevant clinical context",
    "with appropriate medical abbreviations", "in a detailed but readable style"
]

DOC_TYPES = [
    "discharge summary", 
    # "radiology report",
    # "consultation note", "progress note", "operative report",
    # "emergency room note", "pathology report", "nursing note",
    # "physician's order", "admission note", "clinical discharge note", "outpatient clinic note"
    ]

SPECIALTIES = [
    "cardiology", "neurology", "oncology", "pediatrics", "orthopedics",
    "internal medicine", "general surgery", "psychiatry", "endocrinology",
    "pulmonology", "gastroenterology", "nephrology"
]

INSTRUCTION_TEMPLATES = [
        lambda doc, spec, sty, lbl: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 

        Add sections like Patient Identification, Chief Complaint, 
        History of Present Illness, Assessment and Plan, Any relevant procedures or treatments and others.
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline (\n) between every section
        - Use ==== dividers between major sections

        """,
    


        lambda doc, spec, sty, lbl: f"""Write a synthetic {doc} in narrative form {sty} for a {spec} patient. 
        Generate it in the exact style of medical discharge letter.
    Don't structure it too much. It should be a natural medical live recording.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets like this [[LABEL]], use only this labels: {lbl}. 
    Begin with patient presentation, then describe:
    - The clinical reasoning process
    - Diagnostic findings
    - Therapeutic interventions
    - Follow-up plans
    
    CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions. Put all Protected Health Information into the [[label]] format 
        - Newline (\n) between every section
        - Use ==== dividers between major sections

    """,

        lambda doc, spec, sty, lbl: f"""Generate a synthetic {doc} for a {spec} case {sty} following medical discharge letter structure.

    REQUIRED SECTIONS (maybe not all of them and you can add more different) (separated by newlines):
    1. Patient header (Name, Unit#, Admission/Discharge dates, DOB, Sex, Service)
    2. Allergies
    3. Chief Complaint
    4. History of Present Illness (with detailed timeline)
    5. Past Medical History
    6. Social History
    7. Physical Exam (with system-based bullet points)
    8. Pertinent Results (lab format with timestamps)
    9. Brief Hospital Course
    10. Discharge Diagnoses
    11. Discharge Medications (formatted list)
    12. Discharge Instructions
    13. Follow-up Information
    14. Signature line

    PHI TAGGING RULES:
    - Use ONLY these [[LABEL]] formats: {lbl}
    - Tag ALL instances of: names, dates, IDs, contacts, locations
    - Tag all other Protected Health Information but only with labels from labels list: {lbl}
    - Include at least 8 [[TAG]] instances throughout document

    FORMATTING REQUIREMENTS:
    - Newline (\n) between every section
    - Use ==== dividers between major sections
    - Bulleted physical exam findings
    - Indented medication lists

    CONTENT GUIDELINES:
    - Maintain realistic clinical flow for {spec}
    - Use appropriate medical terminology
    - Make [[TAGS]] blend naturally into text""",

        lambda doc, spec, sty, lbl: f"""Create a {doc} for {spec} {sty} that perfectly mimics medical discharge letter documentation style.

    You can use this DOCUMENT STRUCTURE or you can change it, do in a medical discharge letter style:
    1. HEADER SECTION (demographics with name, date and so on tags)
    2. Chief Complaint (1-2 sentences)
    3. HPI (paragraph with [[DATE]]-referenced events)
    4. PMH/SocialHx/FamilyHx ([[OTHER]] for sensitive social details)
    5. Physical Exam (system-based with bullet points)
    6. Results (lab/imaging)
    7. Hospital Course (narrative with [[HOSPITAL]] references)
    8. Discharge Plan 
    9. Follow-up 

    Key requirements: all PHI (Protected Health Information) must be tagged in double brackets [[ ]]. 
    And use only these PHI labels: {lbl}

    CRITICAL INSTRUCTIONS:
    - ALL PHI must use [[LABEL]] format - no exceptions
    - ==== dividers between major sections
    - Newline (\n) between every section
    - Maintain natural clinical narrative flow""",

        lambda doc, spec, sty, lbl: f"""Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.

    You can write with some of the following sections:
    • Patient header with name, age and so on
    • Chief Complaint section
    • Detailed HPI with [[DATE]]-anchored symptoms
    • PMH/Allergies/SocialHx (use [[OTHER]] for sensitive info)
    • System-based Physical Exam (bullet points)
    • Lab/Imaging results
    • Hospital Course narrative
    • Discharge Meds list 
    • Follow-up instructions 

    You should put all PHI TAGGING in double brackets [[ ]], 
    so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

    FORMATTING RULES:
    1. ALL PHI must use [[LABEL]] format - no exceptions
    2. \n between all sections
    3. ==== dividers after key sections"""

]


INSTRUCTION_TEMPLATES_WITH_SUMMARIES = [
    lambda doc, spec, sty, lbl, summary: f"""You are writing a synthetic {doc} for a {spec} patient in the style of a medical discharge letter.

    BEGINNING SUMMARY (REAL, FOR REFERENCE):
    ---
    {summary}
    ---

    Use this summary as inspiration. You MUST expand it significantly into a full-length document. Match the tone and structure, but increase the level of detail and medical reasoning.

    REQUIREMENTS:
    - Include all standard discharge summary sections
    - Use appropriate clinical language and reasoning
    - Make the document **longer and more detailed** than the summary
    - Replace ALL sensitive data using the following PHI tags in double brackets [[LABEL]]: {lbl}

    FORMATTING:
    - Use newline (\n) between sections
    - Use ==== dividers between major sections
    - Include at least 8 instances of [[PHI]] tags

    Tone: {sty}""",


    lambda doc, spec, sty, lbl, summary: f"""Write a detailed, narrative-style synthetic {doc} for a {spec} case, using the real summary below as a guide:

    REAL SUMMARY:
    ---
    {summary}
    ---

    TASK:
    Expand this summary into a much longer document in the form of a medical discharge letter. Use it as a **base for structure and content ideas**, but enrich it with in-depth clinical reasoning, findings, diagnostics, and outcomes.

    INSTRUCTIONS:
    - The final output should be significantly **longer and more thorough** than the summary.
    - Use only these PHI labels in [[LABEL]] format: {lbl}
    - Mimic realistic clinical tone and terminology

    RECOMMENDED FLOW:
    - Patient Presentation
    - HPI with Timeline
    - Diagnostics and Exam
    - Clinical Reasoning
    - Hospital Course
    - Medications, Follow-up Plan, and Education

    FORMATTING:
    - Use \n between sections
    - Use ==== as dividers for major blocks
    - Include at least 8 different [[PHI]] placeholders throughout""",


    lambda doc, spec, sty, lbl, summary: f"""Generate a synthetic {doc} in the style of a professional medical discharge letter for a {spec} case.

    Below is a real discharge summary:
    ---
    {summary}
    ---

    Your task is to reconstruct a **full clinical case** from this short summary. Expand it into a longer, structured, and fully detailed discharge summary.

    CRITICAL GUIDELINES:
    - Write more than the original summary; include nuanced details, timelines, findings, and follow-up care
    - Replace all PHI using only the following tags in double brackets [[LABEL]]: {lbl}
    - Maintain realistic medical structure, language, and progression

    FORMATTING:
    - \n between sections
    - ==== dividers between major blocks
    - At least 8 instances of [[PHI]] labels

    Tone should be: {sty}""",


        lambda doc, spec, sty, lbl, summary: f"""Simulate a full medical discharge encounter as a synthetic {doc} for a {spec} case, written in {sty}.

    REAL CLINICAL SUMMARY (USED AS A BACKDROP):
    ---
    {summary}
    ---

    Do NOT repeat the summary. Instead, simulate what the **full encounter** might have looked like based on it. Include realistic expansion of each phase of the patient's hospital stay.

    IMPORTANT:
    - The generated document should be significantly **longer and richer** than the summary
    - Tag all PHI using double brackets with ONLY these labels: {lbl}

    SUGGESTED SECTIONS:
    - Patient Header
    - Chief Complaint
    - History of Present Illness
    - Physical Exam
    - Results
    - Hospital Course
    - Discharge Diagnoses, Medications, and Plan
    - Follow-up Instructions""",


    lambda doc, spec, sty, lbl, summary: f"""Generate a synthetic {doc} for a {spec} case {sty}, using the exact style of a real medical discharge letter.

    Your task is to reconstruct a **full clinical case** from this short summary.
    USE THIS REAL SUMMARY AS YOUR GUIDE:
    ---
    {summary}
    ---

    USE IT AS A BASE to match:
    - Structure and section flow
    - Style and tone
    - Clinical phrasing

    Do not copy it directly. Use it to inspire the synthetic case.

    REQUIREMENTS:
    - Replace all PHI and sensitive information with labels from this list: {lbl}. Use double brackets like  [[DATE]], etc.
    - Maintain realism, clinical logic, and coherent progression.
    - Use professional medical terminology.

    RECOMMENDED SECTIONS:
    1. Patient Header (demographics)
    2. Chief Complaint
    3. History of Present Illness
    4. Past Medical History
    5. Physical Exam
    6. Hospital Course
    7. Labs/Imaging
    8. Discharge Diagnosis
    9. Discharge Medications
    10. Discharge Instructions
    11. Follow-up Plan

    CRITICAL INSTRUCTIONS:
    - ALL PHI must be tagged in [[LABEL]] format, no exceptions
    - Stick to the tone, structure, and detail level shown in the summary above.
    """


]