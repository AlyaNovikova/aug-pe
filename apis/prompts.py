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
        Generate it in the exact style of MIMIC-IV dataset.
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
        Generate it in the exact style of MIMIC dataset.
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

        lambda doc, spec, sty, lbl: f"""Generate a synthetic {doc} for a {spec} case {sty} following MIMIC-IV structure exactly.

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

        lambda doc, spec, sty, lbl: f"""Create a {doc} for {spec} {sty} that perfectly mimics MIMIC's documentation style.

    You can use this DOCUMENT STRUCTURE or you can change it, but do in a mimic style:
    1. HEADER SECTION (demographics with name, date and so on tags)
    2. Chief Complaint (1-2 sentences)
    3. HPI (paragraph with [[DATE]]-referenced events)
    4. PMH/SocialHx/FamilyHx ([[OTHER]] for sensitive social details)
    5. Physical Exam (system-based with bullet points)
    6. Results (lab/imaging in MIMIC's raw data format)
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

        lambda doc, spec, sty, lbl: f"""Generate a {doc} for {spec} {sty} adhering strictly to MIMIC conventions.

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