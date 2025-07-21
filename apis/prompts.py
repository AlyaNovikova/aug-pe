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
    "[[NAME]]", "[[AGE]]", "[[CONTACT]]", "[[DATE]]",
    "[[HOSPITAL]]", "[[ID]]", "[[LANGUAGE]]", "[[LOCATION]]",
    "[[PROFESSION]]", "[[OTHER]]"
]

phi_list = [
    "NAME", "AGE", "CONTACT", "DATE",
    "HOSPITAL", "ID", "LANGUAGE", "LOCATION",
    "PROFESSION"
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

# SPECIALTIES = [
#     "cardiology", "neurology", "oncology", "pediatrics", "orthopedics",
#     "internal medicine", "general surgery", "psychiatry", "endocrinology",
#     "pulmonology", "gastroenterology", "nephrology"
# ]

SPECIALTIES = [
    "neuroradiology", 
    "musculoskeletal radiology",
    "oncology",
    "radiology oncology"
    "abdominal radiology",
    "chest radiology",
    "breast imaging",
    "interventional radiology",
    "pediatric radiology",
    "cardiothoracic radiology",
    "emergency radiology",
    "nuclear medicine"
]

MIMIC_SECTIONS = """
1. Header with Name and Unit No, ID, Admission Date, Discharge Date, Date of Birth, Sex, Service
2. Allergies
3. Attending Physician
4. Chief Complaint
5. Major Surgical Procedure
6. History of Present Illness
7. Past Medical History
8. Social History
9. Family History
10. Physical Exam (Admission)
11. Physical Exam (Discharge)
12. Pertinent Results (Admission)
13. Pertinent Results (Discharge)
14. Microbiology Results
15. Imaging Results
16. Brief Hospital Course
17. Medications on Admission
18. Discharge Medications
19. Discharge Disposition
20. Discharge Facility
21. Discharge Diagnosis
22. Discharge Condition
23. Discharge Instructions
24. Followup Instructions
"""

INSTRUCTION_TEMPLATES = [
        lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Write a long text, use approximatly {length} words.
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 

        Add sections like 
            - Patient Demographics
            - Medical History
            - Present Illness
            - Physical Exams
            - Test Results
            - Hospital Course
            - Discharge Information
            - Follow-up Plan.
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline (\n) between every section
        - Use ==== dividers between major sections

        """,
    


        lambda doc, spec, sty, lbl, length: f"""Write a synthetic {doc} in narrative form {sty} for a {spec} patient. 
        Generate it in the exact style of medical discharge letter. Write a long text, use approximatly {length} words.
    Don't structure it too much. It should be a natural medical live recording.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets like this [[LABEL]], use only this labels: {lbl}. 
    Begin with patient presentation, then describe:
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
        - ALL PHI must use [[LABEL]] format - no exceptions. Put all Protected Health Information into the [[label]] format 
        - Newline (\n) between every section
        - Use ==== dividers between major sections

    """,

        lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty} following medical discharge letter structure.
Write a long text, use approximatly {length} words.
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

        lambda doc, spec, sty, lbl, length: f"""Create a {doc} for {spec} {sty} that perfectly mimics medical discharge letter documentation style.
Write a long text, use approximatly {length} words.
    You can use this DOCUMENT STRUCTURE or you can change it, do in a medical discharge letter style:
        1. Header with Name and Unit No, ID, Admission Date, Discharge Date, Date of Birth, Sex, Service
        2. Allergies
        3. Attending Physician
        4. Chief Complaint
        5. Major Surgical Procedure
        6. History of Present Illness
        7. Past Medical History
        8. Social History
        9. Family History
        10. Physical Exam (Admission)
        11. Physical Exam (Discharge)
        12. Pertinent Results (Admission)
        13. Pertinent Results (Discharge)
        14. Microbiology Results
        15. Imaging Results
        16. Brief Hospital Course
        17. Medications on Admission
        18. Discharge Medications
        19. Discharge Disposition
        20. Discharge Facility
        21. Discharge Diagnosis
        22. Discharge Condition
        23. Discharge Instructions
        24. Followup Instructions

    Key requirements: all PHI (Protected Health Information) must be tagged in double brackets [[ ]]. 
    And use only these PHI labels: {lbl}

    CRITICAL INSTRUCTIONS:
    - ALL PHI must use [[LABEL]] format - no exceptions
    - ==== dividers between major sections
    - Newline (\n) between every section
    - Maintain natural clinical narrative flow""",

        lambda doc, spec, sty, lbl, length : f"""Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
Write a long text, use approximatly {length} words.
    You can write with some of the following sections:
        1. Header with Name and Unit No, ID, Admission Date, Discharge Date, Date of Birth, Sex, Service
        2. Allergies
        3. Attending Physician
        4. Chief Complaint
        5. Major Surgical Procedure
        6. History of Present Illness
        7. Past Medical History
        8. Social History
        9. Family History
        10. Physical Exam (Admission)
        11. Physical Exam (Discharge)
        12. Pertinent Results (Admission)
        13. Pertinent Results (Discharge)
        14. Microbiology Results
        15. Imaging Results
        16. Brief Hospital Course
        17. Medications on Admission
        18. Discharge Medications
        19. Discharge Disposition
        20. Discharge Facility
        21. Discharge Diagnosis
        22. Discharge Condition
        23. Discharge Instructions
        24. Followup Instructions

    You should put all PHI TAGGING in double brackets [[ ]], 
    so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

    FORMATTING RULES:
    1. ALL PHI must use [[LABEL]] format - no exceptions
    2. \n between all sections
    3. ==== dividers after key sections"""

]


INSTRUCTION_TEMPLATES_WITH_SUMMARIES = [
    lambda doc, spec, sty, lbl, length, summary: f"""You are writing a synthetic {doc} for a {spec} patient in the style of a medical discharge letter.
Write a long text, use approximatly {length} words.
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


    lambda doc, spec, sty, lbl, length, summary: f"""Write a detailed, narrative-style synthetic {doc} for a {spec} case, using the real summary below as a guide:
Write a long text, use approximatly {length} words.
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

    FORMATTING:
    - Use \n between sections
    - Use ==== as dividers for major blocks
    - Include at least 8 different [[PHI]] placeholders throughout""",


    lambda doc, spec, sty, lbl, length, summary: f"""Generate a synthetic {doc} in the style of a professional medical discharge letter for a {spec} case.
Write a long text, use approximatly {length} words.
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


        lambda doc, spec, sty, lbl, length, summary: f"""Simulate a full medical discharge encounter as a synthetic {doc} for a {spec} case, written in {sty}.
Write a long text, use approximatly {length} words.
    REAL CLINICAL SUMMARY (USED AS A BACKDROP):
    ---
    {summary}
    ---

    Do NOT repeat the summary. Instead, simulate what the **full encounter** might have looked like based on it. Include realistic expansion of each phase of the patient's hospital stay.

    IMPORTANT:
    - The generated document should be significantly **longer and richer** than the summary
    - Tag all PHI using double brackets with ONLY these labels: {lbl}

    SUGGESTED SECTIONS:
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
    13. Follow-up Information""",


    lambda doc, spec, sty, lbl, length, summary: f"""Generate a synthetic {doc} for a {spec} case {sty}, using the exact style of a real medical discharge letter.
Write a long text, use approximatly {length} words.
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


FEW_SHOT_EXAMPLES = [
    """ 
Name: [[NAME-1M]] Unit No: [[ID]]

Admission Date: [[DATE]] Discharge Date: [[DATE]]

Date of Birth: [[DATE]] Sex: M

Service: MEDICINE

Allergies:
Sulfonamides

Attending: [[NAME-2A]].

Chief Complaint:
Shortness of Breath, Fever

Major Surgical or Invasive Procedure:
Right Pleural Chest Tube Placement

History of Present Illness:
Mr. [[NAME-1M]] is a [[AGE]] y/o man with a history notable only for
hypertension and hyperlipidemia who presented to [[HOSPITAL]] with
five days of progressive dyspnea on exertion, pleuritic right
sided chest pain, productive cough with rust‑colored sputum,
subjective fevers, and generalized malaise. He also noted new
polyuria, polydipsia, and unintentional 10‑pound weight loss
over two months. No prior diagnosis of diabetes. Denied recent
travel, hemoptysis, syncope, orthopnea, leg swelling, abdominal
pain, nausea, vomiting, diarrhea, or dysuria. No prior chronic
lung disease. Vaccination history for pneumococcus uncertain.

At an urgent care earlier the day of admission he was febrile
(39.1°C) and tachycardic; fingerstick glucose was reported >400
mg/dL prompting referral to the ED. On arrival he was hypoxic to
88% on room air, improved to 95% on 2 L nasal cannula.

Initial labs demonstrated leukocytosis, hyperglycemia with mild
anion gap metabolic acidosis, elevated serum beta‑hydroxybutyrate
consistent with mild diabetic ketoacidosis (DKA), and acute
kidney injury likely prerenal. Chest radiograph and CT chest
showed multilobar right‑predominant pneumonia with a moderate
loculated right pleural effusion concerning for complicated
parapneumonic effusion/empyema. Broad‑spectrum antibiotics were
initiated (ceftriaxone plus azithromycin) after cultures. He
received IV fluids and insulin infusion for DKA. Interventional
Radiology placed a right pleural pigtail catheter with drainage
of purulent fluid. Pleural fluid studies were exudative with low
pH and elevated LDH. Blood cultures grew Streptococcus
pneumoniae in 2/2 sets (pan‑sensitive). Antibiotics were narrowed
to high‑dose IV penicillin G then transitioned to oral high‑dose
amoxicillin upon sterilization of blood cultures.

He developed transient atrial flutter with rapid ventricular
response (rates 140s) on hospital day 2 during sepsis, which was
chemically rate‑controlled with IV then PO metoprolol. No prior
arrhythmia history. Transthoracic echocardiogram revealed normal
biventricular systolic function and no valvular vegetations.

Over his hospitalization his oxygen requirement resolved, kidney
function normalized with volume repletion, the anion gap closed,
and the chest tube was removed after drainage diminished and
follow‑up imaging showed resolution of the loculated effusion.
Newly diagnosed type 2 diabetes was transitioned to a basal‑
bolus regimen and diabetes education completed.

In the ED, initial VS were: 102.4 128 102/58 24 95% 2L NC
Exam notable for: Febrile, mildly diaphoretic. Mild increased
work of breathing with right lower lung field dullness and
decreased breath sounds. No JVD. Mild dry mucous membranes. No
peripheral edema. Oriented x3.
ECG: Sinus tachycardia; later atrial flutter (resolved).
Labs showed: Na 132, K 4.8, Cl 95, HCO3 18, BUN/Cr 32/1.6; WBC
18.4 H/H 12.1/36.0, Plt 265; Glucose 428; AG 19; beta‑OH
butyrate elevated; lactate 2.1; VBG pH 7.31.
Imaging showed:

CXR: Right middle and lower lobe consolidation with moderate
right pleural effusion.

CT Chest (non‑contrast): Multilobar right‑predominant
consolidation; moderate partly loculated right pleural effusion;
no pulmonary embolus (limited non‑contrast study); reactive
mediastinal nodes; mild hepatic steatosis.

CT Abdomen/Pelvis (screening for alternative source): No
intra‑abdominal source of sepsis; incidental colonic diverticulosis
without diverticulitis; no hydronephrosis.

Consults: Pulmonology; Interventional Radiology; Endocrinology

Pulmonology: Recommended diagnostic/therapeutic drainage of
complicated effusion and daily airway clearance regimen.

IR: Placed 14 Fr pigtail catheter into right pleural space on
[[DATE]] with immediate return of 650 mL purulent exudate; no
procedural complications.

Endocrinology: Assisted with transition from insulin infusion
to basal‑bolus regimen and outpatient diabetes management plan.

Patient received:
[[DATE]] 15:10 IV Ceftriaxone 2 g
[[DATE]] 15:15 IV Azithromycin 500 mg
[[DATE]] 15:25 30 mL/kg Isotonic IV fluid bolus
[[DATE]] 16:05 IV Regular Insulin infusion started

Transfer VS were: 99.1 116/70 90 20 97% 2L NC

On arrival to the floor, patient reported pleuritic pain improved
with positioning, denied new dyspnea at rest, and was tolerating
clear liquids.

Past Medical History:

Hypertension

Hyperlipidemia (no known ASCVD)

Social History:
Lives in [[LOCATION]] with his spouse [[NAME-3F]]. Works part‑time as a
retired teacher tutor. Never smoker. Rare alcohol (1–2 drinks at
family events). No illicit drug use. Exercises by walking dog
daily prior to illness. Diet previously high in refined
carbohydrates; limited formal exercise past month due to fatigue.

Family History:
Father with type 2 diabetes (dx in his 60s)
Mother deceased at [[AGE]] from stroke
Older brother with coronary artery disease s/p PCI
No family history of early sudden cardiac death or chronic lung
disease

Physical Exam:
ADMISSION
VS: 101.9 102/58 128 24 95 2L NC
GENERAL: Ill‑appearing but in no acute distress at rest
HEENT: EOMI, PERRL, mildly dry mucous membranes, no scleral icterus
NECK: Supple, no JVD
HEART: Tachycardic, regular initially then episodic flutter; no murmurs
LUNGS: Decreased breath sounds and dullness RLL/RML; scattered
crackles; no wheezes
ABDOMEN: Soft, NTND, normal bowel sounds
GU: No CVA tenderness; external exam deferred
EXTREMITIES: No clubbing or cyanosis; trace ankle edema
PULSES: 2+ radial/dorsalis pedis bilaterally
NEURO: A&Ox3, no focal deficits, normal speech
SKIN: Warm, slightly diaphoretic; no rash

DISCHARGE
VITALS: 97.8 118/72 78 16 97/Ra

PHYSICAL EXAM:
GENERAL: Comfortable, conversant, ambulating with assistance
HEENT: EOMI, PERRL, moist mucous membranes
NECK: No JVD
HEART: Regular rate and rhythm, no murmurs or gallops
LUNGS: Improved aeration; minimal residual crackles right base;
no wheezes; no use of accessory muscles
ABDOMEN: Soft, NTND
GU: Voiding spontaneously without difficulty
EXTREMITIES: No edema; full passive ROM
PULSES: 2+ DP pulses bilaterally
NEURO: A&Ox3, strength grossly intact, steady gait with walker
SKIN: Warm and well perfused; chest tube site clean, dry, intact

Pertinent Results:
ADMISSION/PERTINENT
[[DATE]] 15:05PM BLOOD WBC-18.4* RBC-4.10 Hgb-12.1* Hct-36.0*
MCV-88 MCH-29.5 MCHC-33.6 RDW-13.2 Plt-265
[[DATE]] 15:05PM BLOOD Neuts-88.0* Lymphs-6.0* Monos-5.5
Eos-0.2 Baso-0.3 Im-0.0 AbsNeut-16.2* AbsLymp-1.1
AbsMono-1.0* AbsEos-0.04 AbsBaso-0.05
[[DATE]] 15:05PM BLOOD PT-13.8 INR-1.1 PTT-31.2
[[DATE]] 15:05PM BLOOD Glucose-428* UreaN-32* Creat-1.6* Na-132*
K-4.8 Cl-95* HCO3-18* AnGap-19*
[[DATE]] 15:05PM BLOOD ALT-42 AST-50* AlkPhos-118* TotBili-0.6
Albumin-3.2* Lactate-2.1*
[[DATE]] 15:05PM BLOOD BetaHydroxyBut-3.2* (mmol/L)
[[DATE]] 17:40PM BLOOD VBG-pH-7.31 pCO2-35 HCO3-17*
[[DATE]] 18:15PM PLEURAL Color-Turbid* WBC-68,000* (90% Neut)
Protein-4.5 LDH-1650* Glucose-38* pH-6.9* GramStain-GPC pairs*
[[DATE]] 19:20PM BLOOD A1c-10.4*
[[DATE]] 21:10PM URINE Ketone-LG* Glucose-LG* Nitrite-NEG Protein-TR
Leuks-NEG RBC-0-2 WBC-0-2
[[DATE]] 23:55PM BLOOD TroponinT-0.012 (non-elevated)

DISCHARGE
[[DATE]] 06:30AM BLOOD WBC-7.4 RBC-4.35 Hgb-12.9 Hct-38.4
MCV-88 MCH-29.7 MCHC-33.6 RDW-13.0 Plt-322
[[DATE]] 06:30AM BLOOD PT-13.2 INR-1.0 PTT-30.1
[[DATE]] 06:30AM BLOOD Glucose-142* UreaN-14 Creat-0.9 Na-138
K-4.2 Cl-101 HCO3-25 AnGap-12
[[DATE]] 06:30AM BLOOD ALT-38 AST-32 AlkPhos-104 TotBili-0.5
Albumin-3.5
[[DATE]] 06:30AM BLOOD CRP-28* (down from 156*)

MICRO
[[DATE]] BLOOD CULTURE: Streptococcus pneumoniae (2/2 sets) – pan
sensitive (FINAL)
[[DATE]] BLOOD CULTURE: No growth (post-therapy repeat) (FINAL)
[[DATE]] SPUTUM CULTURE: Moderate S. pneumoniae; normal oral flora
[[DATE]] PLEURAL FLUID CULTURE: S. pneumoniae (matching blood isolate)
[[DATE]] URINE ANTIGEN: S. pneumoniae POSITIVE
[[DATE]] URINE ANTIGEN: Legionella pneumophila Serogroup 1 NEGATIVE

IMAGING
CXR (Admission): Right middle and lower lobe consolidation with
moderate right pleural effusion; no pneumothorax.
CXR (Pre‑discharge): Marked reduction of right effusion; mild
residual basilar atelectasis; clear left lung; no pneumothorax.
CT Chest (Initial): Multilobar right‑predominant consolidation;
loculated right pleural effusion; no cavitation; no central
pulmonary embolus on limited non‑contrast evaluation. Mild
diffuse bronchial wall thickening.
CT Chest (Interval [[DATE]]): Decreased size of right pleural effusion
with pigtail catheter in situ; improved aeration of RML; no new
consolidation.
TTE: Normal LV size and systolic function (LVEF ~60%). No wall
motion abnormalities. Normal RV function. No significant valvular
regurgitation or stenosis. No vegetations. Mild left atrial
enlargement. Estimated PASP normal. No pericardial effusion.
Abdominal Ultrasound: Mild hepatic steatosis; normal biliary
tree; kidneys normal size without hydronephrosis.
Right Pleural Ultrasound (Guidance): Complex septated anechoic
collection consistent with loculated effusion; successful
catheter placement.
Lower Extremity Venous Duplex: No DVT.

Brief Hospital Course:
This is an [[AGE]]-year-old man with HTN and HLD presenting with
severe community-acquired pneumonia complicated by sepsis,
loculated right parapneumonic effusion/empyema, mild DKA from
previously undiagnosed type 2 diabetes, transient atrial flutter
with RVR, and prerenal AKI. Managed with source control (pleural
drainage), targeted antimicrobials, insulin therapy, and
supportive care with clinical improvement and resolution of
organ dysfunction.

Severe Sepsis / Community-Acquired Pneumonia (Streptococcus pneumoniae):
Met criteria for sepsis with fever, tachycardia, leukocytosis,
and hypoxia. Blood and pleural cultures grew S. pneumoniae.
Initial broad therapy (ceftriaxone + azithromycin) narrowed to IV
penicillin G then to oral high-dose amoxicillin to complete a
total 14-day course (day 1 = first negative blood culture). Daily
clinical improvement; afebrile >72h prior to discharge.

Right Parapneumonic Effusion / Empyema:
Loculated effusion (low pH, high LDH) consistent with empyema.
IR placed pigtail catheter on [[DATE]] with drainage and serial
flushes. No need for intrapleural fibrinolytics as output steadily
declined and imaging improved. Catheter removed on [[DATE]] after
<50 mL drainage/24h and radiographic resolution.

Acute Hypoxic Respiratory Failure:
Secondary to multilobar pneumonia and effusion; required up to 4
L NC initially; weaned to room air by hospital day 4. Incentive
spirometry and airway clearance employed.

New Onset Type 2 Diabetes Mellitus with Mild DKA (Resolved):
Presented with glucose 428, AG 19, beta-hydroxybutyrate elevated,
mild acidosis. Managed with insulin infusion and IV fluids;
anion gap closed within 18 hours. Transitioned to basal insulin
(glargine) plus prandial lispro and started on metformin (renal
function normalized; counseled on GI side effects). A1c 10.4%.
Provided diabetes education and glucometer teaching. Outpatient
endocrinology follow-up arranged.

Atrial Flutter with RVR (Transient):
Occurred during sepsis/dyspnea (HD2) with rates 140s. Converted
to sinus rhythm after rate control (IV metoprolol). No recurrent
episodes after stabilization. CHADS-VASc 2 (age + HTN). Started
on low-dose anticoagulation (apixaban) after excluding empyema
drain bleeding risk; tolerated without bleeding. Outpatient
cardiology follow-up for rhythm surveillance.

Acute Kidney Injury, Prerenal (Resolved):
Admission Cr 1.6 (baseline ~0.9) with BUN/Cr ratio elevation;
improved to 0.9 after fluids and sepsis control. No intrinsic
disease suspected. Avoided nephrotoxins; monitored daily BMP.

Iron Deficiency Anemia (Mild):
Admission Hgb 12.1 (baseline unknown) with ferritin 58, Tsat 12%.
Likely chronic dietary; no overt bleeding. Began oral iron every
other day for absorption; plan outpatient colon cancer screening
per age if not current (deferred inpatient).

Mild Transaminitis / Hepatic Steatosis:
AST/ALT peaked 50/42; ultrasound consistent with steatosis; values
trended down with sepsis resolution. Lifestyle modification
counseled.

Deconditioning:
Early PT involvement; ambulating with rolling walker >200 ft by
discharge; home exercise plan provided.

TRANSITIONAL ISSUES:
[] Complete oral amoxicillin course to total 14 days (end date:
[[DATE]]).
[] Monitor blood glucose QID; adjust insulin with endocrinology.
[] Follow up A1c in ~3 months; consider adding GLP-1 RA if weight
goal not met.
[] Repeat CXR in 6–8 weeks to ensure radiographic resolution.
[] Continue iron supplementation every other day for 3 months;
repeat CBC and iron studies thereafter.
[] Cardiology follow-up for atrial flutter surveillance and
long-term anticoagulation assessment.
[] Encourage weight management, structured exercise, and low
refined carbohydrate diet for diabetes and steatosis.

Medications on Admission:
The Preadmission Medication list is accurate and complete.

Lisinopril 20 mg PO DAILY

Atorvastatin 40 mg PO QHS

Aspirin 81 mg PO DAILY

Vitamin D3 1000 IU PO DAILY

Multivitamin (Centrum Silver) 1 TAB PO DAILY

Fish Oil 1000 mg PO DAILY

Discharge Medications:

Amoxicillin 1 g PO TID (to complete pneumonia/empyema course)

Metformin 500 mg PO BID (with meals)

Insulin Glargine 14 units SUBQ QHS

Insulin Lispro 4 units SUBQ TID with meals + correction scale

Ferrous Sulfate 325 mg PO QOD (empty stomach w/vitamin C)

Apixaban 5 mg PO BID

Metoprolol Succinate XL 50 mg PO DAILY

Atorvastatin 40 mg PO QHS

Lisinopril 20 mg PO DAILY (resume; renal function normalized)

Vitamin D3 1000 IU PO DAILY

Multivitamin (Centrum Silver) 1 TAB PO DAILY

Acetaminophen 650 mg PO Q6H:PRN pain/fever (NTE 3 g/24h)

Discharge Disposition:
Extended Care

Facility:
[[HOSPITAL]]

Discharge Diagnosis:
Severe Community-Acquired Pneumonia (Streptococcus pneumoniae)
Right Parapneumonic Effusion / Empyema s/p Drainage
New Onset Type 2 Diabetes Mellitus (A1c 10.4%)
Atrial Flutter (transient)
Acute Hypoxic Respiratory Failure (resolved)
Acute Kidney Injury, Prerenal (resolved)
Iron Deficiency Anemia (mild)
Hepatic Steatosis
Physical Deconditioning

Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - requires assistance or aid (walker
or cane).

Discharge Instructions:
Dear Mr. [[NAME-1M]],

It was a pleasure caring for you at [[HOSPITAL]].

WHY WAS I IN THE HOSPITAL?

You had a serious lung infection (pneumonia) with bacteria in
your bloodstream and fluid infected around your right lung.
This caused breathing trouble, fever, and low oxygen. You also
were found to have high blood sugar and new diabetes.

WHAT HAPPENED TO ME IN THE HOSPITAL?

You received IV antibiotics and then a pill antibiotic after
cultures identified the germ (Streptococcus pneumoniae).

A small tube was placed into your right chest to drain the
infected fluid; it was removed once the infection cleared.

You were on oxygen briefly; you are now breathing well on room
air.

Your blood sugars were very high with mild acid build‑up; IV
insulin and fluids corrected this. You learned how to use basal
(long‑acting) and mealtime insulin and started metformin.

You had a short episode of a fast heart rhythm (atrial
flutter) that stabilized with medication. You were started on a
blood thinner to reduce stroke risk.

Your kidneys were mildly stressed from dehydration and
infection but recovered fully with fluids.

Mild low iron was found; you started iron supplements.

WHAT SHOULD I DO AFTER I LEAVE THE HOSPITAL?

Finish your full antibiotic course (do not miss doses).

Check blood sugar before meals and at bedtime; record results
for your diabetes visit.

Call a doctor or return to care for: worsening shortness of
breath, fever, chest pain, confusion, uncontrolled high or low
blood sugars, bleeding (nose/gums, dark stools), or severe
abdominal pain.

Use your incentive spirometer and walk several times daily to
improve lung recovery.

Keep all follow-up appointments (pulmonology, endocrinology,
cardiology, primary care). Bring your medication list and
glucose log.

We wish you the best!

Sincerely,
Your [[HOSPITAL]] Team

Followup Instructions:
Department: HEMATOLOGY/ONCOLOGY
When: [[DATE]]
With: [[NAME-4A]]
Building: [[LOCATION]]

Completed by: [[NAME-5A]] MD [[DATE]] @ 2203
""",

"""
Name: [[NAME-1M]] Unit No: [[ID]]

Admission Date: [[DATE]] Discharge Date: [[DATE]]

Date of Birth: [[DATE]] Sex: M

Service: MEDICINE

Allergies:
No Known Allergies / Adverse Drug Reactions

Attending: [[NAME-2A]].

Chief Complaint:
Severe headache and photophobia

Major Surgical or Invasive Procedure:
Lumbar puncture

History of Present Illness:
[[AGE]]-year-old man with history of well-controlled hypertension (on no
home medication after intentional weight loss) presenting with 5 days
of escalating bitemporal → diffuse throbbing headache associated with
photophobia, nausea without emesis, neck stiffness, and fevers to
102–103 at home beginning on [[DATE]].

Headache began insidiously as a “tight band” then intensified; lying in
a dark room gave partial relief. Over-the-counter ibuprofen provided
minimal benefit. On morning of presentation he noted difficulty
flexing his neck forward and mild subjective confusion per his partner
(“slower to respond”). No focal weakness, vision loss, speech change,
seizure activity, or rash. Denies recent travel outside [[LOCATION]], no
sick contacts, no tick bites recalled, no recent antibiotics. No sinus
pain or ear symptoms. Appetite decreased; oral intake modest. Denies
diarrhea, dysuria, cough, chest pain, or abdominal pain. No illicit
drug use, does not vape, drinks socially (last drink >1 week prior).
Monogamous with long-term partner; no history of STIs.

Presented to urgent care on [[DATE]] where he was given IM ketorolac and
told likely viral syndrome. Symptoms progressed; came to ED on [[DATE]]
for persistent fever + meningeal symptoms.

In the ED, initial VS were:
102.4
96
142/82
18
99% RA

Exam notable for: photophobia, mild nuchal rigidity, no focal
neurologic deficits, no papilledema, no rash.

ECG: NSR @ 88, normal intervals, no ischemic changes.

Labs: WBC 6.2 with lymphocyte predominance, Na 131, mild glucose
142 (non‑fasting), normal transaminases, lactate 1.1. Serum procalcitonin low.

Non-contrast head CT obtained prior to LP: no mass or bleed.
Lumbar puncture performed (see data below). Empiric ceftriaxone,
vancomycin, and acyclovir initiated pending results. Dexamethasone
not given (low suspicion bacterial on initial CSF profile).

He defervesced after antipyretics. Enterovirus PCR later returned
positive; bacterial cultures remained negative at 48h; antimicrobials
were discontinued. Headache improved with scheduled acetaminophen,
IV fluids, and gradual mobilization. Hyponatremia corrected with fluid
management (thought mild SIADH related to meningitis). Mild transient
confusion resolved within first 12 hours of admission.

He denies residual neck stiffness at discharge; tolerating regular
diet; ambulating independently. No other complaints.

Past Medical History:
Hypertension (diet/exercise controlled)

Social History:
[[OTHER]]

Family History:
Father with type 2 diabetes
No family history of early cardiovascular or autoimmune disease

Physical Exam:
ADMISSION EXAM:
VS: reviewed in eflowsheets
GENERAL: Uncomfortable, prefers dim light, no respiratory distress
HEENT: AT/NC, EOMI, PERRL, anicteric sclera, conjunctiva pink, MMM.
Fundi sharp without papilledema. Oropharynx clear.
NECK: Mild nuchal rigidity, no LAD, trachea midline
HEART: RRR, normal S1/S2, no murmurs, rubs, or gallops
LUNGS: CTAB, no adventitious sounds, normal work of breathing
ABDOMEN: Soft, nondistended, nontender, no rebound/guarding, no HSM
EXTREMITIES: No edema, cyanosis, or clubbing
PULSES: 2+ radial bilaterally
NEURO: A&Ox3, cranial nerves II–XII intact, strength 5/5 throughout,
sensation intact, normal finger-nose, negative pronator drift
SKIN: Warm, well-perfused, no rash or petechiae

DISCHARGE EXAM:
VS:
[[DATE]] 1527 Temp: 98.2 PO BP: 118/76 L Lying HR: 64 RR: 16 O2
sat: 99% O2 delivery: Ra

GENERAL: Comfortable, no acute distress
HEENT: AT/NC, EOMI, PERRL, anicteric sclera, MMM
NECK: Supple, full ROM, no meningismus
HEART: RRR, S1/S2, no murmurs, gallops, or rubs
LUNGS: CTAB, no wheezes, rales, or rhonchi
ABDOMEN: Soft, nondistended, nontender, no HSM
EXTREMITIES: No edema, cyanosis, or clubbing
NEURO: A&Ox3, normal speech, moves all extremities purposefully
SKIN: Warm, intact, no lesions

Pertinent Results:
ADMISSION LABS:
[[DATE]] 09:40PM BLOOD WBC-6.2 RBC-4.95 Hgb-14.6 Hct-43.5
MCV-88 MCH-29.5 MCHC-33.6 RDW-11.9 RDWSD-37.8 Plt Ct-248
[[DATE]] 09:40PM BLOOD Neuts-41.2 Lymphs-46.0* Monos-9.8
Eos-2.4 Baso-0.6 NRBC-0.0 Im Gran-0.0 AbsNeut-2.56
AbsLymp-2.86 AbsMono-0.61 AbsEos-0.15 AbsBaso-0.04
[[DATE]] 09:55PM BLOOD PT-12.6 PTT-30.1 INR(PT)-1.1
[[DATE]] 09:40PM BLOOD Glucose-142* UreaN-15 Creat-0.9 Na-131*
K-4.1 Cl-97* HCO3-25 AnGap-9
[[DATE]] 09:55PM BLOOD ALT-32 AST-28 AlkPhos-72 TotBili-0.6
[[DATE]] 09:40PM BLOOD Calcium-9.3 Phos-3.4 Mg-1.9
[[DATE]] 09:55PM BLOOD Albumin-4.1

IMPORTANT LABS:
[[DATE]] 05:20AM BLOOD Serum Osm-272*
[[DATE]] 07:04AM BLOOD Procalcitonin-0.09
[[DATE]] 07:15AM BLOOD TSH-1.5
[[DATE]] 09:55PM BLOOD CRP-4.8
ESR: 6
[[DATE]] 10:15PM CEREBROSPINAL FLUID (CSF) TNC-182* RBC-4 Polys-12
Lymphs-78 Monos-10
[[DATE]] 10:15PM CEREBROSPINAL FLUID (CSF) TotProt-64* Glucose-58
CSF Enterovirus PCR POSITIVE; HSV PCR NEGATIVE; Bacterial Gram stain: no organisms; Cultures: no growth to date

CHEST X-RAY ([[DATE]]):
FINDINGS:
Clear lung fields without consolidation, effusion, or edema.
Cardiomediastinal silhouette within normal limits. No acute osseous abnormality.

IMPRESSION:
No acute cardiopulmonary process.

ABDOMINAL U/S ([[DATE]]):
FINDINGS:
LIVER: Normal echotexture, smooth contour, no focal lesion. Portal vein patent with hepatopetal flow. No ascites.
BILE DUCTS: No intrahepatic dilation. Common hepatic duct 4 mm.
GALLBLADDER: Thin-walled, no stones, no pericholecystic fluid.
PANCREAS: Visualized portions unremarkable; tail partially obscured by bowel gas.
SPLEEN: 11.8 cm, homogeneous.
KIDNEYS: Right 11.2 cm, Left 11.0 cm, normal cortical echogenicity, no hydronephrosis, masses, or stones.
RETROPERITONEUM: Visualized aorta and IVC normal caliber.

IMPRESSION:
Unremarkable abdominal ultrasound.

CT HEAD NON-CONTRAST ([[DATE]]):
FINDINGS:
No acute infarct, hemorrhage, mass effect, or hydrocephalus. Ventricles and sulci age-appropriate. No extra-axial collection.
Calvarium intact. Paranasal sinuses and mastoid air cells clear. Orbits unremarkable.

IMPRESSION:
No acute intracranial abnormality.

TRANSTHORACIC ECHO ([[DATE]]):
Indication: Baseline evaluation in patient with transient tachycardia.
Findings: Normal left ventricular size and wall thickness with normal systolic function (EF ~60%). Normal right ventricular size and function. Atria normal size. No interatrial shunt by color Doppler. Valves structurally normal: trivial physiologic mitral and tricuspid regurgitation only. No stenosis. No vegetations or masses. Estimated RV systolic pressure normal. No pericardial effusion. Aortic root and ascending aorta normal dimensions.
IMPRESSION: Normal transthoracic echocardiogram.

DISCHARGE LABS:
[[DATE]] 05:30AM BLOOD WBC-5.4 RBC-4.78 Hgb-14.1 Hct-42.2
MCV-88 MCH-29.5 MCHC-33.4 RDW-11.8 RDWSD-37.5 Plt Ct-231
[[DATE]] 05:30AM BLOOD Neuts-43.0 Lymphs-44.2 Monos-8.7
Eos-3.4 Baso-0.7 Im Gran-0.0 AbsNeut-2.33 AbsLymp-2.39
AbsMono-0.47 AbsEos-0.18 AbsBaso-0.04
[[DATE]] 09:40AM BLOOD PT-12.4 PTT-30.0 INR(PT)-1.1
[[DATE]] 05:30AM BLOOD Glucose-104* UreaN-14 Creat-0.9 Na-137
K-4.2 Cl-101 HCO3-26 AnGap-10
[[DATE]] 05:30AM BLOOD ALT-30 AST-24 AlkPhos-70 TotBili-0.5
[[DATE]] 05:30AM BLOOD Serum Osm-279

Brief Hospital Course:
[[AGE]]M with diet-controlled hypertension presented with febrile
headache, photophobia, and meningismus. Evaluation demonstrated
lymphocytic pleocytosis with mildly elevated CSF protein and normal
glucose; enterovirus PCR positive; bacterial/HSV studies negative.
Initial empiric broad-spectrum antimicrobials discontinued once viral
etiology confirmed and cultures remained negative. Managed with IV
fluids, antipyretics, scheduled acetaminophen, and gradual activity.
Mild hyponatremia (likely SIADH secondary to CNS inflammation)
corrected with fluid restriction liberalization as symptoms improved.
Neurologic status remained stable without focal deficits or seizures.
No evidence of increased intracranial pressure. Echocardiogram normal.
He improved clinically—afebrile >24h prior to discharge, headache
mild and controlled with oral agents, tolerating diet, ambulatory.

TRANSITIONAL ISSUES
[] Confirm final CSF culture and bacterial culture results (should be negative)
[] Monitor for any recurrent fever, worsening headache, neurologic deficits
[] Outpatient PCP follow up for blood pressure trend and reassessment of sodium
[] Counsel on return precautions (severe recurrent headache, confusion, seizure, focal weakness)

Greater than 1/2 hour spent on care on day of discharge.

Medications on Admission:
The Preadmission Medication list is accurate and complete.

Ibuprofen 400 mg PO Q6H PRN headache

Multivitamin daily

Discharge Medications:

Acetaminophen 650 mg PO Q6H PRN headache (do not exceed 3000 mg/24h)

Ibuprofen 400 mg PO Q8H PRN breakthrough headache (take with food; avoid if GI upset)

Discharge Disposition:
Home

Discharge Diagnosis:
PRIMARY

viral meningitis (enterovirus)

Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.

Discharge Instructions:
Dear Mr. [[NAME-1M]],

It was a pleasure caring for you at [[HOSPITAL]]. You were admitted
because of fever, severe headache, and neck stiffness. Testing of your
spinal fluid showed viral meningitis (enterovirus). No bacterial
infection was found, so intravenous antibiotics were stopped. Your
symptoms improved with supportive care. Continue hydration, rest, and
take acetaminophen as directed for residual headache. Avoid exceeding
the recommended daily dose. Return immediately or seek emergency care
if you develop recurrent high fever, confusion, stiff neck with
worsening headache, vomiting preventing fluids, a seizure, or any new
weakness or numbness.

Please follow up with your primary care provider for reassessment and
to review final culture reports.

Sincerely,
Your [[HOSPITAL]] team

Followup Instructions:
[[OTHER]]

"""

]



INSTRUCTION_TEMPLATES_WITH_EXAMPLES = [
    lambda doc, spec, sty, lbl, length, exmpls: f"""Generate a synthetic {doc} for a {spec} case {sty} following medical discharge letter structure.
Write a long text, use approximatly {length} words.
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

    # bad prompts
    lambda doc, spec, sty, lbl, length,exmpls: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Write a long text, use approximatly {length} words.
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline between every section
        """,

        lambda doc, spec, sty, lbl, length, exmpls: f"""
        Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
 Write a long text, use approximatly {length} words.
        
        You should put all PHI TAGGING in double brackets [[ ]], 
     so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

     FORMATTING RULES:
     ALL PHI must use [[LABEL]] format - no exceptions
        """,

        lambda doc, spec, sty, lbl, length, exmpls: f"""
        You are an expert medical writer. Your task is to generate a synthetic {doc} for a {spec} case {sty}.
        Write a long text, use approximatly {length} words.
        
        Below is EXAMPLE of well-written discharge letter for reference:

        EXAMPLE: {exmpls[0]}


        Now generate a NEW {doc} in the exact style of the examples above.

        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 

        Add sections like 
            - Patient Demographics
            - Medical History
            - Present Illness
            - Physical Exams
            - Test Results
            - Hospital Course
            - Discharge Information
            - Follow-up Plan.
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline (\n) between every section
        - Use ==== dividers between major sections

        """,
    


        lambda doc, spec, sty, lbl, length, exmpls: f"""
        You are an expert medical writer. 
        Your task is to write a long text, use approximatly {length} words.
        Write a synthetic {doc} in narrative form {sty} for a {spec} patient. 

        Below is EXAMPLE of well-written discharge letter for reference:

        EXAMPLE: {exmpls[1]}

        Now generate a NEW {doc},  generate it in the exact style of an example.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets like this [[LABEL]], use only this labels: {lbl}. 
    
    CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions. Put all Protected Health Information into the [[label]] format 

    """,

        lambda doc, spec, sty, lbl, length, exmpls: f"""You are an expert medical writer. 
        Your task is to generate a synthetic {doc} for a {spec} case {sty} following medical discharge letter structure.
Write a long text, use approximatly {length} words.

        Below is EXAMPLE of well-written discharge letter for reference:

        EXAMPLE: {exmpls[0]}


        Now generate a NEW {doc} in the exact style of the examples above.


    PHI TAGGING RULES:
    - Use ONLY these [[LABEL]] formats: {lbl}
    - Tag ALL instances of: names, dates, IDs, contacts, locations 
    - Tag all other Protected Health Information but only with labels from labels list: {lbl}

    CONTENT GUIDELINES:
    - Maintain realistic clinical flow for {spec}
    - Use appropriate medical terminology
    - Make [[LABEL]] blend naturally into text""",

        lambda doc, spec, sty, lbl, length, exmpls: f"""You are an expert medical writer. 
        Your task is to create a {doc} for {spec} {sty} that perfectly mimics medical discharge letter documentation style.
Write a long text, use approximatly {length} words.

        Below is EXAMPLE of well-written discharge letter for reference:

        EXAMPLE: {exmpls[1]}


        Now generate a NEW {doc} in the exact style of the examples above.

    You can use this DOCUMENT STRUCTURE or you can change it, do in a medical discharge letter style:
        1. Header with Name and Unit No, ID, Admission Date, Discharge Date, Date of Birth, Sex, Service
        2. Allergies
        3. Attending Physician
        4. Chief Complaint
        5. Major Surgical Procedure
        6. History of Present Illness
        7. Past Medical History
        8. Social History
        9. Family History
        10. Physical Exam (Admission)
        11. Physical Exam (Discharge)
        12. Pertinent Results (Admission)
        13. Pertinent Results (Discharge)
        14. Microbiology Results
        15. Imaging Results
        16. Brief Hospital Course
        17. Medications on Admission
        18. Discharge Medications
        19. Discharge Disposition
        20. Discharge Facility
        21. Discharge Diagnosis
        22. Discharge Condition
        23. Discharge Instructions
        24. Followup Instructions

    Key requirements: all PHI (Protected Health Information) must be tagged in double brackets [[ ]]. 
    And use only these PHI labels: {lbl}

    CRITICAL INSTRUCTIONS:
    - ALL PHI must use [[LABEL]] format - no exceptions""",

        lambda doc, spec, sty, lbl, length, exmpls : f"""You are an expert medical writer. 
        Your task is to generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
Write a long text, use approximatly {length} words.


        Below is EXAMPLE of well-written discharge letter for reference:

        EXAMPLE: {exmpls[0]}


        Now generate a NEW {doc} in the exact style of the examples above.

    You should put all PHI TAGGING in double brackets [[ ]], 
    so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

    FORMATTING RULES:
    1. ALL PHI must use [[LABEL]] format - no exceptions
    2. \n between all sections
    3. ==== dividers after key sections"""

]

SELF_REF_PROMPTS = [
        lambda doc, lbl: f"""

Review this synthetic medical discharge letter and perform the following corrections:

1. **PHI Validation**:
   - Scan the entire text for any unprotected PHI ({phi_list}).
   - Ensure ALL PHI uses double brackets [[LABEL]] format. 
   - Cross-check each [[LABEL]] against this allowed list: {lbl}. Replace any incorrect/unknown labels. 
   - All labels in double brackets must be from the list only: {lbl}. Any other information must not be hidden.
   - If raw PHI exists (e.g., "John Smith"), replace it with [[NAME]] for all label types.

2. **Output**: 
   Return the corrected text.

Discharge letter:
        {doc}
        """,
    


        lambda doc, lbl: f"""
        You are a PHI audit tool. Analyze this medical discharge letter for:

1. **Label Errors**:
   - Identify any:
     - PHI *not* in [[ ]] brackets (e.g., "Room 205" → [[LOCATION]]).
     - [[LABELS]] not in the approved list: {lbl}.
    - All labels in double brackets must be from the list only: {lbl}. Any other information must not be hidden.

     - Missing labels (e.g., dates without [[DATE]] tags).

2. **Action**:
   - Rewrite the text with ALL corrections applied.

   Discharge letter:
        {doc}

    """,

        lambda doc, lbl: f"""
Act as a post-processing fixer for a synthetic medical discharge letter.

CHECK:
- All PHI types ({phi_list}) MUST be replaced by an allowed [[LABEL]] token.
- Check each [[LABEL]] is in the list: {lbl}. Replace any incorrect/unknown labels.
- All labels in double brackets must be from the list only: {lbl}. Any other information must not be hidden.
- Tokens must be exactly [[LABEL]] (double brackets, uppercase, no spaces).
- Fix malformed bracket patterns (e.g. [ [LABEL], [[LABEL ]], [[[LABEL]] → [[LABEL]]).

Perform minimal edits else.

OUTPUT corercted text.

Original discharge letter: {doc}
"""

]


INSTRUCTION_TEMPLATES_BAD_PROMPTS = [
        lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Write a long text, use approximatly {length} words.
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 

        Add sections like 
            - Patient Demographics
            - Medical History
            - Present Illness
            - Physical Exams
            - Test Results
            - Hospital Course
            - Discharge Information
            - Follow-up Plan.
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline (\n) between every section
        - Use ==== dividers between major sections

        """,
    


        lambda doc, spec, sty, lbl, length: f"""Write a synthetic {doc} in narrative form {sty} for a {spec} patient. 
        Generate it in the exact style of medical discharge letter. Write a long text, use approximatly {length} words.
    Don't structure it too much. It should be a natural medical live recording.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets like this [[LABEL]], use only this labels: {lbl}. 
    Begin with patient presentation, then describe:
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
        - ALL PHI must use [[LABEL]] format - no exceptions. Put all Protected Health Information into the [[label]] format 
        - Newline (\n) between every section
        - Use ==== dividers between major sections

    """,

        lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty} following medical discharge letter structure.
Write a long text, use approximatly {length} words.
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

    # bad prompts
    lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Write a long text, use approximatly {length} words.
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline between every section
        """,

        lambda doc, spec, sty, lbl, length: f"""
        Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
 Write a long text, use approximatly {length} words.
        
        You should put all PHI TAGGING in double brackets [[ ]], 
     so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

     FORMATTING RULES:
     ALL PHI must use [[LABEL]] format - no exceptions
        """,

        lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Write a long text, use approximatly {length} words.
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline between every section
        """,

        lambda doc, spec, sty, lbl, length: f"""
        Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
 Write a long text, use approximatly {length} words.
        
        You should put all PHI TAGGING in double brackets [[ ]], 
     so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

     FORMATTING RULES:
     ALL PHI must use [[LABEL]] format - no exceptions
        """,

        lambda doc, spec, sty, lbl, length: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Write a long text, use approximatly {length} words.
        Generate it in the exact style of medical discharge letter.
        The text should be realistic and resemble actual medical documentation.
        Replace all PHI and sensitive data with labels from this list in double brackets [[label]]: {lbl}. 
        
        Make sure the text flows naturally and maintains proper medical terminology.
        
        CRITICAL INSTRUCTIONS:
        - ALL PHI must use [[LABEL]] format - no exceptions
        - Newline between every section
        """,

        lambda doc, spec, sty, lbl, length: f"""
        Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
 Write a long text, use approximatly {length} words.
        
        You should put all PHI TAGGING in double brackets [[ ]], 
     so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

     FORMATTING RULES:
     ALL PHI must use [[LABEL]] format - no exceptions
        """,

#         lambda doc, spec, sty, lbl, length: f"""Create a {doc} for {spec} {sty} that perfectly mimics medical discharge letter documentation style.
# Write a long text, use approximatly {length} words.
#     You can use this DOCUMENT STRUCTURE or you can change it, do in a medical discharge letter style:
#         1. Header with Name and Unit No, ID, Admission Date, Discharge Date, Date of Birth, Sex, Service
#         2. Allergies
#         3. Attending Physician
#         4. Chief Complaint
#         5. Major Surgical Procedure
#         6. History of Present Illness
#         7. Past Medical History
#         8. Social History
#         9. Family History
#         10. Physical Exam (Admission)
#         11. Physical Exam (Discharge)
#         12. Pertinent Results (Admission)
#         13. Pertinent Results (Discharge)
#         14. Microbiology Results
#         15. Imaging Results
#         16. Brief Hospital Course
#         17. Medications on Admission
#         18. Discharge Medications
#         19. Discharge Disposition
#         20. Discharge Facility
#         21. Discharge Diagnosis
#         22. Discharge Condition
#         23. Discharge Instructions
#         24. Followup Instructions

#     Key requirements: all PHI (Protected Health Information) must be tagged in double brackets [[ ]]. 
#     And use only these PHI labels: {lbl}

#     CRITICAL INSTRUCTIONS:
#     - ALL PHI must use [[LABEL]] format - no exceptions
#     - ==== dividers between major sections
#     - Newline (\n) between every section
#     - Maintain natural clinical narrative flow""",

#         lambda doc, spec, sty, lbl, length : f"""Generate a {doc} for {spec} {sty} adhering strictly to medical discharge letter conventions.
# Write a long text, use approximatly {length} words.
#     You can write with some of the following sections:
#         1. Header with Name and Unit No, ID, Admission Date, Discharge Date, Date of Birth, Sex, Service
#         2. Allergies
#         3. Attending Physician
#         4. Chief Complaint
#         5. Major Surgical Procedure
#         6. History of Present Illness
#         7. Past Medical History
#         8. Social History
#         9. Family History
#         10. Physical Exam (Admission)
#         11. Physical Exam (Discharge)
#         12. Pertinent Results (Admission)
#         13. Pertinent Results (Discharge)
#         14. Microbiology Results
#         15. Imaging Results
#         16. Brief Hospital Course
#         17. Medications on Admission
#         18. Discharge Medications
#         19. Discharge Disposition
#         20. Discharge Facility
#         21. Discharge Diagnosis
#         22. Discharge Condition
#         23. Discharge Instructions
#         24. Followup Instructions

#     You should put all PHI TAGGING in double brackets [[ ]], 
#     so all Protected Health Information must be classified with one of the labels with the format of [[]]:  {lbl} 

#     FORMATTING RULES:
#     1. ALL PHI must use [[LABEL]] format - no exceptions
#     2. \n between all sections
#     3. ==== dividers after key sections"""

]