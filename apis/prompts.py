LABELS = [
    "[[DÉMOGRAPHIE:ÂGE]]",
    "[[EMPLACEMENT:NUMÉRO_HABITATION]]",
    "[[EMPLACEMENT:RUE]]",
    "[[ORGANISATION]]",
    "[[EMPLACEMENT:EMPLACEMENT_GÉOGRAPHIQUE]]",
    "[[PERSONNES:LIEN_DE_PARENTÉ]]",
    "[[CONTACT:FAX]]",
    "[[EMPLACEMENT:CODE_POSTAL]]",
    "[[CHUV:STRUCTURE_RÉFÉRENCE]]",
    "[[NOM:PERSONNEL_MÉDICAL]]",
    "[[EMPLACEMENT:PAYS]]",
    "[[TEMPORAL:DATE]]",
    "[[NOM:PATIENT_E]]",
    "[[TEMPORAL:TEMPS]]",
    "[[CHUV:BÂTIMENT_CHAMBRE_OU_LIT]]",
    "[[ID:NUMÉRO_BON]]",
    "[[EMPLACEMENT:CODE_CANTON]]",
    "[[CONTACT:TÉLÉPHONE]]",
    "[[ID:NUMÉRO_SÉJOUR]]"
]

STYLES = [
    "de manière professionnelle",
    "sur un ton professionnel",
    "dans un style professionnel",
    "sur un ton clinique professionnel",
    "en utilisant une terminologie médicale concise",
    "avec des détails cliniques approfondis",
    "dans un récit clinique structuré mais naturel",
    "avec des observations médicales précises",
    "y compris le contexte clinique pertinent",
    "avec les abréviations médicales appropriées",
    "dans un style détaillé mais lisible"
]

DOC_TYPES = [
    "compte rendu de sortie",
    "compte rendu de radiologie",
    "note de sortie clinique",
]

SPECIALTIES = [
    "neuroradiologie",
    "radiologie musculo-squelettique",
    "oncologie",
    "oncologie radiologique",
    "radiologie abdominale",
    "radiologie thoracique",
    "imagerie du sein",
    "radiologie interventionnelle",
    "radiologie pédiatrique",
    "radiologie cardiothoracique",
    "radiologie d'urgence",
    "médecine nucléaire"
]

INSTRUCTION_TEMPLATES = [
    lambda doc, spec, sty, lbl, length: f"""Générez un {doc} synthétique en français pour un cas de {spec} {sty}.
    Écrivez un long texte, utilisez environ {length} mots.
    Générez-le dans le style exact d'une lettre de sortie médicale.
    Le texte doit être réaliste et ressembler à une véritable documentation médicale.
    Remplacez toutes les informations de santé protégées (Informations de Santé Protégées (ISP)) et les données sensibles par des étiquettes de cette liste entre doubles crochets [[étiquette]]: {lbl}.
    Ajoutez des sections comme :
        - Démographie du patient
        - Historique médical
        - Maladie actuelle
        - Examens physiques
        - Résultats des tests
        - Parcours hospitalier
        - Informations de sortie
        - Plan de suivi.

    Assurez-vous que le texte coule naturellement et maintient une terminologie médicale appropriée.

    INSTRUCTIONS CRITIQUES :
    - TOUTES LES Informations de Santé Protégées (ISP) doivent utiliser le format [[ÉTIQUETTE]] - sans exception
    """,

    lambda doc, spec, sty, lbl, length: f"""Écrivez un {doc} synthétique en français sous forme narrative {sty} pour un patient {spec}.
    Générez-le dans le style exact d'une lettre de sortie médicale. Écrivez un long texte, utilisez environ {length} mots.
    Ne le structurez pas trop. Cela doit être un enregistrement médical naturel.
    Incluez et remplacez toutes les Informations de Santé Protégées (ISP)/données sensibles par des étiquettes de cette liste entre doubles crochets comme ceci [[ÉTIQUETTE]], utilisez uniquement ces étiquettes : {lbl}.
    Commencez par la présentation du patient, puis décrivez :
        1. En-tête du patient (démographie)
        2. Plainte principale
        3. Historique de la maladie actuelle
        4. Antécédents médicaux
        5. Examen physique
        6. Parcours hospitalier
        7. Laboratoires/Imagerie
        8. Diagnostic de sortie
        9. Médicaments de sortie
        10. Instructions de sortie
        11. Plan de suivi

    INSTRUCTIONS CRITIQUES :
        - TOUTES LES Informations de Santé Protégées (ISP) doivent utiliser le format [[ÉTIQUETTE]] - sans exception. Mettez toutes les informations de santé protégées au format [[étiquette]]
    """,

    lambda doc, spec, sty, lbl, length: f"""Générez un {doc} synthétique en français pour un cas de {spec} {sty} en suivant la structure d'une lettre de sortie médicale.
    Écrivez un long texte, utilisez environ {length} mots.
    SECTIONS REQUISES (peut-être pas toutes et vous pouvez en ajouter d'autres) (séparées par des sauts de ligne) :
    1. En-tête du patient (Nom, Numéro d'unité, Dates d'admission/de sortie, Date de naissance, Sexe, Service)
    2. Allergies
    3. Plainte principale
    4. Historique de la maladie actuelle (avec chronologie détaillée)
    5. Antécédents médicaux
    6. Historique social
    7. Examen physique (avec des points à puces basés sur les systèmes)
    8. Résultats pertinents (format de laboratoire avec horodatages)
    9. Parcours hospitalier bref
    10. Diagnostics de sortie
    11. Médicaments de sortie (liste formatée)
    12. Instructions de sortie
    13. Informations de suivi
    RÈGLES D'ÉTIQUETAGE DES Informations de Santé Protégées (ISP) :
    - Utilisez UNIQUEMENT ces formats [[ÉTIQUETTE]] : {lbl}
    - Étiquetez TOUTES les instances de : noms, dates, identifiants, contacts, lieux
    - Étiquetez toutes les autres informations de santé protégées mais uniquement avec les étiquettes de la liste des étiquettes : {lbl}
    - Incluez au moins 10 instances [[ÉTIQUETTE]] tout au long du document
    DIRECTIVES DE CONTENU :
    - Maintenez un flux clinique réaliste pour {spec}
    - Utilisez une terminologie médicale appropriée
    - Faites en sorte que les [[ÉTIQUETTES]] se fondent naturellement dans le texte""",

    lambda doc, spec, sty, lbl, length: f"""Créez un {doc} pour {spec} {sty} qui imite parfaitement le style de documentation d'une lettre de sortie médicale.
    Écrivez un long texte, utilisez environ {length} mots.
    Vous pouvez utiliser cette STRUCTURE DE DOCUMENT ou vous pouvez la modifier, faites-le dans le style d'une lettre de sortie médicale :
        1. En-tête avec Nom et Numéro d'unité, ID, Date d'admission, Date de sortie, Date de naissance, Sexe, Service
        2. Allergies
        3. Médecin traitant
        4. Plainte principale
        5. Procédure chirurgicale majeure
        6. Historique de la maladie actuelle
        7. Antécédents médicaux
        8. Historique social
        9. Antécédents familiaux
        10. Examen physique (Admission)
        11. Examen physique (Sortie)
        12. Résultats pertinents (Admission)
        13. Résultats pertinents (Sortie)
        14. Résultats de microbiologie
        15. Résultats d'imagerie
        16. Parcours hospitalier bref
        17. Médicaments à l'admission
        18. Médicaments de sortie
        19. Disposition de sortie
        20. Établissement de sortie
        21. Diagnostic de sortie
        22. Condition de sortie
        23. Instructions de sortie
        24. Instructions de suivi
    Exigences clés : toutes les Informations de Santé Protégées (ISP) (Informations de santé protégées) doivent être étiquetées entre doubles crochets [[ ]].
    Et utilisez uniquement ces étiquettes Informations de Santé Protégées (ISP) : {lbl}
    INSTRUCTIONS CRITIQUES :
    - TOUTES LES Informations de Santé Protégées (ISP) doivent utiliser le format [[ÉTIQUETTE]] - sans exception
    - Maintenez un flux narratif clinique naturel""",

    lambda doc, spec, sty, lbl, length: f"""Générez un {doc} en français pour {spec} {sty} en adhérant strictement aux conventions de la lettre de sortie médicale.
    Écrivez un long texte, utilisez environ {length} mots.
    Vous pouvez écrire avec certaines des sections suivantes :
        1. En-tête avec Nom et Numéro d'unité, ID, Date d'admission, Date de sortie, Date de naissance, Sexe, Service
        2. Allergies
        3. Médecin traitant
        4. Plainte principale
        5. Procédure chirurgicale majeure
        6. Historique de la maladie actuelle
        7. Antécédents médicaux
        8. Historique social
        9. Antécédents familiaux
        10. Examen physique (Admission)
        11. Examen physique (Sortie)
        12. Résultats pertinents (Admission)
        13. Résultats pertinents (Sortie)
        14. Résultats de microbiologie
        15. Résultats d'imagerie
        16. Parcours hospitalier bref
        17. Médicaments à l'admission
        18. Médicaments de sortie
        19. Disposition de sortie
        20. Établissement de sortie
        21. Diagnostic de sortie
        22. Condition de sortie
        23. Instructions de sortie
        24. Instructions de suivi
    Vous devez mettre tout l'ÉTIQUETAGE DES Informations de Santé Protégées (ISP) entre doubles crochets [[ ]],
    donc toutes les informations de santé protégées doivent être classées avec l'une des étiquettes au format [[ ]] : {lbl}
    RÈGLES DE FORMATAGE :
    1. TOUTES LES Informations de Santé Protégées (ISP) doivent utiliser le format [[ÉTIQUETTE]] - sans exception"""
]
