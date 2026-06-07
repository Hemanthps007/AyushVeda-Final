import pandas as pd
import os

data = [
    {
        "Disease": "Common Cold",
        "Medicine": "Acetaminophen, Diphenhydramine, Pseudoephedrine",
        "Dosage": "Acetaminophen: 500mg every 6 hours; Diphenhydramine: 25mg at night",
        "Duration": "3-5 days as needed",
        "Diet_Advice": "Stay hydrated, warm fluids, avoid heavy meals",
        "Lifestyle": "Rest, use a humidifier, avoid contact with others"
    },
    {
        "Disease": "Influenza",
        "Medicine": "Oseltamivir, Ibuprofen, Dextromethorphan",
        "Dosage": "Oseltamivir: 75mg twice daily (if started early); Ibuprofen: 400mg every 6-8 hours",
        "Duration": "5 days",
        "Diet_Advice": "High fluid intake, light soups",
        "Lifestyle": "Strict bed rest, isolate to prevent spread"
    },
    {
        "Disease": "Typhoid Fever",
        "Medicine": "Ciprofloxacin or Azithromycin, Paracetamol",
        "Dosage": "Ciprofloxacin: 500mg twice daily; Paracetamol: 500mg as needed for fever",
        "Duration": "7-14 days based on culture results",
        "Diet_Advice": "Easily digestible bland foods, boiled water",
        "Lifestyle": "Rest, strict hand hygiene"
    },
    {
        "Disease": "Malaria",
        "Medicine": "Artemisinin-based combination therapy (ACT), Chloroquine (if susceptible)",
        "Dosage": "Artemether/Lumefantrine: strictly as prescribed based on weight",
        "Duration": "3 days ACT regimen",
        "Diet_Advice": "Hydration, manageable light solid foods",
        "Lifestyle": "Rest, monitor temperature, use mosquito nets"
    },
    {
        "Disease": "Dengue Fever",
        "Medicine": "Paracetamol, Oral Rehydration Salts (ORS)",
        "Dosage": "Paracetamol: 500mg strictly (DO NOT use Aspirin/Ibuprofen); ORS: continuous",
        "Duration": "Until symptoms resolve (typically 7-10 days)",
        "Diet_Advice": "Vigorous hydration, avoid dark colored foods (to prevent mistaken blood in stool)",
        "Lifestyle": "Strict bed rest, daily monitoring of hematocrit and platelets"
    },
    {
        "Disease": "Diabetes",
        "Medicine": "Metformin, Insulin (if required), Sulfonylureas",
        "Dosage": "Metformin: 500mg twice daily with meals (titrated as needed)",
        "Duration": "Lifelong, requires regular monitoring",
        "Diet_Advice": "Diabetic plate method, limit simple sugars, high fiber, watch carb intake",
        "Lifestyle": "150 mins moderate exercise per week, foot care, regular HbA1c testing"
    },
    {
        "Disease": "Hypertension",
        "Medicine": "Lisinopril, Amlodipine, Hydrochlorothiazide",
        "Dosage": "Lisinopril: 10mg daily starting dose; Amlodipine: 5mg daily",
        "Duration": "Lifelong management",
        "Diet_Advice": "DASH diet: low sodium (<2300mg), rich in fruits and vegetables",
        "Lifestyle": "Weight management, limit alcohol, regular blood pressure checks"
    },
    {
        "Disease": "Anxiety Disorder",
        "Medicine": "SSRIs (Escitalopram), Benzodiazepines (Diazepam - short term)",
        "Dosage": "Escitalopram: 10mg daily; Diazepam: 2-5mg precisely as needed",
        "Duration": "Long term for SSRIs, strictly short term for Benzodiazepines",
        "Diet_Advice": "Limit caffeine and alcohol, regular meals",
        "Lifestyle": "Cognitive Behavioral Therapy (CBT), progressive muscle relaxation, regular sleep"
    },
    {
        "Disease": "Depression",
        "Medicine": "Fluoxetine, Sertraline, Bupropion",
        "Dosage": "Fluoxetine: 20mg daily in morning",
        "Duration": "6-12 months typically, may be longer",
        "Diet_Advice": "Balanced diet rich in Omega-3 fatty acids",
        "Lifestyle": "Psychotherapy, regular routine, sunlight exposure, avoid isolation"
    },
    {
        "Disease": "Migraine",
        "Medicine": "Sumatriptan, Ibuprofen, Propranolol (preventive)",
        "Dosage": "Sumatriptan: 50mg at onset of headache; Ibuprofen: 400mg",
        "Duration": "As needed for abortive therapy; daily for preventive",
        "Diet_Advice": "Identify and avoid dietary triggers (e.g., tyramine, MSG, alcohol)",
        "Lifestyle": "Dark quiet room during attacks, maintain regular sleep schedule"
    },
    {
        "Disease": "Asthma",
        "Medicine": "Albuterol Inhaler (reliever), Fluticasone Inhaler (controller)",
        "Dosage": "Albuterol: 2 puffs as needed; Fluticasone: 2 puffs twice daily",
        "Duration": "Chronic management",
        "Diet_Advice": "No specific diet, maintain healthy weight",
        "Lifestyle": "Avoid asthma triggers (dust, pollen), carry rescue inhaler always, peak flow monitoring"
    },
    {
        "Disease": "Pneumonia",
        "Medicine": "Amoxicillin, Azithromycin, Levofloxacin",
        "Dosage": "Amoxicillin: 500mg thrice daily or as prescribed",
        "Duration": "5-7 days (community-acquired)",
        "Diet_Advice": "Stay hydrated to loosen secretions",
        "Lifestyle": "Rest, smoking cessation, deep breathing exercises"
    },
    {
        "Disease": "Gastritis",
        "Medicine": "Omeprazole, Ranitidine, Antacids (Magnesium hydroxide)",
        "Dosage": "Omeprazole: 20mg daily before breakfast",
        "Duration": "14-28 days",
        "Diet_Advice": "Avoid spicy, acidic, and fatty foods, avoid coffee and alcohol",
        "Lifestyle": "Elevate head of bed, quit smoking, manage stress"
    },
    {
        "Disease": "Urinary Tract Infection",
        "Medicine": "Nitrofurantoin, Trimethoprim/Sulfamethoxazole",
        "Dosage": "Nitrofurantoin: 100mg twice daily for 5 days",
        "Duration": "3-7 days depending on antibiotic",
        "Diet_Advice": "High water intake, cranberry juice (adjunct, not curative)",
        "Lifestyle": "Frequent voiding, wipe front to back, empty bladder after intercourse"
    },
    {
        "Disease": "Arthritis",
        "Medicine": "NSAIDs (Ibuprofen, Naproxen), Methotrexate (for RA), Corticosteroids",
        "Dosage": "Ibuprofen: 400-800mg every 6-8 hours as needed",
        "Duration": "Chronic management",
        "Diet_Advice": "Rich in omega-3s, maintain healthy weight to reduce joint stress",
        "Lifestyle": "Physical therapy, regular low-impact exercise (swimming)"
    },
    {
        "Disease": "Anemia",
        "Medicine": "Ferrous Sulfate, Vitamin B12, Folic Acid",
        "Dosage": "Ferrous Sulfate: 325mg daily to thrice daily based on severity",
        "Duration": "3-6 months to replenish stores",
        "Diet_Advice": "Iron-rich foods (red meat, spinach) accompanied by Vitamin C (citrus) for absorption",
        "Lifestyle": "Monitor for black stools (normal side effect of iron), avoid taking iron with calcium"
    },
    {
        "Disease": "Thyroid Disorder",
        "Medicine": "Levothyroxine (Hypo), Methimazole (Hyper)",
        "Dosage": "Levothyroxine: individualized (e.g., 50mcg daily) taken on empty stomach in morning",
        "Duration": "Usually lifelong",
        "Diet_Advice": "Consistent iodine intake, avoid taking medication with iron or calcium supplements",
        "Lifestyle": "Regular TSH monitoring every 6-12 months once stable"
    },
    {
        "Disease": "Liver Disease",
        "Medicine": "Spironolactone, Lactulose, Ursodeoxycholic acid",
        "Dosage": "Depends heavily on specific condition and stage",
        "Duration": "Chronic management",
        "Diet_Advice": "Strictly NO alcohol, low sodium if ascites present, adequate protein",
        "Lifestyle": "Hepatitis vaccination, avoid hepatotoxic drugs (e.g., excessive acetaminophen)"
    },
    {
        "Disease": "Kidney Disease",
        "Medicine": "ACE inhibitors (Lisinopril), Erythropoietin, Diuretics (Furosemide)",
        "Dosage": "Dosed according to GFR",
        "Duration": "Chronic management",
        "Diet_Advice": "Low protein, low potassium, low phosphorus, low sodium based on stage",
        "Lifestyle": "Strict blood pressure and blood sugar control, nephrologist follow-up"
    },
    {
        "Disease": "Skin Allergy",
        "Medicine": "Cetirizine, Loratadine, Topical Corticosteroids (Hydrocortisone)",
        "Dosage": "Cetirizine: 10mg daily; Hydrocortisone: apply a thin layer twice daily",
        "Duration": "As long as exposure to allergen continues, short term for topical steroids",
        "Diet_Advice": "Avoid known food allergens",
        "Lifestyle": "Identify and avoid triggers (soaps, pets), use gentle hypoallergenic moisturizers"
    },
    {
        "Disease": "Covid-19",
        "Medicine": "Paracetamol, Paxlovid (if high risk), Dexamethasone (severe)",
        "Dosage": "Paracetamol: 500mg as needed for fever; Paxlovid: 300mg/100mg twice daily",
        "Duration": "5-10 days",
        "Diet_Advice": "High protein, hydration, Vitamin C & Zinc supplements",
        "Lifestyle": "Strict isolation, monitor oxygen saturation"
    },
    {
        "Disease": "Tuberculosis",
        "Medicine": "Isoniazid, Rifampin, Ethambutol, Pyrazinamide",
        "Dosage": "Strict DOTS protocol based on weight",
        "Duration": "6 months minimum",
        "Diet_Advice": "High calorie, high protein diet to recover weight",
        "Lifestyle": "Isolation during infectious phase, strict medication adherence"
    },
    {
        "Disease": "Measles",
        "Medicine": "Vitamin A, Paracetamol, Ibuprofen",
        "Dosage": "Vitamin A: 2 doses 24 hours apart",
        "Duration": "7-10 days",
        "Diet_Advice": "Fluid replacement, easily digested foods",
        "Lifestyle": "Isolation, dim lights for photophobia"
    },
    {
        "Disease": "Chickenpox",
        "Medicine": "Acyclovir, Calamine lotion, Antihistamines",
        "Dosage": "Acyclovir: 800mg 5 times a day (for adults)",
        "Duration": "5-7 days",
        "Diet_Advice": "Soft, bland foods (especially if mouth sores are present)",
        "Lifestyle": "Do not scratch lesions, avoid public spaces"
    },
    {
        "Disease": "Psoriasis",
        "Medicine": "Topical Corticosteroids, Methotrexate, Biologics (Adalimumab)",
        "Dosage": "Topical: Apply thinly once or twice daily",
        "Duration": "Chronic condition management",
        "Diet_Advice": "Anti-inflammatory diet, omega-3 supplements",
        "Lifestyle": "Moisturize daily, safe sun exposure, stress management"
    },
    {
        "Disease": "Eczema",
        "Medicine": "Emollients, Topical Corticosteroids, Tacrolimus ointment",
        "Dosage": "Apply frequently throughout the day",
        "Duration": "Ongoing management for flare-ups",
        "Diet_Advice": "Identify and avoid trigger foods if allergies present",
        "Lifestyle": "Short warm baths, avoid harsh soaps"
    },
    {
        "Disease": "Osteoarthritis",
        "Medicine": "Acetaminophen, NSAIDs, Intra-articular Corticosteroids",
        "Dosage": "Acetaminophen: Up to 3000mg/day maximum",
        "Duration": "Chronic management",
        "Diet_Advice": "Weight loss diet to reduce joint load, calcium",
        "Lifestyle": "Physical therapy, low-impact exercise"
    },
    {
        "Disease": "Gout",
        "Medicine": "Allopurinol, Colchicine, Indomethacin",
        "Dosage": "Colchicine: 1.2mg initially, then 0.6mg an hour later during acute attack",
        "Duration": "Acute treatment 3-5 days, preventive (Allopurinol) lifelong",
        "Diet_Advice": "Low purine diet (avoid red meat, shellfish), avoid alcohol",
        "Lifestyle": "Maintain hydration, weight management"
    },
    {
        "Disease": "Irritable Bowel Syndrome",
        "Medicine": "Loperamide (IBS-D), Lubiprostone (IBS-C), Antispasmodics (Dicyclomine)",
        "Dosage": "Dicyclomine: 20mg up to 4 times a day",
        "Duration": "Chronic management",
        "Diet_Advice": "Low FODMAP diet, soluble fiber supplements",
        "Lifestyle": "Stress management, regular exercise"
    },
    {
        "Disease": "Peptic Ulcer",
        "Medicine": "PPIs (Omeprazole), Clarithromycin + Amoxicillin (if H. Pylori positive)",
        "Dosage": "Omeprazole: 40mg daily; Antibiotics: twice daily",
        "Duration": "14 days for eradication, 4-8 weeks for healing",
        "Diet_Advice": "Avoid spicy food, caffeine, and NSAIDs",
        "Lifestyle": "Smoking cessation, limit alcohol"
    },
    {
        "Disease": "Hepatitis A",
        "Medicine": "Supportive care (No specific antiviral)",
        "Dosage": "Symptomatic treatment for fever/nausea",
        "Duration": "1-2 months",
        "Diet_Advice": "Small, frequent, low-fat meals",
        "Lifestyle": "Strict hygiene, avoid alcohol entirely"
    },
    {
        "Disease": "Hepatitis B",
        "Medicine": "Entecavir, Tenofovir, Pegylated interferon",
        "Dosage": "Entecavir: 0.5mg daily",
        "Duration": "Often lifelong for chronic infection",
        "Diet_Advice": "Healthy balanced diet, strict abstention from alcohol",
        "Lifestyle": "Regular liver cancer screening, safe sex practices"
    },
    {
        "Disease": "Alzheimers Disease",
        "Medicine": "Donepezil, Memantine",
        "Dosage": "Donepezil: 5mg-10mg daily at bedtime",
        "Duration": "Lifelong symptomatic management",
        "Diet_Advice": "Mediterranean or MIND diet",
        "Lifestyle": "Cognitive stimulation, consistent routine, fall-proofing home"
    },
    {
        "Disease": "Parkinsons Disease",
        "Medicine": "Carbidopa/Levodopa, Pramipexole",
        "Dosage": "Carbidopa/Levodopa: 25/100mg three times daily initially",
        "Duration": "Lifelong management",
        "Diet_Advice": "High fiber diet for constipation, avoid high protein meals concurrently with medication",
        "Lifestyle": "Physical therapy, speech therapy, fall precautions"
    },
    {
        "Disease": "Multiple Sclerosis",
        "Medicine": "Ocrelizumab, Interferon beta, Corticosteroids (for acute relapse)",
        "Dosage": "Methylprednisolone: 1g IV daily for 3-5 days during relapse",
        "Duration": "Lifelong management",
        "Diet_Advice": "Vitamin D supplementation",
        "Lifestyle": "Avoid heat exposure (Uhthoff's phenomenon), physical therapy"
    },
    {
        "Disease": "Cataracts",
        "Medicine": "Surgical Intervention (Phacoemulsification)",
        "Dosage": "N/A",
        "Duration": "Outpatient procedure",
        "Diet_Advice": "Antioxidant-rich diet (Vitamins C, E)",
        "Lifestyle": "UV protection (sunglasses)"
    },
    {
        "Disease": "Glaucoma",
        "Medicine": "Latanoprost eye drops, Timolol, Brimonidine",
        "Dosage": "Latanoprost: 1 drop in affected eye(s) once daily in evening",
        "Duration": "Lifelong management to prevent blindness",
        "Diet_Advice": "Normal healthy diet",
        "Lifestyle": "Regular intraocular pressure monitoring"
    },
    {
        "Disease": "Appendicitis",
        "Medicine": "Surgical Appendectomy, IV Antibiotics",
        "Dosage": "Cefoxitin or Cefazolin IV pre-operatively",
        "Duration": "Surgical intervention required immediately",
        "Diet_Advice": "NPO (nothing by mouth) until evaluation",
        "Lifestyle": "Post-operative rest"
    },
    {
        "Disease": "Hemorrhoids",
        "Medicine": "Hydrocortisone cream, Docusate sodium (Stool softener)",
        "Dosage": "Apply cream twice daily; Docusate: 100mg twice daily",
        "Duration": "1-2 weeks",
        "Diet_Advice": "High fiber diet (25-30g/day), ample hydration",
        "Lifestyle": "Sitz baths, avoid straining during bowel movements"
    },
    {
        "Disease": "Kidney Stones",
        "Medicine": "Tamsulosin, NSAIDs (Ketorolac), Ondansetron (for nausea)",
        "Dosage": "Tamsulosin: 0.4mg daily to facilitate passage within 4 weeks",
        "Duration": "Until stone is passed or surgically removed",
        "Diet_Advice": "Hydration (>2.5L/day), limit sodium, normal dietary calcium",
        "Lifestyle": "Pain management, strain urine to catch stone for analysis"
    },
    {
        "Disease": "Bronchitis",
        "Medicine": "Albuterol inhaler, Dextromethorphan (cough suppressant)",
        "Dosage": "Albuterol: 2 puffs every 4-6 hours as needed for wheezing",
        "Duration": "1-3 weeks",
        "Diet_Advice": "Warm fluids, honey",
        "Lifestyle": "Rest, avoid smoke and irritants"
    },
    {
        "Disease": "Sinusitis",
        "Medicine": "Amoxicillin/Clavulanate (if bacterial), Fluticasone nasal spray",
        "Dosage": "Augmentin: 875/125mg twice daily for 5-7 days",
        "Duration": "7 days",
        "Diet_Advice": "High hydration to thin mucus",
        "Lifestyle": "Saline nasal irrigation, warm compresses"
    },
    {
        "Disease": "Tonsillitis",
        "Medicine": "Penicillin V, Ibuprofen, Acetaminophen",
        "Dosage": "Penicillin V: 500mg twice daily for 10 days (if Strep positive)",
        "Duration": "10 days",
        "Diet_Advice": "Soft foods, cold liquids, avoid acidic foods",
        "Lifestyle": "Rest, salt water gargles"
    },
    {
        "Disease": "Meningitis",
        "Medicine": "Ceftriaxone + Vancomycin IV (Bacterial), Acyclovir (Viral)",
        "Dosage": "Administered immediately in hospital setting",
        "Duration": "10-21 days depending on pathogen",
        "Diet_Advice": "IV hydration",
        "Lifestyle": "Medical emergency, requires ICU monitoring"
    },
    {
        "Disease": "Rabies",
        "Medicine": "Human Rabies Immune Globulin (HRIG), Rabies Vaccine",
        "Dosage": "Vaccine on days 0, 3, 7, and 14 post-exposure",
        "Duration": "14-day post-exposure prophylaxis",
        "Diet_Advice": "Normal diet",
        "Lifestyle": "Immediate emergency room visit following animal bite"
    },
    {
        "Disease": "Tetanus",
        "Medicine": "Metronidazole IV, Tetanus Immune Globulin (TIG), Diazepam (for spasms)",
        "Dosage": "Hospital administration",
        "Duration": "Intensive care required",
        "Diet_Advice": "Parenteral nutrition if unable to swallow",
        "Lifestyle": "Maintain routine vaccinations (Tdap every 10 years)"
    },
    {
        "Disease": "Syphilis",
        "Medicine": "Benzathine Penicillin G",
        "Dosage": "2.4 million units IM single dose (for early syphilis)",
        "Duration": "Single injection (early), or 3 weekly injections (late)",
        "Diet_Advice": "Normal diet",
        "Lifestyle": "Abstain from sexual contact until sores heal and treatment completes"
    },
    {
        "Disease": "Gonorrhea",
        "Medicine": "Ceftriaxone",
        "Dosage": "500 mg IM as a single dose",
        "Duration": "Single dose",
        "Diet_Advice": "Normal diet",
        "Lifestyle": "Partner notification and treatment, abstain from sex for 7 days"
    },
    {
        "Disease": "HIV/AIDS",
        "Medicine": "Antiretroviral Therapy (ART) e.g., Biktarvy (Bictegravir/Emtricitabine/Tenofovir alafenamide)",
        "Dosage": "1 tablet daily",
        "Duration": "Lifelong",
        "Diet_Advice": "Food safety precautions, balanced nutrition",
        "Lifestyle": "Strict daily adherence, safe sex, routine CD4/Viral load checks"
    },
    {
        "Disease": "Cholera",
        "Medicine": "Oral Rehydration Salts (ORS), Doxycycline or Azithromycin",
        "Dosage": "Doxycycline: 300mg single dose; ORS: massive continuous volume replacement",
        "Duration": "Until diarrhea resolves (few days)",
        "Diet_Advice": "Strict continuous hydration, zinc supplementation",
        "Lifestyle": "Strict hygiene and sanitation, boil water"
    }
]

df = pd.DataFrame(data)
output_path = os.path.join(os.path.dirname(__file__), 'allopathic_medicines.xlsx')
df.to_excel(output_path, index=False)
print("Allopathic Excel file created with", len(data), "disease entries at", output_path)
