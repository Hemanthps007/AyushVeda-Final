import pandas as pd
import os

data = [
    {
        "Disease": "Common Cold",
        "Medicine": "Sitopaladi Churna, Trikatu Churna, Tulsi Ark",
        "Dosage": "Sitopaladi Churna: 3g twice daily with honey; Trikatu Churna: 1g thrice daily; Tulsi Ark: 10ml with warm water",
        "Duration": "5-7 days",
        "Diet_Advice": "Warm soups, ginger tea, avoid cold foods and dairy",
        "Lifestyle": "Steam inhalation, adequate rest, warm clothing"
    },
    {
        "Disease": "Influenza",
        "Medicine": "Mahasudarshan Churna, Sanshamani Vati, Giloy Ghanvati",
        "Dosage": "Mahasudarshan Churna: 3g twice daily; Sanshamani Vati: 2 tabs thrice daily; Giloy: 500mg twice daily",
        "Duration": "7-10 days",
        "Diet_Advice": "Light diet, warm fluids, herbal teas, avoid heavy meals",
        "Lifestyle": "Complete bed rest, steam inhalation with eucalyptus"
    },
    {
        "Disease": "Typhoid Fever",
        "Medicine": "Sudarshan Ghan Vati, Punarnava Mandur, Kutki Churna",
        "Dosage": "Sudarshan: 2 tabs thrice daily; Punarnava Mandur: 2 tabs twice daily; Kutki: 1g twice daily",
        "Duration": "14-21 days",
        "Diet_Advice": "Light easily digestible food, avoid spicy and oily food, plenty of fluids",
        "Lifestyle": "Complete rest, avoid strenuous activity"
    },
    {
        "Disease": "Malaria",
        "Medicine": "Saptaparna Bark, Kutki, Chirayata Kwath",
        "Dosage": "Saptaparna: 3g twice daily; Kutki: 1g thrice daily; Chirayata: 50ml decoction twice daily",
        "Duration": "14 days",
        "Diet_Advice": "Light diet with warm foods, bitter gourd juice, avoid dairy",
        "Lifestyle": "Mosquito protection, adequate rest, cool environment"
    },
    {
        "Disease": "Dengue Fever",
        "Medicine": "Papaya Leaf Extract, Giloy Ghanvati, Carica Papaya",
        "Dosage": "Papaya Leaf: 30ml juice twice daily; Giloy Ghanvati: 2 tabs thrice daily",
        "Duration": "7-10 days",
        "Diet_Advice": "Papaya leaf juice, pomegranate juice, hydrate well",
        "Lifestyle": "Complete rest, avoid aspirin, monitor platelet count"
    },
    {
        "Disease": "Diabetes",
        "Medicine": "Karela Churna, Jamun Seed Powder, Gudmar (Gurmar) Churna, Vijaysar",
        "Dosage": "Karela: 5g twice daily before meals; Jamun Seed: 3g twice daily; Gudmar: 3g twice daily",
        "Duration": "Ongoing (3 months minimum)",
        "Diet_Advice": "Low glycemic diet, bitter vegetables, avoid sweets and refined carbs",
        "Lifestyle": "30 minutes daily walk, yoga (especially Pranayama), stress management"
    },
    {
        "Disease": "Hypertension",
        "Medicine": "Sarpagandha Vati, Arjun Bark Powder, Brahmi Vati, Ashwagandha",
        "Dosage": "Sarpagandha: 1 tab twice daily; Arjun: 3g twice daily; Brahmi: 500mg twice daily",
        "Duration": "3 months minimum, under supervision",
        "Diet_Advice": "Low sodium diet, garlic, leafy greens, avoid caffeine and alcohol",
        "Lifestyle": "Daily meditation, light exercise, stress reduction techniques"
    },
    {
        "Disease": "Anxiety Disorder",
        "Medicine": "Ashwagandha Churna, Brahmi Ghrita, Jatamansi Churna, Tagara",
        "Dosage": "Ashwagandha: 3g twice daily with milk; Brahmi: 500mg thrice daily; Jatamansi: 1g twice daily",
        "Duration": "45-90 days",
        "Diet_Advice": "Sattvic diet, warm milk with turmeric, avoid stimulants",
        "Lifestyle": "Daily pranayama, meditation, adequate sleep routine"
    },
    {
        "Disease": "Depression",
        "Medicine": "Brahmi Churna, Shankhpushpi, Ashwagandha, Shatavari",
        "Dosage": "Brahmi: 3g twice daily; Shankhpushpi: 5ml syrup twice daily; Ashwagandha: 3g twice daily",
        "Duration": "90 days minimum",
        "Diet_Advice": "Nourishing foods, sesame, ghee, warm milk, avoid processed food",
        "Lifestyle": "Sunlight exposure, social activity, gentle exercise, proper sleep"
    },
    {
        "Disease": "Migraine",
        "Medicine": "Shirahshooladivajra Rasa, Pathyadi Kwath, Godanti Bhasma",
        "Dosage": "Shirahshoola: 2 tabs twice daily; Pathyadi: 50ml twice daily; Godanti: 250mg twice daily",
        "Duration": "30-60 days",
        "Diet_Advice": "Avoid trigger foods (cheese, chocolate, citrus), eat on schedule",
        "Lifestyle": "Regular sleep, avoid bright lights, stress management, Shirodhara therapy"
    },
    {
        "Disease": "Asthma",
        "Medicine": "Vasavaleha, Sitopaladi Churna, Kantakari Ghrita, Pushkarmool",
        "Dosage": "Vasavaleha: 10g twice daily; Sitopaladi: 3g thrice daily; Kantakari Ghrita: 10ml twice daily",
        "Duration": "90 days minimum",
        "Diet_Advice": "Light warm food, honey, turmeric milk, avoid cold foods and dairy",
        "Lifestyle": "Pranayama breathing exercises, avoid allergens, steam inhalation"
    },
    {
        "Disease": "Pneumonia",
        "Medicine": "Talisadi Churna, Mahasudarshan Churna, Vasarishta",
        "Dosage": "Talisadi: 3g thrice daily with honey; Mahasudarshan: 3g twice daily; Vasarishta: 20ml twice daily",
        "Duration": "14-21 days with medical supervision",
        "Diet_Advice": "Warm liquids, light easily digestible food, ginger-turmeric tea",
        "Lifestyle": "Complete rest, steam inhalation, chest physiotherapy"
    },
    {
        "Disease": "Gastritis",
        "Medicine": "Avipattikar Churna, Sutshekhar Rasa, Kamadudha Rasa, Shankha Bhasma",
        "Dosage": "Avipattikar: 3g twice daily before meals; Sutshekhar: 1 tab twice daily; Kamadudha: 500mg thrice daily",
        "Duration": "21-30 days",
        "Diet_Advice": "Light diet, coconut water, pomegranate, avoid spicy and acidic foods",
        "Lifestyle": "Eat small frequent meals, avoid stress eating, no alcohol"
    },
    {
        "Disease": "Urinary Tract Infection",
        "Medicine": "Chandraprabha Vati, Gokshura Churna, Punarnavadi Mandur",
        "Dosage": "Chandraprabha: 2 tabs thrice daily; Gokshura: 3g twice daily; Punarnavadi: 2 tabs twice daily",
        "Duration": "7-14 days",
        "Diet_Advice": "Drink 8-10 glasses water daily, coconut water, cranberry juice, avoid spices",
        "Lifestyle": "Proper hygiene, avoid holding urine, cotton undergarments"
    },
    {
        "Disease": "Arthritis",
        "Medicine": "Yograj Guggulu, Rasnasaptaka Kwath, Ashwagandha, Nirgundi Oil",
        "Dosage": "Yograj Guggulu: 2 tabs thrice daily; Rasnasaptaka: 50ml twice daily; Ashwagandha: 3g twice daily",
        "Duration": "3-6 months",
        "Diet_Advice": "Anti-inflammatory foods, turmeric milk, ginger, avoid nightshades",
        "Lifestyle": "Gentle yoga, warm oil massage (Abhyanga), avoid cold and damp environments"
    },
    {
        "Disease": "Anemia",
        "Medicine": "Lohasava, Punarnava Mandur, Dhatri Loha, Ashwagandha Lehyam",
        "Dosage": "Lohasava: 20ml twice daily after meals; Punarnava Mandur: 2 tabs twice daily; Dhatri Loha: 2 tabs twice daily",
        "Duration": "60-90 days",
        "Diet_Advice": "Iron-rich foods (dates, raisins, pomegranate, sesame), amla juice, jaggery",
        "Lifestyle": "Moderate exercise, sunlight exposure, adequate sleep"
    },
    {
        "Disease": "Thyroid Disorder",
        "Medicine": "Kanchanar Guggulu, Triphala Churna, Ashwagandha, Punarnava",
        "Dosage": "Kanchanar Guggulu: 2 tabs thrice daily; Triphala: 3g at night; Ashwagandha: 3g twice daily",
        "Duration": "90 days minimum",
        "Diet_Advice": "Coconut oil, seaweed, iodine-rich foods, avoid cruciferous vegetables raw",
        "Lifestyle": "Yoga (especially Sarvangasana), stress reduction, regular sleep schedule"
    },
    {
        "Disease": "Liver Disease",
        "Medicine": "Arogyavardhini Vati, Phyllanthus (Bhumi Amla), Kalmegh Churna, Punarnava",
        "Dosage": "Arogyavardhini: 2 tabs thrice daily; Bhumi Amla: 3g twice daily; Kalmegh: 1g twice daily",
        "Duration": "90 days under supervision",
        "Diet_Advice": "Avoid alcohol completely, light diet, plenty of water, beetroot juice",
        "Lifestyle": "No alcohol, avoid fatty foods, gentle exercise, adequate rest"
    },
    {
        "Disease": "Kidney Disease",
        "Medicine": "Gokshuradi Guggulu, Punarnava Mandur, Chandraprabha Vati, Varuna Bark",
        "Dosage": "Gokshuradi: 2 tabs thrice daily; Punarnava Mandur: 2 tabs twice daily; Chandraprabha: 2 tabs twice daily",
        "Duration": "90 days under medical supervision",
        "Diet_Advice": "Low protein, low sodium diet, adequate hydration, avoid processed foods",
        "Lifestyle": "Monitor blood pressure, avoid NSAIDs, regular kidney function tests"
    },
    {
        "Disease": "Skin Allergy",
        "Medicine": "Mahamanjishthadi Kwath, Gandhak Rasayana, Neem Churna, Haridra Khand",
        "Dosage": "Mahamanjishthadi: 50ml twice daily; Gandhak Rasayana: 2 tabs twice daily; Haridra Khand: 3g twice daily",
        "Duration": "30-60 days",
        "Diet_Advice": "Avoid allergen foods, cool and light diet, bitter gourd, neem leaves",
        "Lifestyle": "Cotton clothing, avoid synthetic fabrics, cold water bathing, neem paste application"
    },
    {
        "Disease": "Covid-19",
        "Medicine": "Ayush Kwath, Ashwagandha, Giloy Ghanvati, Anu Taila",
        "Dosage": "Ayush Kwath: 50ml twice daily; Giloy: 500mg twice daily; Anu Taila: 2 drops per nostril daily",
        "Duration": "14-21 days",
        "Diet_Advice": "Warm fluids, ginger, turmeric milk, light soups",
        "Lifestyle": "Infection isolation, proper rest, steam inhalation strictly"
    },
    {
        "Disease": "Tuberculosis",
        "Medicine": "Swarna Vasant Malti Rasa, Chyawanprash, Pippali Rasayana",
        "Dosage": "Swarna Vasant Malti: 125mg once daily; Chyawanprash: 10g twice daily",
        "Duration": "Ongoing along with Allopathic DOTS",
        "Diet_Advice": "Highly nutritious diet, ghee, milk, dry fruits, avoid junk food",
        "Lifestyle": "Complete rest, sunlight exposure, avoid cold environment"
    },
    {
        "Disease": "Measles",
        "Medicine": "Sanjivani Vati, Godanti Bhasma, Neem juice applications",
        "Dosage": "Sanjivani Vati: 1 tab twice daily; Godanti: 250mg twice daily with honey",
        "Duration": "10-15 days",
        "Diet_Advice": "Liquid diet, barley water, avoid heavy foods",
        "Lifestyle": "Isolation, rest, neem water baths for rash"
    },
    {
        "Disease": "Chickenpox",
        "Medicine": "Neem Kwath, Swarna Makshik Bhasma, Giloy Satva",
        "Dosage": "Neem Kwath: 20ml twice daily; Giloy Satva: 500mg twice daily",
        "Duration": "14 days",
        "Diet_Advice": "Cooling foods, tender coconut water, mung bean soup",
        "Lifestyle": "Neem baths, avoid scratching, cotton clothing"
    },
    {
        "Disease": "Psoriasis",
        "Medicine": "Panchatikta Ghrita Guggulu, Khadirarishta, Neem Churna",
        "Dosage": "Panchatikta: 2 tabs twice daily; Khadirarishta: 20ml twice daily",
        "Duration": "3-6 months",
        "Diet_Advice": "Avoid dairy, spicy, and sour foods. Eat bitter gourd, leafy greens",
        "Lifestyle": "Stress management, daily moisturizer (coconut oil), regular detox"
    },
    {
        "Disease": "Eczema",
        "Medicine": "Khadirarishta, Arogyavardhini Vati, Manjistha, Neem",
        "Dosage": "Khadirarishta: 20ml twice daily; Arogyavardhini: 2 tabs twice daily",
        "Duration": "2-4 months",
        "Diet_Advice": "Avoid artificial colors, preservatives, and excessive salt",
        "Lifestyle": "Avoid harsh soaps, use plain water or oatmeal baths"
    },
    {
        "Disease": "Osteoarthritis",
        "Medicine": "Lakshadi Guggulu, Shallaki (Boswellia), Ashwagandha",
        "Dosage": "Lakshadi: 2 tabs twice daily; Shallaki: 1 cap twice daily",
        "Duration": "3-6 months",
        "Diet_Advice": "Calcium-rich diet, avoid deep fried and vata-aggravating foods",
        "Lifestyle": "Gentle movements, avoid cold drafts, warm oil massages"
    },
    {
        "Disease": "Gout",
        "Medicine": "Kaishore Guggulu, Giloy, Punarnava, Guduchi",
        "Dosage": "Kaishore Guggulu: 2 tabs twice daily; Giloy juice: 20ml twice daily",
        "Duration": "3 months",
        "Diet_Advice": "Strict vegetarian diet, avoid spinach, tomatoes, and high purine foods",
        "Lifestyle": "Increased water intake, avoid sudden intense exercises"
    },
    {
        "Disease": "Irritable Bowel Syndrome",
        "Medicine": "Bilwadi Churna, Kutajarishta, Takrarishta, Musta",
        "Dosage": "Bilwadi Churna: 3g twice daily; Kutajarishta: 20ml twice daily",
        "Duration": "3-6 months",
        "Diet_Advice": "Fresh buttermilk (Takra), avoid excessive raw foods and dairy",
        "Lifestyle": "Mindful eating, slow chewing, manage anxiety and stress"
    },
    {
        "Disease": "Peptic Ulcer",
        "Medicine": "Kamadudha Rasa, Sutshekhar Rasa, Amalaki (Amla) Churna",
        "Dosage": "Kamadudha: 500mg twice daily; Amalaki: 3g twice daily with water",
        "Duration": "2-3 months",
        "Diet_Advice": "Milk, ghee, cooling foods. Avoid spicy, acidic, and fermented items",
        "Lifestyle": "Avoid emotional stress, regular meal timings"
    },
    {
        "Disease": "Hepatitis A",
        "Medicine": "Bhui Amla (Phyllanthus), Kalmegh, Kumaryasava",
        "Dosage": "Bhui Amla: 3g twice daily; Kumaryasava: 20ml twice daily",
        "Duration": "1-2 months",
        "Diet_Advice": "Fat-free diet, sugarcane juice, boiled vegetables",
        "Lifestyle": "Strict bed rest, avoid alcohol permanently, sanitation"
    },
    {
        "Disease": "Hepatitis B",
        "Medicine": "Phalghrita, Arogyavardhini Vati, Punarnavadi Kwath",
        "Dosage": "Arogyavardhini: 2 tabs twice daily; Punarnavadi: 30ml twice daily",
        "Duration": "6+ months",
        "Diet_Advice": "Antioxidant-rich foods, low protein, strict avoidance of alcohol",
        "Lifestyle": "Liver protection, avoid extreme exertion"
    },
    {
        "Disease": "Alzheimers Disease",
        "Medicine": "Brahmi, Shankhpushpi, Jyotishmati, Ashwagandha",
        "Dosage": "Brahmi: 500mg twice daily; Jyotishmati oil drops in milk",
        "Duration": "Ongoing support",
        "Diet_Advice": "Almonds, walnuts, ghee, warm digesting spices",
        "Lifestyle": "Mental stimulation, routine structuring, emotional support"
    },
    {
        "Disease": "Parkinsons Disease",
        "Medicine": "Kapikacchu, Ashwagandha, Mashabaladi Kwath",
        "Dosage": "Kapikacchu: 3g twice daily with milk; Ashwagandha: 2 tabs twice daily",
        "Duration": "Ongoing management",
        "Diet_Advice": "Nourishing, unctuous foods (Vata pacifying)",
        "Lifestyle": "Daily oil massage, physical therapy, gentle yoga"
    },
    {
        "Disease": "Multiple Sclerosis",
        "Medicine": "Bala (Sida cordifolia), Ashwagandha, Guduchi",
        "Dosage": "Bala Churna: 3g twice daily with milk; Guduchi: 500mg daily",
        "Duration": "Ongoing management",
        "Diet_Advice": "Anti-inflammatory diet, avoid processed sugars",
        "Lifestyle": "Avoid heat exposure, maintain low-stress routine"
    },
    {
        "Disease": "Cataracts",
        "Medicine": "Triphala Churna, Mahatriphaladi Ghrita",
        "Dosage": "Triphala: 3g daily at night; Mahatriphaladi: 1tsp at bedtime",
        "Duration": "3-6 months",
        "Diet_Advice": "Vitamin A rich foods, carrots, spinach",
        "Lifestyle": "Eye wash with Triphala water, protect from UV"
    },
    {
        "Disease": "Glaucoma",
        "Medicine": "Saptamrit Lauh, Punarnavadi Kwath",
        "Dosage": "Saptamrit Lauh: 250mg twice daily; Kwath: 20ml twice daily",
        "Duration": "Ongoing",
        "Diet_Advice": "Avoid excessive fluids at once, eat light foods",
        "Lifestyle": "Avoid inverted yoga poses, avoid heavy lifting"
    },
    {
        "Disease": "Appendicitis",
        "Medicine": "Emergency surgical evaluation required",
        "Dosage": "N/A",
        "Duration": "Urgent",
        "Diet_Advice": "Fasting until evaluation",
        "Lifestyle": "Immediate hospitalization"
    },
    {
        "Disease": "Hemorrhoids",
        "Medicine": "Arshoghni Vati, Abhayarishta, Triphala",
        "Dosage": "Arshoghni Vati: 2 tabs twice daily; Abhayarishta: 20ml post meals",
        "Duration": "1-3 months",
        "Diet_Advice": "High fiber diet, papaya, buttermilk, avoid spicy/fried foods",
        "Lifestyle": "Avoid sitting for long hours, maintain bowel regularity"
    },
    {
        "Disease": "Kidney Stones",
        "Medicine": "Pashanbhed, Varuna, Gokshura, Hajrul Yahood Bhasma",
        "Dosage": "Pashanbhed Kwath: 40ml twice daily; Gokshura: 2 tabs twice daily",
        "Duration": "1-2 months",
        "Diet_Advice": "High water intake, avoid tomatoes/spinach, restrict sodium",
        "Lifestyle": "Active lifestyle, avoid dehydration"
    },
    {
        "Disease": "Bronchitis",
        "Medicine": "Kantakari Avaleha, Sitopaladi Churna, Kaphaketu Rasa",
        "Dosage": "Kantakari: 10g twice daily; Sitopaladi: 3g thrice daily",
        "Duration": "4-6 weeks",
        "Diet_Advice": "Warm liquids, honey, ginger, avoid dairy and cold foods",
        "Lifestyle": "Breathing exercises, steam inhalation, avoid dust/smoke"
    },
    {
        "Disease": "Sinusitis",
        "Medicine": "Laxmivilas Rasa, Tribhuvan Kirti Rasa, Anu Taila",
        "Dosage": "Laxmivilas Rasa: 1 tab twice daily; Anu Taila: nasal drops",
        "Duration": "3-4 weeks",
        "Diet_Advice": "Hot soups, garlic, turmeric, avoid cold and heavy food",
        "Lifestyle": "Steam inhalation daily, avoid cold breezes"
    },
    {
        "Disease": "Tonsillitis",
        "Medicine": "Khadiradi Vati, Kanchanar Guggulu, Sphatika Bhasma (Gargle)",
        "Dosage": "Khadiradi Vati: chew 1 tab thrice daily; Kanchanar: 2 tabs twice daily",
        "Duration": "2-3 weeks",
        "Diet_Advice": "Soft warm foods, ginger/honey tea, avoid sour and cold foods",
        "Lifestyle": "Warm saline gargles, protect neck from cold"
    },
    {
        "Disease": "Meningitis",
        "Medicine": "Supportive only: Brahmi, Suvarna Bhasma. Emergency medical care needed.",
        "Dosage": "Hospitalization",
        "Duration": "Emergency",
        "Diet_Advice": "As advised by hospital",
        "Lifestyle": "Intensive Care"
    },
    {
        "Disease": "Rabies",
        "Medicine": "No Ayurvedic cure. Requires immediate Allopathic vaccination.",
        "Dosage": "Emergency vaccination",
        "Duration": "Emergency",
        "Diet_Advice": "N/A",
        "Lifestyle": "Preventive animal vaccination, seek immediate medical help post bite"
    },
    {
        "Disease": "Tetanus",
        "Medicine": "Emergency medical care + Tetanus Toxoid. Supportive: Ashwagandha.",
        "Dosage": "Hospitalization",
        "Duration": "Emergency",
        "Diet_Advice": "N/A",
        "Lifestyle": "Immunization"
    },
    {
        "Disease": "Syphilis",
        "Medicine": "Requires Antibiotics. Supportive Ayurveda: Rasmanikya, Chopchini.",
        "Dosage": "Chopchini: 3g twice daily",
        "Duration": "Supportive 1-2 months",
        "Diet_Advice": "Avoid spicy, salty, sour foods. Light diet.",
        "Lifestyle": "Hygiene, abstention"
    },
    {
        "Disease": "Gonorrhea",
        "Medicine": "Requires Antibiotics. Supportive Ayurveda: Chandanasava, Gokshura.",
        "Dosage": "Chandanasava: 20ml twice daily; Gokshura: 3g twice daily",
        "Duration": "Supportive 1 month",
        "Diet_Advice": "Plenty of fluids, barley water, avoid spices",
        "Lifestyle": "Hygiene, abstention"
    },
    {
        "Disease": "HIV/AIDS",
        "Medicine": "Requires ART. Supportive: Guduchi, Ashwagandha, Chyawanprash (Rasayana).",
        "Dosage": "Giloy: 500mg twice daily; Chyawanprash: 10g daily",
        "Duration": "Lifelong supportive",
        "Diet_Advice": "Immunity-boosting balanced diet, well-cooked clean food",
        "Lifestyle": "Infection prevention, stress management, regular medical checkups"
    },
    {
        "Disease": "Cholera",
        "Medicine": "Requires immediate ORS/IV fluids. Supportive: Sanjivani Vati, Karpura Rasa.",
        "Dosage": "Sanjivani Vati: 1 tab twice daily",
        "Duration": "Supportive",
        "Diet_Advice": "Strict fluid replacement, coconut water, rice gruel",
        "Lifestyle": "Sanitized water, strict hygiene"
    }
]

df = pd.DataFrame(data)

# FIX: Relative path using current module directory
out_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(out_dir, 'ayurvedic_medicines.xlsx')

df.to_excel(excel_path, index=False)
print("Excel file created with", len(data), "disease entries at", excel_path)
