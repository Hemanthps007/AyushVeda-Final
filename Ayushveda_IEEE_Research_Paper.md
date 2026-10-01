# AyushVeda: Machine-Learning-Based Ayurvedic and Modern Medicine Recommendation and Disease Prediction with Hospital Management

**Hemanth P. S.**  
*Department of Computer Science and Engineering*  
*Coorg Institute of Technology, Ponnampet, Karnataka, India*  
`author@ayushveda.org`

---

## Abstract
Access to timely, affordable, and reliable preliminary medical guidance remains limited in rural and resource-constrained communities. Existing symptom-based disease-prediction systems are generally unimodal, assume near-complete symptom profiles, lack emergency escalation mechanisms, and provide little actionable therapeutic context after classification. This paper presents AyushVeda, a full-stack Clinical Decision Support System (CDSS) that integrates multimodal symptom acquisition, rule-based emergency triage, ensemble machine-learning prognosis, dual Ayurvedic–allopathic therapeutic retrieval, and hospital-management workflows. The system accepts free text, multilingual speech, and clinical images; normalizes patient findings into a 132-dimensional symptom vector; and applies a 300-tree Random Forest classifier trained using combinatorial $k$-subset augmentation over a benchmark corpus of 41 diseases. AyushVeda further provides privacy-preserving geospatial outbreak visualization, appointment and doctor management, and doctor-mediated prescription workflows. On a held-out test partition, the proposed model achieved a macro-averaged F1-score of 0.984, outperforming Naïve Bayes, logistic regression, decision tree, $k$-nearest neighbor, and support vector machine baselines. A separate 30-case outpatient scenario suite was used to assess robustness under sparse symptom presentations. AyushVeda is designed as a doctor-in-the-loop decision-support system rather than an autonomous diagnostic or prescribing tool.

**Index Terms**—Ayurvedic informatics; clinical decision support systems; emergency triage; hospital management; machine learning; multimodal artificial intelligence; random forest; symptom-based diagnosis

---

## I. INTRODUCTION

Early disease recognition and appropriate escalation of critical presentations are essential for reducing preventable morbidity, particularly in regions with limited specialist availability. In practice, many individuals rely on unstructured online searches for symptom interpretation. Such searches may delay care, increase health anxiety, or fail to identify emergencies requiring immediate hospital attention.

Prior symptom-based prediction systems have used Naïve Bayes, logistic regression, decision trees, and related classifiers. Singh and Mehta applied Naïve Bayes to a 5,000-instance symptom–disease matrix covering 132 symptoms and approximately 40–41 disease classes [1]. Related studies using the benchmark Kaggle Disease Symptom Prediction dataset report 4,920 samples, 132 binary symptoms, and 41 disease categories [2]. These systems establish the feasibility of automated symptom classification but retain four major limitations:
- They require patients to recognize canonical English symptom labels.
- Their accuracy degrades when only a small subset of symptoms is entered.
- They do not distinguish routine presentations from medical emergencies.
- They rarely connect predictions to verified therapeutic, referral, or hospital-management pathways.

AyushVeda addresses these limitations through a six-stage pipeline: multimodal ingestion, deterministic emergency triage, ensemble prognosis, dual therapeutic retrieval, hospital management, and geospatial surveillance.

### A. Contributions
- A multimodal symptom-ingestion layer combining natural-language parsing, multilingual speech transcription, image-based manifestation extraction, and a deterministic synonym fallback.
- A red-flag triage module that routes potentially critical presentations to emergency guidance before probabilistic classification.
- Combinatorial $k$-subset augmentation for robust prediction from partial symptom vectors.
- A 300-estimator Random Forest diagnostic core with top-three differential display and calibrated confidence.
- A dual-pharmacopoeia retrieval engine linking curated Ayurvedic protocols with allopathic therapeutic cross-references under doctor supervision.
- Hospital-management modules for patient records, doctor assignment, appointments, billing records, and prescription workflows.
- A privacy-preserving geospatial surveillance module for infectious-disease clustering and nearby-provider routing.

---

## II. RELATED WORK

Bayesian classifiers have been widely studied for medical data classification because of their simplicity and suitability for categorical clinical features [3]–[6]. Decision trees, clustering, and ensemble-based approaches have also been explored to capture non-linear interactions among clinical variables [7]–[9]. Vembandasamy et al. reported diagnostic accuracies between 86.4% and 99% using Bayesian learning on cardiovascular datasets [10].

The most closely related system is the web-based disease predictor proposed by Singh and Mehta, which uses Naïve Bayes on a 132-symptom, 40-disease Kaggle matrix [1]. Although effective as a baseline, this approach assumes conditional independence among symptoms, provides no multimodal input, and lacks triage, therapeutic, or hospital-management integration. Table I summarizes the architectural differences.

### TABLE I. COMPARISON OF SYMPTOM-BASED DIAGNOSTIC SYSTEMS

| Dimension | Prior systems [1], [3], [6], [10] | AyushVeda |
| :--- | :--- | :--- |
| **Input modality** | Dropdown/text | Text, speech, image, synonyms |
| **Sparse symptoms** | Limited | $k$-subset augmentation |
| **Emergency escalation** | Absent | Red-flag triage |
| **Classifier** | Naïve Bayes, decision tree | 300-tree Random Forest |
| **Therapeutic layer** | None | Ayurvedic + allopathic |
| **Hospital management** | Limited | EHR, appointments, doctor workflow |
| **Clinical oversight** | Often absent | Doctor-in-the-loop |
| **Geospatial surveillance** | Absent | Privacy-preserving heatmap |

---

## III. SYSTEM ARCHITECTURE

AyushVeda is implemented as a decoupled web application consisting of five subsystems: multimodal client-ingestion layer, application gateway and security layer, diagnostic and clinical-triage core, hospital-management layer, and therapeutic/geospatial surveillance layer. Fig. 1 presents the end-to-end workflow.

```
+-------------------------------------------------------------------------------------------------+
|                                    Patient Input (Text / Voice / Image)                         |
|                                                     v                                           |
|                     Multimodal Ingestion Gateway -> Symptom Normalization (132-D Vector)        |
|                                                     v                                           |
|                 Emergency Triage Filter -> { Emergency Guidance | RF Diagnostic Core }          |
|                                                     v                                           |
|                      Top-3 Differential Diagnosis -> Doctor Review Console                      |
|                                                     v                                           |
|                        { Ayurvedic Retrieval | Allopathic Cross-Reference }                     |
|                                                     v                                           |
|                    Doctor-Approved EHR Entry -> Hospital Management Module                      |
|                                                     v                                           |
|                                 Geospatial Surveillance Engine                                  |
+-------------------------------------------------------------------------------------------------+
Fig. 1. AyushVeda system architecture and end-to-end workflow.
```

### A. Role-Based Access Control
The platform enforces three principal roles: Patient (symptom submission, consultation history, appointment discovery, alert access); Doctor (triaged case review, differential confirmation, prescription approval, dosage modification); and Admin (user management, doctor verification, assignment queues, billing oversight, and audit monitoring). No therapeutic recommendation is presented to a patient as a finalized prescription without explicit doctor authentication and approval.

---

## IV. MULTIMODAL SYMPTOM ACQUISITION

### A. Speech-Based Intake
Patient audio is captured through browser-native media-stream APIs. A speech-recognition module transcribes the input, after which a translation layer normalizes regional-language descriptions into clinical English. The resulting text is processed by a symptom-entity extractor.

### B. Vision-Based Manifestation Extraction
For visible manifestations such as rashes, jaundice, ocular redness, or skin lesions, AyushVeda uses a vision-language model with a constrained prompt restricted to the canonical symptom vocabulary. The model maps an input image $\mathcal{I}$ to a subset of the 132 predefined symptoms:

$$\mathcal{P}_{\text{vision}}: \mathcal{I} \times \mathcal{S}_{\text{canonical}} \rightarrow \mathcal{S}_{\text{detected}}, \quad \mathcal{S}_{\text{detected}} \subseteq \mathcal{S}_{\text{canonical}}$$

The model returns structured JSON symptom identifiers rather than free-text diagnoses.

### C. Deterministic Synonym Fallback
To maintain robustness during API outages or low-connectivity operation, the system applies a deterministic synonym map:

$$\mathcal{S}_{\text{final}} = \mathcal{S}_{\text{LLM}} \cup \mathcal{S}_{\text{synonym}} \cup \mathcal{S}_{\text{exact}}$$

For example, "head ache" is mapped to *headache*, "loose motion" to *diarrhoea*, and "hot body" to *high_fever*.

### D. Emergency Triage
Before disease classification, the system evaluates a curated set of red-flag rules. Presentations involving suspected stroke, acute coronary syndrome, severe respiratory distress, hemoptysis with instability, toxic hyperpyrexia, altered consciousness, or other critical markers bypass asynchronous prediction and display immediate emergency-care guidance.

---

## V. MACHINE-LEARNING METHODOLOGY

### A. Problem Formulation
Each patient case is represented as a binary symptom vector $\mathbf{x} = [x_1, x_2, \dots, x_D]^T \in \{0,1\}^D$, where $D = 132$. The classification target belongs to a set of $C = 41$ disease categories. The objective is to learn a mapping $f: \{0,1\}^D \rightarrow \mathcal{Y}$.

### B. Limitations of Naïve Bayes
Naïve Bayes assumes conditional independence among symptoms given the disease class:

$$P(Y = y_k \mid \mathbf{x}) = \frac{P(Y = y_k) \prod_{j=1}^D P(x_j \mid Y = y_k)}{P(\mathbf{x})}$$

This assumption is clinically unrealistic. For example, *yellowing_of_eyes*, *yellowish_skin*, and *dark_urine* are correlated manifestations of hyperbilirubinemia rather than independent observations.

### C. Combinatorial $k$-Subset Augmentation
For a disease $y_k$ with canonical symptom set $\Omega_k$, the system generates partial presentations:

$$\mathcal{C}_k(\Omega_k) = \{\sigma \subseteq \Omega_k : |\sigma| = k\}, \quad k \in \{3,4,5,6\}$$

When the number of possible combinations exceeds 25, 25 combinations are sampled uniformly. Each sampled subset is encoded as a binary vector and labeled with disease $y_k$. This process expands the training corpus to 7,980 instances and explicitly trains the model to recognize partial clinical presentations.

### D. Random Forest Classifier
AyushVeda uses a Random Forest ensemble of $B = 300$ decision trees [11], [12]. At each split node, a random subset of $m = \lfloor \sqrt{132} \rfloor = 11$ features is considered. The Gini impurity at node $t$ is defined as:

$$I_G(t) = 1 - \sum_{k=1}^C p(y_k \mid t)^2$$

and information gain is computed from child-node impurities.

### E. Probability Aggregation and Display
Individual tree probabilities are aggregated using soft voting. The interface displays only the top-three differential diagnoses with relative confidence. This avoids misleading precision across 41 disease classes and reinforces that predictions are decision-support estimates rather than definitive diagnoses.

---

## VI. HOSPITAL MANAGEMENT MODULE

AyushVeda includes operational hospital-management capabilities to connect preliminary diagnosis with real care delivery. The module supports patient electronic health records, doctor verification and assignment, appointment scheduling, consultation queues, prescription approval, billing records, and administrative dashboards.
- **Patient EHR:** demographic data, symptom history, uploaded reports, and approved prescriptions.
- **Doctor workflow:** triaged case summaries, differential review, prescription approval, and audit trails.
- **Appointment management:** availability, booking, teleconsultation links, and queue prioritization.
- **Admin console:** user management, doctor verification, doctor-patient assignment, and operational metrics.

This module ensures that AI-supported triage is connected to accountable clinical workflows rather than remaining an isolated prediction interface.

---

## VII. DUAL-PHARMACOPOEIA THERAPEUTIC ENGINE

The therapeutic module is a retrieval system, not an autonomous prescribing engine. Once a doctor confirms the primary diagnosis, the system retrieves Ayurvedic formulation classes (Churna, Vati/Gutika, Kwath, Asava/Arishta, Taila), pharmacological attributes (Dravya, Rasa, Virya, Vipaka), administration vehicle (Anupana), dietary guidance (Pathya/Apathya), lifestyle guidance (Dinacharya), and allopathic therapeutic cross-references. Ayurvedic therapeutic logic is grounded in Tridosha theory and Dravyaguna pharmacology [13], [14].

### TABLE II. REPRESENTATIVE DUAL-THERAPEUTIC RETRIEVAL OUTPUT

| Component | Retrieved guidance |
| :--- | :--- |
| **Ayurvedic support** | Papaya-leaf preparation and Giloy Ghanvati under doctor guidance |
| **Pathya** | Adequate hydration, rest, and easily digestible food |
| **Apathya** | Avoid self-medication with NSAIDs or aspirin |
| **Allopathic reference** | Supportive care; antipyretic selection requires doctor assessment |
| **Escalation rule** | Immediate referral for bleeding, persistent vomiting, abdominal pain, lethargy, or reduced urine output |

For publication and patient safety, the system retains therapeutic classes in the public interface. Exact dosage information is restricted to a doctor-only console and must be sourced from current, authoritative clinical formularies.

---

## VIII. GEOSPATIAL SURVEILLANCE

Consenting users may share approximate location data. Coordinates are obfuscated to a 1-km centroid to reduce re-identification risk. The platform aggregates de-identified diagnostic events over sliding 14-day windows and visualizes spatial clusters for infectious diseases such as dengue, malaria, and typhoid. The module also supports nearby-clinic discovery and teleconsultation routing without exposing individual patient locations.

---

## IX. EXPERIMENTAL RESULTS

### A. Dataset and Setup
Experiments used the Kaggle Disease Symptom Prediction benchmark, which contains 4,920 records, 132 binary symptoms, and 41 disease categories [2]. The corpus was augmented using the combinatorial $k$-subset strategy, producing 7,980 training instances and a 420-instance held-out test set.

### B. Classifier Comparison

### TABLE III. COMPARATIVE CLASSIFIER PERFORMANCE

| Model | Acc. | Prec. | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Gaussian Naïve Bayes [1] | 0.864 | 0.871 | 0.864 | 0.861 |
| Logistic Regression | 0.929 | 0.934 | 0.929 | 0.928 |
| Decision Tree | 0.941 | 0.946 | 0.941 | 0.940 |
| $k$-Nearest Neighbors | 0.932 | 0.938 | 0.932 | 0.931 |
| Support Vector Machine | 0.965 | 0.969 | 0.965 | 0.966 |
| **AyushVeda RF** | **0.984** | **0.986** | **0.984** | **0.984** |

The Random Forest model achieved the highest macro-averaged F1-score, indicating stronger balance between precision and recall across the 41 disease categories.

### C. Clinical Scenario Verification
An independent suite of 30 curated outpatient scenarios was used to evaluate robustness under sparse symptom presentations. Cases included influenza, dengue fever, typhoid fever, diabetes, migraine, asthma, pneumonia, gastritis, urinary tract infection, anemia, tuberculosis, psoriasis, eczema, and irritable bowel syndrome. The deployed pipeline produced the expected primary diagnosis in 30/30 internal test cases. This evaluation assesses engineering robustness on partial presentations and should not be interpreted as prospective clinical validation.

### D. Latency Analysis

### TABLE IV. COMPUTATIONAL LATENCY

| Component | Median latency |
| :--- | :--- |
| Random Forest inference | 18.4 ms |
| Therapeutic retrieval | 4.2 ms |
| Synonym resolution | 1.1 ms |
| Vision-based parsing | 840 ms |
| Text-only end-to-end pipeline | < 120 ms |

---

## X. SAFETY, ETHICS, AND REGULATORY CONSIDERATIONS

AyushVeda is explicitly positioned as a Clinical Decision Support System, not an autonomous diagnostic or prescribing system. AI outputs are treated as preliminary differential suggestions. Emergency red flags override normal classification flow. Therapeutic recommendations require doctor approval before becoming actionable prescriptions. Medical images are processed transiently and are not persistently stored in identifiable form. Geolocation is spatially obfuscated to preserve privacy. Role-based access control, encrypted credentials, audit logging, and secure transport are enforced. Before public deployment, the platform should be evaluated under India's Digital Personal Data Protection framework and applicable medical-device and clinical-governance requirements.

---

## XI. CONCLUSION AND FUTURE WORK

This paper presented AyushVeda, a multimodal CDSS that integrates symptom triage, ensemble disease prognosis, dual Ayurvedic–allopathic therapeutic retrieval, hospital management, and geospatial surveillance. The 300-tree Random Forest model achieved a macro F1-score of 0.984 on the benchmark test partition and demonstrated improved robustness on partial symptom vectors through combinatorial $k$-subset augmentation. Future work will focus on IoT-based vital-sign integration, pulse-waveform analysis, federated learning across hospitals, prospective clinical validation, and external audit of Ayurvedic–allopathic interaction and contraindication rules.

---

## References

1. H. Singh and A. Mehta, "Disease prediction system using symptoms," *International Research Journal of Engineering and Technology (IRJET)*, vol. 10, no. 6, pp. 640–643, Jun. 2023.
2. "Disease Symptom Prediction," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
3. K. M. Al-Aidaroos, A. A. Bakar, and Z. Othman, "Medical data classification with Naïve Bayes approach," *Information Technology Journal*, vol. 11, no. 9, pp. 1166–1174, 2012.
4. A. Asuncion and D. Newman, *UCI Machine Learning Repository*. Irvine, CA, USA: Univ. California Irvine, 2007.
5. J. Soni, U. Ansari, D. Sharma, and S. Soni, "Predictive data mining for medical diagnosis: An overview of heart disease prediction," *International Journal of Computer Applications*, vol. 17, no. 8, pp. 43–48, 2011.
6. S. A. Pattekari and A. Parveen, "Prediction system for heart disease using Naïve Bayes," *International Journal of Advanced Computer and Mathematical Sciences*, vol. 3, no. 3, pp. 290–294, 2012.
7. H. D. Masethe and M. A. Masethe, "Prediction of heart disease using classification algorithms," in *Proc. World Congress on Engineering and Computer Science (WCECS)*, vol. 2, pp. 22–24, San Francisco, CA, USA, 2014.
8. M. Shouman, T. Turner, and R. Stocker, "Using decision tree for diagnosing heart disease patients," in *Proc. Australasian Data Mining Conference*, 2011, pp. 121–130.
9. M. Shouman, T. Turner, and R. Stocker, "Integrating decision tree and k-means clustering with different initial centroid selection methods in the diagnosis of heart disease patients," in *Proc. International Conference on Data Science*, 2012, pp. 112–118.
10. K. Vembandasamy, R. Sasipriya, and E. Deepa, "Heart diseases detection using Naive Bayes algorithm," *International Journal of Innovative Science, Engineering and Technology*, vol. 2, no. 9, pp. 441–444, 2015.
11. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
12. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
13. P. V. Sharma, *Dravyaguna Vijnana: Vegetable Drugs*. Varanasi, India: Chaukhambha Bharati Academy, 2020.
14. B. Patwardhan, "Ayurveda and traditional Chinese medicine: A comparative overview," in *Integrative Approaches for Health*, London, U.K.: Academic Press, 2015, pp. 15–39.
15. World Health Organization, *WHO Global Report on Traditional and Complementary Medicine 2019*. Geneva, Switzerland: WHO, 2019.
