## 1. High-Level Summary

AstroVital (Deep Space Medical OS) is an autonomous clinical operating system and multimodal Clinical Decision Support System (CDSS) designed for long-duration crewed missions in deep space. 

In interplanetary missions to Mars, the one-way communication latency of up to 22 minutes (44 minutes round trip) and potential solar blackouts make real-time telemedicine from Earth unfeasible. If an astronaut experiences an acute decompensation, a fracture during an extravehicular activity (EVA), or a radiological event, the crew needs immediate onboard diagnostic capabilities.

We addressed this challenge by developing a 100% offline Edge AI ecosystem that acts as a virtual flight surgeon. Unlike conventional biometric monitors that evaluate variables in isolation, AstroVital fuses two inference engines: a tabular machine learning classifier for continuous physiological telemetry and a network of pre-trained Vision Transformers (ViT) for medical imaging diagnostics (X-rays, CT scans, and facial biometrics). Both streams converge into a Temporal Causal Correlation Engine, capable of cross-referencing morphological findings in images with the biochemical and dosimetric history of the same operational day, delivering comprehensive clinical verdicts without relying on an internet connection or specialized graphical hardware (GPUs).

## 2. Project Details

What exactly does it do?
AstroVital centralizes bioastronautic medical surveillance through interconnected operational modules:
* Mission Telemetry Monitoring: Real-time health tracking of the entire crew using key indicators: heart rate (bpm), systolic blood pressure (mmHg), sleep hygiene (hours/24h), bone mineral density (T-Score), and daily/accumulated ionizing radiation dosimetry (mSv).
* Interactive Anatomical Biometric Intake: Captures vital signs via an interactive spatial coordinate anatomical model (Head/Nervous System, Torso/Cardiovascular, Extremities/Musculoskeletal, and Environment/Dosimetry) or a unified clinical form, evaluating deviations instantly.
* Computer Vision Diagnostics (Transfer Learning): Processes real medical studies uploaded by the crew in three critical modalities:
  * Bone Densitometry and Traumatology (X-Rays): Detection of microfractures and trabecular loss.
  * Computed Tomography (CT): Identification of brain anomalies and signs of intracranial hypertension.
  * Behavioral and Facial Biometrics: Recognition of facial expressions to quantify circadian fatigue, severe exhaustion, or acute stress.
* Causal Correlation Inference: Automatically cross-references the neural network's visual verdict with the physiological telemetry recorded on that exact mission date (YYYY-MM-DD), determining whether a visual anomaly is an isolated primary finding or a systemic consequence of microgravity and radiation.
* Clinical Comparator against Healthy Control Standard: Contrasts the historical evolution of astronauts against an ideal physiological baseline (Healthy Control Parameter), using 7-day moving averages to filter daily noise and highlight clinical degradation trends.

How does it work?
The software architecture operates under a modular, three-layer local paradigm:
* Tabular Inference Layer (Vital Signs): When a crew member's biomarkers are entered, a Random Forest model evaluates the multidimensional input vector and issues a systemic risk classification across three levels: Level 0 (Nominal), Level 1 (Preventive Alert), and Level 2 (Critical State).
* Deep Vision Layer (ViT Pipelines): Upon uploading a medical image (PNG/JPG), the system injects it into image classification pipelines based on Vision Transformer (ViT) architectures. An intermediate logical translator maps the model's raw probabilities (logits/scores) to AstroVital's standardized clinical scale, eliminating false positives through semantic label filtering.
* Causal Correlation Engine & Relational Persistence: Once the visual diagnosis is obtained, the system executes a SQL query indexed by astronaut and mission day on a local SQLite3 database. If the visual model detects a bone anomaly and the database reports a T-Score < -1.0 alongside accumulated_radiation > 150 mSv and sleep deficit (< 5.5 h), the engine automatically drafts a causal clinical hypothesis explaining how radiation and lack of osteoblastic regeneration triggered the lesion.

What benefits does it have?
* Total Medical Autonomy (Zero Latency): Eliminates reliance on Earth during critical emergencies, delivering multimodal diagnoses in under 2 seconds.
* Lightweight CPU Execution (Zero-GPU Dependency): Unlike traditional medical AI systems that require energy-intensive servers with graphics cards, AstroVital was optimized using CPU-only PyTorch binaries. This drastically reduces the environment's footprint and power consumption, making it viable for spacecraft onboard computers.
* Explainable AI (XAI): It does not operate as a "black box"; every alert is accompanied by exact causal factors, violated thresholds, and the correlation between vital signs and imaging, facilitating crew decision-making.
* 100% Offline Operation: Once the model weights are initialized in the local cache, the system operates indefinitely in complete network isolation.

What do you hope to achieve?
Our goal is to establish an open-source standard for space medical operating systems, demonstrating that State-of-the-Art Artificial Intelligence can be deployed on lightweight, austere hardware. We aim for this architecture to serve as a functional prototype for Artemis missions, orbital stations like the Lunar Gateway, and future Martian habitats, as well as having direct terrestrial applications in rural medicine, Antarctic bases, or disaster zones lacking connectivity.

What tools, coding languages, hardware, or software did you use?
* Programming Language: Python 3.
* Frontend & UI: Streamlit, custom HTML5/CSS3 for clinical styling, and streamlit-image-coordinates for interactive spatial anatomical capture.
* Scientific Data Visualization: Altair (declarative multi-layer charts with critical/preventive threshold bands and 7-day moving averages).
* Machine Learning & Data Processing: Scikit-Learn (RandomForestClassifier), Pandas, NumPy, and Joblib for model serialization.
* Deep Learning & Computer Vision: PyTorch (CPU-only compilation for storage and memory optimization), Torchvision, Hugging Face transformers (pipeline), and Pillow (PIL).
* Database & Persistence: Relational SQLite3 (astrovital.db) with normalized tables.
* Hardware & Environment: Developed and validated natively on a Linux environment (Ubuntu 24.04 LTS / Linux Mint) using Visual Studio Code on a Huawei MateBook D14 laptop, empirically proving the smooth performance of Transformer models relying solely on native CPU processing.

How is your project creative?
AstroVital moves beyond standard telemetry dashboards by introducing multimodal Causal AI on the edge. Instead of just displaying a heart rate or flagging a broken bone in isolation, our creative approach lies in the Temporal Causal Correlation Engine. By fusing tabular telemetry with Computer Vision, the system doesn't just state "there is a bone anomaly"; it connects the dots to diagnose "there is a bone anomaly likely exacerbated by a recent drop in T-Score and accumulated radiation exceeding 150 mSv." Furthermore, achieving this by orchestrating Hugging Face Vision Transformers on a CPU-only architecture proves that advanced Deep Learning can be creatively adapted for extreme, resource-constrained environments.

What factors did your team consider?
We heavily considered the physical and operational constraints of deep space exploration:
* Communication Latency: Knowing that a 44-minute round-trip delay to Mars makes real-time Earth support impossible, we engineered the system to be 100% offline.
* Hardware and Energy Limitations: Spacecraft and early Martian habitats cannot rely on energy-hungry GPUs or cloud computing. We explicitly optimized our models (using PyTorch CPU-only binaries) to run natively, smoothly, and efficiently on standard consumer-grade processors (developed and validated on a Huawei laptop running a native Linux OS).
* Medical Validity: We considered the strict standards of space medicine, ensuring our clinical thresholds, dosimetric alerts, and control parameters strictly aligned with the NASA Human Research Program (HRP) guidelines.

## 3. Use of Artificial Intelligence or NASA Data

Artificial Intelligence Implementation:
AstroVital's diagnostic core employs a hybrid AI architecture:
1. Tabular Supervised Learning: We trained a Random Forest ensemble model in Scikit-Learn designed to learn the non-linear decision boundaries between normal physiological wear in microgravity and critical multi-organ decompensations.
2. Transfer Learning with Vision Transformers (ViT): Instead of training convolutional networks from scratch with limited synthetic datasets, we implemented deep vision models pre-trained on tens of thousands of real images via the Hugging Face infrastructure:
  * Hemg/bone-fracture-detection-using-x-rays (ViT architecture) for binary classification of cortical and trabecular integrity in X-rays.
  * google/vit-base-patch16-224 (Vision Transformer pre-trained on ImageNet-21k with 16x16 pixel patches) adapted in our pipeline for the structural inspection of tomographic slices.
  * dima806/facial_emotions_image_detection for the biometric analysis of microexpressions associated with acute fatigue, sadness, or operational stress.

Inspiration from NASA Data and Standards:
AstroVital's biomedical parameters, causal inference rules, and alert thresholds were modeled directly from the scientific literature of the NASA Human Research Program (HRP) and the NASA Spaceflight Human-System Standard (NASA-STD-3001):
* Space Radiation Risks: Implemented monitoring in millisieverts (mSv) with preventive thresholds at 150 mSv and critical limits at 300 mSv accumulated, as well as solar particle storm detection (> 20 mSv/day).
* Microgravity Bone Loss: Tracking bone mineral density T-Score to prevent osteopenia (< -1.0) and accelerated osteoporosis (< -2.5), correlating it with exercise countermeasures (ARED).
* Cephalic Fluid Shift & SANS: Clinical rules linking systolic pressure elevation (> 125 mmHg) and tachycardia (> 85 bpm) with tomographic alterations associated with Spaceflight-Associated Neuro-ocular Syndrome (SANS).
* Healthy Control Parameter: Creation of an ideal physiological reference vector (68 bpm, 115 mmHg, 7.8 h of sleep, T-Score +0.25) to quantify the exact deviation of each crew member.

Compliance Note: In strict adherence to hackathon guidelines, this project does not use, contain, or modify official NASA logos, insignias, flags, or registered trademark identifiers in its interface or demonstration materials.

## 4. References

NASA Bioastronautic Standards and Research:
* NASA Human Research Program (HRP) – Human Research Roadmap (Risks of Space Radiation, Bone Fracture, and Autonomous Medical Care): https://humanresearchroadmap.nasa.gov/
* NASA Spaceflight Human-System Standard (NASA-STD-3001, Volume 1: Crew Health): https://www.nasa.gov/ochmo/health-and-medical/nasa-std-3001/

Open-Source Pre-trained Models (Hugging Face Hub):
* Bone Fracture Detection ViT Model (Hemg/bone-fracture-detection-using-x-rays): https://huggingface.co/Hemg/bone-fracture-detection-using-x-rays
* Base Vision Transformer Model (google/vit-base-patch16-224): https://huggingface.co/google/vit-base-patch16-224
* Facial Emotional Biometrics ViT Model (dima806/facial_emotions_image_detection): https://huggingface.co/dima806/facial_emotions_image_detection

Open-Source Libraries and Frameworks:
* Streamlit Framework: https://streamlit.io/
* PyTorch (CPU-only Linux Binaries): https://pytorch.org/
* Hugging Face Transformers: https://huggingface.co/docs/transformers/
* Scikit-Learn Machine Learning in Python: https://scikit-learn.org/
* Vega-Altair Declarative Visualization: https://altair-viz.github.io/
