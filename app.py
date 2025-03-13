import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from joblib import load



model = load("models/disease_prediction_model.joblib")
scaler = load("models/scaler.joblib")


label_mapping = pd.read_csv("models/label_mapping.csv", index_col=0).squeeze().to_dict()
inverse_label_mapping = {v: k for k, v in label_mapping.items()}


df = pd.read_csv("data/processed_data.csv")
trained_features = [col for col in df.columns if col != "Disease"]

medical_advice = {
            
}


st.set_page_config(page_title="Dr. 98.5👨🏻‍⚕️", page_icon="🦾", layout="centered")

st.markdown(
    """
    <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding: 10px 20px;
        }
        .logo {
            font-size: 50px;
            margin-right: 15px;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            color: #003366;
            margin: 0;
        }
        .main-content {
            text-align: center;
            margin-top: 20px;
        }
        .description {
            font-size: 20px;
            color: #003366;
            margin-top: 10px;
        }
        .bmi-info {
            font-size: 18px;
            color: #555;
            margin-top: 15px;
            background-color: #E8F4FC;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
        }
    </style>

    <div class="header-container">
        <span class="logo">🦾🩺</span>
        <h1 class="title">Dr. 98.5</h1>
    </div>

    <div class="main-content">
        <h2>Welcome User <br> To An AI-Powered Medical Diagnosis System</h2>
        <p class="description">
            This system uses Advanced Machine Learning techniques to analyze symptoms and predict possible diseases.<br>
            Also, make use of our <b>BMI Calculator</b> to check if you're in a healthy weight range!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        body {
            background-color: white;
            color: #003366; 
        }
        .stApp {
            background-color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #003366; 
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #00509e; 
        }
        .stSelectbox, .stMultiselect, .stTextInput {
            border: 2px solid #003366;
        }
        .stMarkdown {
            color: #003366;
        }
    </style>
    """,
    unsafe_allow_html=True
)
health_tips = [
    "Chewing gum can boost memory and focus by increasing blood flow to the brain.",
    "Spending time in nature reduces cortisol levels and enhances mental clarity.",
    "Cold showers can increase dopamine by 250%, reducing stress and depression.",
    "Taking deep nasal breaths activates the parasympathetic nervous system, lowering stress.",
    "Writing by hand improves cognitive function and helps process emotions better than typing.",
    "Standing on one leg for 10 seconds daily can help predict overall lifespan and health.",
    "Skipping dinner once a week gives your digestive system a break and promotes longevity.",
    "Drinking black coffee before a workout can increase fat-burning by up to 30%.",
    "Mewing (proper tongue posture) can improve breathing, jawline definition, and prevent sleep apnea.",
    "Walking backward strengthens different leg muscles and improves coordination and brain function.",
    "Bitter foods like dark chocolate and bitter melon stimulate digestion and liver detoxification.",
    "Eating fermented foods daily boosts gut bacteria, improving mood and immunity.",
    "Adding black pepper to turmeric can increase curcumin absorption by 2000%.",
    "Drinking fennel seed water helps with bloating, digestion, and hormonal balance.",
    "Eating your largest meal at lunch aligns with your body’s natural digestive peak for better metabolism.",
    "Wearing blue-light blocking glasses at night can boost melatonin production and improve sleep.",
    "Sleeping on your left side improves digestion, lymphatic drainage, and heart health.",
    "Exposing yourself to morning sunlight within 30 minutes of waking up resets circadian rhythm.",
    "Keeping your room temperature at 18°C (65°F) promotes deeper and more restful sleep.",
    "A 90-minute nap is better than a short one because it allows a full sleep cycle, preventing grogginess.", 
    "Oil pulling with coconut oil for 10-15 minutes daily removes toxins and whitens teeth naturally.",
    "Using a tongue scraper first thing in the morning removes bacteria and improves digestion.",
    "Brushing your teeth before breakfast (not after) protects your enamel from acid erosion.",
    "Chewing sugar-free gum after meals stimulates saliva production, preventing cavities.",
    "Drinking green tea can reduce bad breath by neutralizing sulfur compounds in the mouth.",
    "Jumping (like skipping rope) for 2 minutes daily improves bone density and reduces osteoporosis risk.",
    "Squatting while going to the toilet helps with complete bowel elimination and prevents hemorrhoids.",
    "Swimming is the best exercise for joint health, reducing arthritis and improving flexibility.",
    "Soaking in Epsom salt water reduces muscle soreness and replenishes magnesium levels.",
    "Holding a plank for 60 seconds daily strengthens core muscles and prevents back pain.",
    "The 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds to reduce eye strain.",
    "Consuming leafy greens & orange bell peppers can protect against macular degeneration.",
    "Blinking consciously while using screens prevents dry eyes and digital eye strain.",
    "Rubbing your palms together and placing them over your eyes relaxes eye muscles and reduces stress.",
    "Sitting near a window during the day helps regulate melatonin and improves visual health.",
    "Drinking a glass of water immediately after waking up jumpstarts metabolism and removes toxins.",
    "Listening to 432 Hz or 528 Hz frequency music can enhance mental clarity and relaxation.",
    "Standing for 2-3 minutes every hour can counteract the negative effects of prolonged sitting.",
    "Holding your breath after exhaling fully can train CO2 tolerance, improving stamina and focus.",
    "Doodling while listening enhances memory retention and creativity.",    
    "Sniffing rosemary can improve memory retention by up to 75%.",
    "Eating dark-colored fruits like blueberries and black grapes can protect against cognitive decline.",
    "A 30-second burst of high-intensity exercise before a meal can help stabilize blood sugar levels.",
    "Watching funny videos for 10 minutes lowers cortisol and boosts immune function.",
    "Writing down worries before bed can reduce insomnia and improve sleep quality.",
    "Rolling a frozen bottle under your feet can reduce stress and improve circulation.",
    "Sunlight exposure on bare skin for 10 minutes boosts testosterone in men and vitamin D levels in everyone.",
    "Avoiding tight socks and underwear at night improves blood circulation and hormone balance.",
    "Sitting in a squatting position (Malasana) for a few minutes daily can improve digestion and flexibility.",
    "Humming for 5 minutes daily increases nitric oxide in the sinuses, improving breathing and reducing allergies."
]

random_health_tip = random.choice(health_tips)

st.markdown(
    f"""
    <div style="text-align: center; font-size: 20px; padding: 10px; background-color: #E8F4FC; border-radius: 10px;">
        💡 <b>Today's Tip:</b> {random_health_tip}
    </div>
    """, 
    unsafe_allow_html=True
)
       
st.divider()
medical_advice = {
    "Influenza": "A contagious viral infection causing fever, cough, fatigue, and chills. It spreads through respiratory droplets.  <br>"
                 "🔴 <b>Severity Level:</b> Mild to Moderate (Can be severe in high-risk groups)  <br>"
                 "🟢 <b>Treatable at Home:</b> Yes, unless complications arise  <br>"
                 "Rest well, stay hydrated, and take antiviral medications if prescribed.",

    "COVID-19": "A viral illness with fever, cough, loss of smell, and fatigue. Can cause severe respiratory distress in some cases.  <br>"
                "🔴 <b>Severity Level:</b> Mild to Severe (Can be fatal)  <br>"
                "🟡 <b>Consult Doctor:</b> If symptoms worsen (breathing issues, high fever)  <br>"
                "Isolate, monitor oxygen levels, and seek medical attention if symptoms worsen.",
    
    "Dysentery": "Severe diarrhea with blood and abdominal pain, often caused by bacteria or parasites.  <br>"
                 "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                 "🛑 <b>Consult Doctor Now:</b> Can cause dehydration if untreated  <br>"
                 "Drink ORS solution to prevent dehydration and maintain hygiene.",

    "Typhoid Fever": "Bacterial infection causing high fever, headache, abdominal pain, and vomiting. Spread through contaminated food and water.  <br>"
                     "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                     "🛑 <b>Consult Doctor Now:</b> Requires antibiotic treatment  <br>"
                     "Eat soft foods, take antibiotics, and avoid contaminated water.",

    "Hepatitis B": "A viral infection affecting the liver, causing jaundice, fatigue, and nausea. Can become chronic.  <br>"
                   "🔴 <b>Severity Level:</b> Severe (Chronic cases can lead to liver damage)  <br>"
                   "🛑 <b>Consult Doctor Now:</b> Regular monitoring is necessary  <br>"
                   "Avoid alcohol, take antiviral drugs, and get regular liver check-ups.",

    "Chronic Bronchitis": "A long-term inflammation of the bronchi, leading to cough and mucus production.  <br>"
                          "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                          "🟡 <b>Consult Doctor:</b> If persistent cough or difficulty breathing  <br>"
                          "Avoid smoking, use inhalers, and stay in a pollution-free environment.",

    "Pneumonia": "An infection causing lung inflammation, leading to fever, cough, and breathing difficulty.  <br>"
                 "🔴 <b>Severity Level:</b> Severe  <br>"
                 "🛑 <b>Consult Doctor Now:</b> May require hospitalization  <br>"
                 "Take antibiotics if bacterial, stay hydrated, and use oxygen therapy if needed.",

    "Tuberculosis": "A bacterial infection that primarily affects the lungs, causing cough, weight loss, and night sweats.  <br>"
                    "🔴 <b>Severity Level:</b> Severe (Highly contagious)  <br>"
                    "🛑 <b>Consult Doctor Now:</b> Requires prolonged treatment  <br>"
                    "Take a full course of antibiotics (DOTS therapy) and maintain good nutrition.",

    "Gastroenteritis": "Inflammation of the stomach and intestines, causing vomiting and diarrhea.  <br>"
                       "🔴 <b>Severity Level:</b> Mild to Moderate  <br>"
                       "🟢 <b>Treatable at Home:</b> Yes, unless dehydration occurs  <br>"
                       "Drink plenty of fluids, eat light meals, and avoid dairy and caffeine.",

    "Asthma": "A chronic condition causing airway inflammation, leading to wheezing and breathlessness.  <br>"
              "🔴 <b>Severity Level:</b> Mild to Severe  <br>"
              "🟡 <b>Consult Doctor:</b> If frequent attacks or severe breathlessness  <br>"
              "Use prescribed inhalers, avoid triggers, and practice breathing exercises.",

    "Hypertension": "High blood pressure that increases the risk of heart disease and stroke.  <br>"
                    "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                    "🟡 <b>Consult Doctor:</b> If persistently high readings  <br>"
                    "Maintain a low-sodium diet, exercise regularly, and take prescribed medication.",

    "Diabetes Type 2": "A chronic condition affecting blood sugar regulation, leading to fatigue and frequent urination.  <br>"
                       "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                       "🟡 <b>Consult Doctor:</b> For regular glucose monitoring  <br>"
                       "Follow a healthy diet, exercise, and take insulin or oral medication if needed.",

    "Arthritis": "Inflammation of joints, leading to pain, stiffness, and reduced mobility.  <br>"
                 "🔴 <b>Severity Level:</b> Mild to Severe  <br>"
                 "🟡 <b>Consult Doctor:</b> If persistent pain and swelling  <br>"
                 "Exercise regularly, maintain a healthy weight, and use anti-inflammatory medications.",

    "Migraine": "A neurological condition causing severe headaches, nausea, and sensitivity to light.  <br>"
                "🔴 <b>Severity Level:</b> Mild to Severe  <br>"
                "🟢 <b>Treatable at Home:</b> Yes, unless frequent and debilitating  <br>"
                "Rest in a dark room, stay hydrated, and take prescribed pain relievers.",

    "Anemia": "A condition where there are not enough red blood cells, causing fatigue and weakness.  <br>"
              "🔴 <b>Severity Level:</b> Mild to Moderate  <br>"
              "🟡 <b>Consult Doctor:</b> If persistent fatigue and pale skin  <br>"
              "Increase iron-rich foods, take supplements, and treat underlying causes.",

    "Epilepsy": "A neurological disorder causing recurrent seizures.  <br>"
                "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                "🛑 <b>Consult Doctor Now:</b> Requires long-term management  <br>"
                "Take prescribed anti-seizure medications and avoid triggers.",

    "Meningitis": "Inflammation of brain and spinal cord membranes, causing severe headache and fever.  <br>"
                  "🔴 <b>Severity Level:</b> Severe (Life-threatening)  <br>"
                  "🛑 <b>Consult Doctor Now:</b> Requires urgent treatment  <br>"
                  "Get immediate medical attention and take prescribed antibiotics or antivirals.",

    "Heart Attack": "A medical emergency where blood flow to the heart is blocked, causing chest pain.  <br>"
                    "🔴 <b>Severity Level:</b> Critical  <br>"
                    "🛑 <b>Consult Doctor Now:</b> Call emergency services immediately  <br>"
                    "Chew aspirin (if advised) and seek immediate medical help.",

    "Stroke": "A condition where blood flow to the brain is interrupted, leading to paralysis and speech issues.  <br>"
              "🔴 <b>Severity Level:</b> Critical  <br>"
              "🛑 <b>Consult Doctor Now:</b> Call emergency services immediately  <br>"
              "Get immediate medical attention to reduce brain damage.",

    "Kidney Failure": "Loss of kidney function leading to fluid buildup and waste accumulation.  <br>"
                      "🔴 <b>Severity Level:</b> Severe  <br>"
                      "🛑 <b>Consult Doctor Now:</b> Requires dialysis or transplant  <br>"
                      "Monitor fluid intake, follow a kidney-friendly diet, and take prescribed medications.",
    
    "Common Cold": "A viral infection causing sneezing, sore throat, runny nose, and mild fever. <br>"
                   "🔴 <b>Severity Level:</b> Mild  <br>"
                   "🟢 <b>Treatable at Home:</b> Yes  <br>"
                   "Rest, drink warm fluids, and take over-the-counter medications for relief.",
    
    "Malaria": "A mosquito-borne disease causing high fever, chills, and sweating.  <br>"
               "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
               "🛑 <b>Consult Doctor Now:</b> Can cause complications if untreated  <br>"
               "Take antimalarial medication and prevent mosquito bites.",

    "Dengue": "A viral infection spread by mosquitoes, causing high fever, rash, and severe body pain.  <br>"
              "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
              "🛑 <b>Consult Doctor Now:</b> Risk of internal bleeding and shock  <br>"
              "Stay hydrated, avoid NSAIDs, and monitor platelet count.",

    "Chickenpox": "A contagious viral infection causing itchy blisters, fever, and fatigue.  <br>"
                  "🔴 <b>Severity Level:</b> Mild to Moderate  <br>"
                  "🟢 <b>Treatable at Home:</b> Yes  <br>"
                  "Apply calamine lotion, avoid scratching, and take antihistamines for itching.",

    "Measles": "A highly contagious viral disease causing fever, cough, rash, and conjunctivitis.  <br>"
               "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
               "🛑 <b>Consult Doctor Now:</b> Can cause complications in children  <br>"
               "Get vaccinated, stay hydrated, and take fever-reducing medications.",

    "HIV/AIDS": "A chronic viral infection that weakens the immune system, leading to opportunistic infections.  <br>"
                "🔴 <b>Severity Level:</b> Severe  <br>"
                "🛑 <b>Consult Doctor Now:</b> Requires lifelong management  <br>"
                "Take antiretroviral therapy (ART) and maintain a healthy lifestyle.",

    "Bronchitis": "Inflammation of the bronchial tubes, causing cough with mucus and chest discomfort.  <br>"
                  "🔴 <b>Severity Level:</b> Mild to Moderate  <br>"
                  "🟡 <b>Consult Doctor:</b> If symptoms persist for more than three weeks  <br>"
                  "Avoid smoke, drink warm fluids, and use humidifiers.",

    "Appendicitis": "Inflammation of the appendix causing severe lower right abdominal pain.  <br>"
                    "🔴 <b>Severity Level:</b> Severe  <br>"
                    "🛑 <b>Consult Doctor Now:</b> Requires surgery  <br>"
                    "Seek immediate medical attention for possible appendectomy.",

    "Food Poisoning": "Illness caused by consuming contaminated food, leading to nausea, vomiting, and diarrhea.  <br>"
                      "🔴 <b>Severity Level:</b> Mild to Moderate  <br>"
                      "🟢 <b>Treatable at Home:</b> Yes, unless severe dehydration occurs  <br>"
                      "Drink ORS, rest, and avoid solid foods until symptoms subside.",

    "Pancreatitis": "Inflammation of the pancreas causing severe abdominal pain and digestive issues.  <br>"
                    "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                    "🛑 <b>Consult Doctor Now:</b> Can lead to serious complications  <br>"
                    "Avoid alcohol, follow a low-fat diet, and take prescribed medications.",

    "Liver Cirrhosis": "Chronic liver damage leading to scarring and liver failure.  <br>"
                        "🔴 <b>Severity Level:</b> Severe  <br>"
                        "🛑 <b>Consult Doctor Now:</b> Requires medical supervision  <br>"
                        "Avoid alcohol, maintain a healthy diet, and take prescribed medications.",

    "Gallstones": "Hardened deposits in the gallbladder causing pain and digestive issues.  <br>"
                  "🔴 <b>Severity Level:</b> Mild to Severe  <br>"
                  "🟡 <b>Consult Doctor:</b> If pain is persistent or severe  <br>"
                  "Eat a low-fat diet, stay hydrated, and consider surgery if needed.",

    "Gout": "A form of arthritis caused by uric acid buildup, leading to joint pain and swelling.  <br>"
            "🔴 <b>Severity Level:</b> Mild to Moderate  <br>"
            "🟡 <b>Consult Doctor:</b> If recurrent attacks occur  <br>"
            "Limit purine-rich foods, stay hydrated, and take prescribed medication.",

    "Multiple Sclerosis": "A neurological disorder affecting the brain and spinal cord, leading to muscle weakness and coordination issues.  <br>"
                          "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                          "🛑 <b>Consult Doctor Now:</b> Requires long-term management  <br>"
                          "Follow prescribed therapy, stay active, and manage stress.",

    "Parkinson’s Disease": "A neurodegenerative disorder affecting movement, causing tremors and muscle stiffness.  <br>"
                            "🔴 <b>Severity Level:</b> Moderate to Severe  <br>"
                            "🛑 <b>Consult Doctor Now:</b> Requires medical management  <br>"
                            "Take prescribed medications and engage in physical therapy.",

    "Alzheimer’s Disease": "A progressive brain disorder affecting memory, thinking, and behavior.  <br>"
                            "🔴 <b>Severity Level:</b> Severe  <br>"
                            "🛑 <b>Consult Doctor Now:</b> Requires long-term care  <br>"
                            "Maintain cognitive activities and follow prescribed medications."            
}

st.subheader("Select Symptoms")
selected_symptoms = st.multiselect(
    "Choose from the list:", trained_features, placeholder="Select Symptoms..."
)


def predict_disease(symptoms):
    test_input = pd.DataFrame(np.zeros((1, len(trained_features))), columns=trained_features)
    for symptom in symptoms:
        if symptom in trained_features:
            test_input[symptom] = 1

    test_input_scaled = scaler.transform(test_input)
    probabilities = model.predict_proba(test_input_scaled)[0]

    
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = [(inverse_label_mapping.get(int(i), "Unknown"), probabilities[i]) for i in top_3_indices]


    total_prob = sum(prob[1] for prob in top_3_diseases)
    normalized_diseases = [(disease, (conf / total_prob) * 100) for disease, conf in top_3_diseases]

    return normalized_diseases


if st.button("🔍 Predict Disease"):
    if selected_symptoms:
        with st.spinner("🔄 Analyzing symptoms..."):
            time.sleep(2)

        predictions = predict_disease(selected_symptoms)

        st.success("**Top Possible Diseases:**")
        for i, (disease, confidence) in enumerate(predictions):
            st.markdown(f"<h4 style='text-align:center;'>🔹 {i+1}. {disease} (Confidence: {confidence:.2f}%)</h4>", unsafe_allow_html=True)

            if disease in medical_advice:
                st.markdown(
                    f'<p style="font-size:20px; color:black;">{medical_advice[disease]}</p>',
                    unsafe_allow_html=True
                )

    else:
        st.warning("⚠️ Please select at least one symptom to proceed.")

st.divider()

st.subheader("🧮 BMI Calculator")

height = st.number_input("Enter your height (cm):", min_value=50, max_value=250, step=1)
weight = st.number_input("Enter your weight (kg):", min_value=10, max_value=300, step=1)

if st.button("Calculate"):
    if height and weight:
        bmi = weight / ((height / 100) ** 2)
        st.success(f"**Your BMI: {bmi:.2f}**")

        if bmi < 18.5:
            st.warning("🔹 **Underweight** – Consider a balanced diet with more calories.")
            st.info("Tip: Eat protein-rich foods, whole grains, and healthy fats.")
        elif 18.5 <= bmi < 24.9:
            st.success("✅ **Normal weight** – Keep up the good work!")
            st.info("Tip: Maintain your weight with regular exercise and a balanced diet.")
        elif 25 <= bmi < 29.9:
            st.warning("⚠️ **Overweight** – Consider lifestyle changes.")
            st.info("Tip: Try a mix of cardio and strength training with a calorie-controlled diet.")
        else:
            st.error("🚨 **Obese** – Consult a healthcare provider for guidance.")
            st.info("Tip: Focus on portion control, avoid processed foods, and increase activity.")

        min_ideal_weight = 18.5 * ((height / 100) ** 2)
        max_ideal_weight = 24.9 * ((height / 100) ** 2)
        st.markdown(f"🔹 **Healthy weight range for your height:** {min_ideal_weight:.1f} kg - {max_ideal_weight:.1f} kg")

st.divider()

if st.button("🔄 Refresh All"):
    st.rerun()

st.markdown("<p style='text-align:center;'><font style='font-size:20px'>Powered by Streamlit</font><br><b><font style='font-size:18px'>Ⓒ 2025 Dr.98.5. All Rights Reserved</b></p>", unsafe_allow_html=True)
