import json
import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class MedicalChatbot:
    def __init__(self, remedies_data_path: str):
        """Initialize the chatbot with remedies data"""
        self.disclaimers = [
            "This chatbot provides general information about home remedies only.",
            "It is not a substitute for professional medical advice.",
            "Please consult a healthcare provider for serious symptoms.",
            "In case of emergency, contact emergency services immediately."
        ]
        
        # Load and process remedies data
        self.remedies_data = self._load_remedies_data(remedies_data_path)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._prepare_vectors()

    def _load_remedies_data(self, filepath: str) -> Dict:
        """Load remedies data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _prepare_vectors(self):
        """Prepare TF-IDF vectors for symptom matching"""
        symptoms = [entry['symptoms'] for entry in self.remedies_data['remedies']]
        self.symptoms_vector = self.vectorizer.fit_transform(symptoms)
    
    def _preprocess_input(self, user_input: str) -> str:
        """Clean and standardize user input"""
        text = user_input.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _find_best_match(self, user_input: str) -> Tuple[int, float]:
        """Find best matching remedy based on symptoms"""
        input_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vector, self.symptoms_vector)[0]
        best_match = np.argmax(similarities)
        return best_match, similarities[best_match]
    
    def get_response(self, user_input: str) -> Dict:
        """Generate response based on user input"""
        processed_input = self._preprocess_input(user_input)
        match_index, confidence = self._find_best_match(processed_input)
        
        if confidence < 0.2:
            return {
                'response_type': 'no_match',
                'message': "I'm sorry, I couldn't find a specific remedy for those symptoms. Please consult a healthcare provider.",
                'disclaimers': self.disclaimers
            }
            
        remedy = self.remedies_data['remedies'][match_index]
        return {
            'response_type': 'remedy',
            'symptoms': remedy['symptoms'],
            'remedies': remedy['remedies'],
            'warnings': remedy['warnings'],
            'confidence': float(confidence),
            'disclaimers': self.disclaimers
        }

# Example remedies data structure
example_remedies = {
    "remedies": [
        {
            "symptoms": "headache with tension around temples",
            "remedies": [
                "Apply peppermint oil to temples",
                "Stay hydrated",
                "Practice relaxation techniques",
                "Use cold or warm compress"
            ],
            "warnings": [
                "Seek medical attention if headache is severe or persistent",
                "Don't use peppermint oil if you have sensitive skin"
            ]
        },
        {
            "symptoms": "sore throat, dry cough",
            "remedies": [
            "Gargle with warm salt water three times a day",
            "Drink honey and lemon tea to soothe the throat",
            "Stay hydrated and avoid caffeine"
            ],
            "warnings": [
                "Consult a doctor if symptoms persist for more than a week",
                "Avoid honey for children under 1 year"
            ]
        },
        {
            "symptoms": "headache, mild fever",
            "remedies": [
                "Take rest and avoid screen time",
                "Apply a cold or warm compress to the forehead",
                "Drink plenty of water to stay hydrated"
            ],
            "warnings": [
                "Seek medical attention if fever exceeds 102°F",
                "Avoid caffeine and alcohol"
            ]
        },
        {
            "symptoms": "nausea, upset stomach",
            "remedies": [
                "Sip on ginger tea or chew fresh ginger",
                "Eat small, light meals and avoid greasy food",
                "Stay hydrated with electrolyte drinks or water"
            ],
            "warnings": [
                "Contact a doctor if nausea lasts more than 24 hours",
                "Avoid solid foods until symptoms improve"
            ]
        },
        {
            "symptoms": "nasal congestion, runny nose",
            "remedies": [
                "Use a saline nasal spray to relieve congestion",
                "Inhale steam from hot water to open nasal passages",
                "Drink warm fluids like herbal tea or chicken broth"
            ],
            "warnings": [
                "Avoid decongestants if you have high blood pressure",
                "Seek medical attention if congestion lasts more than a week"
            ]
        },
        {
            "symptoms": "toothache",
            "remedies": [
                "Apply a cold compress to the cheek near the tooth",
                "Rinse with warm salt water to reduce inflammation",
                "Use clove oil as a natural anesthetic"
            ],
            "warnings": [
                "See a dentist if pain persists or worsens",
                "Avoid very hot or cold foods"
            ]
        },
        {
            "symptoms": "muscle pain, stiffness",
            "remedies": [
                "Apply a warm compress to the affected area",
                "Do gentle stretching exercises",
                "Take a warm bath with Epsom salts"
            ],
            "warnings": [
                "Consult a doctor if pain is severe or lasts more than a week",
                "Avoid heavy lifting until pain subsides"
            ]
        },
        {
            "symptoms": "itchy skin, minor rash",
            "remedies": [
                "Apply a cold compress to reduce itching",
                "Use aloe vera gel to soothe the skin",
                "Avoid scratching the affected area"
            ],
            "warnings": [
                "See a doctor if rash spreads or shows signs of infection",
                "Avoid scented lotions and creams"
            ]
        },
        {
            "symptoms": "constipation",
            "remedies": [
                "Drink plenty of water throughout the day",
                "Increase fiber intake with fruits and vegetables",
                "Exercise regularly to stimulate digestion"
            ],
            "warnings": [
                "Seek medical help if constipation lasts more than a week",
                "Avoid overuse of laxatives"
            ]
        },
        {
            "symptoms": "insomnia, trouble sleeping",
            "remedies": [
                "Establish a regular sleep schedule",
                "Avoid caffeine and screen time in the evening",
                "Practice deep breathing or meditation before bed"
            ],
            "warnings": [
                "Consult a doctor if insomnia persists or affects daily life",
                "Avoid alcohol as a sleep aid"
            ]
        },
        {
            "symptoms": "anxiety, restlessness",
            "remedies": [
                "Practice deep breathing exercises",
                "Try mindfulness meditation for relaxation",
                "Take a short walk to clear your mind"
            ],
            "warnings": [
                "Seek professional help if anxiety interferes with daily activities",
                "Avoid caffeine, as it can increase anxiety"
            ]
        },
        {
            "symptoms": "cold hands and feet",
            "remedies": [
                "Wear warm gloves and socks",
                "Drink warm beverages to increase body temperature",
                "Do gentle exercise to improve blood circulation"
            ],
            "warnings": [
                "If the condition is persistent, consult a doctor for circulation issues",
                "Avoid smoking as it can worsen circulation"
            ]
        },
        {
            "symptoms": "lower back pain",
            "remedies": [
                "Apply a hot compress to the lower back",
                "Do gentle stretching exercises",
                "Consider low-impact activities like walking or swimming"
            ],
            "warnings": [
                "Seek medical help if pain is accompanied by numbness",
                "Avoid heavy lifting until pain subsides"
            ]
        },
        {
            "symptoms": "chapped lips",
            "remedies": [
                "Apply a natural lip balm with beeswax or shea butter",
                "Drink plenty of water to stay hydrated",
                "Avoid licking lips, as it can worsen dryness"
            ],
            "warnings": [
                "Consult a dermatologist if condition persists",
                "Avoid lip products with alcohol"
            ]
        },
        {
            "symptoms": "indigestion, bloating",
            "remedies": [
                "Avoid spicy and greasy foods",
                "Eat smaller meals throughout the day",
                "Drink ginger or peppermint tea"
            ],
            "warnings": [
                "Seek medical advice if symptoms last more than a week",
                "Avoid lying down immediately after eating"
            ]
        },
        {
            "symptoms": "sunburn",
            "remedies": [
                "Apply aloe vera gel to soothe the burn",
                "Drink water to stay hydrated",
                "Use a cool compress to relieve pain"
            ],
            "warnings": [
                "Avoid further sun exposure until healed",
                "Consult a doctor if blisters form"
            ]
        },
        {
            "symptoms": "dry eyes",
            "remedies": [
                "Use artificial tears to keep eyes moist",
                "Rest your eyes frequently when using screens",
                "Blink often to spread natural moisture"
            ],
            "warnings": [
                "Consult an eye doctor if condition persists",
                "Avoid air conditioning, which can dry eyes further"
            ]
        },
        {
            "symptoms": "earache",
            "remedies": [
                "Apply a warm compress to the outside of the ear",
                "Use over-the-counter ear drops for pain relief",
                "Gently massage the area around the ear"
            ],
            "warnings": [
                "See a doctor if pain is severe or lasts more than two days",
                "Avoid inserting objects in the ear"
            ]
        },
        {
            "symptoms": "hives, itchy bumps on skin",
            "remedies": [
                "Apply a cold compress to relieve itching",
                "Take an antihistamine if necessary",
                "Avoid hot showers, which can irritate hives"
            ],
            "warnings": [
                "Seek immediate medical help if hives are accompanied by swelling of the lips or face",
                "Avoid scratching, which may worsen the condition"
            ]
        },
        {
            "symptoms": "eye strain from screens",
            "remedies": [
                "Follow the 20-20-20 rule (look 20 feet away every 20 minutes for 20 seconds)",
                "Adjust screen brightness to match the room lighting",
                "Use artificial tears if eyes feel dry"
            ],
            "warnings": [
                "If symptoms persist, consult an eye doctor",
                "Avoid prolonged screen time without breaks"
            ]
        },
        {
            "symptoms": "acid reflux",
            "remedies": [
                "Avoid acidic and spicy foods",
                "Eat smaller meals throughout the day",
                "Elevate your head while sleeping"
            ],
            "warnings": [
                "Consult a doctor if symptoms occur frequently",
                "Avoid lying down immediately after meals"
            ]
        }
    ]
}

# Save example data
with open('remedies_data.json', 'w') as f:
    json.dump(example_remedies, f, indent=2)

# Usage example
def main():
    chatbot = MedicalChatbot('remedies_data.json')
    
    print("Medical Home Remedies Chatbot")
    print("Type 'quit' to exit")
    print("\nDisclaimer:", *chatbot.disclaimers, sep='\n- ')
    
    while True:
        user_input = input("\nPlease describe your symptoms: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.get_response(user_input)
        
        if response['response_type'] == 'remedy':
            print("\nBased on your symptoms, here are some home remedies:")
            print("\nRemedies:", *response['remedies'], sep='\n- ')
            print("\nWarnings:", *response['warnings'], sep='\n- ')
        else:
            print("\n", response['message'])

if __name__ == "__main__":
    main()
