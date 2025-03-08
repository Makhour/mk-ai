# main.py
import os
import whisper
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from camel_tools.utils.dediac import dediac_ar
from google.cloud import texttospeech
from google.auth.exceptions import DefaultCredentialsError
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# Initialize components
class MKAI:
    def __init__(self):
        # Speech-to-Text (Whisper)
        self.stt_model = whisper.load_model("base")
        
        # Text-to-Speech (Google Cloud with fallback)
        self.tts_client = None
        self.use_google_tts = True
        
        try:
            # Try to initialize Google Cloud TTS
            script_dir = os.path.dirname(__file__)
            cred_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'stoked-door-452101-u4-4339410ca33c.json')
            
            if os.path.exists(cred_path):
                self.tts_client = texttospeech.TextToSpeechClient(
                    client_options={'credentials_file': cred_path}
                )
                print("Google Cloud TTS initialized successfully")
            else:
                self.use_google_tts = False
                print("Google Cloud credentials not found. Using fallback TTS.")
        except Exception as e:
            self.use_google_tts = False
            print(f"Error initializing Google Cloud TTS: {str(e)}. Using fallback TTS.")
        
        # Sentiment Analysis (Arabic)
        self.sentiment_analyzer = pipeline(
            "text-classification", 
            model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
        )
        
        # Dialect Normalization
        self.dialect_map = {
            'egyptian': ['مش', 'هعمل', 'عاوز'],
            'gulf': ['شلونك', 'وينك', 'ابشر'],
            'levantine': ['شو أخبارك', 'مشان الله', 'يما']
        }
        
        # Memory Storage (Simple implementation)
        self.feedback_db = []
        self.emotion_log = []

    def speech_to_text(self, audio_path):
        """Convert Arabic speech to text with dialect detection"""
        result = self.stt_model.transcribe(audio_path)
        text = result["text"]
        
        # Detect dialect
        dialect = self.detect_dialect(text)
        print(f"Detected dialect: {dialect}")
        
        return dediac_ar(text), dialect

    def detect_dialect(self, text):
        """Simple dialect detection based on keywords"""
        for dialect, keywords in self.dialect_map.items():
            if any(keyword in text for keyword in keywords):
                return dialect
        return 'msa'

    def analyze_emotion(self, text):
        """Analyze text emotion using Arabic sentiment analysis"""
        result = self.sentiment_analyzer(text)[0]
        return result['label'], result['score']

    def text_to_speech(self, text, emotion):
        """Convert text to Arabic speech with emotion-appropriate prosody"""
        if self.use_google_tts and self.tts_client:
            try:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="ar-XA",
                    name="ar-XA-Standard-B" if emotion == 'positive' else "ar-XA-Standard-D",
                    ssml_gender=texttospeech.SsmlVoiceGender.MALE
                )
                
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=1.2 if emotion == 'happy' else 1.0,
                    pitch=2 if emotion == 'excited' else 0
                )

                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )

                with open("response.mp3", "wb") as out:
                    out.write(response.audio_content)
                return "response.mp3"
            except Exception as e:
                print(f"Google TTS failed: {str(e)}. Using fallback TTS.")
                self.use_google_tts = False
                
        # Fallback TTS using gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='ar')
            tts.save("response.mp3")
            return "response.mp3"
        except Exception as e:
            print(f"Fallback TTS failed: {str(e)}")
            return None

    def generate_response(self, text, emotion):
        """Generate appropriate response based on emotion and context"""
        responses = {
            'positive': [
                "أمر رائع! هل تريد المزيد من المساعدة؟",
                "هذا ممتع! ماذا تريد أن تفعل بعد ذلك؟"
            ],
            'negative': [
                "أنا آسف لسماع ذلك. هل تريد التحدث أكثر؟",
                "يبدو أن هذا صعب. كيف يمكنني المساعدة؟"
            ],
            'neutral': [
                "فهمت. ماذا تريد أن تفعل الآن؟",
                "حسنًا، ما الخطوة التالية؟"
            ]
        }
        
        # Select response based on emotion
        return responses[emotion][0]

    def save_feedback(self, user_input, bot_response, correction):
        """Store user feedback for later training"""
        self.feedback_db.append({
            'timestamp': datetime.now(),
            'input': user_input,
            'response': bot_response,
            'correction': correction
        })

    def record_audio(self):
        """Record audio from microphone and save to file"""
        try:
            fs = 44100  # Sample rate
            duration = 5  # Recording duration in seconds
            
            print("Recording...")
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished
            print("Recording complete")
            
            audio_path = "input.wav"
            sf.write(audio_path, audio, fs)
            return audio_path
        except Exception as e:
            print(f"Error recording audio: {str(e)}")
            print("Please ensure your microphone is connected and permissions are granted.")
            return None

    def get_text_input(self):
        """Get text input from user"""
        while True:
            try:
                text = input("\nEnter Arabic text (or 'exit' to quit): ")
                if text.lower() == 'exit':
                    return None, None
                if text.strip():
                    return text, 'msa'  # Return text with MSA as default dialect
                print("Please enter some text.")
            except KeyboardInterrupt:
                return None, None

# Main execution
if __name__ == "__main__":
    bot = MKAI()
    
    try:
        while True:
            # Get user input - try audio first, fallback to text
            audio_path = bot.record_audio()
            if audio_path:
                # Convert speech to text
                text, dialect = bot.speech_to_text(audio_path)
                print(f"User (voice): {text} ({dialect})")
            else:
                # Fallback to text input
                text, dialect = bot.get_text_input()
                if text is None:  # User wants to exit
                    break
                print(f"User (text): {text} ({dialect})")
            
            # Analyze emotion
            emotion, confidence = bot.analyze_emotion(text)
            print(f"Detected emotion: {emotion} ({confidence:.2f})")
            
            # Generate response
            response = bot.generate_response(text, emotion)
            print(f"Bot: {response}")
            
            # Convert response to speech
            response_audio = bot.text_to_speech(response, emotion)
            
            # Play response
            audio, fs = sf.read(response_audio)
            sd.play(audio, fs)
            sd.wait()
            
            # Get feedback
            feedback = input("Was this response good? (y/n/correct): ").lower()
            if feedback != 'y':
                correction = input("Enter correct response: ")
                bot.save_feedback(text, response, correction)
                
    except KeyboardInterrupt:
        print("\nTraining with feedback data...")
        # Here you would add code to retrain the model with feedback_db
        print(f"Saved {len(bot.feedback_db)} feedback entries for training")
