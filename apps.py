from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.utils import platform

import speech_recognition as sr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import threading

# Ensure the app has the necessary permissions on iOS
if platform == 'ios':
    from pyobjus import autoclass
    AVAudioSession = autoclass('AVAudioSession')
    audio_session = AVAudioSession.sharedInstance()
    audio_session.setCategory_error_('AVAudioSessionCategoryPlayAndRecord', None)
    audio_session.setActive_error_(True, None)

# Initialize M2M100 model and tokenizer
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Supported languages
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "Korean": "ko"
}

class TranslationApp(BoxLayout):
    def __init__(self, **kwargs):
        super(TranslationApp, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10

        self.source_lang_spinner = Spinner(text='English', values=list(LANGUAGES.keys()))
        self.target_lang_spinner = Spinner(text='Japanese', values=list(LANGUAGES.keys()))
        self.add_widget(self.source_lang_spinner)
        self.add_widget(self.target_lang_spinner)

        self.source_text = TextInput(hint_text='Source Text', multiline=True)
        self.target_text = TextInput(hint_text='Translated Text', multiline=True)
        self.add_widget(self.source_text)
        self.add_widget(self.target_text)

        self.record_button = Button(text='Start Recording', on_press=self.toggle_recording)
        self.add_widget(self.record_button)

        self.is_recording = False
        self.recognizer = sr.Recognizer()
        self.audio_thread = None

    def toggle_recording(self, instance):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.text = 'Stop Recording'
        self.audio_thread = threading.Thread(target=self.audio_processing_loop)
        self.audio_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.text = 'Start Recording'
        if self.audio_thread:
            self.audio_thread.join()

    def audio_processing_loop(self):
        while self.is_recording:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = self.recognizer.recognize_google(audio, language=LANGUAGES[self.source_lang_spinner.text])
                Clock.schedule_once(lambda dt: self.update_source_text(text))
                Clock.schedule_once(lambda dt: self.translate_text())
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

    def update_source_text(self, text):
        self.source_text.text = text

    def translate_text(self):
        source_lang = LANGUAGES[self.source_lang_spinner.text]
        target_lang = LANGUAGES[self.target_lang_spinner.text]
        
        m2m_tokenizer.src_lang = source_lang
        encoded_text = m2m_tokenizer(self.source_text.text, return_tensors="pt")
        
        generated_tokens = m2m_model.generate(
            **encoded_text, 
            forced_bos_token_id=m2m_tokenizer.get_lang_id(target_lang)
        )
        
        translation = m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        self.target_text.text = translation

class TranslationKivyApp(App):
    def build(self):
        return TranslationApp()

if __name__ == '__main__':
    TranslationKivyApp().run()