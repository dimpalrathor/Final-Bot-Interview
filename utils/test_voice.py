# test_voice.py
import pyttsx3
import speech_recognition as sr
import sys
import os

# Add the utils directory to Python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_tts():
    """Test Text-to-Speech functionality"""
    print("=" * 50)
    print("TESTING TEXT-TO-SPEECH (TTS) SYSTEM")
    print("=" * 50)
    
    try:
        # Try to initialize TTS engine
        print("🔄 Initializing TTS engine...")
        engine = pyttsx3.init()
        print("✅ pyttsx3 initialized successfully")
        
        # List available voices
        voices = engine.getProperty('voices')
        print(f"✅ Found {len(voices)} voice(s):")
        for i, voice in enumerate(voices):
            print(f"   {i}: {voice.name} (ID: {voice.id})")
        
        # Configure engine
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        
        # Try to set a voice
        if voices:
            engine.setProperty('voice', voices[0].id)
            print(f"🎙️ Using voice: {voices[0].name}")
        
        # Test speaking
        print("🔊 Speaking test message...")
        test_message = "Hello, this is a voice test. If you can hear this, TTS is working properly."
        engine.say(test_message)
        engine.runAndWait()
        print("✅ TTS test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if speakers are connected and not muted")
        print("   2. Try installing: pip install --force-reinstall pyttsx3")
        print("   3. On Windows, try: pip install comtypes pywin32")
        print("   4. Check system audio settings")
        return False

def test_microphone():
    """Test Speech Recognition (microphone) functionality"""
    print("\n" + "=" * 50)
    print("TESTING MICROPHONE & SPEECH RECOGNITION")
    print("=" * 50)
    
    try:
        # List available microphones
        print("🔍 Looking for microphones...")
        try:
            mics = sr.Microphone.list_microphone_names()
            print(f"✅ Found {len(mics)} microphone(s):")
            for i, mic in enumerate(mics):
                print(f"   {i}: {mic}")
        except:
            print("⚠️ Could not list microphones")
        
        # Initialize microphone
        print("🔄 Initializing microphone...")
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Calibrate for ambient noise
        with microphone as source:
            print("🔊 Calibrating microphone for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Microphone calibrated")
        
        print("🎤 Microphone test passed - ready for speech recognition")
        return True
        
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if microphone is connected and not muted")
        print("   2. Try: pip install --force-reinstall pyaudio")
        print("   3. On Windows, you may need to install Microsoft Visual C++ Redistributable")
        print("   4. Check system microphone permissions")
        return False

def test_audio_utils():
    """Test the custom AudioHandler class"""
    print("\n" + "=" * 50)
    print("TESTING CUSTOM AUDIO UTILS")
    print("=" * 50)
    
    try:
        from audio_utils import AudioHandler
        
        print("🔄 Creating AudioHandler instance...")
        audio = AudioHandler()
        
        # Get status
        status = audio.get_voice_status()
        print(f"📊 Audio System Status:")
        print(f"   TTS Available: {'✅' if status['tts_available'] else '❌'}")
        print(f"   Microphone Available: {'✅' if status['microphone_available'] else '❌'}")
        
        # Test TTS if available
        if status['tts_available']:
            print("🔊 Testing AudioHandler TTS...")
            success = audio.speak_blocking("Audio handler test message.")
            print(f"   TTS Test: {'✅ Success' if success else '❌ Failed'}")
        
        audio.stop_all()
        print("✅ Audio utils test completed")
        return True
        
    except Exception as e:
        print(f"❌ Audio utils test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 AI Interview Bot - Voice System Diagnostics")
    print("This will test your voice setup step by step...\n")
    
    # Run tests
    tts_ok = test_tts()
    mic_ok = test_microphone()
    utils_ok = test_audio_utils()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Text-to-Speech: {'✅ WORKING' if tts_ok else '❌ BROKEN'}")
    print(f"Microphone: {'✅ WORKING' if mic_ok else '❌ BROKEN'}")
    print(f"Audio Utils: {'✅ WORKING' if utils_ok else '❌ BROKEN'}")
    
    if tts_ok and mic_ok:
        print("\n🎉 All voice systems are working! You can run the interview bot.")
    else:
        print("\n🔧 Some voice components need attention. Check the troubleshooting tips above.")
    
    input("\nPress Enter to exit...")