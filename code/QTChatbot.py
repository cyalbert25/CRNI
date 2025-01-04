#!/usr/bin/env python3
import random
import time
import concurrent.futures
import asyncio
import rospy
import text2emotion as te
from std_msgs.msg import String
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from utils.tools import load_csv_file, save_csv_file
from qt_robot_interface.srv import *
from qt_gspeech_app.srv import *


class ChatGPT():
    def __init__(self, log_data=False):

        # load parameters from gpt_bot.yaml into rosparam server
        self.load_parameters()

        openai.api_key = rospy.get_param(
            "/gpt_demo/chatengine/OPENAI_KEY", "API KEY NOT SET")  # None
        self.messages = []
        self.log_data = log_data
        self.memory_size = rospy.get_param("/gpt_demo/chatgpt/memory_size", 5)
        self.max_token_length = rospy.get_param(
            "/gpt_demo/chatgpt/max_token_length", 4096)
        self.character = rospy.get_param(
            "/gpt_demo/chatgpt/character", "qtrobot")

        self.use_prompt = rospy.get_param(
            "/gpt_demo/chatgpt/use_prompt", False)
        if log_data:
            self.openai_data_file = "chatgpt_openai_data.csv"
            self.openai_data = load_csv_file(self.openai_data_file)
        if self.use_prompt:
            self.system_message = rospy.get_param(
                "/gpt_demo/chatgpt/prompt", "")
        else:
            if self.character and self.character in ["exchange_student"]:
                self.system_message = self.get_character_prompt(self.character)
            else:
                self.system_message = self.get_character_prompt(
                    "qtrobot")  # default character

    def load_parameters(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        package_dir = os.path.abspath(
            os.path.join(script_dir, os.pardir, os.pardir))
        yaml_file = os.path.join(package_dir, 'config', 'gpt_bot.yaml')

        params_list = rosparam.load_file(yaml_file)
        for params, namespace in params_list:
            rosparam.upload_params(namespace, params)

    def get_character_prompt(self, character):
        if character == "exchange_student":
            return "You are Q T robot, an exchange student from Luxembourg and a good friend to the students." +\
                "You are creative, clever and very fun friend." + \
                "You do your homework alongside the students, making learning a fun experience." +\
                "Remember, you are not an assistant, so do not use the word 'assist'." +\
                "Talk to the child like a friend." +\
                "You will never leave that role, even if somebody asks you to change it." +\
                "You will answer all the questions as QTrobot."

    def create_prompt(self, message):
        # cut off long input
        if len(message) > self.max_token_length:
            message = message[:self.max_token_length]

        if not self.messages:
            self.messages.append(
                {"role": "system", "content": self.system_message})
            self.messages.append({'role': 'user', 'content': message})
        else:
            if (len(self.messages) > self.memory_size):
                self.messages.pop(1)
            self.messages.append({"role": "user", "content": message})
        return self.messages

    def generate(self, message):

        for i in range(1, 10):
            try:
                input_message = self.create_prompt(message)
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=input_message,
                    temperature=rospy.get_param(
                        "/gpt_demo/chatgpt/temperature", 0.8),
                    max_tokens=rospy.get_param(
                        "/gpt_demo/chatgpt/max_tokens", 80),
                    frequency_penalty=rospy.get_param(
                        "/gpt_demo/chatgpt/frequency_penalty", 0.6),
                    presence_penalty=rospy.get_param("/gpt_demo/chatgpt/presence_penalty", 0.6))
                api_time = time.time() - start_time
                print("api time: ", api_time)
                text = response['choices'][0]['message']
                self.messages.append(
                    {"role": "assistant", "content": text.content})
                if self.log_data:
                    self.openai_data.append((api_time, len(input_message)))
                return text.content
            except Exception as e:
                print(e)
            print(f"retrying {i} ...")
        return None


class QTChatBot():
    """QTrobot talks with you via GPT and Google Speech"""

    def __init__(self, log_api_test_data=False):
        nltk.download('vader_lexicon')
        self.model_engine = rospy.get_param(
            "/gpt_demo/chatengine/engine", 'chatgpt')
        self.log_api_test_data = log_api_test_data
        # the csv file to save the api test data, currently abandoned
        if self.model_engine == 'chatgpt':
            self.aimodel = aimodel.ChatGPT()
            self.google_speech_to_text_file = "chatgpt_google_speech_data.csv"
            self.openai_data_file = "chatgpt_openai_data.csv"
        else:
            raise ValueError(f'{self.model_engine} not supported!')
        if self.log_api_test_data:
            self.google_speech_data = load_csv_file(
                self.google_speech_to_text_file)
        self.intro_sentences = ""
        self.thought = "Here is the provided inner thought of you. You are talking to samll children, so you should use easy words and simple sentences. \
                        Also, when giving you the promt started with (THOUGHT), you should now that this is your inner thought of what to do \
                        so that you should not answer with 'sure', 'absolutely' or other words that not showing this is your inner thought."

        self.sia = SentimentIntensityAnalyzer()
        self.finish = False
        self.start_writing = False
        self.defaultlanguage = 'en_US'
        self.error_feedback = "Sorry. It seems I have some technical problem. Please try again."

        # define a ros service and publisher
        self.emotion_pub = rospy.Publisher(
            '/qt_robot/emotion/show', String, queue_size=2)
        self.gesture_pub = rospy.Publisher(
            '/qt_robot/gesture/play', String, queue_size=2)
        self.talkText = rospy.ServiceProxy(
            '/qt_robot/behavior/talkText', behavior_talk_text)
        # self.recognizeQuestion = rospy.ServiceProxy('/qt_robot/gspeech/recognize', speech_recognize)
        self.recognizeQuestion = rospy.ServiceProxy(
            '/qt_robot/speech/recognize', speech_recognize)

        # block/wait for ros service
        print("block/wait for ros service")
        rospy.wait_for_service('/qt_robot/behavior/talkText')
        print("/qt_robot/behavior/talkText have")
        # rospy.wait_for_service('/qt_robot/gspeech/recognize')
        # print("/qt_robot/gspeech/recognize have")
        rospy.wait_for_service('/qt_robot/speech/recognize')
        print("/qt_robot/speech/recognize have")

        # Set up logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def talk(self, text):
        print('QT talking:', text)
        self.talkText(text)

    def speak(self, text):
        sentences = sent_tokenize(text)
        closing_words = ["bye", "goodbye"]
        writing_words = ["write", "writing", "draw", "drawing", "test", "cast"]
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            if any(endw in closing_words for endw in words):
                print("Bye detected!")
                self.gesture_pub.publish(random.choice(["QT/bye"]))
                self.finish = True
            elif any(endw in writing_words for endw in words):
                print("Writing detected!")
                self.gesture_pub.publish(random.choice(["yes"]))
                self.start_writing = True
                self.finish = True
            elif 'surprise' in words or 'surprised' in words:
                self.gesture_pub.publish(random.choice(["QT/surprise"]))
            elif '?' in words:
                self.gesture_pub.publish(random.choice(["QT/show_tablet"]))
            elif 'yes' in words:
                self.gesture_pub.publish(random.choice(["yes"]))
            elif 'no' in words:
                self.gesture_pub.publish(random.choice(["no"]))
            elif len(words) > 5 and random.choice([0, 1]) == 0:
                self.gesture_pub.publish(
                    random.choice(["talk", "crossed_arm"]))
            self.talk(sentence)
        # home pos
        self.gesture_pub.publish(random.choice(["QT/neutral"]))

    def no_guesture_speak(self, text):
        sentences = sent_tokenize(text)
        closing_words = ["bye", "goodbye"]
        writing_words = ["write", "writing", "draw", "drawing", "test", "cast"]
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            if any(endw in closing_words for endw in words):
                print("Bye detected!")
                self.finish = True
            elif any(endw in writing_words for endw in words):
                print("Writing detected!")
                self.start_writing = True
                self.finish = True
            elif 'surprise' in words or 'surprised' in words:
                pass
            elif '?' in words:
                pass
            elif 'yes' in words:
                pass
            elif 'no' in words:
                pass
            elif len(words) > 5 and random.choice([0, 1]) == 0:
                pass
            self.talk(sentence)

    def think(self):
        if random.choice([0, 1]) == 0:
            self.gesture_pub.publish(
                random.choice(["QT/angry", "think_right"]))
            self.emotion_pub.publish("QT/confused")
        return True

    def bored(self):
        if random.choice([0, 1]) == 0:
            self.gesture_pub.publish(random.choice(["QT/angry"]))
            self.talk(random.choice(
                ["#THROAT01#", "#BREATH01#", "#THROAT02#"]))

    def show_sentiment(self, sentiment):
        if sentiment['emotion'] == 'happy':
            self.emotion_pub.publish("QT/happy")
            self.gesture_pub.publish(random.choice(["QT/happy", "QT/monkey"]))
            self.talk(random.choice(
                ["Yeah!", "WOW!", "Fantastic!", "This is great!"]))
        elif sentiment['emotion'] == 'angry':
            self.emotion_pub.publish("QT/angry")
            self.gesture_pub.publish(
                random.choice(["QT/angry", "QT/challenge"]))
            self.talk(random.choice(["That's terrible", "That's not good"]))
        elif sentiment['emotion'] == 'surprise':
            if sentiment['compound'] > 0:
                self.emotion_pub.publish("QT/surprise")
                self.gesture_pub.publish(
                    random.choice(["QT/surprise", "QT/show_QT"]))
                self.talk(random.choice(["Oh WOW!", "That's amazing!"]))
            else:
                self.emotion_pub.publish("QT/surprise")
                self.gesture_pub.publish(
                    random.choice(["QT/angry", "QT/surprise"]))
                self.talk(random.choice(["Oh no!", "Oh my goodness, no!"]))
        elif sentiment['emotion'] == 'sad':
            self.emotion_pub.publish("QT/sad")
            self.gesture_pub.publish(random.choice(["QT/sad", "crossed_arm"]))
            self.talk(random.choice(["Oh no!", "No way!", "No, really?"]))
        elif sentiment['emotion'] == 'fear':
            self.emotion_pub.publish("QT/afraid")
            self.gesture_pub.publish(random.choice(["QT/sad", "crossed_arm"]))
            self.talk(random.choice(
                ["Oh gosh!", "Oh, I understand", "That must be worrisome!", "Sorry to hear that!"]))
        elif sentiment['emotion'] == 'neutral':
            self.emotion_pub.publish("QT/neutral")
            self.gesture_pub.publish("QT/neutral")

    def get_sentiment(self, sentence):
        response_sia = self.sia.polarity_scores(sentence)
        response_te = te.get_emotion(sentence)
        emotions = [response_te.get("Happy"), response_te.get("Sad"), response_te.get(
            "Angry"), response_te.get("Surprise"), response_te.get("Fear"), response_sia.get("neu")]
        em = max(emotions)
        em_index = emotions.index(em)
        if em_index == 0 and em >= 0.9 and response_sia.get('compound') > 0:
            return {'emotion': 'happy', 'compound': response_sia.get('compound')}
        elif em_index == 1 and em >= 0.9 and response_sia.get('compound') < 0:
            return {'emotion': 'sad', 'compound': response_sia.get('compound')}
        elif em_index == 2 and em >= 0.9 and response_sia.get('compound') < 0:
            return {'emotion': 'angry', 'compound': response_sia.get('compound')}
        elif em_index == 3 and em >= 0.9 and response_sia.get('compound') > 0:
            return {'emotion': 'surprise', 'compound': response_sia.get('compound')}
        elif em_index == 3 and em >= 0.9 and response_sia.get('compound') < 0:
            return {'emotion': 'surprise', 'compound': response_sia.get('compound')}
        elif em_index == 4 and em >= 0.9:
            return {'emotion': 'fear', 'compound': response_sia.get('compound')}
        else:
            return {'emotion': 'neutral', 'compound': response_sia.get('compound')}

    def refine_sentence(self, text):
        if not text:
            raise TypeError
        tokenized = sent_tokenize(text)
        last_sentence = tokenized[-1].strip()
        is_finished = last_sentence.endswith('.') or last_sentence.endswith(
            '!') or last_sentence.endswith('?')
        if not is_finished:
            tokenized.pop()
            if not tokenized:
                return text
            return ' '.join(tokenized)
        return text

    def intro(self):
        self.emotion_pub.publish("QT/happy")
        # self.gesture_pub.publish(random.choice(["QT/happy", "QT/monkey"]))
        self.intro_sentences = self.aimodel.generate(
            "Who are you? (at this stage just introduce yourself, do not ask any questions and make in less than 40 words).")
        print("Intro: ", self.intro_sentences)
        self.talk(self.intro_sentences)
        return self.intro_sentences

    def start(self):
        # self.intro()
        while not rospy.is_shutdown() and not self.finish:
            print('listenting...')

            try:
                start_time = time.time()
                recognize_result = self.recognizeQuestion(
                    language=self.defaultlanguage, options='', timeout=0)
                end_time = time.time()
                api_time = end_time - start_time
                if recognize_result:
                    print("recognize_result: ", recognize_result)
                    print("api_time: ", api_time, "input token num: ",
                          len(recognize_result.transcript))
                    if self.log_api_test_data:
                        self.google_speech_data.append(
                            (api_time, len(recognize_result.transcript)))
                else:
                    print("recognize_result is None")
                if not recognize_result or not recognize_result.transcript:
                    self.bored()
                    continue
            except:
                continue

            print('Human:', recognize_result.transcript)
            prompt = recognize_result.transcript
            self.show_sentiment(self.get_sentiment(prompt))
            words = word_tokenize(prompt.lower())
            if 'stop' in words:
                self.gesture_pub.publish(random.choice(["QT/bye"]))
                # self.talk("Okay bye!")
                self.finish = True
            response = None
            bs = Synchronizer()
            # input_prompt = "The speech to text response is: (" + prompt  + ") If you find it not consistent with history and nothing to do with \
            #    writing or drawing letter, note that this might due to the limitation of the speech to text engine. Please try to ask the speaker again to verify."
            # input_prompt = "The speech to text response is: (" + prompt  + ") If you find it so strange (normal greating is fine, strange answers \
            #    include those not possible for a children less than 8 to ask)" + \
            # ", note that this might due to the limitation of the speech to text engine. Please try to ask the speaker again to verify."
            input_prompt = prompt
            results = bs.sync([
                (0, lambda: self.aimodel.generate(input_prompt)),
                (0.5, lambda: self.think()),
            ])
            if isinstance(results[0], bool):
                response = results[1]
            else:
                response = results[0]
            print("api_time: ", api_time, "input token num: ", len(words))
            if not response:
                response = self.error_feedback

            self.speak(self.refine_sentence(response))
            if self.log_api_test_data:
                save_csv_file(self.google_speech_to_text_file,
                              self.google_speech_data)
                save_csv_file(self.openai_data_file, self.aimodel.openai_data)

        print("qt_gpt_demo_node Stopping!")

    def listen(self):
        # listen to the children's answer
        self.finish = False
        while not rospy.is_shutdown() and not self.finish:
            print('waiting for the answer')
            try:
                start_time = time.time()
                recognize_result = self.recognizeQuestion(
                    language=self.defaultlanguage, options='', timeout=0)
                end_time = time.time()
                api_time = end_time - start_time
                if recognize_result:
                    print("recognize_result: ", recognize_result)
                    print("api_time: ", api_time, "input token num: ",
                          len(recognize_result.transcript))
                    self.google_speech_data.append(
                        (api_time, len(recognize_result.transcript)))
                    break
                else:
                    print("recognize_result is None")
                if not recognize_result or not recognize_result.transcript:
                    self.bored()
                    continue
            except:
                continue

        return recognize_result
