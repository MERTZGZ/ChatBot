import nltk
import numpy as np
import random
import string

# download punkt and wordnet modules
nltk.download('punkt')
nltk.download('wordnet')

# read and lowercase the text
with open('chatbot.txt', 'r', errors='ignore') as f:
    raw = f.read()
    raw = raw.lower()

# tokenize the text into sentences and words
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# print the first two sentences and words
sent_tokens[:2]
word_tokens[:2]

# create a lemmatizer object
lemmer = nltk.stem.WordNetLemmatizer()

# lemmatize tokens
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# lemmatize a string
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# greeting inputs and responses
GREETING_INPUTS = ("greetings", "sup", "what's up", "hey", "hello", "hi")
GREETING_RESPONSES = ["greetings", "sup", "what's up", "hey", "hello", "hi"]

# check for greeting words in a sentence
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# response system
def response(user_response):
    # response sentence
    robo_response = ''
    # add user sentence to the list
    sent_tokens.append(user_response)
    # create tf-idf vector
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='English')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # calculate similarity values
    vals = cosine_similarity(tfidf[-1], tfidf)
    # find the index of the sentence with the highest similarity
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    # if similarity is 0, return a message saying we don't understand
    if (req_tfidf == 0):
        robo_response = robo_response + "I'm sorry, but I'm unable to understand what you're trying to say. " \
                                        "Could you please provide more context or explain your question in a different way?"
        return robo_response
    # if similarity is non-zero, return the sentence with the highest similarity
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

# loop control variable
flag = True


# chatbot welcome message
print("My name is Chatbot and I am here to assist you with any questions you may have about chatbots."
      "If you want to end our conversation, simply type 'bye' and I will stop responding."
      " I am here to help, so please feel free to ask me anything.")

# main loop
while (flag == True):
    # get a sentence from the user
    user_response = input()
    # lowercase the sentence
    user_response = user_response.lower()
    # if the sentence is bye, exit the loop
    if (user_response != 'bye'):
        # if the sentence is thank you or thanks, exit the loop
        if (user_response == 'thank you' or user_response == 'thanks'):
            flag = False
            print("Chatbot: You're welcome.")
        else:
            # if the sentence is a greeting, respond
            if (greeting(user_response) != None):
                print("Chatbot: " + greeting(user_response))
            # if the sentence is not a greeting, use the response system to respond
            else:
                print("Chatbot: ", end="")
                print(response(user_response))
                # remove the user's sentence from the list
                sent_tokens.remove(user_response)
    # if the sentence is bye, exit the loop
    else:
        flag = False
        print("Goodbye! I hope you have a great day."
              " I look forward to speaking with you again in the future. Take care.")


