import random
import nltk
from nltk.tokenize import word_tokenize


from semantic_analysis import data

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def generate_simple_sentence(gloss):
    #words = gloss.lower().split()
    
    #POS tag the words
    tagged = nltk.pos_tag([gloss])[0]
    word, pos = tagged

    # #Initialize sentence components
    # subject = "I"
    # verb = ""
    # object = ""
    # location = ""


    # #build sentence and ASL gloss
    # english_words = []
    # asl_words = []

    # for word, pos in tagged:
    #     if pos.startswith('V'): #verb
    #         verb = word
    #         english_words.append(verb)
    #         asl_words.append(word.upper())
    #         #sentence.append(f"I {word}")
    #     elif pos.startswith('N'): #Noun
    #         if not object:
    #             object = f"the {word}"
    #             english_words.append(object)
    #         else:
    #             location = f"the {word}"
    #             english_words.append(f"on {location}")
    #         asl_words.append(word.upper())
    #         #sentence.append(f"the {word}")
    #     elif pos.startswith("JJ"):  #adjective
    #         english_words.append(word)
    #         asl_words.append(word.upper())
    #         #sentence.append(word)
    
    # # Construct sentences
    # if verb:
    #     english = f"{subject} {' '.join(english_words)}"
    # else:
    #     english = f"The {' '.join(english_words)}"
    
    # asl = " ".join(asl_words)

    # return english.capitalize(), asl


    if pos.startswith('NN'):  # Noun
        english = f"The {word} is here"
        asl = f"{word.upper()} HERE"
    elif pos.startswith('VB'):  # Verb
        english = f"I {word} now"
        asl = f"I {word.upper()} NOW"
    elif pos.startswith('JJ'):  # Adjective
        english = f"It is {word}"
        asl = f"IT {word.upper()}"
    else:  # Default case
        english = f"This is {word}"
        asl = f"THIS {word.upper()}"

    return english.capitalize(), asl



#Generate parallel data
unique_glosses = set(data.values())
parallel_data = [generate_simple_sentence(word) for word in unique_glosses]

# for gloss in unique_glosses:
#     english = generate_simple_sentence(gloss)
#     parallel_data.append((english, gloss))


#Shuffle the data
random.shuffle(parallel_data)

print(parallel_data)
print("")
with open('parallel.txt', 'w') as f:
    for data in parallel_data:
        f.write(f"{data}\n")

#Split into train and test sets
split = int(0.8 * len(parallel_data))
train_data = parallel_data[:split]
test_data = parallel_data[split:]

print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples.")

#Print examples
print("\nExample parallel data:")
for i in range(5):
    print(f"English: {train_data[i][0]}")
    print(f"ASL Gloss: {train_data[i][1]}")
    print()