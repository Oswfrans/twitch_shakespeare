import keras
import numpy as np

#load model
model = keras.models.load_model('C:\\Users\\o.frans\\Python Scripts\\twitch-master\\model.h5')

#initial_text = 'kappa sir'
#initial_text = [dictionary[c] for c in initial_text.split(' ')]
initial_text= [21, 237, 21, 237, 21, 237, 0, 1, 100,  21, 237, 0, 1, 100,  21, 237, 0, 1, 100, 21]

vocabulary_size = 100000 
GENERATED_LENGTH = 200
SEQ_LENGTH=20

test_text = np.array(initial_text)
generated_text = []

for i in range(200):
    #change input to what the model expects, unsure if current implementation works
    X = np.reshape(test_text, (1, 1, SEQ_LENGTH))

    #why the division??
    next_character = model.predict(X) #(X/float(SEQ_LENGTH))
    
    index = np.argmax(next_character)
    generated_text.append(index) #dictionary[index]
    test_text=np.append(test_text, index)
    #test_text.append(index)
    test_text = test_text[1:]

print(generated_text)
#print(''.join(generated_text))
