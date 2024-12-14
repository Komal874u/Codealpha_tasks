import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, Activation # type: ignore
from keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder
import glob

# Step 1: Load and Preprocess MIDI Data
def load_midi_data():
    notes = []
    # Load all MIDI files from a directory
    for file in glob.glob("midi_songs/*.mid"):  # Change the folder path as needed
        midi = converter.parse(file)
        print(f"Parsing {file}")

        # Extract all notes and chords
        for element in midi.flat.notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Step 2: Prepare Data for the Model
def prepare_sequences(notes, sequence_length=50):
    # Encode notes to integers
    le = LabelEncoder()
    encoded_notes = le.fit_transform(notes)

    # Prepare input and output sequences
    input_sequences = []
    output_sequences = []

    for i in range(len(encoded_notes) - sequence_length):
        input_sequences.append(encoded_notes[i:i+sequence_length])
        output_sequences.append(encoded_notes[i+sequence_length])

    # Reshape input for LSTM and one-hot encode output
    X = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1)) / float(len(set(encoded_notes)))
    y = to_categorical(output_sequences, num_classes=len(set(encoded_notes)))
    
    return X, y, le

# Step 3: Build the RNN Model
def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 4: Generate Music
def generate_music(model, le, sequence_length=50, num_notes=500):
    start_index = np.random.randint(0, len(X) - 1)  # Random starting point
    pattern = X[start_index]
    generated_notes = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction = model.predict(prediction_input, verbose=0)
        note_index = np.argmax(prediction)
        note = le.inverse_transform([note_index])[0]
        generated_notes.append(note)

        # Update the pattern
        pattern = np.append(pattern, note_index / float(len(set(le.classes_))))
        pattern = pattern[1:]

    return generated_notes

# Step 5: Convert Notes to MIDI File
def create_midi(generated_notes, output_file="generated_music.mid"):
    output_stream = stream.Stream()
    for pattern in generated_notes:
        if '.' in pattern:  # Chord
            chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
            new_chord = chord.Chord(chord_notes)
            output_stream.append(new_chord)
        else:  # Note
            new_note = note.Note(pattern)
            output_stream.append(new_note)

    output_stream.write('midi', fp=output_file)
    print(f"Generated music saved to {output_file}")

# Main Execution
if __name__== "_main_":
    print("Loading and processing MIDI data...")
    notes = load_midi_data()
    X, y, le = prepare_sequences(notes)

    print("Building and training the model...")
    model = build_model(X.shape[1:], y.shape[1])
    model.fit(X, y, epochs=50, batch_size=64)

    print("Generating music...")
    generated_notes = generate_music(model, le)
    create_midi(generated_notes)
